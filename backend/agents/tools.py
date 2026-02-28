"""Agent tools for the AI tutoring platform.

Provides BaseTool classes for:
- Professor turn response scaffolding (strict JSON schema)
- RAG knowledge-base retrieval and document management

Tools
-----
ProfessorRespondTool     - Professor turn response generation scaffold
RetrieveContextTool      - Semantic search over Bedrock Knowledge Base
UploadDocumentTool       - PDF/slide upload to S3 + KB ingestion trigger
CheckIngestionStatusTool - Poll ingestion job progress
ListDocumentsTool        - List available documents (metadata only)
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from typing import Any, Union

import boto3
from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_core.tools import BaseTool

from db.models import (
    Citation,
    ProfessorMode,
    ProfessorNextAction,
    ProfessorTurnRequest,
    ProfessorTurnResponse,
    ProfessorTurnStrategy,
)
from utils.helpers import parse_tool_input

logger = logging.getLogger(__name__)

PROFESSOR_CONFIG_PATH = "agents.professor"


# ---------------------------------------------------------------------------
# 0. ProfessorRespondTool
# ---------------------------------------------------------------------------

class ProfessorRespondTool(BaseTool):
    """Professor tutoring tool with strict request/response schema handling."""

    name: str = "professor_respond"
    description: str = (
        "Generate one tutoring response turn for the Professor agent.\n"
        "\n"
        "INPUT (JSON object):\n"
        "  session_id (str, REQUIRED)\n"
        "  message    (str, REQUIRED)\n"
        "  topic      (str, REQUIRED)\n"
        "  mode       (str, optional) - strict | convenience (default: strict)\n"
        "  profile    (object, REQUIRED) - level/learning_style/pace\n"
        "\n"
        "OUTPUT (JSON object):\n"
        "  assistant_response (str)\n"
        "  strategy           (str)\n"
        "  revealed_final_answer (false)\n"
        "  next_action        (str)\n"
        "  citations          (list)\n"
        "\n"
        "Always returns strict JSON matching ProfessorTurnResponse schema."
    )

    professor_config: Any = None
    citations_enabled: bool = True
    socratic_default: bool = True

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.professor_config = self._load_professor_runtime_config(config)
        tutoring_cfg = self.professor_config.get("tutoring", {})
        self.citations_enabled = bool(tutoring_cfg.get("citations_enabled", True))
        self.socratic_default = bool(tutoring_cfg.get("socratic_default", True))

    @staticmethod
    def _load_professor_runtime_config(config: Any) -> dict[str, Any]:
        professor_config = config.get(PROFESSOR_CONFIG_PATH)
        if not isinstance(professor_config, dict):
            raise ValueError(
                f"Missing or invalid config mapping at '{PROFESSOR_CONFIG_PATH}'"
            )

        llm_cfg = professor_config.get("llm")
        tutoring_cfg = professor_config.get("tutoring")
        if not isinstance(llm_cfg, dict):
            raise ValueError("Missing or invalid config mapping at 'agents.professor.llm'")
        if not isinstance(tutoring_cfg, dict):
            raise ValueError("Missing or invalid config mapping at 'agents.professor.tutoring'")
        if not llm_cfg.get("provider"):
            raise ValueError("Missing required config key: 'agents.professor.llm.provider'")
        if not llm_cfg.get("model_id"):
            raise ValueError("Missing required config key: 'agents.professor.llm.model_id'")
        return professor_config

    @staticmethod
    def map_professor_mode_to_rag_mode(mode: ProfessorMode) -> str:
        """Normalize Professor mode to the equivalent RAG mode."""
        if mode is ProfessorMode.STRICT:
            return "internal_only"
        return "external_ok"

    @staticmethod
    def sanitize_for_log(request: ProfessorTurnRequest) -> dict[str, Any]:
        """Return metadata-only logs (no raw student message)."""
        return {
            "session_id": request.session_id,
            "mode": request.mode.value,
            "rag_mode": ProfessorRespondTool.map_professor_mode_to_rag_mode(request.mode),
            "topic": request.topic,
            "message_length": len(request.student_message),
            "profile_level": request.profile.level,
        }

    @staticmethod
    def get_professor_json_schemas() -> dict[str, dict[str, Any]]:
        """Expose strict transport schemas for validation in caller layers."""
        return {
            "ProfessorTurnRequest": ProfessorTurnRequest.model_json_schema(),
            "ProfessorTurnResponse": ProfessorTurnResponse.model_json_schema(),
        }

    def _retrieve_citations(self, request: ProfessorTurnRequest) -> list[Citation]:
        """Return citations from RAG when wired; empty list for now."""
        if not self.citations_enabled:
            return []
        # Do not fabricate citations. Until RAG retrieval is connected,
        # return an empty list instead of synthetic sources.
        _ = self.map_professor_mode_to_rag_mode(request.mode)
        return []

    def _build_response(self, request: ProfessorTurnRequest) -> ProfessorTurnResponse:
        citations = self._retrieve_citations(request)
        if self.socratic_default:
            strategy = ProfessorTurnStrategy.SOCRATIC_QUESTION
            response_text = (
                f"Before we solve it, can you explain in one sentence what the key idea in "
                f"{request.topic} is?"
            )
        else:
            strategy = ProfessorTurnStrategy.CONCEPT_EXPLAIN
            response_text = (
                f"Let's do a short recap: in {request.topic}, focus on the core principle "
                "first, then we can apply it to your problem."
            )

        return ProfessorTurnResponse(
            assistant_response=response_text,
            strategy=strategy,
            revealed_final_answer=False,
            next_action=ProfessorNextAction.CONTINUE,
            citations=citations,
        )

    def _run(self, tool_input: Union[str, dict]) -> str:
        params = parse_tool_input(tool_input)
        request = ProfessorTurnRequest.model_validate(params)
        response = self._build_response(request)
        return response.model_dump_json()


# ---------------------------------------------------------------------------
# 1. RetrieveContextTool
# ---------------------------------------------------------------------------

class RetrieveContextTool(BaseTool):
    """Semantic search over Bedrock Knowledge Base.

    Returns ranked, citation-annotated text chunks -- never raw full
    documents.  Used by every calling agent for different purposes:
    - Professor Agent: concept explanations, lecture material, examples
    - TA Agent: practice problems, worked solutions, difficulty context
    - Manager Agent: content verification before document operations
    """

    name: str = "retrieve_context"
    description: str = (
        "Search the Knowledge Base for relevant study-material chunks.\n"
        "\n"
        "WHEN TO USE:\n"
        "- Professor caller needs concept explanations, definitions, lecture "
        "material, or illustrative examples from course documents.\n"
        "- TA caller needs practice problems, worked examples, solution "
        "steps, or difficulty-level context from course documents.\n"
        "- Any caller needs to verify what content exists before answering "
        "a student query.\n"
        "- ALWAYS call this tool BEFORE attempting to answer a content "
        "question; never guess from memory alone.\n"
        "\n"
        "INPUT (JSON object):\n"
        "  query   (str, REQUIRED) - Natural-language search query.\n"
        "  subject (str, optional) - Filter by subject/course name to "
        "narrow results (e.g. 'linear_algebra', 'cs101').\n"
        "  top_k   (int, optional) - Number of chunks to return "
        "(default: configured value, typically 5).\n"
        "\n"
        "OUTPUT (JSON object):\n"
        "  found    (bool)         - Whether any matching chunks were found.\n"
        "  context  (str)          - Numbered text excerpts, e.g. "
        "'[1] ...\\n\\n[2] ...'.\n"
        "  citations (list[dict])  - Each has 'index', 'doc' (filename), "
        "'page' (page number or '?').\n"
        "\n"
        "If nothing is found, returns {found: false, context: '', citations: []}."
    )

    # Instance fields -- set from config during __init__
    kb_id: Any = None
    region: Any = None
    top_k: Any = None

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.kb_id = config.bedrock.knowledge_base_id
        self.region = config.aws.region
        self.top_k = config.rag.top_k

    def _run(self, tool_input: Union[str, dict]) -> str:
        """Execute synchronous retrieval against Bedrock Knowledge Base."""
        params = parse_tool_input(tool_input)
        query: str = params.get("query", "")
        subject: str = params.get("subject", "")
        top_k: int = int(params.get("top_k", self.top_k))

        if not query.strip():
            return json.dumps({"context": "", "citations": [], "found": False,
                               "error": "Empty query -- provide a search query."})

        # Build optional metadata filter for subject scoping
        vector_config: dict[str, Any] = {"numberOfResults": top_k}
        if subject:
            vector_config["filter"] = {
                "equals": {"key": "subject", "value": subject}
            }

        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=self.kb_id,
            region_name=self.region,
            retrieval_config={"vectorSearchConfiguration": vector_config},
        )

        try:
            docs = retriever.invoke(query)
        except Exception as exc:
            logger.error("Knowledge Base retrieval failed: %s", exc)
            return json.dumps({"context": "", "citations": [], "found": False,
                               "error": f"Retrieval error: {exc}"})

        if not docs:
            return json.dumps({"context": "", "citations": [], "found": False})

        context_parts: list[str] = []
        citations: list[dict[str, Any]] = []
        for idx, doc in enumerate(docs, start=1):
            metadata = doc.metadata or {}
            source = metadata.get("source", "unknown")
            page = metadata.get("page_number", "?")
            doc_name = source.split("/")[-1]
            score = metadata.get("score")

            context_parts.append(f"[{idx}] {doc.page_content.strip()}")
            citation: dict[str, Any] = {
                "index": idx,
                "doc": doc_name,
                "page": page,
            }
            if score is not None:
                citation["score"] = round(float(score), 4)
            citations.append(citation)

        return json.dumps({
            "context": "\n\n".join(context_parts),
            "citations": citations,
            "found": True,
        })


# ---------------------------------------------------------------------------
# 2. UploadDocumentTool
# ---------------------------------------------------------------------------

class UploadDocumentTool(BaseTool):
    """Upload a document to S3 and trigger Bedrock Knowledge Base ingestion.

    File bytes are processed in memory and discarded after upload.
    Raw file is stored encrypted (AES-256) in S3; only vector
    embeddings are stored in the Knowledge Base.
    """

    name: str = "upload_document"
    description: str = (
        "Upload a PDF or slide document to S3, then trigger a Knowledge "
        "Base ingestion sync so the content becomes searchable.\n"
        "\n"
        "WHEN TO USE:\n"
        "- Manager caller needs to add new course material to the Knowledge "
        "Base (e.g. professor uploaded a new lecture PDF).\n"
        "- This is an administrative action; Professor and TA callers "
        "should NOT call this tool directly.\n"
        "\n"
        "INPUT (JSON object):\n"
        "  file_bytes_b64 (str, REQUIRED) - Base64-encoded file content.\n"
        "  filename       (str, REQUIRED) - Original filename with extension "
        "(e.g. 'lecture_3.pdf').\n"
        "  subject        (str, REQUIRED) - Subject/course identifier "
        "(e.g. 'linear_algebra').\n"
        "  uploader_id    (str, REQUIRED) - ID of the user performing the "
        "upload.\n"
        "  permissions    (str, optional)  - Access level: 'student' | "
        "'teacher' | 'admin' (default: 'student').\n"
        "\n"
        "OUTPUT (JSON object):\n"
        "  doc_id           (str) - Unique document identifier.\n"
        "  s3_key           (str) - S3 object key where file was stored.\n"
        "  ingestion_job_id (str) - Job ID to poll with "
        "check_ingestion_status.\n"
        "  status           (str) - Always 'syncing' on success.\n"
        "On failure: {error: '<message>'}."
    )

    bucket: Any = None
    kb_id: Any = None
    ds_id: Any = None
    region: Any = None
    s3: Any = None
    bedrock: Any = None

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.bucket = config.aws.s3_bucket
        self.kb_id = config.bedrock.knowledge_base_id
        self.ds_id = config.bedrock.data_source_id
        self.region = config.aws.region
        self.s3 = boto3.client("s3", region_name=self.region)
        self.bedrock = boto3.client("bedrock-agent", region_name=self.region)

    def _run(self, tool_input: Union[str, dict]) -> str:
        """Upload file to S3 and start KB ingestion job."""
        params = parse_tool_input(tool_input)

        # Validate required fields
        required = ("file_bytes_b64", "filename", "subject", "uploader_id")
        missing = [f for f in required if not params.get(f)]
        if missing:
            return json.dumps({"error": f"Missing required fields: {missing}"})

        filename: str = params["filename"]
        subject: str = params["subject"]
        uploader_id: str = params["uploader_id"]
        permissions: str = params.get("permissions", "student")

        try:
            file_bytes = base64.b64decode(params["file_bytes_b64"])
        except Exception as exc:
            return json.dumps({"error": f"Invalid base64 encoding: {exc}"})

        try:
            doc_id = str(uuid.uuid4())
            s3_key = f"docs/{subject}/{doc_id}/{filename}"

            self.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=file_bytes,
                ServerSideEncryption="AES256",
                Metadata={
                    "subject": subject,
                    "uploader_id": uploader_id,
                    "permissions": permissions,
                    "doc_id": doc_id,
                },
            )

            sync = self.bedrock.start_ingestion_job(
                knowledgeBaseId=self.kb_id,
                dataSourceId=self.ds_id,
            )
            job_id = sync["ingestionJob"]["ingestionJobId"]

            logger.info("Document uploaded: doc_id=%s, s3_key=%s, job_id=%s",
                        doc_id, s3_key, job_id)

            return json.dumps({
                "doc_id": doc_id,
                "s3_key": s3_key,
                "ingestion_job_id": job_id,
                "status": "syncing",
            })

        except Exception as exc:
            logger.error("Upload failed for '%s': %s", filename, exc)
            return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# 3. CheckIngestionStatusTool
# ---------------------------------------------------------------------------

class CheckIngestionStatusTool(BaseTool):
    """Poll the ingestion status of a previously uploaded document."""

    name: str = "check_ingestion_status"
    description: str = (
        "Check the sync/ingestion status of a document that was uploaded "
        "with upload_document.\n"
        "\n"
        "WHEN TO USE:\n"
        "- Manager caller needs to confirm a document has been fully "
        "indexed and is ready for retrieval.\n"
        "- Call this AFTER upload_document returns an ingestion_job_id.\n"
        "- May need to be called multiple times until status is COMPLETE.\n"
        "\n"
        "INPUT (JSON object):\n"
        "  ingestion_job_id (str, REQUIRED) - The job ID returned by "
        "upload_document.\n"
        "\n"
        "OUTPUT (JSON object):\n"
        "  status        (str) - One of: STARTING | IN_PROGRESS | "
        "COMPLETE | FAILED.\n"
        "  indexed_count (int) - Number of documents successfully indexed.\n"
        "  failed_count  (int) - Number of documents that failed indexing.\n"
        "On failure: {error: '<message>'}."
    )

    kb_id: Any = None
    ds_id: Any = None
    bedrock: Any = None

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.kb_id = config.bedrock.knowledge_base_id
        self.ds_id = config.bedrock.data_source_id
        self.bedrock = boto3.client("bedrock-agent", region_name=config.aws.region)

    def _run(self, tool_input: Union[str, dict]) -> str:
        """Query Bedrock for ingestion job status."""
        params = parse_tool_input(tool_input)
        ingestion_job_id: str = params.get("ingestion_job_id", "")

        if not ingestion_job_id:
            return json.dumps({"error": "Missing required field: ingestion_job_id"})

        try:
            res = self.bedrock.get_ingestion_job(
                knowledgeBaseId=self.kb_id,
                dataSourceId=self.ds_id,
                ingestionJobId=ingestion_job_id,
            )
            job = res["ingestionJob"]
            stats = job.get("statistics", {})

            return json.dumps({
                "status": job["status"],
                "indexed_count": stats.get("numberOfDocumentsIndexed", 0),
                "failed_count": stats.get("numberOfDocumentsFailed", 0),
            })

        except Exception as exc:
            logger.error("Ingestion status check failed for job '%s': %s",
                         ingestion_job_id, exc)
            return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# 4. ListDocumentsTool
# ---------------------------------------------------------------------------

class ListDocumentsTool(BaseTool):
    """List documents available in the Knowledge Base -- metadata only."""

    name: str = "list_available_documents"
    description: str = (
        "List all documents currently stored in the Knowledge Base.\n"
        "\n"
        "WHEN TO USE:\n"
        "- Manager caller needs to see what course material is available.\n"
        "- Professor or TA caller wants to know which documents exist "
        "for a subject before retrieving content.\n"
        "- Useful for verifying a document was uploaded successfully.\n"
        "\n"
        "INPUT (JSON object):\n"
        "  subject (str, optional) - Filter by subject/course name. "
        "If omitted, lists ALL documents across all subjects.\n"
        "\n"
        "OUTPUT (JSON object):\n"
        "  documents (list[dict]) - Each has 'filename', 's3_key', "
        "'last_modified' (ISO 8601), 'size_kb' (float).\n"
        "  total     (int)        - Total number of documents returned.\n"
        "Never returns raw document content -- metadata only.\n"
        "On failure: {error: '<message>'}."
    )

    bucket: Any = None
    s3: Any = None

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.bucket = config.aws.s3_bucket
        self.s3 = boto3.client("s3", region_name=config.aws.region)

    def _run(self, tool_input: Union[str, dict]) -> str:
        """List S3 objects under the docs/ prefix, returning metadata only."""
        params = parse_tool_input(tool_input)
        subject: str = params.get("subject", "")

        prefix = f"docs/{subject}/" if subject else "docs/"

        try:
            documents: list[dict[str, Any]] = []
            continuation_token: str | None = None

            # Paginate through all results (S3 returns max 1000 per call)
            while True:
                list_kwargs: dict[str, Any] = {
                    "Bucket": self.bucket,
                    "Prefix": prefix,
                }
                if continuation_token:
                    list_kwargs["ContinuationToken"] = continuation_token

                response = self.s3.list_objects_v2(**list_kwargs)

                for obj in response.get("Contents", []):
                    key: str = obj["Key"]
                    if key.endswith("/"):
                        continue  # skip folder-marker entries
                    documents.append({
                        "filename": key.split("/")[-1],
                        "s3_key": key,
                        "last_modified": obj["LastModified"].isoformat(),
                        "size_kb": round(obj["Size"] / 1024, 1),
                    })

                if not response.get("IsTruncated"):
                    break
                continuation_token = response.get("NextContinuationToken")

            return json.dumps({"documents": documents, "total": len(documents)})

        except Exception as exc:
            logger.error("Failed to list documents (prefix='%s'): %s", prefix, exc)
            return json.dumps({"error": str(exc)})
