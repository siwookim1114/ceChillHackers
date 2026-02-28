"""Professor agent tools and schema helpers.

This module intentionally avoids persistence and raw-content logging.
"""

from __future__ import annotations

import json
import uuid
import base64
import boto3
from typing import Any, Union

from langchain_core.tools import BaseTool
from langchain_aws import AmazonKnowledgeBasesRetriever
from backend.agents.schemas.professor import (
    Citation,
    ProfessorMode,
    ProfessorNextAction,
    ProfessorTurnRequest,
    ProfessorTurnResponse,
    ProfessorTurnStrategy,
)
from backend.utils.helpers import parse_tool_input

PROFESSOR_CONFIG_PATH = "agents.professor"


class ProfessorRespondTool(BaseTool):
    """Professor tutoring tool with strict request/response schema handling."""

    name: str = "professor_respond"
    description: str = (
        "Generate a tutoring response for one professor turn. "
        "Uses strict JSON schema, avoids revealing final answers, and returns citations."
    )

    professor_config: Any = None
    citations_enabled: bool = True
    socratic_default: bool = True

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.professor_config = self._load_professor_runtime_config(config)
        tutoring_cfg = self.professor_config.get("tutoring", {})
        self.citations_enabled = bool(tutoring_cfg.get("citations_enabled", True))
        self.socratic_default = bool(tutoring_cfg.get("socratic_default", True))

    @staticmethod
    def _load_professor_runtime_config(config) -> dict[str, Any]:
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
    def sanitize_for_log(request: ProfessorTurnRequest) -> dict[str, Any]:
        return {
            "session_id": request.session_id,
            "mode": request.mode.value,
            "topic": request.topic,
            "message_length": len(request.student_message),
            "profile_level": request.profile.level,
        }

    @staticmethod
    def get_professor_json_schemas() -> dict[str, dict[str, Any]]:
        return {
            "ProfessorTurnRequest": ProfessorTurnRequest.model_json_schema(),
            "ProfessorTurnResponse": ProfessorTurnResponse.model_json_schema(),
        }

    def _retrieve_citations(self, request: ProfessorTurnRequest) -> list[Citation]:
        if not self.citations_enabled:
            return []

        source_id = (
            "local_stub_strict"
            if request.mode is ProfessorMode.STRICT
            else "local_stub_convenience"
        )
        return [
            Citation(
                source_id=source_id,
                title=f"Intro to {request.topic}",
                snippet=f"Core concept recap for {request.topic}.",
                url=None,
            )
        ]

    def _build_response(self, request: ProfessorTurnRequest) -> ProfessorTurnResponse:
        citations = self._retrieve_citations(request)
        if self.socratic_default:
            strategy = ProfessorTurnStrategy.SOCRATIC_QUESTION
            question = (
                f"Before we solve it, can you explain in one sentence what the key idea in "
                f"{request.topic} is?"
            )
        else:
            strategy = ProfessorTurnStrategy.CONCEPT_EXPLAIN
            question = (
                f"Let's do a short recap: in {request.topic}, focus on the core principle "
                "first, then we can apply it to your problem."
            )

        return ProfessorTurnResponse(
            assistant_response=question,
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


# Rag Tools
class RetrieveContextTool(BaseTool):
    """
    Retrieve relevant chunks from Bedrock Knowledge Base.
    Returns summarized context with source citations.
    Never returns raw full documents.
    """
    name: str = "retrieve_context"
    description: str = (
        "Retrieve relevant study material chunks for a given query. "
        "Returns summarized context with citations (doc name + page). "
        "Use this tool when you need content from uploaded study materials. "
        "Never returns raw full documents — excerpts and summaries only."
    )

    kb_id: Any = None
    region: Any = None
    top_k: Any = None

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.kb_id = config.bedrock.knowledge_base_id
        self.region = config.aws.region
        self.top_k = config.rag.top_k

    def _run(self, tool_input: Union[str, dict]) -> str:
        params = parse_tool_input(tool_input)
        query = params.get("query", "")
        subject = params.get("subject", "")
        top_k = params.get("top_k", self.top_k)

        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=self.kb_id,
            region_name=self.region,
            retrieval_config={
                "vectorSearchConfiguration": {
                    "numberOfResults": top_k,
                    **(
                        {"filter": {"equals": {"key": "subject", "value": subject}}}
                        if subject else {}
                    )
                }
            }
        )

        docs = retriever.invoke(query)
        if not docs:
            return json.dumps({"context": "", "citations": [], "found": False})
        
        context_parts, citations = [], []
        for i, doc in enumerate(docs):
            metadata = doc.metadata or {}
            source = metadata.get("source", "unknown")
            page = metadata.get("page_number", "?")
            doc_name = source.split("/")[-1]

            context_parts.append(f"[{i+1}] {doc.page_content.strip()}")
            citations.append({"index": i + 1, "doc": doc_name, "page": page})

        return json.dumps({
            "context": "\n\n".join(context_parts),
            "citations": citations,
            "found": True
        })


class UploadDocumentTool(BaseTool):
    """
    Upload a PDF to S3 and trigger Bedrock Knowledge Base sync.
    File bytes processed in memory and discarded after upload.
    Raw file stored encrypted in S3. Vectors only in Knowledge Base. 
    """

    name: str = "upload_document"
    description: str = (
        "Upload a PDF or slide document to S3 then sync into Knowledge Base. "
        "Returns doc_id and ingestion_job_id for sync tracking. "
        "permissions: 'student' | 'teacher' | 'admin'"
    )

    bucket: Any = None
    kb_id: Any = None
    ds_id: Any = None
    region: Any = None
    s3: Any = None
    bedrock: Any = None

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.bucket = config.aws.s3_bucket
        self.kb_id = config.bedrock.knowledge_base_id
        self.ds_id = config.bedrock.data_source_id
        self.region = config.aws.region
        self.s3 = boto3.client("s3", region_name=self.region)
        self.bedrock = boto3.client("bedrock-agent", region_name=self.region)

    def _run(self, tool_input: Union[str, dict]) -> str:
        params = parse_tool_input(tool_input)
        file_bytes = base64.b64decode(params["file_bytes_b64"])
        filename = params["filename"]
        subject = params["subject"]
        uploader_id = params["uploader_id"]
        permissions = params.get("permissions", "student")

        try:
            doc_id = str(uuid.uuid4())
            s3_key = f"docs/{subject}/{doc_id}/{filename}"

            self.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=file_bytes,
                ServerSideEncryption="AES256",
                Metadata={
                    "subject":     subject,
                    "uploader_id": uploader_id,
                    "permissions": permissions,
                    "doc_id":      doc_id,
                }
            )
            sync = self.bedrock.start_ingestion_job(
                knowledgeBaseId=self.kb_id,
                dataSourceId=self.ds_id,
            )
            job_id = sync["ingestionJob"]["ingestionJobId"]

            return json.dumps({
                "doc_id":           doc_id,
                "s3_key":           s3_key,
                "ingestion_job_id": job_id,
                "status":           "syncing"
            })

        except Exception as e:
            return json.dumps({"error": str(e)})


class CheckIngestionStatusTool(BaseTool):
    """Check if a document has finished syncing into the Knowledge Base."""

    name: str = "check_ingestion_status"
    description: str = (
        "Check sync status of an uploaded document. "
        "Returns: STARTING | IN_PROGRESS | COMPLETE | FAILED "
        "Use this tool after upload_document to confirm the doc is ready for retrieval."
    )

    kb_id: Any = None
    ds_id: Any = None
    bedrock: Any = None

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.kb_id = config.bedrock.knowledge_base_id
        self.ds_id = config.bedrock.data_source_id
        self.bedrock = boto3.client("bedrock-agent", region_name=config.aws.region)

    def _run(self, tool_input: Union[str, dict]) -> str:
        params = parse_tool_input(tool_input)
        ingestion_job_id = params.get("ingestion_job_id", "")

        try:
            res = self.bedrock.get_ingestion_job(
                knowledgeBaseId=self.kb_id,
                dataSourceId=self.ds_id,
                ingestionJobId=ingestion_job_id
            )
            job = res["ingestionJob"]
            stats = job.get("statistics", {})

            return json.dumps({
                "status": job["status"],
                "indexed_count": stats.get("numberOfDocumentsIndexed", 0),
                "failed_count": stats.get("numberOfDocumentsFailed", 0),
            })

        except Exception as e:
            return json.dumps({"error": str(e)})
        
class ListDocumentsTool(BaseTool):
    """List documents in the Knowledge Base — metadata only, no raw content."""

    name: str = "list_available_documents"
    description: str = (
        "List all documents available in the Knowledge Base. "
        "Returns filename, subject, size, last modified. "
        "Never returns raw document content — metadata only."
    )

    bucket: Any = None
    s3: Any = None

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.bucket = config.aws.s3_bucket
        self.s3 = boto3.client("s3", region_name=config.aws.region)

    def _run(self, tool_input: Union[str, dict]) -> str:
        params = parse_tool_input(tool_input)
        subject = params.get("subject", "")

        try:
            prefix = f"docs/{subject}/" if subject else "docs/"
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)

            docs = [
                {
                    "filename": obj["Key"].split("/")[-1],
                    "s3_key": obj["Key"],
                    "last_modified": obj["LastModified"].isoformat(),
                    "size_kb": round(obj["Size"] / 1024, 1),
                }
                for obj in response.get("Contents", [])
                if not obj["Key"].endswith("/")   # skip folder entries
            ]

            return json.dumps({"documents": docs, "total": len(docs)})

        except Exception as e:
            return json.dumps({"error": str(e)})
