"""RAG agent tools for the AI tutoring platform.

Provides the three BaseTool classes used by RagAgent for knowledge-base
retrieval and document management.  These tools are shared across all
calling agents (Professor, TA, Manager) -- no agent-specific logic here.

Document uploads are handled directly by the FastAPI layer (not the agent).

Tools
-----
RetrieveContextTool      - Semantic search via local FAISS (S3 -> PDF -> embeddings)
CheckIngestionStatusTool - Poll ingestion job progress
ListDocumentsTool        - List available documents (metadata only)
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Union

import boto3
import pdfplumber
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helpers import parse_tool_input

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LocalVectorStore — S3 download → PDF parse → chunk → embed → FAISS
# ---------------------------------------------------------------------------

class LocalVectorStore:
    """Downloads PDFs from S3, parses, chunks, embeds, and provides FAISS search.

    Built once at tool init time. Supports subject-filtered or full-corpus indexing.
    """

    def __init__(
        self,
        s3_client: Any,
        bucket: str,
        docs_prefix: str,
        embedding_model_name: str,
        chunk_size: int,
        chunk_overlap: int,
        cache_dir: str,
    ) -> None:
        self.s3 = s3_client
        self.bucket = bucket
        self.docs_prefix = docs_prefix
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
        )

        # Cache: {subject_or_"__all__": (FAISS, set_of_s3_keys)}
        self._indexes: dict[str, tuple[FAISS, set[str]]] = {}

    def _list_s3_pdfs(self, prefix: str) -> list[str]:
        """List all PDF keys under a given S3 prefix."""
        keys: list[str] = []
        continuation_token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            resp = self.s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                if obj["Key"].lower().endswith(".pdf"):
                    keys.append(obj["Key"])
            if not resp.get("IsTruncated"):
                break
            continuation_token = resp.get("NextContinuationToken")
        return keys

    def _download_pdf(self, s3_key: str) -> Path:
        """Download PDF to local cache, skip if already cached."""
        safe_name = s3_key.replace("/", "_")
        local_path = self.cache_dir / safe_name
        if not local_path.exists():
            logger.info("Downloading %s from S3...", s3_key)
            self.s3.download_file(self.bucket, s3_key, str(local_path))
        return local_path

    def _parse_and_chunk(self, pdf_path: Path, s3_key: str) -> list[Document]:
        """Extract text page-by-page, then chunk with page number tracking.

        Each chunk is prefixed with ``[doc_name | Page N]`` so that the
        embedding model can distinguish document types (e.g. "Tutorial"
        vs "Lecture") — metadata alone is invisible to FAISS similarity
        search.
        """
        documents: list[Document] = []
        doc_name = s3_key.split("/")[-1]

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue

                metadata_prefix = f"[{doc_name} | Page {page_num}]\n"

                for chunk in self._chunk_text(text):
                    documents.append(Document(
                        page_content=f"{metadata_prefix}{chunk}",
                        metadata={
                            "source": s3_key,
                            "doc_name": doc_name,
                            "page_number": page_num,
                        },
                    ))
        return documents

    def _chunk_text(self, text: str) -> list[str]:
        """Paragraph/sentence-aware chunking via RecursiveCharacterTextSplitter.

        Splits on paragraph breaks first, then sentences, then words —
        keeps math problems and solution pairs intact instead of cutting
        mid-sentence at a fixed character offset.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]

    def get_or_build_index(self, subject: str = "") -> FAISS | None:
        """Return a FAISS index for the given subject (or all docs).

        Builds on first call. Auto-rebuilds when S3 contents change
        (new uploads or deletions detected).
        """
        cache_key = subject or "__all__"

        prefix = f"{self.docs_prefix}{subject}/" if subject else self.docs_prefix
        pdf_keys = self._list_s3_pdfs(prefix)

        # Fall back to all docs if subject-specific search finds nothing
        if not pdf_keys and subject:
            logger.info("No PDFs under '%s', falling back to all docs", prefix)
            pdf_keys = self._list_s3_pdfs(self.docs_prefix)

        if not pdf_keys:
            return None

        current_keys = set(pdf_keys)

        # Use cache if S3 contents haven't changed
        if cache_key in self._indexes:
            cached_index, cached_keys = self._indexes[cache_key]
            if cached_keys == current_keys:
                return cached_index
            logger.info("S3 contents changed for '%s', rebuilding FAISS index", cache_key)

        all_docs: list[Document] = []
        for s3_key in pdf_keys:
            local_path = self._download_pdf(s3_key)
            chunks = self._parse_and_chunk(local_path, s3_key)
            all_docs.extend(chunks)
            logger.info("Parsed %s: %d chunks", s3_key, len(chunks))

        if not all_docs:
            return None

        index = FAISS.from_documents(all_docs, self.embeddings)
        self._indexes[cache_key] = (index, current_keys)
        logger.info("FAISS index built: %d chunks for '%s'", len(all_docs), cache_key)
        return index


# ---------------------------------------------------------------------------
# RetrieveContextTool — semantic search via local FAISS
# ---------------------------------------------------------------------------

class RetrieveContextTool(BaseTool):
    """Semantic search via local FAISS vector store (S3 -> PDF -> embeddings).

    Returns ranked, citation-annotated text chunks -- never raw full
    documents.  Used by every calling agent for different purposes:
    - Professor Agent: concept explanations, lecture material, examples
    - TA Agent: practice problems, worked solutions, difficulty context
    - Manager Agent: content verification before document operations
    """

    name: str = "retrieve_context"
    description: str = (
        "Semantic search over the Knowledge Base. "
        "Call this for ANY content question — concepts, problems, examples, lecture material. "
        "Always call before answering; never rely on memory alone. "
        "Args: query (str, required), subject (str, optional — course filter), top_k (int, optional). "
        "Returns: {found, context (numbered [1]...[n] chunks), citations [{index, doc, page}]}."
    )
    top_k: Any = None
    vector_store: Any = None

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.top_k = config.rag.top_k

        s3 = boto3.client("s3", region_name=config.aws.region)
        self.vector_store = LocalVectorStore(
            s3_client=s3,
            bucket=config.aws.s3_bucket,
            docs_prefix=config.get("rag.s3_docs_prefix", "docs/"),
            embedding_model_name=config.get("rag.embedding_model", "all-MiniLM-L6-v2"),
            chunk_size=int(config.get("rag.chunk_size", 500)),
            chunk_overlap=int(config.get("rag.chunk_overlap", 100)),
            cache_dir=config.get("rag.cache_dir", "/tmp/cechillhackers_cache"),
        )

    def _run(self, tool_input: Union[str, dict]) -> str:
        """Execute retrieval against local FAISS vector store."""
        params = parse_tool_input(tool_input)
        query: str = params.get("query", "")
        subject: str = params.get("subject", "")
        top_k: int = int(params.get("top_k", self.top_k))

        if not query.strip():
            return json.dumps({"context": "", "citations": [], "found": False,
                               "error": "Empty query -- provide a search query."})

        try:
            index = self.vector_store.get_or_build_index(subject)
        except Exception as exc:
            logger.error("Failed to build vector index: %s", exc)
            return json.dumps({"context": "", "citations": [], "found": False,
                               "error": f"Index build error: {exc}"})

        if index is None:
            return json.dumps({"context": "", "citations": [], "found": False})

        try:
            results = index.similarity_search_with_score(query, k=top_k)
        except Exception as exc:
            logger.error("FAISS search failed: %s", exc)
            return json.dumps({"context": "", "citations": [], "found": False,
                               "error": f"Search error: {exc}"})

        if not results:
            return json.dumps({"context": "", "citations": [], "found": False})

        context_parts: list[str] = []
        citations: list[dict[str, Any]] = []
        for idx, (doc, l2_dist) in enumerate(results, start=1):
            metadata = doc.metadata or {}
            doc_name = metadata.get("doc_name", "unknown")
            page = metadata.get("page_number", "?")

            # Convert squared L2 distance to cosine similarity.
            # FAISS returns d_sq = ||a - b||^2 (already squared).
            # For normalized vectors: d_sq = 2(1 - cos_sim)
            # So cos_sim = 1 - d_sq / 2
            cos_sim = max(0.0, 1.0 - float(l2_dist) / 2.0)

            context_parts.append(f"[{idx}] {doc.page_content.strip()}")
            citation: dict[str, Any] = {
                "index": idx,
                "doc": doc_name,
                "page": page,
                "score": round(cos_sim, 4),
            }
            citations.append(citation)

        return json.dumps({
            "context": "\n\n".join(context_parts),
            "citations": citations,
            "found": True,
        })


# ---------------------------------------------------------------------------
# CheckIngestionStatusTool
# ---------------------------------------------------------------------------

class CheckIngestionStatusTool(BaseTool):
    """Poll the ingestion status of a previously uploaded document."""

    name: str = "check_ingestion_status"
    description: str = (
        "Poll the sync status of a document after it has been uploaded to S3. "
        "Call this with an ingestion_job_id; repeat until status=COMPLETE or FAILED. "
        "Args: ingestion_job_id (str, required). "
        "Returns: {status (STARTING|IN_PROGRESS|COMPLETE|FAILED), indexed_count, failed_count} or {error}."
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
# ListDocumentsTool
# ---------------------------------------------------------------------------

class ListDocumentsTool(BaseTool):
    """List documents available in the Knowledge Base -- metadata only."""

    name: str = "list_available_documents"
    description: str = (
        "List documents stored in the Knowledge Base (metadata only, no content). "
        "Call this to see what course material exists or to verify a recent upload. "
        "Args: subject (str, optional — omit to list all subjects). "
        "Returns: {documents [{filename, s3_key, last_modified, size_kb}], total} or {error}."
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
                        continue
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


# ---------------------------------------------------------------------------
# ExaWebSearchTool — web search via Exa API for EXTERNAL_OK mode
# ---------------------------------------------------------------------------

class ExaWebSearchTool(BaseTool):
    """Web search via Exa API for supplementing KB content in EXTERNAL_OK mode.

    Caller-aware: professor queries target concepts/theory, TA queries
    target problems/exercises. Level-aware: query complexity adjusts
    to beginner/intermediate/advanced.
    """

    name: str = "exa_web_search"
    description: str = (
        "Search the web via Exa for educational content when the Knowledge Base "
        "is insufficient. Returns web results with text highlights and URLs. "
        "Args: query (str, required), caller (str), level (str), subject (str). "
        "Returns: {found, context, web_citations [{index, title, url, snippet}]}."
    )

    exa_client: Any = None
    num_results: int = 5

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from exa_py import Exa

        api_key = os.environ.get("EXA_API_KEY", "")
        if not api_key:
            api_key = config.get("exa.api_key", "")
        if not api_key:
            raise ValueError(
                "Missing EXA_API_KEY in environment or config -- "
                "add it to your .env file or config.yaml."
            )
        self.exa_client = Exa(api_key=api_key)
        self.num_results = int(config.get("exa.num_results", 5))

    def _run(self, tool_input: Union[str, dict]) -> str:
        """Execute web search via Exa, returning formatted context + citations."""
        params = parse_tool_input(tool_input)
        query: str = params.get("query", "")
        caller: str = params.get("caller", "")
        level: str = params.get("level", "intermediate")
        subject: str = params.get("subject", "")

        if not query.strip():
            return json.dumps({"context": "", "web_citations": [], "found": False,
                               "error": "Empty query."})

        search_query = self._build_search_query(query, caller, level, subject)

        try:
            response = self.exa_client.search(
                search_query,
                type="auto",
                num_results=self.num_results,
                contents={
                    "text": {"max_characters": 2000},
                    "highlights": {
                        "query": query,
                        "num_sentences": 5,
                    },
                },
            )
        except Exception as exc:
            logger.error("Exa search failed: %s", exc)
            return json.dumps({"context": "", "web_citations": [], "found": False,
                               "error": f"Web search error: {exc}"})

        if not response.results:
            return json.dumps({"context": "", "web_citations": [], "found": False})

        context_parts: list[str] = []
        web_citations: list[dict[str, Any]] = []

        for idx, result in enumerate(response.results, start=1):
            content = ""
            if hasattr(result, "highlights") and result.highlights:
                content = " ... ".join(result.highlights)
            elif hasattr(result, "text") and result.text:
                content = result.text[:1500]

            if not content.strip():
                continue

            context_parts.append(f"[W{idx}] {content.strip()}")
            web_citations.append({
                "index": f"W{idx}",
                "title": getattr(result, "title", "Untitled") or "Untitled",
                "url": getattr(result, "url", ""),
                "snippet": content[:300],
            })

        return json.dumps({
            "context": "\n\n".join(context_parts),
            "web_citations": web_citations,
            "found": len(context_parts) > 0,
        })

    @staticmethod
    def _build_search_query(
        query: str, caller: str, level: str, subject: str
    ) -> str:
        """Construct a search query based on caller role and student level."""
        parts: list[str] = []

        if subject:
            parts.append(subject)

        if caller == "professor":
            parts.append("concept explanation definition theory")
        elif caller == "ta":
            parts.append("practice problems worked examples exercises solutions")

        _LEVEL_MODIFIERS = {
            "beginner": "introductory basic simple explanation",
            "intermediate": "detailed comprehensive with examples",
            "advanced": "rigorous in-depth proof derivation graduate-level",
        }
        parts.append(_LEVEL_MODIFIERS.get(level, _LEVEL_MODIFIERS["intermediate"]))
        parts.append(query)

        return " ".join(parts)
