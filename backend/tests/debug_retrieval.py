"""
Diagnostic script for FAISS retrieval quality.

Investigates why FAISS scores are low and why Tutorial_1.pdf chunks
don't rank higher for practice problem queries.
"""
import json
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent          # backend/tests/
_BACKEND_DIR = _SCRIPT_DIR.parent                      # backend/
_ROOT_DIR = _BACKEND_DIR.parent                        # ceChillHackers/

for p in (_ROOT_DIR, _BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dotenv import load_dotenv
load_dotenv(_ROOT_DIR / ".env")

from config.config_loader import config
from agents.tools import RetrieveContextTool, LocalVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("debug_retrieval")


def separator(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def run_diagnostics() -> None:
    separator("1. Initializing RetrieveContextTool")
    tool = RetrieveContextTool(config)
    vs: LocalVectorStore = tool.vector_store
    print(f"  Embedding model : {config.get('rag.embedding_model', 'all-MiniLM-L6-v2')}")
    print(f"  Chunk size      : {vs.chunk_size}")
    print(f"  Chunk overlap   : {vs.chunk_overlap}")
    print(f"  Top-k           : {tool.top_k}")
    print(f"  S3 bucket       : {vs.bucket}")
    print(f"  S3 docs prefix  : {vs.docs_prefix}")
    print(f"  Cache dir       : {vs.cache_dir}")

    separator("2. Building FAISS index")
    index = vs.get_or_build_index(subject="")
    if index is None:
        print("  ERROR: FAISS index is None -- no PDFs found in S3!")
        return

    # Access the internal FAISS index to get total vector count
    total_chunks = index.index.ntotal
    print(f"  Total chunks in FAISS index: {total_chunks}")

    # List all unique source documents
    docstore = index.docstore
    doc_sources = {}
    for doc_id in index.index_to_docstore_id.values():
        doc = docstore.search(doc_id)
        if hasattr(doc, "metadata"):
            src = doc.metadata.get("doc_name", "unknown")
            doc_sources[src] = doc_sources.get(src, 0) + 1
    print(f"\n  Documents indexed (chunk counts):")
    for doc_name, count in sorted(doc_sources.items()):
        print(f"    {doc_name}: {count} chunks")

    separator("3. Checking embedding normalization")
    import numpy as np

    sample_text = "This is a test sentence for normalization check."
    sample_embedding = vs.embeddings.embed_query(sample_text)
    sample_vec = np.array(sample_embedding)
    l2_norm = np.linalg.norm(sample_vec)
    print(f"  Sample embedding dimension: {len(sample_embedding)}")
    print(f"  Sample embedding L2 norm : {l2_norm:.6f}")
    print(f"  Is normalized (norm ~= 1): {'YES' if abs(l2_norm - 1.0) < 0.01 else 'NO -- THIS IS THE PROBLEM'}")

    if abs(l2_norm - 1.0) > 0.01:
        print(f"\n  *** WARNING: Embeddings are NOT unit-normalized!")
        print(f"  *** The cosine similarity formula cos_sim = 1 - d_sq/2")
        print(f"  *** only holds when ||a|| = ||b|| = 1.")
        print(f"  *** Actual norm = {l2_norm:.6f}")
        print(f"  *** For unnormalized vectors, d_sq can exceed 2,")
        print(f"  *** making the cosine formula yield negative/wrong values.")

    queries = [
        ("Explain Bayes' rule", "Should match lec1.pdf well"),
        ("Find practice problems and exercises", "Should match Tutorial_1.pdf"),
        ("practice problems exercises solutions examples", "Enriched TA query"),
    ]

    for query, description in queries:
        separator(f"4. Query: \"{query}\"\n     ({description})")

        results = index.similarity_search_with_score(query, k=5)

        if not results:
            print("  NO RESULTS returned.")
            continue

        for rank, (doc, l2_dist) in enumerate(results, start=1):
            meta = doc.metadata or {}
            doc_name = meta.get("doc_name", "unknown")
            page = meta.get("page_number", "?")

            # The raw score from FAISS (squared L2 distance)
            raw_l2_sq = float(l2_dist)

            # The conversion used in tools.py
            cos_sim_tools = max(0.0, 1.0 - raw_l2_sq / 2.0)

            # True cosine similarity (for comparison)
            query_vec = np.array(vs.embeddings.embed_query(query))
            chunk_vec = np.array(vs.embeddings.embed_query(doc.page_content))
            true_cosine = float(np.dot(query_vec, chunk_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
            ))

            print(f"  --- Rank {rank} ---")
            print(f"  Source        : {doc_name} (page {page})")
            print(f"  Raw L2-sq dist: {raw_l2_sq:.6f}")
            print(f"  cos_sim (tools): {cos_sim_tools:.6f}  (formula: 1 - d/2)")
            print(f"  True cosine   : {true_cosine:.6f}  (dot / norms)")
            print(f"  Chunk length  : {len(doc.page_content)} chars")
            # Truncate for display
            content_preview = doc.page_content[:300]
            if len(doc.page_content) > 300:
                content_preview += "..."
            print(f"  Content:\n    {content_preview}\n")

    separator("5. Sampling Tutorial_1.pdf chunks")

    tutorial_chunks = []
    for doc_id in index.index_to_docstore_id.values():
        doc = docstore.search(doc_id)
        if hasattr(doc, "metadata") and "tutorial" in doc.metadata.get("doc_name", "").lower():
            tutorial_chunks.append(doc)

    if not tutorial_chunks:
        print("  No Tutorial_1.pdf chunks found in the index!")
        print("  Available doc names:")
        for name in sorted(doc_sources.keys()):
            print(f"    - {name}")
    else:
        print(f"  Found {len(tutorial_chunks)} tutorial chunks total.\n")
        # Show first 5 and last 2
        sample = tutorial_chunks[:5]
        if len(tutorial_chunks) > 7:
            sample += tutorial_chunks[-2:]
        for i, doc in enumerate(sample):
            meta = doc.metadata or {}
            page = meta.get("page_number", "?")
            print(f"  [Chunk {i+1}] page={page}, len={len(doc.page_content)} chars")
            content_preview = doc.page_content[:400]
            if len(doc.page_content) > 400:
                content_preview += "..."
            print(f"    {content_preview}\n")

    separator("6. Direct similarity: 'practice problems' vs Tutorial chunks")

    if tutorial_chunks:
        query_text = "Find practice problems and exercises"
        query_vec = np.array(vs.embeddings.embed_query(query_text))
        query_norm = np.linalg.norm(query_vec)

        scored = []
        for doc in tutorial_chunks:
            chunk_vec = np.array(vs.embeddings.embed_query(doc.page_content))
            chunk_norm = np.linalg.norm(chunk_vec)
            cosine = float(np.dot(query_vec, chunk_vec) / (query_norm * chunk_norm))
            l2_sq = float(np.sum((query_vec - chunk_vec) ** 2))
            scored.append((cosine, l2_sq, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        print(f"  Query: \"{query_text}\"")
        print(f"  Top 5 Tutorial chunks by TRUE cosine similarity:\n")
        for rank, (cosine, l2_sq, doc) in enumerate(scored[:5], start=1):
            meta = doc.metadata or {}
            page = meta.get("page_number", "?")
            tools_cosine = max(0.0, 1.0 - l2_sq / 2.0)
            print(f"  #{rank} page={page} | true_cos={cosine:.4f} | l2_sq={l2_sq:.4f} | tools_cos={tools_cosine:.4f}")
            preview = doc.page_content[:200]
            if len(doc.page_content) > 200:
                preview += "..."
            print(f"       {preview}\n")

    separator("7. Summary diagnostics")
    print(f"  Embedding model         : {config.get('rag.embedding_model')}")
    print(f"  Embedding dimension     : {len(sample_embedding)}")
    print(f"  Embedding L2 norm       : {l2_norm:.6f} ({'normalized' if abs(l2_norm - 1.0) < 0.01 else 'NOT normalized'})")
    print(f"  Chunk size (chars)      : {vs.chunk_size}")
    print(f"  Chunk overlap (chars)   : {vs.chunk_overlap}")
    print(f"  Total indexed chunks    : {total_chunks}")
    print(f"  Relevance threshold     : {config.get('rag.relevance_threshold')}")
    print(f"  Top-k                   : {tool.top_k}")

    # Summarize the core issue
    if abs(l2_norm - 1.0) > 0.01:
        print(f"\n  ** KEY FINDING: Embeddings have norm {l2_norm:.4f}, NOT 1.0")
        print(f"     The formula `cos_sim = 1 - d_sq/2` is INVALID for unnormalized vectors.")
        print(f"     This means the 'cosine similarity' scores in tools.py are WRONG.")
        print(f"     With norm ~{l2_norm:.2f}, max possible d_sq = 2*(norm^2 + norm^2) = {4*l2_norm**2:.2f}")
        print(f"     Scores can go negative, and the relevance_threshold comparison is unreliable.")
    else:
        print(f"\n  Embeddings ARE normalized. The cos_sim = 1 - d/2 formula is valid.")


if __name__ == "__main__":
    run_diagnostics()
