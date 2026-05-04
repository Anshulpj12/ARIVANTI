"""
Query Engine — Core online RAG pipeline.
Intent Detection -> Metadata Filter -> FAISS Search -> LLM Answer
"""
import json
import time
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional

from intent_graph import detect_intent
from llm_client import generate_answer

INDEX_DIR = Path(__file__).resolve().parent / "index_store"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

# Global state (loaded once at startup)
_faiss_index = None
_metadata = None
_embed_model = None
_chunk_ids_by_section = {}
_chunk_ids_by_is_number = {}
_all_chunk_ids = set()


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def initialize():
    """Load FAISS index, metadata, and embedding model into memory."""
    global _faiss_index, _metadata, _embed_model
    global _chunk_ids_by_section, _chunk_ids_by_is_number, _all_chunk_ids

    print("[Engine] Loading FAISS index...")
    _faiss_index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    print(f"         {_faiss_index.ntotal} vectors loaded.")

    print("[Engine] Loading metadata...")
    with open(INDEX_DIR / "metadata.json", "r", encoding="utf-8") as f:
        _metadata = json.load(f)

    # Build section and IS code lookups
    _chunk_ids_by_section = {}
    _chunk_ids_by_is_number = {}
    _all_chunk_ids = set()
    for cid, meta in _metadata.items():
        cid_int = int(cid)
        _all_chunk_ids.add(cid_int)
        sid = meta["section_id"]
        _chunk_ids_by_section.setdefault(sid, set()).add(cid_int)
        is_num = meta.get("is_number", "")
        if is_num:
            _chunk_ids_by_is_number[is_num] = cid_int

    print(f"         {len(_metadata)} chunks across {len(_chunk_ids_by_section)} sections.")

    # Load embedding model
    config = load_config()
    model_name = config["embedding"]["active_model"]
    print(f"[Engine] Loading embedding model: {model_name}...")
    from sentence_transformers import SentenceTransformer
    _embed_model = SentenceTransformer(model_name)
    
    # Pre-warm the model with a dummy query so PyTorch allocates inference memory 
    # and compiles the graph BEFORE the evaluation timer starts.
    print("[Engine] Warming up model inference cache...")
    _embed_model.encode(["warmup"], normalize_embeddings=True)
    
    print("         Ready.")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    vec = _embed_model.encode([query], normalize_embeddings=True)
    return vec.astype(np.float32)


def search_chunks(
    query_vec: np.ndarray,
    candidate_ids: Optional[set] = None,
    top_k: int = 5,
    score_threshold: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Search FAISS index, optionally restricted to candidate chunk IDs.
    Uses post-search filtering with a set for O(1) lookup.
    """
    # Always search all vectors, then filter by candidate set
    n_search = min(_faiss_index.ntotal, 448)
    scores, indices = _faiss_index.search(query_vec, n_search)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < score_threshold:
            continue
        idx_int = int(idx)
        # Apply section filter if provided
        if candidate_ids is not None and idx_int not in candidate_ids:
            continue
        meta = _metadata.get(str(idx_int))
        if meta:
            results.append({**meta, "score": float(score)})
        if len(results) >= top_k:
            break

    return results


def build_fallback_answer(question: str, chunks: list) -> str:
    """
    Generate a structured answer from retrieved chunks WITHOUT an LLM.
    Used when no LLM server is running. Extracts and formats the most
    relevant content from retrieved sources.
    """
    if not chunks:
        return "No relevant standards found for this query."

    parts = ["Based on the retrieved Indian Standards:\n"]

    for i, chunk in enumerate(chunks[:3], 1):
        parts.append(f"--- {chunk['is_code']} - {chunk['title']} ---")
        parts.append(f"Section: {chunk['section']}")

        # Include sub-sections if available
        sections = chunk.get("sections", {})
        if sections:
            for key, val in sections.items():
                label = key.replace("_", " ").replace("section ", "").title()
                text = str(val).strip()
                if text and len(text) > 10:
                    # Truncate very long sections
                    if len(text) > 600:
                        text = text[:600] + "..."
                    parts.append(f"\n{label}:\n{text}")
        else:
            # Fallback to content field
            content = chunk.get("content", "")
            if len(content) > 800:
                content = content[:800] + "..."
            parts.append(f"\n{content}")

        # Include tables if available
        tables = chunk.get("tables", [])
        if tables:
            for t in tables[:2]:
                parts.append(f"\nTable: {t.get('caption', 'Data Table')}")
                for row in t.get("rows", [])[:5]:
                    parts.append("  " + " | ".join(str(c) for c in row))

        parts.append("")

    return "\n".join(parts)


async def process_query(question: str) -> Dict[str, Any]:
    """
    Full RAG pipeline:
    1. Intent detection
    2. Metadata pre-filtering
    3. Query embedding + FAISS search
    4. LLM answer generation (with fallback)
    """
    t_start = time.time()
    config = load_config()
    top_k = config.get("search", {}).get("top_k", 5)
    threshold = config.get("search", {}).get("score_threshold", 0.15)

    # Step 1: Intent Detection
    intent = detect_intent(question)

    # Step 2: Determine candidate chunk IDs (as a set for O(1) lookup)
    candidate_ids = None

    # Direct IS code reference
    if intent["is_code_ref"]:
        is_num = intent["is_code_ref"]
        if is_num in _chunk_ids_by_is_number:
            direct_id = _chunk_ids_by_is_number[is_num]
            direct_meta = _metadata.get(str(direct_id))
            if direct_meta:
                candidate_ids = _chunk_ids_by_section.get(direct_meta["section_id"], set())
        # else: full search (candidate_ids stays None)

    elif intent["section_id"]:
        candidate_ids = _chunk_ids_by_section.get(intent["section_id"], set())

    # Step 3: Embed query and search
    query_vec = embed_query(question)
    retrieved = search_chunks(query_vec, candidate_ids, top_k, threshold)

    # Fallback: if section filter returned nothing, search globally
    if not retrieved and candidate_ids is not None:
        intent["fallback_mode"] = "full"
        retrieved = search_chunks(query_vec, None, top_k, threshold)

    t_search = time.time()

    # Step 4: Generate answer via LLM (with fallback)
    llm_used = True
    if retrieved:
        try:
            answer = await generate_answer(question, retrieved)
            # Check if the answer indicates LLM failure — trigger fallback
            if "Could not connect" in answer or answer.startswith("\u274c"):
                llm_used = False
                fallback_note = answer.split('\n')[0]  # Keep the error note
                answer = build_fallback_answer(question, retrieved)
                answer = f"[AI Summary unavailable: {fallback_note}]\n\n{answer}"
        except Exception:
            llm_used = False
            answer = build_fallback_answer(question, retrieved)
    else:
        llm_used = False
        answer = "I could not find any relevant standards matching your query in the SP 21:2005 dataset. Please try rephrasing with specific terms like material names, IS codes, or technical properties."

    t_end = time.time()

    # Build sources summary
    sources = []
    for r in retrieved:
        sources.append({
            "is_code": r["is_code"],
            "title": r["title"],
            "section": r["section"],
            "page": r["page_start"],
            "score": round(r["score"], 4),
        })

    return {
        "answer": answer,
        "sources": sources,
        "intent": intent,
        "llm_used": llm_used,
        "latency_ms": {
            "total": round((t_end - t_start) * 1000),
            "search": round((t_search - t_start) * 1000),
            "llm": round((t_end - t_search) * 1000),
        },
        "chunks_searched": len(candidate_ids) if candidate_ids else _faiss_index.ntotal,
        "chunks_retrieved": len(retrieved),
    }


def get_stats() -> Dict[str, Any]:
    """Return index statistics."""
    embed_info = {}
    info_path = INDEX_DIR / "embed_info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            embed_info = json.load(f)
    section_counts = {
        sid: len(ids) for sid, ids in sorted(_chunk_ids_by_section.items())
    }
    return {
        "total_chunks": _faiss_index.ntotal if _faiss_index else 0,
        "total_sections": len(_chunk_ids_by_section),
        "embedding_model": embed_info.get("model_name", "unknown"),
        "embedding_dimensions": embed_info.get("dimensions", 0),
        "index_built_at": embed_info.get("built_at", "unknown"),
        "section_counts": section_counts,
    }
