"""
Offline Ingestion Pipeline — SP 21:2005 RAG System
===================================================
Reads dataset.json, embeds all 448 chunks using sentence-transformers,
builds a FAISS index, and saves everything to index_store/.

Run once: python ingest.py
Re-run to rebuild with a different embedding model: python ingest.py --rebuild
"""

import json
import os
import sys
import time
import numpy as np
import faiss
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset.json"
CONFIG_PATH = ROOT_DIR / "config.json"
INDEX_DIR = Path(__file__).resolve().parent / "index_store"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"
EMBED_INFO_PATH = INDEX_DIR / "embed_info.json"


def load_config():
    """Load config.json for embedding model selection."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset():
    """Load the structured JSON dataset."""
    print(f"[1/4] Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    standards = data["standards"]
    sections = data["sections"]
    print(f"       Loaded {len(standards)} standards across {len(sections)} sections.")
    return data, standards, sections


def build_chunk_texts(standards):
    """
    Build the text representation for each chunk that will be embedded.
    Combines title, section name, IS code, and full content for maximum
    semantic coverage.
    """
    texts = []
    for std in standards:
        # Build a rich text combining key metadata + content
        parts = [
            f"IS Code: {std['is_code']}",
            f"Title: {std['title']}",
            f"Section: {std['section']}",
        ]
        if std.get("revision"):
            parts.append(f"Revision: {std['revision']}")

        parts.append(f"Content: {std['content']}")

        # Also include parsed sub-sections if available
        if std.get("sections"):
            for key, value in std["sections"].items():
                parts.append(f"{key}: {value}")

        texts.append("\n".join(parts))

    return texts


def embed_chunks(texts, model_name):
    """Embed all chunk texts using sentence-transformers."""
    print(f"[2/4] Loading embedding model: {model_name}...")
    print(f"       (First run will download the model, ~{get_model_size(model_name)})")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    print(f"[3/4] Embedding {len(texts)} chunks...")
    start = time.time()

    # Batch encode with progress
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True  # L2 normalize for cosine similarity via inner product
    )

    elapsed = time.time() - start
    print(f"       Done in {elapsed:.1f}s ({len(texts)/elapsed:.1f} chunks/sec)")
    print(f"       Embedding dimensions: {embeddings.shape[1]}")

    return embeddings.astype(np.float32)


def get_model_size(model_name):
    """Return approximate model download size string."""
    sizes = {
        "all-MiniLM-L6-v2": "80 MB",
        "all-mpnet-base-v2": "420 MB",
        "BAAI/bge-large-en-v1.5": "1.3 GB",
        "BAAI/bge-small-en-v1.5": "130 MB",
    }
    return sizes.get(model_name, "unknown size")


def build_faiss_index(embeddings):
    """Build a FAISS IndexFlatIP (inner product = cosine sim on normalized vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def build_metadata(standards, sections_map):
    """Build metadata dict keyed by chunk_id for fast lookup during query."""
    metadata = {}
    for std in standards:
        chunk_id = std["chunk_id"]
        metadata[str(chunk_id)] = {
            "chunk_id": chunk_id,
            "is_code": std["is_code"],
            "is_number": std.get("is_number", ""),
            "is_year": std.get("is_year", 0),
            "title": std["title"],
            "revision": std.get("revision"),
            "section_id": std["section_id"],
            "section": std["section"],
            "page_start": std.get("page_start", 0),
            "content": std["content"],
            "sections": std.get("sections", {}),
            "tables": std.get("tables", []),
        }
    return metadata


def save_index(index, metadata, model_name, dim):
    """Save FAISS index and metadata to disk."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[4/4] Saving index and metadata to {INDEX_DIR}...")

    # Save FAISS index
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    # Save metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    # Save embedding info (so we know which model was used)
    embed_info = {
        "model_name": model_name,
        "dimensions": dim,
        "num_chunks": len(metadata),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(EMBED_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(embed_info, f, indent=2)

    print(f"       FAISS index:  {FAISS_INDEX_PATH} ({os.path.getsize(FAISS_INDEX_PATH) / 1024:.0f} KB)")
    print(f"       Metadata:     {METADATA_PATH} ({os.path.getsize(METADATA_PATH) / 1024:.0f} KB)")


def main():
    rebuild = "--rebuild" in sys.argv

    # Check if index already exists
    if FAISS_INDEX_PATH.exists() and not rebuild:
        print("[OK] FAISS index already exists. Use --rebuild to re-create.")
        print(f"   Index: {FAISS_INDEX_PATH}")

        # Show current embedding info
        if EMBED_INFO_PATH.exists():
            with open(EMBED_INFO_PATH, "r") as f:
                info = json.load(f)
            print(f"   Model: {info['model_name']}")
            print(f"   Chunks: {info['num_chunks']}")
            print(f"   Built: {info['built_at']}")
        return

    print("=" * 60)
    print("  SP 21:2005 RAG -- Offline Ingestion Pipeline")
    print("=" * 60)

    # Load config
    config = load_config()
    model_name = config["embedding"]["active_model"]

    # Load dataset
    data, standards, sections = load_dataset()

    # Build chunk texts
    texts = build_chunk_texts(standards)

    # Embed
    embeddings = embed_chunks(texts, model_name)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Build metadata
    metadata = build_metadata(standards, data["sections"])

    # Save
    save_index(index, metadata, model_name, embeddings.shape[1])

    print()
    print("=" * 60)
    print(f"  [OK] Ingestion complete!")
    print(f"  {len(standards)} standards indexed with {model_name}")
    print(f"  Index stored at: {INDEX_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
