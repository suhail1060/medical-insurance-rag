"""
End-to-end ingestion pipeline.

Usage:  uv run python -m src.data.ingest

Flow:   load docs  →  chunk  →  embed  →  create Qdrant collection  →  upsert
Expected result: 81 points in the 'medical_insurance' collection.
"""

from src.data.loader import load_documents
from src.data.chunker import build_chunks
from src.embeddings.embedder import embed_texts
from src.retrieval.qdrant_store import create_collection, upsert_points


def run_ingest(recreate: bool = True) -> int:
    """Run the full ingestion pipeline. Returns the number of points upserted."""

    # 1. Load raw docs from data/raw/
    raw = load_documents()
    print(f"Loaded {len(raw)} raw documents.")

    # 2. Chunk (split long docs)
    chunks = build_chunks(raw)
    print(f"Built {len(chunks)} chunks.")

    # 3. Embed all chunk texts
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    print(f"Generated {len(vectors)} embeddings.")

    # 4. Build payloads (metadata stored alongside each vector in Qdrant)
    payloads = [
        {
            "text": c.text,
            "doc_id": c.doc_id,
            "title": c.title,
            "category": c.category,
            "topics": c.topics,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    # 5. Create collection and upsert
    create_collection(recreate=recreate)
    upsert_points(vectors, payloads)

    return len(chunks)


if __name__ == "__main__":
    count = run_ingest()
    print(f"\nDone. {count} points ingested.")
