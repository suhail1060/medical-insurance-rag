"""
Qdrant vector store helpers.

Provides a shared client, collection creation, and point upsertion.
The collection uses cosine distance with 384-dim vectors (bge-small-en-v1.5).
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from src.config import QDRANT_URL, QDRANT_API_KEY

COLLECTION_NAME = "medical_insurance"
VECTOR_SIZE = 384  # bge-small-en-v1.5 output dimension

_client = None


def get_client() -> QdrantClient:
    """Return a singleton Qdrant client."""
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


def create_collection(recreate: bool = True) -> None:
    """Create (or recreate) the vector collection in Qdrant."""
    client = get_client()
    if recreate:
        # Delete if it already exists so we start fresh
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in collections:
            client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created (recreate={recreate}).")


def upsert_points(
    vectors: list[list[float]],
    payloads: list[dict],
) -> None:
    """Upsert points into the collection. IDs are incremental ints starting at 0."""
    client = get_client()
    points = [
        PointStruct(id=i, vector=vec, payload=payload)
        for i, (vec, payload) in enumerate(zip(vectors, payloads))
    ]
    # Upsert in a single batch (81 points is small enough)
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted {len(points)} points into '{COLLECTION_NAME}'.")
