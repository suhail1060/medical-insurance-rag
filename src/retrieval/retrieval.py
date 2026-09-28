from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.embeddings.embedder import embed_query
from src.retrieval.qdrant_store import get_client, COLLECTION_NAME

def search(query: str, top_k: int = 5, category: str | None = None) -> list[dict]:
    client = get_client()
    query_vector = embed_query(query)

    qdrant_filter = None
    if category:
        qdrant_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    ).points

    return [
        {
            "score": r.score,
            "text": r.payload["text"],
            "title": r.payload["title"],
            "doc_id": r.payload["doc_id"],
            "category": r.payload["category"],
        }
        for r in results
    ]