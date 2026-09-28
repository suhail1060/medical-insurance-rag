from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    # bge models recommend a query prefix for queries, but not for documents
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return embeddings.tolist()

def embed_query(query: str) -> list[float]:
    model = get_model()
    # bge models expect this instruction prefix for search queries specifically
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    embedding = model.encode(prefixed, normalize_embeddings=True)
    return embedding.tolist()