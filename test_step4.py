# Chunking test

# from src.data.loader import load_documents
# from src.data.chunker import build_chunks

# docs = load_documents()
# print(f"Loaded {len(docs)} documents")

# chunks = build_chunks(docs)
# print(f"Produced {len(chunks)} chunks")
# print(chunks[0])

# Embedding test
from src.data.loader import load_documents
from src.data.chunker import build_chunks
from src.embeddings.embedder import embed_texts, embed_query

docs = load_documents()
chunks = build_chunks(docs)

texts = [c.text for c in chunks[:5]]  # test on first 5 only, fast
vectors = embed_texts(texts)

print(f"Generated {len(vectors)} embeddings")
print(f"Embedding dimension: {len(vectors[0])}")

q_vec = embed_query("What is a deductible?")
print(f"Query embedding dimension: {len(q_vec)}")