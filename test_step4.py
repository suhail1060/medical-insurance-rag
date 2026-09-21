from src.data.loader import load_documents
from src.data.chunker import build_chunks

docs = load_documents()
print(f"Loaded {len(docs)} documents")

chunks = build_chunks(docs)
print(f"Produced {len(chunks)} chunks")
print(chunks[0])