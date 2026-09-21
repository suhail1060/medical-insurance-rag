from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.data.loader import Chunk

def sub_chunk_if_needed(chunk: Chunk, max_chars: int = 1500, overlap: int = 150) -> list[Chunk]:
    """Only splits if doc exceeds max_chars. Most of our docs won't trigger this."""
    if len(chunk.text) <= max_chars:
        return [chunk]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    pieces = splitter.split_text(chunk.text)
    return [
        Chunk(text=piece, doc_id=chunk.doc_id, title=chunk.title,
              category=chunk.category, topics=chunk.topics, chunk_index=i)
        for i, piece in enumerate(pieces)
    ]

def build_chunks(raw_chunks: list[Chunk]) -> list[Chunk]:
    result = []
    for c in raw_chunks:
        result.extend(sub_chunk_if_needed(c))
    return result