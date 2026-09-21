import glob
import yaml
from dataclasses import dataclass, field

@dataclass
class Chunk:
    text: str
    doc_id: str
    title: str
    category: str
    topics: list[str] = field(default_factory=list)
    chunk_index: int = 0  # position within doc, for sub-chunked docs

def parse_doc(filepath: str) -> tuple[dict, str]:
    """Split a markdown file into (frontmatter_dict, body_text)."""
    content = open(filepath, encoding="utf-8").read()
    _, frontmatter_raw, body = content.split("---", 2)
    meta = yaml.safe_load(frontmatter_raw)
    return meta, body.strip()

def load_documents(data_dir: str = "data/raw") -> list[Chunk]:
    chunks = []
    for filepath in sorted(glob.glob(f"{data_dir}/*.md")):
        meta, body = parse_doc(filepath)
        chunks.append(Chunk(
            text=body,
            doc_id=meta["doc_id"],
            title=meta["title"],
            category=meta["category"],
            topics=meta.get("topics", []),
            chunk_index=0,
        ))
    return chunks