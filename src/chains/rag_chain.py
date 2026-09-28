"""
RAG chain: retriever → prompt → LLM → structured answer with citations.

Supports multiple LLM backends (configurable via model parameter):
  - "gemma"   : Gemma 3 1B via Ollama (local, default)
  - "gemini"  : Google Gemini Flash (hosted, requires API key)

Usage:
    from src.chains.rag_chain import ask
    result = ask("How do I file a claim?")                # uses default (gemma)
    result = ask("How do I file a claim?", model="gemini") # uses hosted Gemini

Why this design:
- We use our own retriever (src.retrieval) rather than LangChain's retriever
  abstraction, so we keep full control over filtering and payload access.
- The prompt is explicit about grounding: answer ONLY from the provided context,
  and say so when the context doesn't cover the question.
- Sources (doc_id + title) are extracted from retrieval results, giving the user
  traceability back to the original documents.
- Model selection is a simple string switch so we can compare Gemini vs fine-tuned
  Gemma side-by-side during evaluation (Phase 4).
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import GOOGLE_API_KEY
from src.retrieval.retrieval import search

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# System message sets the ground rules; human message injects context + query.
# "If the answer is not in the context" → forces the model to refuse gracefully
# rather than hallucinate.

_SYSTEM_MSG = (
    "You are a helpful medical insurance assistant. "
    "Answer the user's question using ONLY the provided context. "
    "If the context does not contain enough information to answer, "
    'say "I don\'t have enough information in the provided documents to answer that." '
    "Be concise, accurate, and cite the document title when referencing specific information."
)

_HUMAN_MSG = """Context:
{context}

Question: {question}"""

_prompt = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_MSG), ("human", _HUMAN_MSG)]
)

# ---------------------------------------------------------------------------
# Output parser (shared across all models)
# ---------------------------------------------------------------------------
_parser = StrOutputParser()

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry returns a LangChain LLM instance.
# New backends are added here as simple factory functions.

def _get_gemini():
    """Gemini 2.0 Flash: hosted, fast, good for grounded Q&A."""
    return ChatGoogleGenerativeAI(
        model="gemini-3.7-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        max_retries=3,  # retry on transient 503/429 errors
    )


def _get_gemma():
    """Gemma 3 1B via Ollama: local, no API key needed, light on resources."""
    return ChatOllama(
        model="gemma3:1b",
        temperature=0.2,
    )


_MODEL_REGISTRY = {
    "gemma": _get_gemma,
    "gemini": _get_gemini,
}

# Cache instantiated models so we don't recreate them on every call
_model_cache: dict = {}


def _get_llm(model: str):
    """Get (or create) the LLM instance for the given model name."""
    if model not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    if model not in _model_cache:
        _model_cache[model] = _MODEL_REGISTRY[model]()
    return _model_cache[model]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def ask(
    question: str,
    top_k: int = 5,
    category: str | None = None,
    model: str = "gemma",
) -> dict:
    """
    End-to-end RAG: retrieve relevant chunks, feed them to the LLM, return
    the answer plus source citations.

    Args:
        question: The user's question.
        top_k:    Number of chunks to retrieve (default 5).
        category: Optional category filter for retrieval (e.g. "faq", "policy").
        model:    Which LLM backend to use ("gemini" or "gemma").

    Returns:
        {
            "question": str,
            "answer":   str,
            "sources":  [{"doc_id": str, "title": str, "score": float}, ...],
            "model":    str,
        }
    """
    # 1. Retrieve relevant chunks
    hits = search(question, top_k=top_k, category=category)

    # 2. Build context string from retrieved chunks
    #    Numbering helps the LLM reference specific passages.
    context_parts = []
    for i, hit in enumerate(hits, 1):
        context_parts.append(
            f"[{i}] (Source: {hit['title']}, ID: {hit['doc_id']})\n{hit['text']}"
        )
    context = "\n\n".join(context_parts)

    # 3. Build and run the chain with the selected model
    llm = _get_llm(model)
    chain = _prompt | llm | _parser
    answer = chain.invoke({"context": context, "question": question})

    # 4. Deduplicate sources (a doc may appear in multiple chunks)
    seen = set()
    sources = []
    for hit in hits:
        key = hit["doc_id"]
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "doc_id": hit["doc_id"],
                    "title": hit["title"],
                    "score": round(hit["score"], 4),
                }
            )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "model": model,
    }
