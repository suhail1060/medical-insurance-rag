"""
Quick sanity check for the RAG chain.

Usage:  uv run python test_step8.py

Tests three scenarios:
  1. A normal in-scope question
  2. A question with category filter
  3. An out-of-scope question (should trigger refusal)
"""

from src.chains.rag_chain import ask


def print_result(result: dict) -> None:
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print("Sources:")
    for s in result["sources"]:
        print(f"  - [{s['score']}] {s['title']} ({s['doc_id']})")
    print("-" * 60)


if __name__ == "__main__":
    # 1. Normal question — should get a grounded answer about filing claims
    print("=== Test 1: In-scope question ===")
    r1 = ask("How do I file a medical insurance claim?")
    print_result(r1)

    # 2. Filtered question — only search FAQ docs
    print("\n=== Test 2: Filtered by category ===")
    r2 = ask("What is the difference between a premium and a deductible?", category="faq")
    print_result(r2)

    # 3. Out-of-scope question — model should refuse gracefully
    print("\n=== Test 3: Out-of-scope question ===")
    r3 = ask("What is the capital of France?")
    print_result(r3)
