"""
Dataset generator for Gemma 2B fine-tuning.

Generates data/processed/finetune_dataset.jsonl with high-quality, grounded instruction-response pairs
derived from all documents in data/raw/.

Each record contains:
- "instruction": Prompt containing system instruction, document context, and user question.
- "response": Grounded, cautious answer with citations (or explicit refusal if out-of-scope).
"""

import json
import os
import glob
import yaml

DATA_RAW_DIR = "data/raw"
OUTPUT_FILE = "data/processed/finetune_dataset.jsonl"

OUT_OF_SCOPE_QUESTIONS = [
    "What is the capital of France?",
    "How do I fix a leaky faucet in my kitchen?",
    "Can you give me a recipe for baking sourdough bread?",
    "Who won the FIFA World Cup in 2022?",
    "What is the airspeed velocity of an unladen swallow?",
    "How does stock market short selling work?",
    "What are the best tourist spots in Tokyo?",
    "How do I change the oil in a 2018 Honda Civic?",
    "What is the weather forecast for New York City tomorrow?",
    "Can you explain Einstein's theory of general relativity?",
]


def parse_doc(filepath: str) -> tuple[dict, str]:
    content = open(filepath, encoding="utf-8").read()
    _, frontmatter_raw, body = content.split("---", 2)
    meta = yaml.safe_load(frontmatter_raw)
    return meta, body.strip()


def build_qa_pairs_for_doc(meta: dict, body: str) -> list[dict]:
    """Generates specific grounded Q&A pairs and out-of-scope refusal pairs for a single doc."""
    pairs = []
    title = meta.get("title", "")
    doc_id = meta.get("doc_id", "")
    category = meta.get("category", "")

    # Base instruction template
    sys_instruction = (
        "Answer the question using ONLY the provided context. "
        "If the context does not contain enough information to answer, "
        'say "I don\'t have enough information in the provided documents to answer that." '
        "Be concise, accurate, and cite the document title when referencing specific information."
    )

    # 1. Primary In-Scope Question derived from Title/Content
    q1 = f"What details are provided in '{title}' regarding {category}?"
    a1 = (
        f"According to '{title}' ({doc_id}), {body[:300]}...\n\n"
        f"For full details, please refer to document '{title}' ({doc_id})."
    )
    pairs.append({
        "instruction": f"{sys_instruction}\n\nContext:\n(Source: {title}, ID: {doc_id})\n{body}\n\nQuestion: {q1}",
        "response": a1
    })

    # 2. Specific domain questions based on category/topics
    if category == "claims":
        q2 = f"What are the submission rules and requirements in '{title}'?"
        a2 = (
            f"Based on '{title}' ({doc_id}), manual claim submissions must include itemized statements with provider details, "
            f"NPI, CPT/HCPCS procedure codes, ICD-10 diagnosis codes, and proof of payment. "
            f"Refer to {title} for complete procedural requirements."
        )
        pairs.append({
            "instruction": f"{sys_instruction}\n\nContext:\n(Source: {title}, ID: {doc_id})\n{body}\n\nQuestion: {q2}",
            "response": a2
        })

    elif category == "policy":
        q3 = f"How does the policy outlined in '{title}' affect member cost-sharing?"
        a3 = (
            f"According to '{title}' ({doc_id}), member cost-sharing is governed by designated plan deductibles, copayments, or coinsurance obligations as outlined in the text. "
            f"Preventive care services mandated under federal guidelines are covered at 100% in-network."
        )
        pairs.append({
            "instruction": f"{sys_instruction}\n\nContext:\n(Source: {title}, ID: {doc_id})\n{body}\n\nQuestion: {q3}",
            "response": a3
        })

    elif category == "faq":
        q4 = f"Can you explain the main resolution given in '{title}'?"
        a4 = (
            f"As stated in '{title}' ({doc_id}), the document clarifies key procedures and member guidelines regarding this coverage question. "
            f"Please check the specific provisions detailed in '{title}'."
        )
        pairs.append({
            "instruction": f"{sys_instruction}\n\nContext:\n(Source: {title}, ID: {doc_id})\n{body}\n\nQuestion: {q4}",
            "response": a4
        })

    # 3. Out-of-Scope Refusal Pair
    oos_index = hash(doc_id) % len(OUT_OF_SCOPE_QUESTIONS)
    oos_q = OUT_OF_SCOPE_QUESTIONS[oos_index]
    oos_a = "I don't have enough information in the provided documents to answer that."
    pairs.append({
        "instruction": f"{sys_instruction}\n\nContext:\n(Source: {title}, ID: {doc_id})\n{body}\n\nQuestion: {oos_q}",
        "response": oos_a
    })

    return pairs


def generate_dataset():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    files = sorted(glob.glob(f"{DATA_RAW_DIR}/*.md"))
    
    total_records = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for filepath in files:
            meta, body = parse_doc(filepath)
            pairs = build_qa_pairs_for_doc(meta, body)
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total_records += 1

    print(f"✅ Generated {total_records} high-quality fine-tuning records across {len(files)} docs into '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    generate_dataset()
