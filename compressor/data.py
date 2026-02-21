"""
FinanceBench loading and train/val/test split.
Context = evidence_text_full_page, deduplicated by (evidence_doc_name, evidence_page_num).
See docs/phase0.md §4.2–4.4.
"""

import json
import random
from pathlib import Path
from typing import Any, List, Tuple, TypedDict


class Example(TypedDict):
    example_id: str
    context: str
    query: str
    gold_answer: str


def load_financebench(jsonl_path: str) -> List[Example]:
    """Load JSONL and extract (example_id, context, query, gold_answer) per row."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"FinanceBench path not found: {jsonl_path}")
    examples: List[Example] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            example_id = row.get("financebench_id", "")
            query = row.get("question", "")
            gold_answer = row.get("answer", "")
            # Context: concat evidence_text_full_page, dedupe by (evidence_doc_name, evidence_page_num)
            seen: set = set()
            parts: List[str] = []
            for ev in row.get("evidence", []) or []:
                key = (ev.get("evidence_doc_name"), ev.get("evidence_page_num"))
                if key in seen:
                    continue
                seen.add(key)
                text = ev.get("evidence_text_full_page", "")
                if text:
                    parts.append(text)
            context = "\n\n".join(parts)
            examples.append(
                Example(
                    example_id=example_id,
                    context=context,
                    query=query,
                    gold_answer=gold_answer,
                )
            )
    return examples


def split_train_val_test(
    examples: List[Example],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Example], List[Example], List[Example]]:
    """Deterministic split: sort by example_id, shuffle with seed, then split by ratios."""
    sorted_ex = sorted(examples, key=lambda e: e["example_id"])
    rng = random.Random(seed)
    rng.shuffle(sorted_ex)
    n = len(sorted_ex)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    train = sorted_ex[:t]
    val = sorted_ex[t : t + v]
    test = sorted_ex[t + v :]
    return train, val, test
