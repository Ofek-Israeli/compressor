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


def parse_indices(indices_str: str) -> List[int]:
    """Parse comma-separated indices string into list of non-negative integers (0-based)."""
    if not indices_str or not str(indices_str).strip():
        return []
    return [int(x.strip()) for x in str(indices_str).strip().split(",") if x.strip()]


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


def split_train_val_test_explicit(
    examples: List[Example],
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
) -> Tuple[List[Example], List[Example], List[Example]]:
    """Split by explicit 0-based indices into the examples list (JSONL order).
    Each index must be in [0, len(examples)); overlapping indices are allowed but usually undesired.
    """
    n = len(examples)
    for idx in train_indices + val_indices + test_indices:
        if idx < 0 or idx >= n:
            raise ValueError(
                f"Index {idx} out of range [0, {n}); examples list has {n} items (0-based JSONL order)"
            )
    train = [examples[i] for i in train_indices]
    val = [examples[i] for i in val_indices]
    test = [examples[i] for i in test_indices]
    return train, val, test
