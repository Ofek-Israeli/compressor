"""
Token filters and frequency-difference computation for Phase 0.
Uses Target tokenizer only. See docs/phase0.md §4.8–4.10.
"""

import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from transformers import AutoTokenizer


def load_tokenizer(model_id: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_id)


def get_stop_token_ids(tok: AutoTokenizer) -> List[int]:
    """
    Stop token IDs for Target generation (EOS + Llama-3 EOT when present).
    Matches minions_channel minions/clients/sglang.py stop_ids logic so
    generation stops at end-of-turn and does not over-decode.
    """
    stop_ids: List[int] = []
    if tok.eos_token_id is not None:
        stop_ids.append(tok.eos_token_id)
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and eot_id != getattr(tok, "unk_token_id", None):
        stop_ids.append(eot_id)
    if not stop_ids:
        stop_ids = [128001, 128009]  # Llama-3 EOS and EOT fallback
    return stop_ids


def tokenize(tok: AutoTokenizer, text: str) -> List[int]:
    return tok.encode(text, add_special_tokens=False)


def decode_one(tok: AutoTokenizer, token_id: int) -> str:
    return tok.decode([token_id])


def _is_whitespace_or_punctuation_only(s: str) -> bool:
    u = s.strip()
    if not u:
        return True
    return all(unicodedata.category(c).startswith("P") for c in u)


def _is_digit_only(s: str) -> bool:
    u = s.strip()
    if not u:
        return False
    return all(c in "0123456789" for c in u)


def filter_candidate_tokens(
    tok: AutoTokenizer,
    drop_special: bool,
    drop_whitespace_only: bool,
    drop_digit_only: bool,
) -> Set[int]:
    """Build candidate set V' (excludes EOS always)."""
    vocab_size = tok.vocab_size
    special_ids = set(tok.all_special_ids) if drop_special else set()
    eos_id = tok.eos_token_id
    if eos_id is not None:
        special_ids.add(eos_id)
    exclude: Set[int] = set(special_ids)
    for tid in range(vocab_size):
        if tid in exclude:
            continue
        s = decode_one(tok, tid)
        if drop_whitespace_only and _is_whitespace_or_punctuation_only(s):
            exclude.add(tid)
            continue
        if drop_digit_only and _is_digit_only(s):
            exclude.add(tid)
    return set(range(vocab_size)) - exclude


def compute_freq_and_delta(
    retained_pairs: List[Tuple[List[int], List[int], float]],
    k: int,
    candidate_ids: Set[int],
) -> Tuple[List[int], Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    retained_pairs: list of (tokens_verbose, tokens_comp, weight).
    Returns (top_k_token_ids, frequency_difference_by_id, freq_raw, freq_comp).
    """
    C_raw: Dict[int, float] = defaultdict(float)
    Z_raw = 0.0
    C_comp: Dict[int, float] = defaultdict(float)
    Z_comp = 0.0
    for (tv, tc, w) in retained_pairs:
        for t in tv:
            C_raw[t] += w
        Z_raw += w * len(tv)
        for t in tc:
            C_comp[t] += w
        Z_comp += w * len(tc)
    freq_raw = {t: (C_raw[t] / Z_raw if Z_raw else 0) for t in C_raw}
    freq_comp = {t: (C_comp[t] / Z_comp if Z_comp else 0) for t in C_comp}
    all_ids = set(freq_raw) | set(freq_comp)
    delta = {t: freq_raw.get(t, 0) - freq_comp.get(t, 0) for t in all_ids}
    candidates = [t for t in all_ids if t in candidate_ids]
    candidates.sort(key=lambda t: -delta[t])
    top = candidates[:k]
    return top, delta, freq_raw, freq_comp


def build_v_steer_and_delta_by_id(
    tok: AutoTokenizer,
    top_ids: List[int],
    delta: Dict[int, float],
) -> Tuple[List[dict], List[int], Dict[str, float]]:
    """Build v_steer list and frequency_difference_by_token_id from top_ids and frequency-difference map."""
    v_steer = [
        {"token_id": t, "token_str": decode_one(tok, t), "frequency_difference": delta[t]}
        for t in top_ids
    ]
    v_steer_token_ids = top_ids
    frequency_difference_by_token_id = {str(t): delta[t] for t in top_ids}
    return v_steer, v_steer_token_ids, frequency_difference_by_token_id
