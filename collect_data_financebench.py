#!/usr/bin/env python3
"""
Phase 0 data collection: FinanceBench → Target (SGLang) verbose → Reflector (OpenAI) compress
→ correctness filter → tokenize → frequency difference → top-k → output JSON.
See docs/phase0.md. No defaults, no fallbacks; all from kconfig.
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone

# Add minions_channel for RemoteVerdictEvaluator
_root = Path(__file__).resolve().parent
_minions = _root.parent / "minions_channel"
if os.environ.get("MINIONS_CHANNEL_PATH"):
    _minions = Path(os.environ["MINIONS_CHANNEL_PATH"])
if _minions.exists():
    sys.path.insert(0, str(_minions))
try:
    from evaluate.correctness import RemoteVerdictEvaluator
    from minions.usage import Usage as MinionsUsage
except ImportError:
    RemoteVerdictEvaluator = None
    MinionsUsage = None

from compressor.config import load_and_validate
from compressor.data import (
    load_financebench,
    split_train_val_test,
    split_train_val_test_explicit,
    parse_indices,
    Example,
)
from compressor.sglang_client import SGLangClient
from compressor.openai_chat import OpenAIChatClient, Usage
from compressor.token_utils import (
    load_tokenizer,
    get_stop_token_ids,
    tokenize,
    filter_candidate_tokens,
    compute_freq_and_delta,
    build_v_steer_and_delta_by_id,
)

LOG = logging.getLogger(__name__)

REFLECTOR_COMPRESS_PROMPT_TEMPLATE = """
You will be given a question and an assistant answer. Rewrite the answer in the fewest words while preserving its meaning. Remove all fluff, padding, hedging, and redundancy. Output ONLY the rewritten answer—no preamble, no bullets unless essential, no explanations.

Question:
{query}

Assistant answer:
{prev_answer}

Rewritten answer:
"""

# Reflector compression prompt (query + verbose answer → rewritten answer). Not configurable.

def _run_target_verbose(
    cfg: dict,
    train: list,
    sglang: SGLangClient,
    resume_verbose_path: Path,
    resume: bool,
) -> dict:
    """Generate M_verbose Target (SGLang) answers per train example. Returns example_id -> list of verbose texts."""
    out: dict = {}
    template = cfg["target"]["prompt_template"]
    M = int(cfg["phase0"]["verbose_num_samples"])
    concurrency = int(cfg["phase0"]["target_concurrency"])
    target_seed = int(cfg["target"]["seed"])

    if not resume and resume_verbose_path.exists():
        resume_verbose_path.unlink()
    if resume and resume_verbose_path.exists():
        with open(resume_verbose_path) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                out[obj["example_id"]] = obj.get("verbose_answers", [])

    todo = [ex for ex in train if ex["example_id"] not in out or len(out[ex["example_id"]]) < M]
    if not todo:
        LOG.info("Target verbose: all %s examples already have answers (skip)", len(train))
        return out

    tok = load_tokenizer(cfg["target"]["model_id"])
    stop_token_ids = get_stop_token_ids(tok)
    LOG.info("Target verbose: stop_token_ids=%s", stop_token_ids)

    LOG.info(
        "Target verbose: generating for %s examples (%s samples each), concurrency=%s",
        len(todo), M, concurrency,
    )

    def one_example(ex: Example):
        eid = ex["example_id"]
        LOG.info("Target verbose: starting example %s", eid)
        c, q = ex["context"], ex["query"]
        user_body = template.format(context=c, query=q)
        messages = [
            {"role": "system", "content": "Answer the question using the provided context."},
            {"role": "user",   "content": user_body},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        LOG.info("Target verbose: sending %s requests for example %s (prompt_len=%s)", M, eid, len(prompt))
        answers = []
        for _ in range(M):
            text = sglang.generate(
                prompt=prompt,
                temperature=float(cfg["target"]["temperature"]),
                top_p=float(cfg["target"]["top_p"]),
                max_new_tokens=int(cfg["target"]["max_new_tokens"]),
                seed=target_seed,
                stop_token_ids=stop_token_ids,
            )
            answers.append(text)
        return ex["example_id"], answers

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as exe:
        futures = {exe.submit(one_example, ex): ex for ex in todo}
        LOG.info("Target verbose: submitted %s tasks, waiting for completions...", len(futures))
        for fut in as_completed(futures):
            eid, answers = fut.result()
            out[eid] = answers
            done += 1
            if done % 10 == 0 or done == len(todo):
                LOG.info("Target verbose: %s / %s examples done", done, len(todo))
            with open(resume_verbose_path, "a") as f:
                f.write(json.dumps({"example_id": eid, "verbose_answers": answers}) + "\n")
    return out


def _run_reflector_compress(
    cfg: dict,
    train: list,
    verbose_by_id: dict,
    reflector_client: OpenAIChatClient,
    compressed_by_key: dict | None = None,
) -> dict:
    """For each (ex, pair_idx) not in compressed_by_key, compress via Reflector. Returns (example_id, pair_idx) -> compressed text."""
    out = compressed_by_key if compressed_by_key is not None else {}
    concurrency = int(cfg["phase0"]["reflector_concurrency"])

    tasks = [(ex, v, i) for ex in train for i, v in enumerate(verbose_by_id.get(ex["example_id"], [])) if (ex["example_id"], i) not in out]
    if not tasks:
        return out

    def one_pair(ex: Example, verbose: str, pair_idx: int):
        prompt = REFLECTOR_COMPRESS_PROMPT_TEMPLATE.format(query=ex["query"], prev_answer=verbose)
        messages = [{"role": "user", "content": prompt}]
        resp, _ = reflector_client.chat(messages)
        return (ex["example_id"], pair_idx), resp[0] if resp else ""

    with ThreadPoolExecutor(max_workers=concurrency) as exe:
        futures = {exe.submit(one_pair, ex, v, i): (ex["example_id"], i) for ex, v, i in tasks}
        for fut in as_completed(futures):
            key, comp = fut.result()
            out[key] = comp
    return out


def _run_correctness(
    cfg: dict,
    train: list,
    verbose_by_id: dict,
    compressed_by_key: dict,
    correctness_by_key: dict | None = None,
) -> dict:
    """Run judge on each compressed answer not already in correctness_by_key. Returns (example_id, pair_idx) -> (is_correct, result)."""
    if RemoteVerdictEvaluator is None:
        raise RuntimeError("RemoteVerdictEvaluator not found; add minions_channel to path or set MINIONS_CHANNEL_PATH")
    results = dict(correctness_by_key) if correctness_by_key else {}
    pc = cfg["phase0"]["correctness"]
    judge_client = OpenAIChatClient(
        api_base=pc["evaluator_api_base"],
        api_key_env=pc["evaluator_api_key_env"],
        model_id=pc["evaluator_model"],
        temperature=float(pc["evaluator_temperature"]),
        top_p=float(pc["evaluator_top_p"]),
        max_tokens=int(pc["evaluator_max_new_tokens"]),
        seed=int(pc["evaluator_seed"]),
        usage_class=MinionsUsage if MinionsUsage is not None else Usage,
    )
    evaluator = RemoteVerdictEvaluator(
        remote_client=judge_client,
        numerical_tolerance=float(pc["tolerance"]),
        qualitative_forgiving=bool(pc["qualitative_forgiving"]),
    )
    concurrency = int(cfg["phase0"]["judge_concurrency"])

    def one_eval(ex: Example, pair_idx: int, comp: str):
        gold = ex["gold_answer"]
        q = ex["query"]
        r = evaluator.evaluate(predicted=comp, ground_truth=gold, question=q)
        return (ex["example_id"], pair_idx), (r.is_correct, r)

    tasks = []
    for ex in train:
        for i, verbose in enumerate(verbose_by_id.get(ex["example_id"], [])):
            key = (ex["example_id"], i)
            if key not in compressed_by_key or key in results:
                continue
            tasks.append((ex, i, compressed_by_key[key]))
    if tasks:
        with ThreadPoolExecutor(max_workers=concurrency) as exe:
            futures = {exe.submit(one_eval, ex, i, comp): (ex["example_id"], i) for ex, i, comp in tasks}
            for fut in as_completed(futures):
                key, (ok, res) = fut.result()
                results[key] = (ok, res)
    return results


def _load_pairs_checkpoint(pairs_path: Path) -> tuple:
    """Load phase0_pairs.jsonl. Returns (compressed_by_key, correctness_by_key). correctness value is (is_correct, SimpleNamespace)."""
    compressed_by_key: dict = {}
    correctness_by_key: dict = {}
    if not pairs_path.exists():
        return compressed_by_key, correctness_by_key
    with open(pairs_path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            eid = obj["example_id"]
            i = obj["pair_idx"]
            key = (eid, i)
            compressed_by_key[key] = obj.get("compressed", "")
            res = SimpleNamespace(
                is_correct=obj.get("is_correct", False),
                confidence=float(obj.get("confidence", 0)),
                reasoning=obj.get("reasoning", ""),
                category=obj.get("category", ""),
            )
            correctness_by_key[key] = (res.is_correct, res)
    return compressed_by_key, correctness_by_key


def _write_pairs_checkpoint(pairs_path: Path, compressed_by_key: dict, correctness_by_key: dict) -> None:
    """Write phase0_pairs.jsonl with one line per (example_id, pair_idx)."""
    with open(pairs_path, "w") as f:
        for (eid, i), comp in sorted(compressed_by_key.items(), key=lambda x: (x[0][0], x[0][1])):
            ok, res = correctness_by_key.get((eid, i), (False, None))
            row = {"example_id": eid, "pair_idx": i, "compressed": comp}
            if res is not None:
                row["is_correct"] = ok
                row["confidence"] = getattr(res, "confidence", 0)
                row["reasoning"] = getattr(res, "reasoning", "")
                row["category"] = getattr(res, "category", "")
            else:
                row["is_correct"] = ok
            f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Phase 0: collect data for delta learning (FinanceBench)")
    ap.add_argument("--config", required=True, help="Path to kconfig YAML")
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint files")
    args = ap.parse_args()

    cfg = load_and_validate(args.config)
    log_level = cfg.get("log_level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level))
    LOG.info("kconfig loaded and validated")

    # Data
    examples = load_financebench(cfg["data"]["financebench_path"])
    data_cfg = cfg["data"]
    seed = int(cfg["seed"])
    if data_cfg.get("use_explicit_indices"):
        train_s = (data_cfg.get("train_indices") or "").strip()
        val_s = (data_cfg.get("val_indices") or "").strip()
        test_s = (data_cfg.get("test_indices") or "").strip()
        if not train_s:
            raise ValueError(
                "data.use_explicit_indices is true but data.train_indices is empty; "
                "set train indices (comma-separated 0-based). Val/test may be empty."
            )
        train_idx = parse_indices(train_s)
        val_idx = parse_indices(val_s) if val_s else []
        test_idx = parse_indices(test_s) if test_s else []
        train, val, test = split_train_val_test_explicit(
            examples, train_idx, val_idx, test_idx
        )
        LOG.info(
            "split (explicit indices): train=%s val=%s test=%s",
            len(train), len(val), len(test),
        )
    else:
        ratios = data_cfg["split_ratios"]
        train, val, test = split_train_val_test(
            examples,
            seed,
            float(ratios["train"]),
            float(ratios["val"]),
            float(ratios["test"]),
        )
        LOG.info("split: train=%s val=%s test=%s", len(train), len(val), len(test))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_verbose = output_dir / "phase0_verbose.jsonl"
    resume_pairs = output_dir / "phase0_pairs.jsonl"
    if not args.resume and resume_pairs.exists():
        resume_pairs.unlink()

    # Target (SGLang)
    sglang = SGLangClient(
        base_url=cfg["target"]["sglang"]["base_url"],
        model_id=cfg["target"]["model_id"],
        timeout_s=float(cfg["target"]["sglang"]["timeout_s"]),
        max_retries=int(cfg["target"]["sglang"]["max_retries"]),
    )
    if not sglang.health_check():
        LOG.error("Cannot reach SGLang server at %s — aborting", cfg["target"]["sglang"]["base_url"])
        return 1
    verbose_by_id = _run_target_verbose(cfg, train, sglang, resume_verbose, resume=args.resume)
    LOG.info("Target verbose: %s examples with answers", len(verbose_by_id))

    # Reflector + Correctness (with optional resume from phase0_pairs.jsonl)
    ref = cfg["reflector"]
    reflector_client = OpenAIChatClient(
        api_base=ref["api_base"],
        api_key_env=ref["api_key_env"],
        model_id=ref["model_id"],
        temperature=float(ref["temperature"]),
        top_p=float(ref["top_p"]),
        max_tokens=int(ref["max_new_tokens"]),
        seed=int(ref["seed"]),
    )
    loaded_comp, loaded_corr = _load_pairs_checkpoint(resume_pairs) if args.resume else ({}, {})
    compressed_by_key = _run_reflector_compress(cfg, train, verbose_by_id, reflector_client, loaded_comp)
    LOG.info("Reflector: %s compressed answers", len(compressed_by_key))

    correctness_by_key = _run_correctness(cfg, train, verbose_by_id, compressed_by_key, loaded_corr)
    _write_pairs_checkpoint(resume_pairs, compressed_by_key, correctness_by_key)

    # Retained and discarded pairs, weights
    retained: list = []
    train_output = []
    for ex in train:
        eid = ex["example_id"]
        verboses = verbose_by_id.get(eid, [])
        pairs_ok = []
        pairs_discarded = []
        for i, v in enumerate(verboses):
            key = (eid, i)
            comp = compressed_by_key.get(key, "")
            ok, res = correctness_by_key.get(key, (False, None))
            corr = {"is_correct": ok, "confidence": getattr(res, "confidence", 0), "category": getattr(res, "category", ""), "reasoning": getattr(res, "reasoning", "")} if res else {}
            if ok:
                pairs_ok.append((v, comp, res))
            else:
                pairs_discarded.append({"verbose_answer": v, "compressed_answer": comp, "correctness": corr})
        m = len(pairs_ok)
        w = 1.0 / m if m else None
        train_output.append({
            "example_id": eid,
            "context": ex["context"],
            "query": ex["query"],
            "gold_answer": ex["gold_answer"],
            "num_verbose_generated": len(verboses),
            "num_retained": m,
            "sample_weight": w,
            "retained_pairs": [
                {
                    "verbose_answer": v,
                    "compressed_answer": c,
                    "correctness": {"is_correct": True, "confidence": getattr(r, "confidence", 0), "category": getattr(r, "category", ""), "reasoning": getattr(r, "reasoning", "")}
                    if r else {},
                }
                for v, c, r in pairs_ok
            ],
            "discarded_pairs": pairs_discarded,
        })
        for v, c, r in pairs_ok:
            retained.append((v, c, w))

    # Tokenize and compute frequency difference (verbose vs compressed)
    tok = load_tokenizer(cfg["target"]["model_id"])
    filters = cfg["phase0"]["filters"]
    candidates = filter_candidate_tokens(
        tok,
        drop_special=bool(filters["drop_special_tokens"]),
        drop_whitespace_only=bool(filters["drop_whitespace_only"]),
        drop_digit_only=bool(filters["drop_digit_only"]),
    )
    retained_with_tokens = [
        (tokenize(tok, v), tokenize(tok, c), w) for v, c, w in retained
    ]
    k = int(cfg["phase0"]["k"])
    top_ids, delta, freq_raw, freq_comp = compute_freq_and_delta(retained_with_tokens, k, candidates)
    v_steer, v_steer_token_ids, frequency_difference_by_token_id = build_v_steer_and_delta_by_id(tok, top_ids, delta)

    # Add token_ids to retained_pairs in train_output
    for ex in train_output:
        for p in ex["retained_pairs"]:
            p["verbose_token_ids"] = tokenize(tok, p["verbose_answer"])
            p["compressed_token_ids"] = tokenize(tok, p["compressed_answer"])

    # Output JSON
    out_path = cfg["phase0"]["output_path"]
    n_retained_ex = sum(1 for ex in train_output if ex["num_retained"] > 0)
    n_discarded_ex = len(train_output) - n_retained_ex
    total_pairs = sum(ex["num_retained"] for ex in train_output)
    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "seed": seed,
            "target_model_id": cfg["target"]["model_id"],
            "reflector_model": cfg["reflector"]["model_id"],
            "evaluator_model": cfg["phase0"]["correctness"]["evaluator_model"],
            "verbose_num_samples": cfg["phase0"]["verbose_num_samples"],
            "k": k,
            "correctness_tolerance": cfg["phase0"]["correctness"]["tolerance"],
            "split_counts": {"train": len(train), "val": len(val), "test": len(test)},
            "train_retained_examples": n_retained_ex,
            "train_discarded_examples": n_discarded_ex,
            "total_retained_pairs": total_pairs,
            "config_snapshot": cfg,
        },
        "token_selection": {
            "k": len(v_steer_token_ids),
            "v_steer": v_steer,
            "v_steer_token_ids": v_steer_token_ids,
            "frequency_difference_by_token_id": frequency_difference_by_token_id,
            "freq_raw": {str(t): freq_raw.get(t, 0) for t in v_steer_token_ids},
            "freq_comp": {str(t): freq_comp.get(t, 0) for t in v_steer_token_ids},
        },
        "splits": {
            "train": train_output,
            "val": [{"example_id": ex["example_id"], "context": ex["context"], "query": ex["query"], "gold_answer": ex["gold_answer"]} for ex in val],
            "test": [{"example_id": ex["example_id"], "context": ex["context"], "query": ex["query"], "gold_answer": ex["gold_answer"]} for ex in test],
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    LOG.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
