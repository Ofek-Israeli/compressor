#!/usr/bin/env python3
"""
Terminal menu to create or edit the phase0 YAML config.
Run via: make menuconfig   or   python scripts/menuconfig.py [CONFIG_PATH]
"""

import sys
from pathlib import Path

# Run from repo root so compressor is importable
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import yaml
from compressor.config import REQUIRED_KEYS, load_and_validate, _get

# Key path -> type for prompts (str, int, float, bool)
KEY_TYPE = {
    "seed": "int",
    "device": "str",
    "output_dir": "str",
    "log_level": "str",
    "target.model_id": "str",
    "target.sglang.base_url": "str",
    "target.sglang.timeout_s": "float",
    "target.sglang.max_retries": "int",
    "target.temperature": "float",
    "target.top_p": "float",
    "target.max_new_tokens": "int",
    "target.seed": "int",
    "target.prompt_template": "str",
    "reflector.model_id": "str",
    "reflector.api": "str",
    "reflector.temperature": "float",
    "reflector.top_p": "float",
    "reflector.max_new_tokens": "int",
    "reflector.seed": "int",
    "reflector.compress_prompt_template": "str",
    "reflector.api_base": "str",
    "reflector.api_key_env": "str",
    "data.financebench_path": "str",
    "data.split_ratios.train": "float",
    "data.split_ratios.val": "float",
    "data.split_ratios.test": "float",
    "phase0.verbose_num_samples": "int",
    "phase0.k": "int",
    "phase0.output_path": "str",
    "phase0.filters.drop_special_tokens": "bool",
    "phase0.filters.drop_whitespace_only": "bool",
    "phase0.filters.drop_digit_only": "bool",
    "phase0.correctness.metric": "str",
    "phase0.correctness.threshold": "float",
    "phase0.correctness.tolerance": "float",
    "phase0.correctness.evaluator_model": "str",
    "phase0.correctness.evaluator_api_base": "str",
    "phase0.correctness.evaluator_api_key_env": "str",
    "phase0.correctness.qualitative_forgiving": "bool",
    "phase0.correctness.evaluator_temperature": "float",
    "phase0.correctness.evaluator_top_p": "float",
    "phase0.correctness.evaluator_max_new_tokens": "int",
    "phase0.correctness.evaluator_seed": "int",
    "phase0.target_concurrency": "int",
    "phase0.reflector_concurrency": "int",
    "phase0.judge_concurrency": "int",
}

# Section name -> list of key paths (order preserved)
SECTIONS = [
    ("General", ["seed", "device", "output_dir", "log_level"]),
    ("Target", [
        "target.model_id", "target.sglang.base_url", "target.sglang.timeout_s",
        "target.sglang.max_retries", "target.temperature", "target.top_p",
        "target.max_new_tokens", "target.seed", "target.prompt_template",
    ]),
    ("Reflector", [
        "reflector.model_id", "reflector.api", "reflector.temperature", "reflector.top_p",
        "reflector.max_new_tokens", "reflector.seed", "reflector.compress_prompt_template",
        "reflector.api_base", "reflector.api_key_env",
    ]),
    ("Data", [
        "data.financebench_path", "data.split_ratios.train", "data.split_ratios.val", "data.split_ratios.test",
    ]),
    ("Phase0", [
        "phase0.verbose_num_samples", "phase0.k", "phase0.output_path",
        "phase0.filters.drop_special_tokens", "phase0.filters.drop_whitespace_only", "phase0.filters.drop_digit_only",
        "phase0.target_concurrency", "phase0.reflector_concurrency", "phase0.judge_concurrency",
    ]),
    ("Phase0 Correctness", [
        "phase0.correctness.metric", "phase0.correctness.threshold", "phase0.correctness.tolerance",
        "phase0.correctness.evaluator_model", "phase0.correctness.evaluator_api_base", "phase0.correctness.evaluator_api_key_env",
        "phase0.correctness.qualitative_forgiving",
        "phase0.correctness.evaluator_temperature", "phase0.correctness.evaluator_top_p",
        "phase0.correctness.evaluator_max_new_tokens", "phase0.correctness.evaluator_seed",
    ]),
]


def _set_by_path(cfg: dict, path: str, value) -> None:
    parts = path.split(".")
    cur = cfg
    for i, k in enumerate(parts[:-1]):
        if k not in cur:
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value


def _default_for_typ(typ: str):
    if typ == "str":
        return ""
    if typ == "int":
        return 0
    if typ == "float":
        return 0.0
    if typ == "bool":
        return False
    return ""


def _parse_value(raw: str, typ: str):
    raw = raw.strip()
    if typ == "str":
        return raw
    if typ == "bool":
        if raw.lower() in ("1", "true", "yes", "on", "y"):
            return True
        if raw.lower() in ("0", "false", "no", "off", "n", ""):
            return False
        raise ValueError("Enter true/false, 1/0, or y/n")
    if typ == "int":
        return int(raw)
    if typ == "float":
        return float(raw)
    return raw


def build_empty_config() -> dict:
    cfg = {}
    for path in REQUIRED_KEYS:
        typ = KEY_TYPE.get(path, "str")
        _set_by_path(cfg, path, _default_for_typ(typ))
    return cfg


def load_config(path: Path) -> dict:
    base = build_empty_config()
    if not path.exists():
        return base
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    for path in REQUIRED_KEYS:
        v = _get(data, path)
        if v is None:
            continue
        if isinstance(v, (str, list, dict)) and len(v) == 0:
            continue
        if isinstance(v, float) and (v != v or v == float("inf")):
            continue
        _set_by_path(base, path, v)
    return base


def save_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def format_display_value(path: str, value) -> str:
    typ = KEY_TYPE.get(path, "str")
    if typ == "bool":
        return "true" if value else "false"
    s = str(value)
    if len(s) > 50:
        return s[:47] + "..."
    return s


def run_menu(config_path: Path) -> None:
    cfg = load_config(config_path)

    while True:
        print()
        print("=== Phase 0 Config ===")
        print(f"  Config file: {config_path}")
        print()
        for i, (name, _) in enumerate(SECTIONS, 1):
            print(f"  {i}. {name}")
        print(f"  {len(SECTIONS) + 1}. Save and exit")
        print(f"  {len(SECTIONS) + 2}. Exit without saving")
        try:
            choice = input("\nSelect option (number): ").strip()
            n = int(choice)
        except (ValueError, EOFError):
            continue
        if n == len(SECTIONS) + 1:
            try:
                save_config(cfg, config_path)
                # Validate after save
                load_and_validate(str(config_path))
                print(f"Saved and validated: {config_path}")
            except Exception as e:
                print(f"Error: {e}")
                if input("Save anyway? [y/N]: ").strip().lower() != "y":
                    continue
                save_config(cfg, config_path)
                print("Saved (validation failed).")
            return
        if n == len(SECTIONS) + 2:
            print("Exited without saving.")
            return
        if 1 <= n <= len(SECTIONS):
            section_name, keys = SECTIONS[n - 1]
            while True:
                print()
                print(f"--- {section_name} ---")
                for i, path in enumerate(keys, 1):
                    val = _get(cfg, path)
                    disp = format_display_value(path, val)
                    print(f"  {i}. {path} = {disp}")
                print(f"  {len(keys) + 1}. Back")
                try:
                    sub = input("\nSelect key to edit (number): ").strip()
                    sn = int(sub)
                except (ValueError, EOFError):
                    continue
                if sn == len(keys) + 1:
                    break
                if 1 <= sn <= len(keys):
                    path = keys[sn - 1]
                    typ = KEY_TYPE.get(path, "str")
                    current = _get(cfg, path)
                    prompt = f"  {path} ({typ})"
                    if typ == "bool":
                        prompt += " [true/false]"
                    prompt += f"\n  Current: {format_display_value(path, current)}\n  New value: "
                    try:
                        raw = input(prompt).strip()
                        if raw == "":
                            continue
                        val = _parse_value(raw, typ)
                        _set_by_path(cfg, path, val)
                        print("  Updated.")
                    except Exception as e:
                        print(f"  Invalid: {e}")


def main():
    default_path = _repo_root / "config.yaml"
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    config_path = config_path.resolve()
    run_menu(config_path)


if __name__ == "__main__":
    main()
