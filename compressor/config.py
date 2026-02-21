"""
Load and validate kconfig (YAML or Kconfig .config). No defaults, no fallbacks.
Any missing or null key terminates the program.
See docs/phase0.md §3 and §4.1.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _get(cfg: Dict[str, Any], path: str) -> Any:
    """Get nested key; path is dot-separated e.g. 'phase0.correctness.tolerance'."""
    v = cfg
    for k in path.split("."):
        if not isinstance(v, dict):
            return None
        v = v.get(k)
    return v


# All required keys (nested paths). All must be present and non-null/non-empty.
REQUIRED_KEYS: List[str] = [
    "seed",
    "device",
    "output_dir",
    "log_level",
    "target.model_id",
    "target.sglang.base_url",
    "target.sglang.timeout_s",
    "target.sglang.max_retries",
    "target.temperature",
    "target.top_p",
    "target.max_new_tokens",
    "target.seed",
    "target.prompt_template",
    "reflector.model_id",
    "reflector.api",
    "reflector.temperature",
    "reflector.top_p",
    "reflector.max_new_tokens",
    "reflector.seed",
    "reflector.api_base",
    "reflector.api_key_env",
    "data.financebench_path",
    "data.split_ratios.train",
    "data.split_ratios.val",
    "data.split_ratios.test",
    "phase0.verbose_num_samples",
    "phase0.k",
    "phase0.output_path",
    "phase0.filters.drop_special_tokens",
    "phase0.filters.drop_whitespace_only",
    "phase0.filters.drop_digit_only",
    "phase0.correctness.metric",
    "phase0.correctness.threshold",
    "phase0.correctness.tolerance",
    "phase0.correctness.evaluator_model",
    "phase0.correctness.evaluator_api_base",
    "phase0.correctness.evaluator_api_key_env",
    "phase0.correctness.qualitative_forgiving",
    "phase0.correctness.evaluator_temperature",
    "phase0.correctness.evaluator_top_p",
    "phase0.correctness.evaluator_max_new_tokens",
    "phase0.correctness.evaluator_seed",
    "phase0.target_concurrency",
    "phase0.reflector_concurrency",
    "phase0.judge_concurrency",
]


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (str, list, dict)) and len(v) == 0:
        return True
    if isinstance(v, float) and (v != v or v == float("inf")):  # NaN or inf
        return True
    return False


def load_and_validate(kconfig_path: str) -> Dict[str, Any]:
    """Load config (YAML or Kconfig .config) and validate every required key; raise ValueError on any missing/null."""
    path = Path(kconfig_path)
    if not path.exists():
        raise ValueError(f"kconfig not found: {kconfig_path}")
    if path.suffix == ".config" or path.name == ".config":
        # kconfig_loader lives at repo root (sibling of compressor package)
        _repo_root = Path(__file__).resolve().parent.parent.parent
        if str(_repo_root) not in __import__("sys").path:
            __import__("sys").path.insert(0, str(_repo_root))
        from kconfig_loader import load_config as load_kconfig
        cfg = load_kconfig(str(path))
    else:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("kconfig must be a YAML object (dict) or .config")

    missing_or_empty: List[str] = []
    for key in REQUIRED_KEYS:
        v = _get(cfg, key)
        if _is_empty(v):
            missing_or_empty.append(key)

    if missing_or_empty:
        raise ValueError(
            "kconfig: missing or null/empty value for required keys (no defaults, no fallbacks): "
            + ", ".join(missing_or_empty)
        )

    # Split ratios must sum to 1.0
    t = float(cfg["data"]["split_ratios"]["train"])
    v = float(cfg["data"]["split_ratios"]["val"])
    s = float(cfg["data"]["split_ratios"]["test"])
    if abs(t + v + s - 1.0) > 1e-6:
        raise ValueError(f"data.split_ratios must sum to 1.0, got train={t} val={v} test={s}")

    return cfg
