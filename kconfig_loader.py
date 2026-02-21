"""
Kconfig loader for Phase 0 data collection.

Loads .config files (Kconfig-style) and converts them into the nested dict
expected by load_and_validate() and collect_data_financebench.py.
"""

import re
from pathlib import Path
from typing import Any, Dict


def _parse_config_file(path: str) -> Dict[str, Any]:
    """Parse a Kconfig-style .config file into CONFIG_NAME -> value.

    Handles CONFIG_FOO=value, CONFIG_FOO=y/n, and "# CONFIG_FOO is not set".
    """
    values: Dict[str, Any] = {}
    path = Path(path)
    if not path.exists():
        return values
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # "# CONFIG_FOO is not set" -> FOO = False
                m = re.match(r"#\s*CONFIG_(\w+)\s+is not set", line)
                if m:
                    values[m.group(1)] = False
                continue
            m = re.match(r"CONFIG_(\w+)=(.+)", line)
            if not m:
                continue
            name, value = m.groups()
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1].replace("\\n", "\n")
            elif value == "y":
                value = True
            elif value == "n":
                value = False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            values[name] = value
    return values


def _set_nested(cfg: Dict[str, Any], path: str, value: Any) -> None:
    """Set cfg[path[0]][path[1]]... = value for dot-separated path."""
    parts = path.split(".")
    cur = cfg
    for k in parts[:-1]:
        if k not in cur:
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value


# Map from CONFIG_ key (without CONFIG_ prefix, as in _parse_config_file output)
# to dot-separated path in the nested dict.
_CONFIG_TO_PATH = {
    "SEED": "seed",
    "DEVICE": "device",
    "OUTPUT_DIR": "output_dir",
    "TARGET_MODEL_ID": "target.model_id",
    "TARGET_SGLANG_BASE_URL": "target.sglang.base_url",
    "TARGET_SGLANG_TIMEOUT_S": "target.sglang.timeout_s",
    "TARGET_SGLANG_MAX_RETRIES": "target.sglang.max_retries",
    "TARGET_TEMPERATURE": "target.temperature",
    "TARGET_TOP_P": "target.top_p",
    "TARGET_MAX_NEW_TOKENS": "target.max_new_tokens",
    "TARGET_SEED": "target.seed",
    "TARGET_PROMPT_TEMPLATE": "target.prompt_template",
    "REFLECTOR_MODEL_ID": "reflector.model_id",
    "REFLECTOR_API": "reflector.api",
    "REFLECTOR_TEMPERATURE": "reflector.temperature",
    "REFLECTOR_TOP_P": "reflector.top_p",
    "REFLECTOR_MAX_NEW_TOKENS": "reflector.max_new_tokens",
    "REFLECTOR_SEED": "reflector.seed",
    "REFLECTOR_API_BASE": "reflector.api_base",
    "REFLECTOR_API_KEY_ENV": "reflector.api_key_env",
    "DATA_FINANCEBENCH_PATH": "data.financebench_path",
    "DATA_SPLIT_RATIOS_TRAIN": "data.split_ratios.train",
    "DATA_SPLIT_RATIOS_VAL": "data.split_ratios.val",
    "DATA_SPLIT_RATIOS_TEST": "data.split_ratios.test",
    "DATA_USE_EXPLICIT_INDICES": "data.use_explicit_indices",
    "DATA_TRAIN_INDICES": "data.train_indices",
    "DATA_VAL_INDICES": "data.val_indices",
    "DATA_TEST_INDICES": "data.test_indices",
    "PHASE0_VERBOSE_NUM_SAMPLES": "phase0.verbose_num_samples",
    "PHASE0_K": "phase0.k",
    "PHASE0_OUTPUT_PATH": "phase0.output_path",
    "PHASE0_FILTERS_DROP_SPECIAL_TOKENS": "phase0.filters.drop_special_tokens",
    "PHASE0_FILTERS_DROP_WHITESPACE_ONLY": "phase0.filters.drop_whitespace_only",
    "PHASE0_FILTERS_DROP_DIGIT_ONLY": "phase0.filters.drop_digit_only",
    "PHASE0_CORRECTNESS_METRIC": "phase0.correctness.metric",
    "PHASE0_CORRECTNESS_THRESHOLD": "phase0.correctness.threshold",
    "PHASE0_CORRECTNESS_TOLERANCE": "phase0.correctness.tolerance",
    "PHASE0_CORRECTNESS_EVALUATOR_MODEL": "phase0.correctness.evaluator_model",
    "PHASE0_CORRECTNESS_EVALUATOR_API_BASE": "phase0.correctness.evaluator_api_base",
    "PHASE0_CORRECTNESS_EVALUATOR_API_KEY_ENV": "phase0.correctness.evaluator_api_key_env",
    "PHASE0_CORRECTNESS_QUALITATIVE_FORGIVING": "phase0.correctness.qualitative_forgiving",
    "PHASE0_CORRECTNESS_EVALUATOR_TEMPERATURE": "phase0.correctness.evaluator_temperature",
    "PHASE0_CORRECTNESS_EVALUATOR_TOP_P": "phase0.correctness.evaluator_top_p",
    "PHASE0_CORRECTNESS_EVALUATOR_MAX_NEW_TOKENS": "phase0.correctness.evaluator_max_new_tokens",
    "PHASE0_CORRECTNESS_EVALUATOR_SEED": "phase0.correctness.evaluator_seed",
    "PHASE0_TARGET_CONCURRENCY": "phase0.target_concurrency",
    "PHASE0_REFLECTOR_CONCURRENCY": "phase0.reflector_concurrency",
    "PHASE0_JUDGE_CONCURRENCY": "phase0.judge_concurrency",
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a .config file and return the nested dict for Phase 0.

    Same shape as YAML config: target.model_id, phase0.k, etc.
    """
    values = _parse_config_file(config_path)

    # Resolve log level from choice (LOG_LEVEL_DEBUG / LOG_LEVEL_INFO / LOG_LEVEL_WARNING)
    if values.get("LOG_LEVEL_DEBUG") is True:
        log_level = "DEBUG"
    elif values.get("LOG_LEVEL_WARNING") is True:
        log_level = "WARNING"
    else:
        log_level = "INFO"
    values["LOG_LEVEL"] = log_level

    cfg: Dict[str, Any] = {}
    for config_name, path in _CONFIG_TO_PATH.items():
        if config_name not in values:
            continue
        val = values[config_name]
        _set_nested(cfg, path, val)

    # Ensure nested structure for split_ratios and coerce to float
    if "data" not in cfg:
        cfg["data"] = {}
    if "split_ratios" not in cfg["data"]:
        cfg["data"]["split_ratios"] = {}
    for k in ("train", "val", "test"):
        v = cfg["data"]["split_ratios"].get(k, 0.0)
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                v = 0.0
        cfg["data"]["split_ratios"][k] = v
    # Explicit split indices (optional; default false / empty)
    cfg["data"].setdefault("use_explicit_indices", False)
    cfg["data"].setdefault("train_indices", "")
    cfg["data"].setdefault("val_indices", "")
    cfg["data"].setdefault("test_indices", "")

    if "phase0" not in cfg:
        cfg["phase0"] = {}
    if "filters" not in cfg["phase0"]:
        cfg["phase0"]["filters"] = {}
    if "correctness" not in cfg["phase0"]:
        cfg["phase0"]["correctness"] = {}

    # Coerce types where needed (strings from .config)
    if "target" in cfg and "sglang" in cfg["target"]:
        for key in ("timeout_s", "max_retries"):
            if key in cfg["target"]["sglang"] and isinstance(cfg["target"]["sglang"][key], str):
                cfg["target"]["sglang"][key] = float(cfg["target"]["sglang"][key]) if key == "timeout_s" else int(cfg["target"]["sglang"][key])
    for path in ("target.temperature", "target.top_p", "reflector.temperature", "reflector.top_p",
                 "phase0.correctness.threshold", "phase0.correctness.tolerance",
                 "phase0.correctness.evaluator_temperature", "phase0.correctness.evaluator_top_p"):
        parts = path.split(".")
        parent = cfg
        for p in parts[:-1]:
            parent = parent.get(p, {})
        if parts[-1] in parent and isinstance(parent[parts[-1]], str):
            try:
                parent[parts[-1]] = float(parent[parts[-1]])
            except ValueError:
                pass

    # Prompt templates: convert | to newline (Kconfig strings don't support \n)
    for key in ("target.prompt_template",):
        parts = key.split(".")
        parent = cfg
        for p in parts[:-1]:
            parent = parent.get(p, {})
        if parts[-1] in parent and isinstance(parent[parts[-1]], str):
            parent[parts[-1]] = parent[parts[-1]].replace("|", "\n")

    # log_level: set from choice
    cfg["log_level"] = log_level

    return cfg
