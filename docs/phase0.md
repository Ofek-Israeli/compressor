# Phase 0: Data Collection Implementation Plan (`collect_data_financebench.py`)

> **Scope:** Phase 0 data collection only — no training, no delta learning.
> The output is a single JSON file containing everything Phase 1 needs.

---

## 1. Overview

`collect_data_financebench.py` implements the full Phase 0 pipeline from
`unigrams_plan.tex` on the FinanceBench dataset:

1. Load FinanceBench, extract `(context, query, gold_answer)` triples.
2. Split deterministically into train / val / test.
3. For each train example: sample **Target** verbose answers (load Target from kconfig, target.\*).
4. Compress each verbose answer with the **Reflector** (load Reflector from kconfig, reflector.\*).
5. Filter on correctness (compressed answer vs. gold).
6. Tokenize retained pairs with the Target tokenizer (`target.model_id`), compute frequency
   statistics, and select the top-k "fluff" tokens.
7. Write a single output JSON file containing the selected tokens and all
   data needed for Phase 1. **Phase 1 reads the delta-learning vocabulary
   directly from `token_selection.v_steer_token_ids` / `delta_by_token_id`;
   Phase 1 must not recompute token discovery.**

Val/test splits are stored as raw `(context, query, gold_answer)` triples
for downstream Phase 1 evaluation; the verbose→compress pipeline runs
**only on the train split**.

---

## 2. Prerequisites

| Dependency | Purpose |
|---|---|
| **Target** | Local SGLang server + our Python SGLang client wrapper (`compressor/sglang_client.py`, see §4.4a). Target is **never** called via OpenAI or any OpenAI-compatible HTTP endpoint. |
| **Reflector + correctness judge** | OpenAI Chat Completions (remote) via the `openai` Python SDK only; model IDs and endpoints from kconfig. |
| `transformers` | `AutoTokenizer.from_pretrained(target.model_id)` for tokenization (Steps 3–5); tokenizer source-of-truth is kconfig (`target.model_id`). |
| `pyyaml` | Parse kconfig YAML. |
| correctness module (`minions_channel/evaluate/correctness.py`) | `RemoteVerdictEvaluator` + `deterministic_numerical_check`. |

All configuration is via kconfig: **no defaults, no fallbacks.** Every
value used at runtime must be set in the kconfig file. Both Target and
Reflector are fully configured via kconfig (no hardcoded model names).
**Reflector (and the correctness judge) use OpenAI Chat Completions;**
**Target uses a local SGLang server via the SGLang client module.** Load
Target from kconfig (target.\*) and Reflector from kconfig (reflector.\*).

---

## 3. kconfig Keys

**No defaults, no fallbacks.** Every value used by the program at runtime
must be explicitly set in the kconfig file. The program must not assume
any default value or fallback. **Any missing key or null/empty value for
any key is an error and must terminate the program immediately.** All keys
listed below are mandatory and must be non-null (no exceptions). Both the
Target model and the Reflector model are fully configured via kconfig;
there are no hardcoded model names anywhere.

### 3.1 Target (parallel config block)

**Target generation is always executed by a local SGLang server (not OpenAI).**

| Key | Type | Description |
|---|---|---|
| `target.model_id` | str | Model identifier (e.g. HuggingFace id). Source of truth for tokenizer: `AutoTokenizer.from_pretrained(target.model_id)`. Same model must be served by the local SGLang server. |
| `target.sglang.base_url` | str | Base URL of the local SGLang server (e.g. `http://localhost:30000`). |
| `target.sglang.timeout_s` | float | Request timeout in seconds for SGLang client. No default; must be set in kconfig. |
| `target.sglang.max_retries` | int | Number of retries on transient failure for SGLang client. No default; must be set in kconfig. |
| `target.temperature` | float | Sampling temperature for Target generation. |
| `target.top_p` | float | Nucleus sampling p for Target. |
| `target.max_new_tokens` | int | Max tokens per Target generation. |
| `target.seed` | int | Seed for Target sampling. Mandatory; no fallback. |
| `target.prompt_template` | str | User-message body template for the Target model. Placeholders: `{context}`, `{query}`. **This is not the full prompt;** the full prompt is produced by rendering a `messages` list through the HuggingFace chat template (see §4.5). Example value: `"Context:\n{context}\n\nQuestion:\n{query}"`. |

### 3.2 Reflector (parallel config block — OpenAI Chat Completions only)

Reflector (and the judge) use the OpenAI API (Chat Completions). No Target keys here.

| Key | Type | Description |
|---|---|---|
| `reflector.model_id` | str | Model identifier for the Reflector (OpenAI API model name). |
| `reflector.api` | enum | Endpoint type: `"openai_chat_completions"` (Reflector uses OpenAI only). |
| `reflector.temperature` | float | Sampling temperature for Reflector compression. |
| `reflector.top_p` | float | Nucleus sampling p for Reflector. |
| `reflector.max_new_tokens` | int | Max tokens per Reflector compression call. |
| `reflector.seed` | int | Seed for Reflector sampling. Mandatory; no fallback. |
| `reflector.compress_prompt_template` | str | Compression prompt template (query + prev_answer, no context). Placeholders: `{query}`, `{prev_answer}`. |
| `reflector.api_base` | str | Base URL for the Reflector (OpenAI Chat Completions, e.g. `https://api.openai.com/v1`). |
| `reflector.api_key_env` | str | Name of the environment variable holding the Reflector (OpenAI) API key. |

### 3.3 General and data keys

| Key | Type | Description |
|---|---|---|
| `seed` | int | Global random seed for dataset splitting (e.g. shuffle). Required. |
| `device` | str | Compute device (not used for API calls; kept for tokenizer/offline use). |
| `output_dir` | str | Root directory for all artifacts. |
| `log_level` | str | Logging verbosity (`"INFO"`, `"DEBUG"`, …). |
| `data.financebench_path` | str | Path to `financebench_open_source.jsonl`. |
| `data.split_ratios.train` | float | Fraction of examples for training (e.g. `0.70`). |
| `data.split_ratios.val` | float | Fraction for validation (e.g. `0.15`). |
| `data.split_ratios.test` | float | Fraction for test (e.g. `0.15`). Must satisfy `train + val + test = 1.0`. |
| `phase0.verbose_num_samples` | int | Number of Target verbose answers M_verbose per (c, q). |
| `phase0.k` | int | Number of top-k fluff tokens to select. |
| `phase0.filters.drop_special_tokens` | bool | Drop special/control tokens before ranking. |
| `phase0.filters.drop_whitespace_only` | bool | Drop whitespace-only and punctuation-only tokens. |
| `phase0.filters.drop_digit_only` | bool | Drop digit-only tokens. |
| `phase0.output_path` | str | Path for the output JSON file. |
| `phase0.correctness.metric` | str | Metric name (e.g. `"remote_verdict"`). |
| `phase0.correctness.threshold` | float | Minimum metric score τ for a compressed answer to be correct. |
| `phase0.correctness.tolerance` | float | Numerical tolerance for correctness (e.g. `0.15`). See §6. |
| `phase0.correctness.evaluator_model` | str | Model for the LLM-based correctness judge (from kconfig). |
| `phase0.correctness.evaluator_api_base` | str | API base URL for the correctness judge. |
| `phase0.correctness.evaluator_api_key_env` | str | Env var name for the judge API key. |
| `phase0.correctness.qualitative_forgiving` | bool | Whether to use forgiving evaluation for qualitative questions (passed to evaluator; no default in code). |
| `phase0.correctness.evaluator_temperature` | float | Sampling temperature for the judge LLM. No default; from kconfig only. |
| `phase0.correctness.evaluator_top_p` | float | Nucleus sampling p for the judge. No default; from kconfig only. |
| `phase0.correctness.evaluator_max_new_tokens` | int | Max tokens per judge call. No default; from kconfig only. |
| `phase0.correctness.evaluator_seed` | int | Seed for judge sampling. Mandatory; no fallback. |
| `phase0.target_concurrency` | int | Max concurrent Target (SGLang) requests. No default; from kconfig only. |
| `phase0.reflector_concurrency` | int | Max concurrent Reflector (OpenAI) requests. No default; from kconfig only. |
| `phase0.judge_concurrency` | int | Max concurrent judge (OpenAI) requests. No default; from kconfig only. |

> **Note on the train split.** The train split is used for **two** purposes:
> (i) discovering fluff tokens in Phase 0 (this module), and
> (ii) learning delta magnitudes in Phase 1 (future module).
> This is why it must be deterministically reproducible from the seed and
> split ratios alone.

---

## 4. Workflow (step by step)

### 4.1 Load and validate kconfig

```
cfg = yaml.safe_load(open(kconfig_path))
```

Validate: every key listed in §3 must be present and have a non-null,
non-empty value. **Any missing key or null/empty value for any key is an
error and must terminate the program immediately** (no fallbacks, no
defaults). Assert `split_ratios.train + split_ratios.val + split_ratios.test == 1.0`
(within floating-point tolerance of 1e-6). Every value used at runtime
must come from this validated config.

### 4.2 Load FinanceBench dataset

Read `data.financebench_path` (JSONL). Each line is a JSON object with
fields including `financebench_id`, `question`, `answer`, and `evidence`
(a list of evidence objects).

### 4.3 Extract (context, query, gold_answer) triples

For each example:

- **query** ← `example["question"]`
- **gold_answer** ← `example["answer"]`
- **context** ← concatenate the `evidence_text_full_page` field from each
  entry in `example["evidence"]`, separated by `"\n\n"`, **deduplicating**
  by `(evidence_doc_name, evidence_page_num)` (FinanceBench field names) so
  the same page is never included twice.
- Preserve `financebench_id` as the example identifier.

> **Important:** use `evidence_text_full_page` — not `evidence_text` and
> not the raw PDF text — as the context sent to the Target model.

### 4.4 Split into train / val / test

1. Sort examples by `financebench_id` (lexicographic) for determinism.
2. Shuffle the sorted list with `random.Random(seed).shuffle(examples)`.
3. Split by ratios: first `⌊N × train_ratio⌋` → train, next
   `⌊N × val_ratio⌋` → val, remainder → test.

Log the split sizes.

### 4.4a SGLang client module (Target only)

A dedicated Python module will be implemented (later) for calling the
**local** SGLang server. No Target generation goes through OpenAI.

- **Module path:** `compressor/sglang_client.py` (use this path consistently).
- **Class:** `SGLangClient`, initialized with (all from kconfig; no defaults):
  - `base_url` (str) — from `target.sglang.base_url`.
  - `model_id` (str) — from `target.model_id`.
  - `timeout_s` (float) — from `target.sglang.timeout_s`.
  - `max_retries` (int) — from `target.sglang.max_retries`.
- **Method:** `generate(prompt: str, temperature: float, top_p: float, max_new_tokens: int, seed: int, stop_token_ids: list[int]) -> str`
  - `seed`: always use `target.seed` from kconfig (mandatory; no fallback).
  - `stop_token_ids`: list of token ids at which generation must stop (see §4.4b).
  - Sends the request to the local SGLang server via the **single-prompt endpoint** (one pre-rendered prompt string; no chat/messages wrapper) and returns the generated text only.
- **Usage:** `collect_data_financebench.py` calls the Target **only** via `SGLangClient.generate(...)`; there are no direct HTTP calls or OpenAI client calls for the Target in the data-collection script.

(This section describes the module and its interface; no code is implemented in this plan.)

### 4.4b Target prompt construction and EOT stopping

The Target model is `meta-llama/Llama-3.1-8B-Instruct` and **must** be
prompted using its HuggingFace chat template. The raw
`target.prompt_template` value is only the **user-message body**; the
full prompt is always produced via `apply_chat_template`.

**Prompt construction:**

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(target.model_id)

# 1. Build the messages list
user_body = target.prompt_template.format(context=c, query=q)
messages = [
    {"role": "system", "content": "Answer the question using the provided context."},
    {"role": "user",   "content": user_body},
]

# 2. Render the full prompt string via the chat template
prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

`prompt_text` is the string sent to `SGLangClient.generate(prompt=prompt_text, ...)`.

**EOT (end-of-turn) stopping:**

We must stop Target generation at the model's end-of-turn token to
prevent runaway continuation (repetition loops). Obtain the stop token
id from the same tokenizer:

```python
EOT_ID = tok.eos_token_id
```

(For Llama-3.1-Instruct, `tok.eos_token_id` is the end-of-turn token.
If the tokenizer exposes a dedicated EOT token id different from
`eos_token_id`, use that id instead.)

Pass `stop_token_ids=[EOT_ID]` in **every** Target generation request:

```python
text = sglang.generate(
    prompt=prompt_text,
    ...,
    stop_token_ids=[EOT_ID],
)
```

This ensures the Target stops cleanly after producing its answer.

### 4.5 Generate Target verbose answers (plan Step 1)

**Runs only on the train split.**

**Target generation is never executed via OpenAI; it is always executed
via the local SGLang server.** Load Target from kconfig (target.\*).

1. Load the Target tokenizer once:
   `tok = AutoTokenizer.from_pretrained(target.model_id)`
2. Obtain the EOT stop token id:
   `EOT_ID = tok.eos_token_id`
3. For each `(c, q)` in the train split, construct the prompt via the
   chat template (see §4.4b):

   ```python
   user_body = target.prompt_template.format(context=c, query=q)
   messages = [
       {"role": "system", "content": "Answer the question using the provided context."},
       {"role": "user",   "content": user_body},
   ]
   prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   ```

4. Call **SGLangClient.generate(prompt=prompt_text,
   temperature=target.temperature, top_p=target.top_p,
   max_new_tokens=target.max_new_tokens, seed=target.seed,
   stop_token_ids=[EOT_ID])** M_verbose times per (c,q).
5. Store each returned string in `verbose_answers: List[str]` per example.

The Reflector is **not** called in this step.
Enforce max in-flight Target requests using
**phase0.target_concurrency** from kconfig (mandatory; no default).

> **Note:** `target.prompt_template` is only the user-message body
> (e.g. `"Context:\n{context}\n\nQuestion:\n{query}"`). The full prompt
> sent to SGLang is always the output of `apply_chat_template`, which
> wraps it in the model's expected header/footer tokens. Do not send the
> raw template directly to the Target model.

### 4.6 Reflector compression (plan Step 2)

Load Reflector from kconfig (reflector.\*). For each verbose answer
`â_verbose` generated in §4.5, call the Reflector via OpenAI Chat
Completions at `reflector.api_base` with:

```
prompt = reflector.compress_prompt_template.format(
    query=q,
    prev_answer=â_verbose
)
```

Parameters: `model` = `reflector.model_id`, `temperature` =
`reflector.temperature`, `top_p` = `reflector.top_p`, `max_tokens` =
`reflector.max_new_tokens`.

The prompt contains **only** query + verbose answer — **no context**.
Store `compressed_answers: List[str]` parallel to `verbose_answers`. Respect
**phase0.reflector_concurrency** (max concurrent Reflector requests) from kconfig; no default.

### 4.7 Correctness filtering (plan Step 2b)

Evaluate each compressed answer against the gold answer using
`RemoteVerdictEvaluator` from
`minions_channel/evaluate/correctness.py`. The judge uses the **same**
OpenAI client pattern as the Reflector (same API, same client type); only
the model and generation parameters come from `phase0.correctness.evaluator_*`:

```python
from correctness import RemoteVerdictEvaluator

evaluator = RemoteVerdictEvaluator(
    remote_client=judge_client,         # same OpenAI client as Reflector; phase0.correctness.* for model/params
    numerical_tolerance=cfg["phase0"]["correctness"]["tolerance"],
    qualitative_forgiving=cfg["phase0"]["correctness"]["qualitative_forgiving"]
)
# All evaluator parameters from kconfig; no defaults or fallbacks.
# Judge calls use phase0.correctness.evaluator_temperature, evaluator_top_p,
# evaluator_max_new_tokens, evaluator_seed. Respect phase0.judge_concurrency.

for j, â_comp in enumerate(compressed_answers):
    result = evaluator.evaluate(
        predicted=â_comp,
        ground_truth=gold_answer,
        question=query
    )
    if result.is_correct:
        # retain the pair (â_verbose_j, â_comp_j)
```

**Retained pair:** a `(â_verbose, â_comp)` pair whose compressed answer
passes the correctness check (`result.is_correct == True`).

**Discarded pair:** `result.is_correct == False`.

**Discarded example:** a `(c, q)` where **all** M_verbose pairs are
discarded (`m_{c,q} = 0`). Log these for analysis but exclude from
all subsequent computation (frequency stats and Phase 1 training).

**Sample weight** for each retained pair of example `(c, q)`:

```
w_{c,q} = 1 / m_{c,q}
```

where `m_{c,q}` is the number of retained pairs for that example.
This ensures every `(c, q)` contributes total weight 1 regardless of
how many pairs survived filtering.

See §6 for full correctness details.

### 4.8 Tokenize retained pairs (plan Step 3)

The Target tokenizer is the single source of truth for all encode/decode
and frequency computation; it is loaded from kconfig only:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(target.model_id)
```

Define: `Tok(â) = tok.encode(â, add_special_tokens=False)` and
`decode(t) = tok.decode([t])` for a single token id `t`. For each
retained pair `(â_verbose, â_comp)`:

```python
tokens_verbose = tok.encode(â_verbose, add_special_tokens=False)  # Tok(â_verbose)
tokens_comp    = tok.encode(â_comp,    add_special_tokens=False)  # Tok(â_comp)
```

Store token-id lists alongside the text.

### 4.9 Compute frequencies and Δ(v) (plan Steps 4–5)

Let `S = { (c,q) : m_{c,q} > 0 }`.

**Raw (verbose) frequency (weighted):**

```
C_raw(v)  = Σ_{(c,q)∈S} Σ_{i=1}^{m_{c,q}} w_{c,q} · count(v in tokens_verbose_i)
Z_raw     = Σ_{(c,q)∈S} Σ_{i=1}^{m_{c,q}} w_{c,q} · len(tokens_verbose_i)
freq_raw(v) = C_raw(v) / Z_raw
```

**Compressed frequency (weighted):**

```
C_comp(v) = Σ_{(c,q)∈S} Σ_{i=1}^{m_{c,q}} w_{c,q} · count(v in tokens_comp_i)
Z_comp    = Σ_{(c,q)∈S} Σ_{i=1}^{m_{c,q}} w_{c,q} · len(tokens_comp_i)
freq_comp(v) = C_comp(v) / Z_comp
```

**Frequency difference:**

```
Δ(v) = freq_raw(v) - freq_comp(v)
```

### 4.10 Token filtering and top-k selection (plan Step 5)

Apply the deterministic filters (all using `decode(t) = tok.decode([t])`
with the tokenizer from `target.model_id`) to obtain candidate set V':

1. **Drop special tokens** (if `phase0.filters.drop_special_tokens`):
   remove all ids in `tok.all_special_ids` plus any reserved control ids.

2. **Drop whitespace-only and punctuation-only** (if
   `phase0.filters.drop_whitespace_only`): decode token → strip → if
   empty, drop; if every character has Unicode category starting with `P`,
   drop.

3. **Drop digit-only** (if `phase0.filters.drop_digit_only`): decode →
   strip → if non-empty and every char is `0-9`, drop.

4. **Always exclude EOS** from V' (even if `drop_special_tokens` is
   false) so biases are not duplicated with the dedicated δ_EOS.

Select:

```
V_steer = top-k tokens from V' by Δ(v), descending
```

If `|V'| < k`, set `V_steer = V'` and log a warning.

### 4.11 Save output JSON

Write to `phase0.output_path`. Schema defined in §5.

---

## 5. Output JSON Schema

The output file is a single JSON object. The **token selection** block is
**required** and must be sufficient for Phase 1 to learn deltas without
recomputing token discovery. **Phase 1 reads the delta-learning vocabulary
directly from `token_selection.v_steer_token_ids` / `delta_by_token_id`;
Phase 1 must not recompute token discovery.** All input and runtime
behavior is driven only by kconfig (no defaults, no fallbacks). In the
schema below, fields marked optional are optional *in the output file* (may
be omitted for analysis); all other fields are required in the output.

```jsonc
{
  // ── Metadata ─────────────────────────────────────────────────
  "metadata": {
    "created_at":        "2026-02-20T14:30:00Z",   // ISO-8601 timestamp
    "seed":              42,                         // int
    "target_model_id":   "<value from target.model_id>",   // str, from kconfig
    "reflector_model":   "<value from reflector.model_id>", // str, from kconfig
    "evaluator_model":   "<value from phase0.correctness.evaluator_model>", // str, from kconfig
    "verbose_num_samples": 5,                        // int (M_verbose)
    "k":                 50,                          // int
    "correctness_tolerance": 0.15,                   // float
    "split_counts": {
      "train": 105, "val": 23, "test": 22           // int
    },
    "train_retained_examples": 98,                   // int, |S|
    "train_discarded_examples": 7,                   // int, examples with m=0
    "total_retained_pairs": 312,                     // int
    "config_snapshot": { }                           // full kconfig dict
  },

  // ── Token selection (REQUIRED for Phase 1 delta learning) ─────
  "token_selection": {
    "k": 50,                                         // int, actual k used
    "v_steer": [                                     // REQUIRED; length = k (or |V'| if < k)
      {
        "token_id":  1234,                           // int, vocab id
        "token_str": " the",                         // str, tok.decode([id])
        "delta":     0.0452                          // float, Δ(v)
      }
      // ... k entries, sorted by delta descending
    ],
    "v_steer_token_ids": [ 1234, 5678, ... ],        // REQUIRED; same ordering as v_steer
    "delta_by_token_id": {                           // REQUIRED; token_id (string) -> delta (float), for all ids in v_steer
      "1234": 0.0452,
      "5678": 0.0310
    },
    "freq_raw":  { "1234": 0.0500, "5678": 0.0310 },   // (optional) token_id→float
    "freq_comp": { "1234": 0.0048, "5678": 0.0120 }    // (optional) token_id→float
  },

  // ── Per-split data ──────────────────────────────────────────
  "splits": {
    "train": [                                       // one entry per (c,q) in train
      {
        "example_id": "financebench_id_03029",       // str
        "context":    "Full page text ...",           // str (evidence_text_full_page)
        "query":      "What is the FY2018 ...",      // str
        "gold_answer": "$1577.00",                   // str
        "num_verbose_generated": 5,                  // int, M_verbose
        "num_retained": 3,                           // int, m_{c,q}
        "sample_weight": 0.3333,                     // float, 1/m  (null if m=0)
        "retained_pairs": [                          // length = num_retained
          {
            "verbose_answer":    "Based on the ...", // str
            "compressed_answer": "$1,577 million",   // str
            "verbose_token_ids":    [101, 234, ...], // List[int]
            "compressed_token_ids": [88, 412, ...],  // List[int]
            "correctness": {                         // evaluation metadata
              "is_correct":  true,                   // bool (always true here)
              "confidence":  0.95,                   // float
              "category":    "numerical",            // str
              "reasoning":   "..."                   // str
            }
          }
          // ... num_retained entries
        ],
        "discarded_pairs": [                         // (optional) for analysis
          {
            "verbose_answer":    "...",
            "compressed_answer": "...",
            "correctness": { "is_correct": false, ... }
          }
        ]
      }
      // ... one object per train example (including discarded examples with m=0)
    ],

    "val": [                                         // raw triples only
      {
        "example_id":  "financebench_id_...",
        "context":     "...",
        "query":       "...",
        "gold_answer": "..."
      }
    ],

    "test": [                                        // same schema as val
      {
        "example_id":  "financebench_id_...",
        "context":     "...",
        "query":       "...",
        "gold_answer": "..."
      }
    ]
  }
}
```

### Field summary

| Field path | Type | Shape / cardinality |
|---|---|---|
| `metadata` | object | 1 |
| `token_selection.v_steer` | array of objects | length k; REQUIRED |
| `token_selection.v_steer_token_ids` | array of int | length k; same order as v_steer; REQUIRED for Phase 1 |
| `token_selection.delta_by_token_id` | object | token_id (str) → delta (float) for all ids in v_steer; REQUIRED for Phase 1 |
| `splits.train` | array of objects | one per train example (retained **and** discarded) |
| `splits.train[i].retained_pairs` | array of objects | 0 to M_verbose per example |
| `splits.train[i].retained_pairs[j].verbose_token_ids` | array of int | variable length |
| `splits.train[i].retained_pairs[j].compressed_token_ids` | array of int | variable length |
| `splits.val` / `splits.test` | array of objects | one per val/test example |

---

## 6. Correctness Filtering Details

### Evaluator

Use `RemoteVerdictEvaluator` from
`minions_channel/evaluate/correctness.py`. All parameters (tolerance,
threshold, model, API base, API key env) must be read from kconfig
(`phase0.correctness.*`); no defaults or fallbacks in code. For example,
numerical tolerance is always `phase0.correctness.tolerance` (e.g. 0.15 for
15% relative error).

### How tolerance = 0.15 is applied

The evaluator runs a three-stage pipeline; tolerance affects all stages:

1. **Deterministic numerical pre-check.**
   Activates only when the gold answer is a pure number (e.g. `"$1577.00"`,
   `"65.4%"`). Extracts numbers from both gold and predicted strings.
   If any extracted predicted number satisfies:

   ```
   |pred − gold| / |gold|  ≤  0.15
   ```

   the answer is judged **CORRECT immediately** (no LLM call).
   The function never short-circuits to False; ambiguous cases fall
   through to the LLM.

2. **LLM verdict with structured output.**
   The prompt instructs the judge model to "allow 15% tolerance for exact
   numerical comparisons" and to normalize units before comparing.
   The LLM returns a JSON with `is_correct`, `normalized_gt`,
   `normalized_pred`, and `relative_error_pct`.

3. **Arithmetic verification override.**
   If the LLM says WRONG for a numerical answer, but its own reported
   `normalized_gt` and `normalized_pred` are within 15% relative error,
   the verdict is overridden to **CORRECT**. This catches LLM arithmetic
   mistakes.

### Configuration

| kconfig key | Value | Meaning |
|---|---|---|
| `phase0.correctness.tolerance` | `0.15` | 15% relative error allowed for numerical answers |
| `phase0.correctness.evaluator_model` | str (from kconfig) | Model used as the correctness judge; no hardcoded model names |
| `phase0.correctness.metric` | `"remote_verdict"` | Selects the `RemoteVerdictEvaluator` path |
| `phase0.correctness.threshold` | float (from kconfig) | Minimum metric score τ for a compressed answer to be correct; no default in code. |

---

## 7. Retained vs. Discarded — Definitions

| Term | Definition |
|---|---|
| **Generated pair** | One `(â_verbose, â_comp)` produced from a single Target sample + Reflector compression. M_verbose pairs are generated per `(c, q)`. |
| **Retained pair** | A generated pair whose compressed answer passes the correctness check (`is_correct == True`). Stored in `retained_pairs`. |
| **Discarded pair** | A generated pair that fails correctness. Stored in `discarded_pairs` (optional, for analysis only). |
| **Retained example** | A `(c, q)` with at least one retained pair (`m_{c,q} > 0`). Contributes to frequency statistics and Phase 1 training. |
| **Discarded example** | A `(c, q)` where all pairs fail (`m_{c,q} = 0`). Excluded from frequency computation and from Phase 1 training. Still present in the output JSON with `num_retained: 0` and empty `retained_pairs`, logged for post-hoc analysis. |
| **Sample weight** | `w_{c,q} = 1 / m_{c,q}` for each retained pair of example `(c,q)`. Ensures every retained example contributes total weight 1 to frequency computation and to the Phase 1 loss, regardless of how many pairs survived. |

---

## 8. Reproducibility Notes

### Seeds and determinism

- **Splitting** is fully deterministic given `seed` and the sorted
  `financebench_id` list. `random.Random(seed).shuffle(...)` is used
  (stdlib, platform-independent).
- **Target generation** uses `temperature` and `top_p` from kconfig.
  When `temperature > 0`, outputs are stochastic; each call yields a
  different sample (this is intentional — we want M_verbose diverse
  answers). If exact reproducibility of the raw texts is needed, set
  `temperature = 0` and `M_verbose = 1` (greedy).
- **Reflector compression** is similarly stochastic when
  `reflector.temperature > 0`.
- **Correctness evaluation** uses a deterministic numerical pre-check
  (always reproducible) plus an LLM judge (stochastic). Judge calls use evaluator parameters from kconfig: **phase0.correctness.evaluator_temperature**, **evaluator_top_p**, **evaluator_max_new_tokens**, **evaluator_seed** (no implied defaults).
- **Token selection** (Steps 3–5) is fully deterministic given the
  retained text pairs and the tokenizer.

### What to log

The script must log (at `INFO` level or higher):

- kconfig snapshot at startup.
- Split sizes after §4.4.
- Per-example: number of verbose answers generated, number retained,
  number discarded, sample weight.
- Aggregate: total retained examples / total, total retained pairs /
  total generated pairs.
- Frequency stats summary: top-20 Δ(v) tokens with their decoded strings.
- Warnings: examples with 0 retained pairs; `|V'| < k`.
- Final output path and file size.

### Checkpointing

Because Reflector and judge (OpenAI) API calls are expensive:

- After §4.5 completes (all verbose answers generated), write an
  intermediate checkpoint: `{output_dir}/phase0_verbose.jsonl`
  (one JSON object per example with `example_id` and `verbose_answers`).
- After §4.6 + §4.7 completes (compression + correctness done), write:
  `{output_dir}/phase0_pairs.jsonl` (includes compressed answers and
  correctness results).
- The script should accept a `--resume` flag that skips already-completed
  examples by checking the checkpoint files.

---

## 9. FinanceBench-Specific Notes

### Context field

Each FinanceBench example has an `evidence` list. Each evidence entry
contains `evidence_text_full_page` — the full text of the PDF page where
the evidence appears. When constructing the context:

1. Iterate over `example["evidence"]`.
2. Deduplicate by `(evidence_doc_name, evidence_page_num)`.
3. Concatenate the unique `evidence_text_full_page` strings separated by
   `"\n\n"`.
4. Use this concatenated text as the `{context}` placeholder in the
   user-message body template (`target.prompt_template`). The full
   Target prompt is then produced via `apply_chat_template` (see §4.4b).

### Dataset size

FinanceBench open-source has **150 examples**. With split ratios set in
kconfig (e.g. `data.split_ratios.train`, `.val`, `.test`; example values
0.70, 0.15, 0.15):

- Train: 105 examples → 105 × M_verbose calls to local SGLang (Target) + up to
  105 × M_verbose calls to Reflector (OpenAI) + up to 105 × M_verbose judge calls (OpenAI).
- Val: 22–23 examples (raw triples only).
- Test: 22–23 examples (raw triples only).

### Cost estimation

For M_verbose = 5 and 105 train examples:
- Target calls (local SGLang): 525
- Reflector calls (OpenAI): 525
- Judge calls (OpenAI): ≤ 525 (some skipped by deterministic pre-check)
- Total: 525 local SGLang calls + up to ~1,050 remote OpenAI calls.

---

## 10. Example kconfig YAML (for reference)

All model identities and endpoints come from kconfig (no hardcoded names).

```yaml
seed: 42
device: "cuda:0"
output_dir: "./outputs/phase0"
log_level: "INFO"

target:
  model_id: "meta-llama/Llama-3.1-8B-Instruct"
  sglang:
    base_url: "http://localhost:8000"
    timeout_s: 120.0
    max_retries: 3
  temperature: 0.7
  top_p: 0.95
  max_new_tokens: 512
  seed: 42
  # User-message body only; full prompt is rendered via apply_chat_template (see §4.4b)
  prompt_template: "Context:\n{context}\n\nQuestion:\n{query}"

reflector:
  model_id: "gpt-4o"   # from kconfig
  api: "openai_chat_completions"
  temperature: 0.0
  top_p: 1.0
  max_new_tokens: 256
  seed: 42
  compress_prompt_template: |
    Question:
    {query}
    Previous answer:
    {prev_answer}
    Rewritten answer:
  api_base: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"

data:
  financebench_path: "/path/to/financebench/data/financebench_open_source.jsonl"
  split_ratios:
    train: 0.70
    val: 0.15
    test: 0.15

phase0:
  verbose_num_samples: 5
  k: 50
  output_path: "./outputs/phase0/phase0_data.json"
  target_concurrency: 1
  reflector_concurrency: 4
  judge_concurrency: 4

  correctness:
    metric: "remote_verdict"
    threshold: 0.5
    tolerance: 0.15
    qualitative_forgiving: true
    evaluator_model: "gpt-4o"
    evaluator_api_base: "https://api.openai.com/v1"
    evaluator_api_key_env: "OPENAI_API_KEY"
    evaluator_temperature: 0.0
    evaluator_top_p: 1.0
    evaluator_max_new_tokens: 512
    evaluator_seed: 42

  filters:
    drop_special_tokens: true
    drop_whitespace_only: true
    drop_digit_only: true
```

---

## 11. Script Interface

```
python collect_data_financebench.py --config path/to/kconfig.yaml [--resume]
```

| Argument | Required | Description |
|---|---|---|
| `--config` | yes | Path to kconfig YAML file |
| `--resume` | no | Skip examples already present in checkpoint files |

Exit codes: `0` on success, `1` on kconfig validation error, `2` on
runtime error (API failure after retries).
