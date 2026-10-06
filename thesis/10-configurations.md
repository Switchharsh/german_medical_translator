# Configurations and protocols

This chapter is the settings reference: every model with its exact runtime
configuration, every experiment with its exact protocol, and what was recorded
and selected on. It exists because the other chapters record *verdicts* and
*reasoning*, and a reader reproducing the work needs the numbers.

Everything below was read out of the code and launchers rather than from memory,
and cites the file it came from.

## 0. There are no training recipes, because nothing was trained

**No model in this project was fine-tuned, adapted or otherwise updated.** The
repository contains no optimiser, no backward pass and no PEFT configuration —
`src/medmt_eval/` is an inference and evaluation harness. Every system was run at
its published weights.

That is a deliberate finding rather than an omission: the project asked whether a
specialised German↔English medical model is *needed*, and concluded that the case
for fine-tuning is not yet made ([06-results](05-results.md) §"Answering the
question"). So the analogue of a training recipe here is the **decoding
configuration** and the **experimental protocol**, and those are what follow.

## 1. Model inventory and runtime configuration

Per-model settings are resolved in
[`scripts/roundtrip/_rt_common.sh`](../scripts/roundtrip/_rt_common.sh). `chunk`
is the sentence-chunking budget in tokens (0 = disabled).

| Name | Adapter | Model id | batch | max-new | chunk |
|---|---|---|---|---|---|
| `identity` | `identity` | — | 32 | 2048 | 0 |
| `opus` | `opus` | `Helsinki-NLP/opus-mt-{de-en,en-de}` | 8 | **480** | **400** |
| `nllb` | `nllb` | `facebook/nllb-200-distilled-1.3B` | 2 | 960 | **400** |
| `hymt2-1.8b` | `hymt2` | `tencent/Hy-MT2-1.8B` | 4 | 2048 | 0 |
| `hymt2-7b` | `hymt2` | `tencent/Hy-MT2-7B` | 2 | 2048 | 0 |
| `hymt2-30b-a3b` | `hymt2` | `tencent/Hy-MT2-30B-A3B` | 1 | 2048 | 0 |
| `translategemma-4b` | `translategemma` | `google/translategemma-4b-it` | 2 | 2048 | 0 |
| `translategemma-27b` | `translategemma` | `google/translategemma-27b-it` | 1 | 2048 | 0 |
| `qwen35-4b` | `prompted-llm` | `Qwen/Qwen3.5-4B` | 2 | 2048 | 0 |
| `qwen35-27b` | `prompted-llm` | `Qwen/Qwen3.5-27B` | 1 | 2048 | 0 |
| `glm-5.2` ☁ | `openai-compat` | served as `z-ai/glm-5.2` | 4 | 2048 | 0 |
| `DeepSeek-V4-Flash` ☁ | `openai-compat` | served as `deepseek-ai/deepseek-v4-flash-0731` | 4 | 2048 | 0 |
| `MiniMax-M3` ☁ | `openai-compat` | served as `minimaxai/minimax-m3` | 4 | 2048 | 0 |
| `deepl` ☁ | `deepl` | none disclosed | 8 | — | — |

`deepl` differs in kind: no GPU, no model id, no generation parameters. Settings that
matter are the target variant (`EN-US`), a batch of 8 documents per request, and the
login-node run ([`scripts/deepl/run_deepl.sh`](../scripts/deepl/run_deepl.sh)). Batch
sizes for the other systems are memory-driven, not tuned for quality: they are the largest that
fit the allocation each model was given ([05-experiments](04-experiments.md)).

### Why three models differ from the rest

- **`opus` — 480 max-new, chunked at 400.** Marian has 512 encoder *and* decoder
  positions. Exceeding the decoder table raises a CUDA device-side assert
  (`marian/modeling_marian.py:596`), so generation is capped below it.
- **`nllb` — chunked at 400.** 1024 encoder positions, still short of a full
  report. The budget was originally 800 tokens (≈2400 characters), longer than
  most documents, so chunking never engaged at all; 400 was the correction.
- **`identity` — batch 32.** It performs no computation; the batch size only
  affects loop overhead.

## 2. Generation defaults

`GenerationConfig` ([`models/base.py`](../src/medmt_eval/models/base.py)):

| Field | Default |
|---|---|
| `batch_size` | 8 |
| `num_beams` | 4 |
| `max_input_tokens` | 512 |
| `max_new_tokens` | 512 |
| `device` | `None` (auto) |

The round-trip runs override these: **`--num-beams 1`** (greedy, for tractability
over 20 passes × 13 systems) and **`--max-input-tokens 4096`**. Hosted models run
at **temperature 0** so results are reproducible, matching the local models.

Adapter-specific behaviour that changes outputs, not just speed:

- **Chat templates, not raw text.** `PromptedLLMTranslator` calls
  `apply_chat_template(..., enable_thinking=False)`, with a `strip_thinking()`
  regex as a second line of defence.
- **Left padding.** `padding_side="left"` for batched decoder-only generation;
  right padding corrupts it, and fixing this moved `qwen35-4b` from 24.66% to
  18.92% critical errors.
- **`direction_specific = True` for Opus only.** One checkpoint per language
  pair, so the round-trip builds two instances for Opus and exactly one for
  every other adapter.
- **Served-model verification** on every hosted call (`_check_served_model`).

## 3. Prompt templates

Verbatim, because a prompt is a setting.

**Plain translation** — local (`models/llm_mt.py`) and hosted
(`models/openai_compat_mt.py`), which additionally forbids quotation marks:

```
Translate the following medical text from {source_lang} to {target_lang}.
Return only the translation, with no explanation[, no preamble, and no quotation marks].

{text}
```

**Glossary arms** ([`inference/glossary_mt.py`](../src/medmt_eval/inference/glossary_mt.py)) —
the block is empty in the `none` arm, so that prompt carries no trace of a
glossary having been considered:

```
You are translating a medical document from {source_lang} to {target_lang}.
{glossary_block}
Return only the translation. No preamble, no commentary, no quotation marks.

{text}
```

The injected block is rendered as `- <source term> = <target term>` lines under
the header *"Approved terminology for this text. Where the source uses the term
on the left, the translation must use the term on the right."*

**Debate and cascade personas** are in
[`inference/debate.py`](../src/medmt_eval/inference/debate.py) and
[`inference/cascade.py`](../src/medmt_eval/inference/cascade.py); the clinical
judge rubric is in [`metrics/llm_judge.py`](../src/medmt_eval/metrics/llm_judge.py).

## 4. Experiment protocols

### 4.1 Single-pass benchmark

All 13 systems over PARROT-DE (296), EMEA (400), HimL 2015 (353), HimL 2017 (119).
Per-adapter defaults; both evaluation layers. DeepL: PARROT-DE only.

### 4.2 Round-trip

| Setting | Value |
|---|---|
| Cycles | 10 (= **20 translation passes**) |
| Sample | 20 reports, deterministic length-stratified, **seed 13** |
| Beams | 1 |
| Term bank | `radiology_en_de_starter.csv` (9 entries) |
| Anchors | English steps → human English reference; German steps → original German |

Every system sees the **same** 20 reports. Full corpus × 10 cycles was ~168
GPU-hours, past every wall-clock limit. Recorded per step: `clinical_vs_origin`
(cumulative drift) and `clinical_vs_input` (which hop introduced the error).

### 4.3 Dictionary injection

| Setting | Value |
|---|---|
| Arms | `none` / `glossary` / `distractor` |
| Sample | 40 (whole-report runs), 109 (medical-text runs), seed 13 |
| Max terms/doc | 30 |
| Glossary | `wikidata_med.csv` (default) or `merged_de_en.csv` / `radlex_findings_de_en.csv` |
| Branch filter | `ACE` (MeSH anatomy / disease / diagnostics) |
| Model | `local:Qwen/Qwen3.5-4B` |

The `distractor` arm injects the **same number** of terms that do *not* occur in
the document — the control that distinguishes "the terms helped" from "the block
helped". `--holdout F` splits the glossary by concept so injected and scored
terms are disjoint.

### 4.4 Multi-agent debate

Three agents, disjoint glossary slices, **2 rounds** plus synthesis, 20–40 reports.
Default backends: `Qwen/Qwen3.5-4B`, `tencent/Hy-MT2-7B`,
`google/medgemma-1.5-4b-it`. Personas map one-to-one onto the critical detectors
(anatomist → laterality, safety → negation/numbers, linguist → terminology).
Convergence is checked on whitespace- and case-normalised text. **Every agent's
round-0 solo proposal is scored alongside the debate output.**

### 4.5 Review cascade

Three stages, scored separately: `translate` (`Hy-MT2-7B`) → `terminologise`
(`Qwen3.5-4B` **with** `merged_de_en.csv`) → `arbitrate` (`Qwen3.5-4B`
**without** the glossary). Only the terminologist sees the dictionary; the
arbiter exists to catch corrections the glossary got wrong.

Reply format requires **`TRANSLATION:` first**, note second, so truncation costs
the optional field rather than the translation — and `parse_failures` is reported
per stage. Both properties are consequences of a run where 17 of 40 replies were
scored as their own commentary.

### 4.6 COMET

`Unbabel/wmt22-comet-da`, reference-based, over existing outputs — no
retranslation. Runs from an isolated `.venv-comet` (transformers 4.57.6,
`setuptools<81`); it cannot share the project environment.

### 4.7 LLM judge

Clinical rubric with eight categories (negation, laterality, measurement,
anatomy, finding, certainty, terminology, omission), severity by effect on
patient management, scored against the **source** only. Refuses to grade a model
against itself; counts unparseable replies separately. Implemented but **not yet
run at scale**.

## 5. Evaluation configuration

**Surface** — sacreBLEU 2.6.0, signatures persisted with every result:

```
BLEU  |nrefs:1|case:mixed|eff:yes|tok:13a|smooth:exp|version:2.6.0
chrF++|nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0
TER   |nrefs:1|case:lc|tok:tercom|norm:no|punct:yes|asian:no|version:2.6.0
```

Corpus scores use sacreBLEU's corpus scorers, not the mean of sentence scores.

**Clinical** — source-vs-output detectors
([`taxonomy/clinical.py`](../src/medmt_eval/taxonomy/clinical.py)):

| Detector | Severity |
|---|---|
| `negation_dropped`, `negation_introduced` | critical |
| `laterality_missing_or_flipped`, `laterality_added_or_flipped` | critical |
| `number_or_measurement_mismatch` | critical |
| `terminology_not_preserved` | **major** — contributes **zero** to `crit%` |

Headline metric = share of **documents** with ≥1 critical finding.

**Statistics** — exact McNemar on paired critical-error incidence; paired
bootstrap (2,000 resamples, seed 13) for BLEU deltas. At n=20 the error rate
moves in 5-point steps and at n=40 in 2.5-point steps; finer differences are not
interpretable.

**Selection** — there are no checkpoints to select. What is "selected" is an
experimental arm or variant, and every arm is reported rather than only the best.

## 6. Environment and reproducibility

| | |
|---|---|
| Main venv | transformers **5.15.1** (required for `qwen3_5`) |
| COMET venv | `.venv-comet`, transformers 4.57.6, `setuptools<81` |
| Offline | `HF_HUB_DISABLE_XET=1`; compute nodes have no outbound network, so checkpoints are pre-fetched on the login node |
| Allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Seeds | `RT_SEED=13`, `EXP_SEED=13` |
| Persisted per row | full `generation_config`, sacreBLEU signature, served model string, `model_substitution` flag |

Every launcher preflights before requesting work: credentials, gated-repo access,
API reachability, chat-template capability, glossary size, cached checkpoints.
