# Systems under test

Thirteen entries: twelve translation systems plus a control.

| Name | Model ID | Class | Params | Adapter |
|---|---|---|---|---|
| `identity` | — | **control** | — | `identity` |
| `opus` | `Helsinki-NLP/opus-mt-{de-en,en-de}` | Dedicated bilingual MT | 74 M | `opus` |
| `nllb` | `facebook/nllb-200-distilled-1.3B` | Massively multilingual MT | 1.3 B | `nllb` |
| `hymt2-1.8b` | `tencent/Hy-MT2-1.8B` | Specialised MT | 1.8 B | `hymt2` |
| `hymt2-7b` | `tencent/Hy-MT2-7B` | Specialised MT | 7 B | `hymt2` |
| `hymt2-30b-a3b` | `tencent/Hy-MT2-30B-A3B` | Specialised MT (MoE) | 30 B / 3 B active | `hymt2` |
| `translategemma-4b` | `google/translategemma-4b-it` | Specialised MT | 4 B | `translategemma` |
| `translategemma-27b` | `google/translategemma-27b-it` | Specialised MT | 27 B | `translategemma` |
| `qwen35-4b` | `Qwen/Qwen3.5-4B` | General LLM, prompted | 4 B | `prompted-llm` |
| `qwen35-27b` | `Qwen/Qwen3.5-27B` | General LLM, prompted | 27 B | `prompted-llm` |
| `glm-5.2` | `z-ai/glm-5.2` | Hosted frontier LLM | — | `openai-compat` |
| `DeepSeek-V4-Flash` | `deepseek-ai/deepseek-v4-flash-0731` | Hosted frontier LLM | — | `openai-compat` |
| `MiniMax-M3` | `minimaxai/minimax-m3` | Hosted frontier LLM | — | `openai-compat` |

The set spans the four hypotheses worth testing: a small dedicated bilingual model, a
massively multilingual one, purpose-built translation LLMs, and general-purpose LLMs
prompted to translate — local and hosted.

## The control

`identity` returns the source unchanged. It is not a translation system; it is the
floor. Its scores establish what "no translation at all" looks like under every metric,
which is the only way to know a metric has a working scale. On German it scores BLEU
3.39 and a 95% critical-error rate — correctly catastrophic. On Turkish it scores 0%
critical errors, which is how the Turkish coverage problem was found
([01-dataset.md](01-dataset.md)).

It is excluded from every figure and reported separately in tables.

## Adapter notes

Each family needed something specific. These are the ones that changed results, not just
plumbing.

**`opus` is direction-specific.** Helsinki-NLP ships one checkpoint per language pair,
and an instance pins itself to the first direction used. It is the only adapter with
`direction_specific = True`, so the round-trip runner builds two instances for it and
exactly one for everything else. Building two of everything doubled GPU memory and
OOM'd a 14 GB model on a 20 GB slice before this was distinguished.

**Prompted LLMs must use the chat template.** `PromptedLLMTranslator` originally fed raw
text to the model. Fixed to call `apply_chat_template` with `enable_thinking=False`,
plus a `strip_thinking()` regex as a second line of defence for templates that emit
`<think>` blocks anyway. Without this, reasoning traces land in the translation output.

**Left padding for batched decoder-only generation.** Right padding corrupts batched
generation for causal LMs. Setting `padding_side="left"` moved `qwen35-4b`'s critical
error rate from 24.66% to 18.92% — a change in the *result*, not just in speed.

**Position limits are a per-model property.** Opus has 512 encoder *and* decoder
positions; NLLB has 1024. Both truncate radiology reports, so both are chunked at
sentence boundaries. Opus additionally needs `max_new_tokens = 480`: exceeding its
decoder table raises a CUDA device-side assert (`marian/modeling_marian.py:596`), which
took three attempts to attribute correctly — the first fix reduced batch size, which
addressed the wrong cause.

**MoE weight storage ≠ compute.** `Hy-MT2-30B-A3B` activates ~3 B parameters but must
*hold* ~60 GB of weights, so it needs 2×80 GB despite being cheap to run. Sizing it by
active parameters would have failed to load.

## Hosted models: the substitution guard

The gateway used for the hosted models has been observed **serving a different model
than requested, with no error**: `DeepSeek-V4-Pro` was answered by
`nvidia/nemotron-3-ultra-550b-a55b`, and `Kimi-K2.6` by `thinkingmachines/inkling`.
The `/v1/models` catalogue is not trustworthy either — every entry claims
`"owned_by": "openai"`.

The only reliable signal is the `model` field of an actual response, which
`_check_served_model()` verifies on **every call**. Results are never attributed to a
model that did not produce them; a mismatch fails the run rather than mislabelling data.

Only three aliases are verified to route honestly and they are the only hosted models
reported:

```
glm-5.2            → z-ai/glm-5.2
DeepSeek-V4-Flash  → deepseek-ai/deepseek-v4-flash-0731
MiniMax-M3         → minimaxai/minimax-m3
```

The guard needed two corrections, both of which blocked legitimate runs rather than
admitting bad ones: first for namespacing (`glm-5.2` → `z-ai/glm-5.2`), then for dated
snapshots (`DeepSeek-V4-Flash` → `…-flash-0731`). Version-tag stripping is deliberately
narrow — digits only, optionally `v`-prefixed — so a differing *name* can never be
excused as a differing version.

## Decoding

Greedy/beam settings are fixed across systems and persisted with every result
(`generation_config`). Round-trip runs use `num_beams=1` for tractability; the
single-pass benchmark uses the per-adapter default. Temperature is 0 for the hosted
models so runs are reproducible, matching the local models.
