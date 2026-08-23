# Experiments

## What was run

**Single-pass benchmark** — all thirteen systems over PARROT-DE (296 reports), EMEA
(400), HimL 2015 (353) and HimL 2017 (119). Both evaluation layers.

**Round-trip** — all thirteen systems, ten cycles (twenty passes) over the 20-report
stratified PARROT-DE subsample. Results in
`results/roundtrip_20260814_132452/`.

**Turkish** — the same pipeline on the 48 TR→EN PARROT pairs. Reported as a limitation
only; see [01-dataset.md](01-dataset.md).

**Dictionary injection** — three arms (`none` / `glossary` / `distractor`) on 40
PARROT-DE reports, run twice: once on a 2,834-entry Wikidata bank (3.0 terms per
document) and once on a 47,997-entry RadLex+Wikidata bank (13.6 terms per
document). See [09](09-experiments-glossary-debate.md).

**Multi-agent debate** — three local instruction-following models with distinct
personas and disjoint glossary slices, two rounds plus synthesis, 20 reports.

**Review cascade** — an asymmetric pipeline (MT drafts → terminologist with the
glossary corrects → radiologist without the glossary arbitrates), every stage
scored separately, 40 reports.

**COMET** — `Unbabel/wmt22-comet-da` over all thirteen systems' existing
single-pass outputs. No retranslation. See [07](07-metric-roadmap.md).

## Hardware and cost

SLURM, mixed A100 partition. The nodes are **not** uniform: 19 carry the `a100_40`
feature and 18 carry `a100_80`, and an unconstrained `--gres=gpu:a100` request lands on
either. Early runs happened to get 40 GB cards, which produced a wrong conclusion —
recorded here because it shaped several sizing decisions before it was corrected — that
all the cluster's A100s were 40 GB. `--constraint=a100_80` pins the large jobs.

Round-trip wall time (20 passes × 20 reports):

| System | Wall | Placement |
|---|---|---|
| `identity` | <1 min | MIG slice |
| `opus` | 5 min | MIG slice |
| `glm-5.2` | 13 min | API (no GPU) |
| `MiniMax-M3` | 13 min | API (no GPU) |
| `nllb` | 27 min | MIG slice |
| `qwen35-4b` | 39 min | MIG slice |
| `hymt2-1.8b` | 44 min | MIG slice |
| `qwen35-27b` | 46 min | 1×80 GB, batch 4 |
| `translategemma-4b` | 48 min | MIG slice |
| `hymt2-7b` | 54 min | MIG slice, batch 4 |
| `translategemma-27b` | 92 min | 1×80 GB, batch 4 |
| `hymt2-30b-a3b` | 137 min | 2×80 GB, batch 8 |
| `DeepSeek-V4-Flash` | 189 min | API (no GPU) |

Moving the 27 B models from 2×40 GB (batch 1, weights sharded) to 1×80 GB (batch 4) cut
`qwen35-27b` from an estimated 2.4 h to 46 min, and a single-GPU request also schedules
sooner on a busy partition.

## Failures worth recording

Each of these changed either the code or the conclusions.

**Gated repositories (jobs 4006927, 4006946).** Both TranslateGemma runs died on an
opaque HTTP 401 after ~55 s. Cause: SLURM inherits the submitting shell's environment,
and `HF_TOKEN` was not exported in it. Fixed with a hard `:?` guard in `_rt_common.sh`
that fails at submission with a readable message.

There is a second lesson here that cost a full round of work. The token *was* present in
the user's `~/.bashrc`, but a non-interactive shell does not source `.bashrc` — bash
skips it unless the shell is interactive or `BASH_ENV` is set, and there was no
`~/.bash_profile` to bridge it. The environment looked empty when it was not. Sourcing
it explicitly resolved it.

**API rate limiting (jobs 4008963, 4008965).** Three hosted-model jobs submitted in
parallel, 8 workers each, one shared key → 24 concurrent requests → HTTP 429. Two of
three died. Two fixes: exponential backoff with jitter honouring `Retry-After`, and —
more importantly — the per-item fallback was *amplifying* the problem, turning one
rate-limited batch of 8 into 8 more requests. The jobs now run as a **serial chain** at
3 workers.

**Origin timeout (job 4009014).** After the 429 fix, `DeepSeek-V4-Flash` failed at 78
minutes with Cloudflare `524 Response Timeout by Origin Server`. Two distinct errors:
524 was not in the retry set, and the no-fanout rule introduced for 429 had been applied
to *all* HTTP errors, which made this case worse. A 429 means "you are sending too
much" — do not send more. A 524 means "this request did not come back" — smaller
requests are precisely the cure. Only 429 now propagates; timeouts fan out as before.
Re-run at batch 2, completed in 189 minutes.

**Round-trip OOM (job 4006929).** `hymt2-7b` (14 GB) OOM'd on a 20 GB MIG slice because
the runner built one translator instance per direction. Only Opus needs that; every
other adapter takes the direction as a call argument. Fixed with the
`direction_specific` flag — one instance unless the adapter is pinned.

**A destructive mistake of my own.** An earlier refactor used `$(dirname "$0")` in a
SLURM script, which does not resolve as expected under SLURM. Correcting it to absolute
paths killed two running jobs (3942019, 3942020). The check performed before editing
looked for writers in `results/` but not in `scripts/`.

**Compute nodes have no outbound network (job 4077499).** COMET's checkpoint
fetch works interactively and fails inside a job — and COMET swallows the real
error, re-raising it as `Model 'Unbabel/wmt22-comet-da' not supported by COMET`,
which points at the checkpoint name rather than at the network. Fixed by
pre-fetching on the login node and loading from a local `.ckpt` path, with a
preflight that fails at submission if the cache is empty.

**Installing a metric broke the models (job 4077516).** `unbabel-comet` 2.2.7
requires `transformers` 4.x, so installing it silently downgraded the shared venv
from 5.15.1 to 4.57.6. That broke `Qwen3.5` loading everywhere
(`model type 'qwen3_5' not recognized`) and killed a cascade job that had nothing
to do with COMET. Upgrading back breaks COMET instead — its XLM-R encoder unpacks
a three-tuple that transformers 5.x no longer returns. They are mutually
exclusive; COMET now lives in `.venv-comet`.

This one is worth dwelling on because the misdiagnosis was instructive. The
symptom was a model failing *only* in the cascade, where Hy-MT2 loads before
Qwen, and not in the debate, where Qwen loads first. That is a perfectly coherent
load-order hypothesis, and it was wrong. What settled it was the dist-info
timestamp on `transformers-4.57.6` matching the COMET install to the minute — a
fact about the environment, not about the code.

**Tokenizer output the model rejects (job 4077495).** `Hy-MT2-7B`'s tokenizer
returns `token_type_ids`; its `generate()` refuses them
(`model_kwargs are not used by the model`). Qwen's and Gemma's tokenizers do not
return the field, so this is per-model and cannot be assumed away. Fixed by
filtering the encoding against the model's actual `forward` signature rather than
a hardcoded deny-list, so the next tokenizer with an extra field does not break
the same way.

**A translation-only model cannot join a debate (job 4077236).**
`translategemma-4b-it`'s chat template requires each message's content to be a
structured mapping carrying `source_lang_code`/`target_lang_code`. It can express
"translate this" and nothing else — no persona, no critique. The job died twelve
minutes in, after loading three models, on a Jinja `TemplateError`. Fixed with a
tokenizer-only capability probe that runs in the preflight, before any weights
load.

**The hosted gateway degraded mid-project.** Between the round-trip runs and the
experiments, `glm-5.2` and `MiniMax-M3` were withdrawn entirely, and of what
remained only `DeepSeek-V4-Flash` still routed honestly — `Qwen3.8-27B` was
served as `meta/muse-glimmer-30b`, `Qwen3.8-35B-A3B` as
`nvidia/nemotron-3.5-lightning-30b-a3b`, `Kimi-K2.6` as
`thinkingmachines/inkling`. The substitution guard caught every case. This is why
the debate and cascade experiments default to local models: a participant that
cannot be named is not reproducible.

**A parser turned commentary into a finding (job 4077545).** The cascade's
terminologist stage appeared to destroy translation quality — BLEU −17.9,
critical errors +22.5. It had not. A 4B model asked for a two-field structured
reply wrote its `ISSUES:` list and never reached `TRANSLATION:` (1 reply in 40
contained the marker), and the parser's fallback returned the whole reply as the
translation on 17 of 40 documents.

Three properties of this failure make it worth recording. It produced a
*plausible* result — one that confirmed an existing finding, which is exactly
when a result gets least scrutiny. The evidence was already in hand: mean output
length jumped from 769 to 3,446 characters. And the guard that would have caught
it had been written into a different module an hour earlier and not carried
across. The fix reorders the reply format so truncation costs the optional field,
and reports `parse_failures` per stage.

## Verification practices adopted

- **Preflight everything.** Credentials, gated-repo access, API reachability,
  chat-template capability, glossary size and cached checkpoints are all probed
  before a long job is submitted. Five separate failure classes in this project
  presented as an opaque error minutes into a job and were each one cheap check
  at submission time.
- **Check the parse rate before reading any score** from a stage that asks a
  model for structured output, and put the critical field first in the required
  format so truncation damages the optional one.
- **Never transcribe numbers.** Figures and published tables are generated
  programmatically from `roundtrip_steps.csv`. This was adopted after several
  intermediate values in a hand-written chart data block turned out to be wrong — they
  had been typed from memory rather than read from the file. Published output is now
  re-parsed and diffed against the source CSV before release.
- **Look at the rendered figure.** Three separate layout defects — colliding scatter
  labels, a legend sitting on top of the data, and a mislabelled series — were only
  visible on inspection, not from the code.
