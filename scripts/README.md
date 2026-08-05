# SLURM scripts

Organised by **corpus**, since that is what determines the data, the direction,
and which clinical detectors are available.

```
scripts/
├── lib/                  shared setup, sourced by the others — not submitted directly
│   ├── bench_common.sh       venv, HF cache, GPU logger, data prep, run_models()
│   ├── parrot_de_common.sh   overrides DATASETS for PARROT German
│   └── parrot_tr_common.sh   overrides DATASETS for PARROT Turkish
├── himl_emea/            HimL + EMEA, EN→DE and DE→EN, one script per model
├── parrot_de/            PARROT German radiology reports, DE→EN, tiered by GPU need
├── parrot_tr/            PARROT Turkish radiology reports, TR→EN, tiered by GPU need
├── utils/                connectivity checks, smoke tests, leaderboard assembly
└── deprecated/           superseded by the per-model / per-tier scripts
```

Paths are relative to the repository root. Submit from there, not from inside
`scripts/`.

## Prerequisites

Both benchmarks need credentials exported in your shell — SLURM inherits them.
Put these in `~/.bashrc`; never hardcode them in a script (a committed token has
already blocked one push to this repo).

```bash
export HF_TOKEN="..."                 # gated repos: TranslateGemma
export OPENAI_COMPAT_API_KEY="..."    # hosted-API benchmarks only
```

## HimL / EMEA

Three corpora, ~872 segments per system. One script per model, each requesting
only the GPU that model actually needs.

```bash
sbatch scripts/himl_emea/hymt2_1.8b.slurm          # a100small MIG
sbatch scripts/himl_emea/hymt2_7b.slurm            # a100med MIG
sbatch scripts/himl_emea/translategemma_4b.slurm   # a100med MIG
sbatch scripts/himl_emea/qwen35_4b.slurm           # a100med MIG
sbatch scripts/himl_emea/hymt2_30b_moe.slurm       # 2x a100 (MoE: ~60GB weights)
sbatch scripts/himl_emea/translategemma_27b.slurm  # 2x a100
sbatch scripts/himl_emea/qwen35_27b.slurm          # 2x a100
```

## PARROT German (DE→EN)

296 radiology reports. Full detector coverage — negation, laterality, numbers
and terminology all active.

```bash
sbatch scripts/parrot_de/small.slurm    # identity, opus, nllb
sbatch scripts/parrot_de/medium.slurm   # hymt2-1.8b, translategemma-4b, qwen35-4b
sbatch scripts/parrot_de/7b.slurm       # hymt2-7b
sbatch scripts/parrot_de/large.slurm    # 27B/30B models, 2x a100

API_MODEL=glm-5.2 sbatch scripts/parrot_de/api.slurm
```

Verified-honest API aliases: `glm-5.2`, `DeepSeek-V4-Flash`, `MiniMax-M3`.
`DeepSeek-V4-Pro` and `Kimi-K2.6` are silently substituted by the gateway; the
adapter aborts on a mismatch rather than mislabelling a result.

## PARROT Turkish (TR→EN)

48 reports. Two caveats, both significant:

**Reduced detector coverage.** The cue lexicons cover EN and DE only, so
negation, laterality and terminology all skip and only the language-agnostic
number check runs. Results are *not* comparable to the German ones, and
`identity` is no longer a meaningful floor — it scores 0 % here versus ~96 % on
German, because copying the source through trivially preserves every number.

**The first run (jobs 3941775–3941778) is invalid.** The tier scripts did not
pass `--max-new-tokens`, so models inherited the 512-token CLI default and were
truncated mid-report — Turkish reports have a median length of ~2,080 characters.
See [`results/parrot_tr/INVALID_RUNS.md`](../results/parrot_tr/INVALID_RUNS.md).
Fixed here; re-run into a **fresh** directory rather than resuming.

```bash
sbatch scripts/parrot_tr/small.slurm
sbatch scripts/parrot_tr/medium.slurm
sbatch scripts/parrot_tr/7b.slurm
sbatch scripts/parrot_tr/large.slurm

API_MODEL=glm-5.2 sbatch scripts/parrot_tr/api.slurm
```

## Output length

`lib/bench_common.sh` passes `--max-new-tokens` explicitly (default **2048**);
`lib/parrot_tr_common.sh` raises it to **3072** for the longer Turkish reports.
Override per run if needed — an explicit value always wins:

```bash
MAX_NEW_TOKENS=4096 sbatch scripts/parrot_tr/large.slurm
```

The CLI default of 512 is too low for whole radiology reports and truncates
silently, which reads as terrible translation quality rather than a config error.

## Utilities

```bash
# Can a compute node reach the hosted API? Run before a long API job.
sbatch scripts/utils/check_api_connectivity.slurm

# One real inference per adapter — catches silent tokenizer/template bugs
# before they waste hours. This is how the MADLAD garbage-output and the
# Hy-MT2 BatchEncoding crash were both caught.
sbatch scripts/utils/smoke_test_new_adapters.slurm

# Interactive GPU session sanity check
salloc --gres=gpu:a40:1 --time=00:30:00 --partition=a40
bash scripts/utils/quick_test_interactive.sh

# Merge a run directory into one leaderboard
RUN_DIR=results/parrot_de/consolidated sbatch scripts/utils/assemble_leaderboard.slurm
```

## Resuming

Every tier script skips models whose output already exists, so a failed or
timed-out run is resumed by pointing at the same directory:

```bash
RESUME_DIR=results/parrot_tr/3941775 sbatch scripts/parrot_tr/small.slurm
```

Use a **fresh** directory instead when the previous output was wrong rather than
missing — resuming would preserve the bad results.
