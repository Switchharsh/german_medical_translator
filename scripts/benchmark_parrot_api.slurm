#!/usr/bin/env bash
#SBATCH --job-name=parrot-api
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100small:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/parrot-api-%j.out
#SBATCH --error=logs/parrot-api-%j.err
#
# PARROT German radiology reports (296 segments, DE->EN) via a hosted
# OpenAI-compatible /v1/chat/completions endpoint.
#
# WHICH MODEL: set API_MODEL (default glm-5.2). Only use aliases whose routing
# has been verified — this gateway silently substitutes models for some names.
# Verified 2026-07-31 by reading the `model` field of a live response:
#
#   glm-5.2            -> z-ai/glm-5.2                        HONEST
#   DeepSeek-V4-Flash  -> deepseek-ai/deepseek-v4-flash       HONEST
#   MiniMax-M3         -> minimaxai/minimax-m3                HONEST
#   DeepSeek-V4-Pro    -> nvidia/nemotron-3-ultra-550b-a55b   SUBSTITUTED
#   Kimi-K2.6          -> thinkingmachines/inkling            SUBSTITUTED
#
# The /v1/models catalogue is NOT a reliable guide (every entry there claims
# "owned_by": "openai"). The adapter re-checks the served model on every
# request and aborts on a mismatch, so a substituted alias fails loudly instead
# of producing a mislabelled leaderboard row. The served model is recorded in
# generation_config either way.
#
# NO GPU IS USED. This adapter is a pure HTTP client; inference happens on the
# remote endpoint. A GPU is requested only because this cluster has no CPU-only
# partition (sinfo lists a40/a100/a100mig/rtxpro6k and nothing else), so the
# smallest MIG slice is the cheapest way to obtain a compute node. If a CPU
# partition is ever added, switch to it and drop the --gres line. Since the GPU
# is idle regardless, this job can run alongside the GPU benchmarks.
#
# Timing: observed ~27 generated tokens/s on this gateway. With batch_size=8
# that is 37 batches; at 8 concurrent workers, expect minutes rather than
# hours. The 6h limit is deliberately generous because a slow or rate-limited
# endpoint is the likeliest failure mode, and the per-item fallback path (used
# when a batch response is malformed) multiplies requests for that batch.
#
# Requires an API key in the environment — export OPENAI_COMPAT_API_KEY (or
# DEEPSEEK_API_KEY) in ~/.bashrc, the same way HF_TOKEN is handled. Never
# hardcode it here: a committed token already blocked one push to this repo.
#
# Submit: API_MODEL=glm-5.2 sbatch scripts/benchmark_parrot_api.slurm
# Resume: RESUME_DIR=results/parrot_XXXX sbatch scripts/benchmark_parrot_api.slurm
#
set -euo pipefail

: "${OPENAI_COMPAT_API_KEY:=${DEEPSEEK_API_KEY:-}}"
: "${OPENAI_COMPAT_API_KEY:?Set OPENAI_COMPAT_API_KEY (or DEEPSEEK_API_KEY) before submitting}"
export OPENAI_COMPAT_API_KEY

API_MODEL="${API_MODEL:-glm-5.2}"
# Leaderboard label, derived from the model name so result files are
# self-describing (e.g. glm-5.2 -> parrot_de_glm-5.2.jsonl).
API_LABEL="${API_LABEL:-$API_MODEL}"

# Tunables:
#   OPENAI_COMPAT_MAX_WORKERS   concurrent in-flight requests (default 8)
#   OPENAI_COMPAT_BASE_URL      override the endpoint
export OPENAI_COMPAT_MAX_WORKERS="${OPENAI_COMPAT_MAX_WORKERS:-8}"

module purge
module load python/3.12-base

PROJECT=/home/atuin/b180dc/b180dc50/german_medical_translator
cd "$PROJECT"
mkdir -p logs data/derived

source .venv/bin/activate

RUN_DIR="${RESUME_DIR:-results/parrot_${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"

PARROT_SRC="${PARROT_SRC:-PARROT_v1.0/data/PARROT_v1_0.jsonl}"
PARROT_JSONL="${PARROT_JSONL:-data/derived/parrot_de.jsonl}"
[ -f "$PARROT_JSONL" ] || medmt-eval convert parrot \
  --input "$PARROT_SRC" --output "$PARROT_JSONL" --src-lang de --tgt-lang en

echo "=== API benchmark on PARROT ==="
echo "    model:    $API_MODEL"
echo "    segments: $(wc -l < "$PARROT_JSONL")"
echo "    workers:  $OPENAI_COMPAT_MAX_WORKERS, batch_size=8"
echo "    results:  $RUN_DIR"

OUT="$RUN_DIR/parrot_de_${API_LABEL}.jsonl"
if [ -f "$OUT" ]; then
  echo "=== already done, skipping ==="
  exit 0
fi

_t0=$(date +%s)
medmt-eval run \
  --input "$PARROT_JSONL" --model openai-compat --model-id "$API_MODEL" \
  --batch-size 8 --num-beams 1 --max-new-tokens 1024 \
  --output "$OUT" \
  --summary "$RUN_DIR/parrot_de_${API_LABEL}.summary.json" \
  --term-bank data/term_banks/radiology_en_de_starter.csv \
  2>> "$RUN_DIR/parrot_de_${API_LABEL}.time.log"
_t1=$(date +%s)
echo "wall_seconds=$(( _t1 - _t0 ))" | tee -a "$RUN_DIR/parrot_de_${API_LABEL}.time.log"

echo "=== Done: $OUT ==="
