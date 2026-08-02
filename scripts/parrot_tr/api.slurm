#!/usr/bin/env bash
#SBATCH --job-name=parrot-tr-api
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100small:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/parrot-tr-api-%j.out
#SBATCH --error=logs/parrot-tr-api-%j.err
#
# PARROT Turkish radiology reports (48 segments, TR->EN) via a hosted
# OpenAI-compatible endpoint. See scripts/lib/parrot_tr_common.sh for the reduced
# detector coverage and small-subset caveats.
#
# Only aliases verified to route honestly (2026-07-31):
#   glm-5.2  DeepSeek-V4-Flash  MiniMax-M3
# DeepSeek-V4-Pro and Kimi-K2.6 are silently substituted; the adapter aborts
# on a mismatch rather than mislabelling a result.
#
# No GPU is used — this is a pure HTTP client. A slice is requested only
# because the cluster has no CPU-only partition.
#
# Submit: API_MODEL=glm-5.2 sbatch scripts/benchmark_parrot_tr_api.slurm
#
set -euo pipefail

: "${OPENAI_COMPAT_API_KEY:=${DEEPSEEK_API_KEY:-}}"
: "${OPENAI_COMPAT_API_KEY:?Set OPENAI_COMPAT_API_KEY (or DEEPSEEK_API_KEY) before submitting}"
export OPENAI_COMPAT_API_KEY

API_MODEL="${API_MODEL:-glm-5.2}"
API_LABEL="${API_LABEL:-$API_MODEL}"
export OPENAI_COMPAT_MAX_WORKERS="${OPENAI_COMPAT_MAX_WORKERS:-8}"

module purge
module load python/3.12-base

PROJECT=/home/atuin/b180dc/b180dc50/german_medical_translator
cd "$PROJECT"
mkdir -p logs data/derived
source .venv/bin/activate

RUN_DIR="${RESUME_DIR:-results/parrot_tr_${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"

PARROT_SRC="${PARROT_SRC:-PARROT_v1.0/data/PARROT_v1_0.jsonl}"
PARROT_TR_JSONL="${PARROT_TR_JSONL:-data/derived/parrot_tr.jsonl}"
[ -f "$PARROT_TR_JSONL" ] || medmt-eval convert parrot \
  --input "$PARROT_SRC" --output "$PARROT_TR_JSONL" --src-lang tr --tgt-lang en

echo "=== PARROT-Turkish API benchmark ==="
echo "    model:    $API_MODEL"
echo "    segments: $(wc -l < "$PARROT_TR_JSONL")  (TR->EN, reduced detector coverage)"
echo "    results:  $RUN_DIR"

OUT="$RUN_DIR/parrot_tr_${API_LABEL}.jsonl"
if [ -f "$OUT" ]; then echo "=== already done, skipping ==="; exit 0; fi

_t0=$(date +%s)
# Turkish reports are long (median 2,080 chars), so batches are small and the
# token ceiling is raised relative to the German run.
medmt-eval run \
  --input "$PARROT_TR_JSONL" --model openai-compat --model-id "$API_MODEL" \
  --batch-size 4 --num-beams 1 --max-new-tokens 2048 \
  --output "$OUT" \
  --summary "$RUN_DIR/parrot_tr_${API_LABEL}.summary.json" \
  --term-bank data/term_banks/radiology_en_de_starter.csv \
  2>> "$RUN_DIR/parrot_tr_${API_LABEL}.time.log"
_t1=$(date +%s)
echo "wall_seconds=$(( _t1 - _t0 ))" | tee -a "$RUN_DIR/parrot_tr_${API_LABEL}.time.log"
echo "=== Done: $OUT ==="
