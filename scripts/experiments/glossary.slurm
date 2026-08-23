#!/usr/bin/env bash
#SBATCH --job-name=exp-glossary
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/exp-%x-%j.out
#SBATCH --error=logs/exp-%x-%j.err
#
# Experiment 1 — dictionary-augmented translation, three arms.
#
#   none        no glossary block          (baseline)
#   glossary    the terms in this document (treatment)
#   distractor  the same number of terms that are NOT in this document (control)
#
# The distractor arm is the one that makes this an experiment. A glossary block
# lengthens and restructures the prompt, and either could move the score on its
# own; only glossary-vs-distractor isolates the terms themselves.
#
# Submit:
#   BACKEND=local:Qwen/Qwen3.5-4B sbatch scripts/experiments/glossary.slurm
#   BACKEND=api:DeepSeek-V4-Flash sbatch scripts/experiments/glossary.slurm
set -euo pipefail
: "${BACKEND:?Set BACKEND, e.g. BACKEND=local:Qwen/Qwen3.5-4B}"
source /home/atuin/b180dc/b180dc50/german_medical_translator/scripts/experiments/_exp_common.sh

case "$BACKEND" in
  api:*) : "${OPENAI_COMPAT_API_KEY:?export OPENAI_COMPAT_API_KEY before submitting}" ;;
esac

SLUG=$(echo "$BACKEND" | tr '/:' '__')
OUT="$RUN_DIR/glossary_${SLUG}.jsonl"
[ -f "$OUT" ] && { echo "=== $BACKEND already done, skipping ==="; exit 0; }

echo "=== glossary experiment: $BACKEND ==="
_t0=$(date +%s)
medmt-eval glossary-run \
  --input "$EXP_INPUT" --src-lang de --tgt-lang en \
  --backend "$BACKEND" --glossary "$EXP_GLOSSARY" --branches "$EXP_BRANCHES" \
  --arms none glossary distractor \
  --sample-size "$EXP_SAMPLE" --seed "$EXP_SEED" --max-terms "$EXP_MAX_TERMS" \
  --holdout "${EXP_HOLDOUT:-0.0}" \
  --term-bank "$EXP_TERM_BANK" \
  --output "$OUT" --summary "$RUN_DIR/glossary_${SLUG}.summary.json"
echo "wall_seconds=$(( $(date +%s) - _t0 ))"
echo "=== done: $OUT ==="
