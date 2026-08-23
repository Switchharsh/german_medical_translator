#!/usr/bin/env bash
#SBATCH --job-name=exp-debate
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/exp-%x-%j.out
#SBATCH --error=logs/exp-%x-%j.err
#
# Experiment 2 — three personas, three glossary slices, N rounds, then sign-off.
#
# Defaults to LOCAL models. The hosted gateway is not usable for this today:
# of the models it lists, only DeepSeek-V4-Flash still routes honestly —
# Qwen3.8-27B is served as meta/muse-glimmer-30b, Qwen3.8-35B-A3B as
# nvidia/nemotron-3.5-lightning-30b-a3b, Kimi-K2.6 as thinkingmachines/inkling,
# and glm-5.2 / MiniMax-M3 have been withdrawn entirely. A debate whose
# participants cannot be named is not reproducible.
#
# The three defaults are different families on purpose (a general LLM, a
# specialised MT model, a translation-tuned Gemma), ~32 GB of bf16 weights in
# total, which fits one 80 GB card with room for activations.
#
# Submit:
#   sbatch scripts/experiments/debate.slurm
#   BACKENDS="api:DeepSeek-V4-Flash api:DeepSeek-V4-Flash api:DeepSeek-V4-Flash" \
#     sbatch scripts/experiments/debate.slurm     # single-model ablation
set -euo pipefail
source /home/atuin/b180dc/b180dc50/german_medical_translator/scripts/experiments/_exp_common.sh

# Three instruction-following families: a general LLM, a specialised MT model,
# and a medical-domain model. ~32 GB of bf16 weights, fits one 80 GB card.
# translategemma is deliberately absent: its chat template is translation-only
# and cannot carry a persona or a critique. The preflight below enforces that.
BACKENDS="${BACKENDS:-local:Qwen/Qwen3.5-4B local:tencent/Hy-MT2-7B local:google/medgemma-1.5-4b-it}"
ROUNDS="${ROUNDS:-2}"

case "$BACKENDS" in
  *api:*) : "${OPENAI_COMPAT_API_KEY:?export OPENAI_COMPAT_API_KEY before submitting}" ;;
esac
case "$BACKENDS" in
  *gemma*) : "${HF_TOKEN:?Gemma repos are gated — export HF_TOKEN before submitting}" ;;
esac

python3 scripts/experiments/probe_backends.py $BACKENDS

OUT="$RUN_DIR/debate.jsonl"
[ -f "$OUT" ] && { echo "=== debate already done, skipping ==="; exit 0; }

echo "=== debate: $BACKENDS  (rounds=$ROUNDS) ==="
_t0=$(date +%s)
medmt-eval debate \
  --input "$EXP_INPUT" --src-lang de --tgt-lang en \
  --backends $BACKENDS \
  ${SYNTHESISER:+--synthesiser "$SYNTHESISER"} \
  --glossary "$EXP_GLOSSARY" --branches "$EXP_BRANCHES" \
  --rounds "$ROUNDS" --sample-size "$EXP_SAMPLE" --seed "$EXP_SEED" \
  --max-terms "$EXP_MAX_TERMS" --term-bank "$EXP_TERM_BANK" \
  --output "$OUT" --summary "$RUN_DIR/debate.summary.json"
echo "wall_seconds=$(( $(date +%s) - _t0 ))"
echo "=== done: $OUT ==="
