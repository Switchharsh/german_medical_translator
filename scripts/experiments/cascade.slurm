#!/usr/bin/env bash
#SBATCH --job-name=exp-cascade
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/exp-%x-%j.out
#SBATCH --error=logs/exp-%x-%j.err
#
# Sequential review pipeline, scored at every stage.
#
#   1 translate      Hy-MT2-7B drafts the translation
#   2 terminologise  Qwen3.5-4B + the RadLex glossary corrects what it thinks is wrong
#   3 arbitrate      Qwen3.5-4B as a German radiologist, WITHOUT the glossary,
#                    weighs the disagreement and issues the final text
#
# Only stage 2 sees the dictionary. That asymmetry is the experiment: stage 3
# exists to catch corrections the glossary got wrong, and the previous run showed
# it has something to catch — 26% of glossary terms the model adopted were terms
# the human reference does not use.
#
# Every stage's output is scored separately, so each is credited only with what
# it changed. A stage that makes things worse is visible rather than averaged in.
#
# Submit: sbatch scripts/experiments/cascade.slurm
set -euo pipefail
source /home/atuin/b180dc/b180dc50/german_medical_translator/scripts/experiments/_exp_common.sh

BACKENDS="${BACKENDS:-local:tencent/Hy-MT2-7B local:Qwen/Qwen3.5-4B local:Qwen/Qwen3.5-4B}"
ROLES="${ROLES:-translate terminologise arbitrate}"
CASCADE_GLOSSARY="${CASCADE_GLOSSARY:-data/term_banks/merged_de_en.csv}"

case "$BACKENDS" in *gemma*) : "${HF_TOKEN:?Gemma repos are gated — export HF_TOKEN}" ;; esac
case "$BACKENDS" in *api:*) : "${OPENAI_COMPAT_API_KEY:?export OPENAI_COMPAT_API_KEY}" ;; esac

python3 scripts/experiments/probe_backends.py $BACKENDS

OUT="$RUN_DIR/cascade.jsonl"
[ -f "$OUT" ] && { echo "=== cascade already done, skipping ==="; exit 0; }

echo "=== cascade: $BACKENDS ==="
echo "    roles: $ROLES"
_t0=$(date +%s)
medmt-eval cascade \
  --input "$EXP_INPUT" --src-lang de --tgt-lang en \
  --backends $BACKENDS --roles $ROLES \
  --glossary "$CASCADE_GLOSSARY" --glossary-stage terminologise \
  --max-terms "$EXP_MAX_TERMS" --sample-size "$EXP_SAMPLE" --seed "$EXP_SEED" \
  --term-bank "$EXP_TERM_BANK" \
  --output "$OUT" --summary "$RUN_DIR/cascade.summary.json"
echo "wall_seconds=$(( $(date +%s) - _t0 ))"
echo "=== done: $OUT ==="
