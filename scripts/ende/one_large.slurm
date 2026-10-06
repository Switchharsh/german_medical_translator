#!/usr/bin/env bash
#SBATCH --job-name=ende-large
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/ende-%x-%j.out
#SBATCH --error=logs/ende-%x-%j.err
#
# PARROT EN->DE single pass for ONE large model. The DE->EN benchmark ran these three
# models in one 2-GPU job; one job per model is far easier to backfill on a busy
# cluster, and a failure no longer takes the other two with it.
#
#   MODEL=translategemma-27b RESUME_DIR=results/parrot_en_de_X sbatch --job-name=ende-tg27 \
#       --constraint=a100_80 scripts/ende/one_large.slurm
#   MODEL=hymt2-30b-a3b ... sbatch --gres=gpu:a100:2 scripts/ende/one_large.slurm
set -euo pipefail
: "${MODEL:?Set MODEL: hymt2-30b-a3b | translategemma-27b | qwen35-27b}"
: "${RESUME_DIR:?Set RESUME_DIR so all systems land in one results directory}"
case "$MODEL" in
  translategemma-*) : "${HF_TOKEN:?$MODEL is a gated repo: export HF_TOKEN before submitting}" ;;
esac
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export RESUME_DIR
export PARROT_JSONL="${PARROT_JSONL:-data/derived/parrot_en_de.jsonl}"

source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/lib/bench_common.sh"
source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/lib/parrot_de_common.sh"

MODELS="$MODEL"
declare -A MODEL_IDS=(
    [hymt2-30b-a3b]="tencent/Hy-MT2-30B-A3B"
    [translategemma-27b]="google/translategemma-27b-it"
    [qwen35-27b]="Qwen/Qwen3.5-27B"
)
# batch 4 on a single 80 GB card for the 27B models (the DE->EN round trip used the
# same); the MoE keeps the DE->EN setting of 2.
declare -A BATCH_SIZE=( [hymt2-30b-a3b]=2 [translategemma-27b]=4 [qwen35-27b]=4 )
declare -A NUM_BEAMS=(  [hymt2-30b-a3b]=1 [translategemma-27b]=1 [qwen35-27b]=1 )

echo "preflight: model=$MODEL id=${MODEL_IDS[$MODEL]} batch=${BATCH_SIZE[$MODEL]} dataset=$PARROT_JSONL dir=$RESUME_DIR"
run_models_with_ids
