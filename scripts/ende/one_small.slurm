#!/usr/bin/env bash
#SBATCH --job-name=ende-small1
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/ende-%x-%j.out
#SBATCH --error=logs/ende-%x-%j.err
#
# PARROT EN->DE single pass for ONE 4B-class model on a MIG slice. The three-model job
# ende-medium timed out at 2h30 (hymt2-1.8b alone took 71 min on a slice), so each model now
# gets its own job and its own 4 h.
#
#   MODEL=qwen35-4b RESUME_DIR=results/parrot_en_de_20261006 sbatch --job-name=ende-qwen4 scripts/ende/one_small.slurm
set -euo pipefail
: "${MODEL:?Set MODEL: translategemma-4b | qwen35-4b | hymt2-1.8b}"
: "${RESUME_DIR:?Set RESUME_DIR}"
case "$MODEL" in translategemma-*) : "${HF_TOKEN:?$MODEL is gated: export HF_TOKEN before submitting}" ;; esac
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}" RESUME_DIR
export PARROT_JSONL="${PARROT_JSONL:-data/derived/parrot_en_de.jsonl}"
source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/lib/bench_common.sh"
source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/lib/parrot_de_common.sh"
MODELS="$MODEL"
declare -A MODEL_IDS=( [hymt2-1.8b]="tencent/Hy-MT2-1.8B" [translategemma-4b]="google/translategemma-4b-it" [qwen35-4b]="Qwen/Qwen3.5-4B" )
declare -A BATCH_SIZE=( [hymt2-1.8b]=8 [translategemma-4b]=4 [qwen35-4b]=4 )
declare -A NUM_BEAMS=(  [hymt2-1.8b]=1 [translategemma-4b]=1 [qwen35-4b]=1 )
echo "preflight: model=$MODEL id=${MODEL_IDS[$MODEL]} batch=${BATCH_SIZE[$MODEL]} dataset=$PARROT_JSONL dir=$RESUME_DIR"
run_models_with_ids
