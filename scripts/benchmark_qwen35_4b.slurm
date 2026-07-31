#!/usr/bin/env bash
#SBATCH --job-name=medmt-qwen35-4b
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/medmt-qwen35-4b-%j.out
#SBATCH --error=logs/medmt-qwen35-4b-%j.err
#
# Qwen3.5-4B (~8GB bf16 weights) — general-purpose instruction-tuned LLM,
# used here as a non-specialized-translation baseline via the prompted-llm
# adapter (dispatch: MODELS name has no hymt2/translategemma prefix, so
# run_models_with_ids' case statement falls through to "prompted-llm").
# Fits the ~20GB a100med MIG slice.
#
# Submit: sbatch scripts/benchmark_qwen35_4b.slurm
# Resume: RESUME_DIR=results/full_XXXX sbatch scripts/benchmark_qwen35_4b.slurm
#
set -euo pipefail
export HF_HOME="$WORK/.cache/huggingface"
source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/_bench_common.sh"

MODELS="qwen35-4b"
declare -A MODEL_IDS=( [qwen35-4b]="Qwen/Qwen3.5-4B" )
declare -A BATCH_SIZE=( [qwen35-4b]=8 )
declare -A NUM_BEAMS=(  [qwen35-4b]=1 )

run_models_with_ids
