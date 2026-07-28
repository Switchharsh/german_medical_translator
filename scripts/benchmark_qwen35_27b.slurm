#!/usr/bin/env bash
#SBATCH --job-name=medmt-qwen35-27b
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/medmt-qwen35-27b-%j.out
#SBATCH --error=logs/medmt-qwen35-27b-%j.err
#
# Qwen3.5-27B (~54GB bf16 weights) — general-purpose instruction-tuned LLM
# baseline via the prompted-llm adapter. Needs a full A100 (80GB); does not
# fit any a100mig MIG slice (max ~20GB).
#
# Submit: sbatch scripts/benchmark_qwen35_27b.slurm
# Resume: RESUME_DIR=results/full_XXXX sbatch scripts/benchmark_qwen35_27b.slurm
#
set -euo pipefail

source "$(dirname "$0")/_bench_common.sh"

MODELS="qwen35-27b"
declare -A MODEL_IDS=( [qwen35-27b]="Qwen/Qwen3.5-27B" )
declare -A BATCH_SIZE=( [qwen35-27b]=2 )
declare -A NUM_BEAMS=(  [qwen35-27b]=1 )

run_models_with_ids
