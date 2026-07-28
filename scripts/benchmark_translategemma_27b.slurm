#!/usr/bin/env bash
#SBATCH --job-name=medmt-tgemma-27b
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/medmt-tgemma-27b-%j.out
#SBATCH --error=logs/medmt-tgemma-27b-%j.err
#
# TranslateGemma-27B (~54GB bf16 weights) — needs a full A100 (80GB); does
# not fit any a100mig MIG slice (max ~20GB).
# Submit: sbatch scripts/benchmark_translategemma_27b.slurm
# Resume: RESUME_DIR=results/full_XXXX sbatch scripts/benchmark_translategemma_27b.slurm
#
set -euo pipefail

source "$(dirname "$0")/_bench_common.sh"

MODELS="translategemma-27b"
declare -A MODEL_IDS=( [translategemma-27b]="google/translategemma-27b-it" )
declare -A BATCH_SIZE=( [translategemma-27b]=2 )
declare -A NUM_BEAMS=(  [translategemma-27b]=1 )

run_models_with_ids
