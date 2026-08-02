#!/usr/bin/env bash
#SBATCH --job-name=medmt-tgemma-4b
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/medmt-tgemma-4b-%j.out
#SBATCH --error=logs/medmt-tgemma-4b-%j.err
#
# TranslateGemma-4B (~8GB bf16 weights) — fits the ~20GB a100med MIG slice.
# Submit: sbatch scripts/benchmark_translategemma_4b.slurm
# Resume: RESUME_DIR=results/full_XXXX sbatch scripts/benchmark_translategemma_4b.slurm
#
set -euo pipefail

source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/lib/bench_common.sh"

MODELS="translategemma-4b"
declare -A MODEL_IDS=( [translategemma-4b]="google/translategemma-4b-it" )
declare -A BATCH_SIZE=( [translategemma-4b]=8 )
declare -A NUM_BEAMS=(  [translategemma-4b]=1 )

run_models_with_ids
