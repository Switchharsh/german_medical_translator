#!/usr/bin/env bash
#SBATCH --job-name=medmt-medium
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/medmt-medium-%j.out
#SBATCH --error=logs/medmt-medium-%j.err
#
# Tier 2 — medium models (1.8B–4B, fits on a 20GB MIG slice).
# Submit: sbatch scripts/benchmark_medium.slurm
# Resume: RESUME_DIR=results/full_XXXX sbatch scripts/benchmark_medium.slurm
#
set -euo pipefail
export HF_HOME="$WORK/.cache/huggingface"
source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/lib/bench_common.sh"

MODELS="hymt2 translategemma"
declare -A BATCH_SIZE=( [hymt2]=16 [translategemma]=8 )
declare -A NUM_BEAMS=(  [hymt2]=1  [translategemma]=1  )

run_models
