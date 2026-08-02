#!/usr/bin/env bash
#SBATCH --job-name=medmt-small
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/medmt-small-%j.out
#SBATCH --error=logs/medmt-small-%j.err
#
# Tier 1 — small models (~300MB–5GB, fits on a 20GB MIG slice).
# Submit: sbatch scripts/benchmark_small.slurm
# Resume: RESUME_DIR=results/full_XXXX sbatch scripts/benchmark_small.slurm
#
set -euo pipefail

source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/lib/bench_common.sh"

MODELS="identity opus nllb"
declare -A BATCH_SIZE=( [identity]=64 [opus]=64 [nllb]=16 )
declare -A NUM_BEAMS=(  [identity]=1  [opus]=4  [nllb]=4  )

run_models
