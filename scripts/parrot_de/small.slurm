#!/usr/bin/env bash
#SBATCH --job-name=parrot-small
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/parrot-small-%j.out
#SBATCH --error=logs/parrot-small-%j.err
#
# PARROT German radiology reports (296 segments, DE->EN).
# Tier 1: identity baseline + the small/generic MT models. All fit a ~20GB MIG slice.
#
# Submit: sbatch scripts/benchmark_parrot_small.slurm
# Resume: RESUME_DIR=results/parrot_XXXX sbatch scripts/benchmark_parrot_small.slurm
#
set -euo pipefail
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"

# Give PARROT its own results namespace BEFORE lib/bench_common.sh runs, so the
# GPU logger it launches writes into results/parrot_<jobid> rather than the
# results/full_<jobid> namespace used by the HimL/EMEA runs. RESUME_DIR is
# lib/bench_common.sh's existing hook for choosing RUN_DIR.
export RESUME_DIR="${RESUME_DIR:-results/parrot_${SLURM_JOB_ID}}"

source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/lib/bench_common.sh"
source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/lib/parrot_de_common.sh"

MODELS="identity opus nllb"
declare -A BATCH_SIZE=( [identity]=64 [opus]=16 [nllb]=8 )
declare -A NUM_BEAMS=(  [identity]=1  [opus]=4  [nllb]=4 )

run_models
