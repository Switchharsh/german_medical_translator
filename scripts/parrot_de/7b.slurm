#!/usr/bin/env bash
#SBATCH --job-name=parrot-7b
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --time=02:30:00
#SBATCH --output=logs/parrot-7b-%j.out
#SBATCH --error=logs/parrot-7b-%j.err
#
# PARROT German radiology reports (296 segments, DE->EN).
# Hy-MT2-7B (~14GB bf16) fits the ~20GB a100med MIG slice.
#
# Submit: sbatch scripts/benchmark_parrot_7b.slurm
# Resume: RESUME_DIR=results/parrot_XXXX sbatch scripts/benchmark_parrot_7b.slurm
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

MODELS="hymt2-7b"
declare -A MODEL_IDS=( [hymt2-7b]="tencent/Hy-MT2-7B" )
declare -A BATCH_SIZE=( [hymt2-7b]=4 )
declare -A NUM_BEAMS=(  [hymt2-7b]=1 )

run_models_with_ids
