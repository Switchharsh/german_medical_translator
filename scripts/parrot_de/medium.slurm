#!/usr/bin/env bash
#SBATCH --job-name=parrot-medium
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --time=02:30:00
#SBATCH --output=logs/parrot-medium-%j.out
#SBATCH --error=logs/parrot-medium-%j.err
#
# PARROT German radiology reports (296 segments, DE->EN).
# Tier 2: 1.8B-4B models. All fit a ~20GB MIG slice.
#
# Radiology reports are longer than the HimL/EMEA sentences these batch sizes
# were tuned on (PARROT median ~588 chars vs single sentences), so batch sizes
# are reduced accordingly to stay within the slice.
#
# Submit: sbatch scripts/benchmark_parrot_medium.slurm
# Resume: RESUME_DIR=results/parrot_XXXX sbatch scripts/benchmark_parrot_medium.slurm
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

MODELS="hymt2-1.8b translategemma-4b qwen35-4b"
declare -A MODEL_IDS=(
    [hymt2-1.8b]="tencent/Hy-MT2-1.8B"
    [translategemma-4b]="google/translategemma-4b-it"
    [qwen35-4b]="Qwen/Qwen3.5-4B"
)
declare -A BATCH_SIZE=( [hymt2-1.8b]=8 [translategemma-4b]=4 [qwen35-4b]=4 )
declare -A NUM_BEAMS=(  [hymt2-1.8b]=1 [translategemma-4b]=1 [qwen35-4b]=1 )

run_models_with_ids
