#!/usr/bin/env bash
#SBATCH --job-name=parrot-tr-medium
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/parrot-tr-medium-%j.out
#SBATCH --error=logs/parrot-tr-medium-%j.err
#
# PARROT Turkish radiology reports (48 segments, TR->EN).
# See scripts/_parrot_tr_common.sh for the two caveats that travel with these
# results: reduced detector coverage (numbers only) and a small, skewed subset.
#
# Submit: sbatch scripts/benchmark_parrot_tr_medium.slurm
# Resume: RESUME_DIR=results/parrot_tr_XXXX sbatch scripts/benchmark_parrot_tr_medium.slurm
#
set -euo pipefail
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"

# Own results namespace, set BEFORE _bench_common.sh so its GPU logger writes
# to the right place.
export RESUME_DIR="${RESUME_DIR:-results/parrot_tr_${SLURM_JOB_ID}}"

source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/_bench_common.sh"
source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/_parrot_tr_common.sh"

MODELS="hymt2-1.8b translategemma-4b qwen35-4b"
declare -A MODEL_IDS=(
    [hymt2-1.8b]="tencent/Hy-MT2-1.8B"
    [translategemma-4b]="google/translategemma-4b-it"
    [qwen35-4b]="Qwen/Qwen3.5-4B"
)
# Turkish reports are ~3.5x longer than the German ones, so batches are halved.
declare -A BATCH_SIZE=( [hymt2-1.8b]=4 [translategemma-4b]=2 [qwen35-4b]=2 )
declare -A NUM_BEAMS=(  [hymt2-1.8b]=1 [translategemma-4b]=1 [qwen35-4b]=1 )

run_models_with_ids
