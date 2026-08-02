#!/usr/bin/env bash
#SBATCH --job-name=parrot-tr-large
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --time=03:00:00
#SBATCH --output=logs/parrot-tr-large-%j.out
#SBATCH --error=logs/parrot-tr-large-%j.err
#
# PARROT Turkish radiology reports (48 segments, TR->EN).
# See scripts/_parrot_tr_common.sh for the two caveats that travel with these
# results: reduced detector coverage (numbers only) and a small, skewed subset.
#
# Submit: sbatch scripts/benchmark_parrot_tr_large.slurm
# Resume: RESUME_DIR=results/parrot_tr_XXXX sbatch scripts/benchmark_parrot_tr_large.slurm
#
set -euo pipefail
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"

# Own results namespace, set BEFORE _bench_common.sh so its GPU logger writes
# to the right place.
export RESUME_DIR="${RESUME_DIR:-results/parrot_tr_${SLURM_JOB_ID}}"

source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/_bench_common.sh"
source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/_parrot_tr_common.sh"

MODELS="hymt2-30b-a3b translategemma-27b qwen35-27b"
declare -A MODEL_IDS=(
    [hymt2-30b-a3b]="tencent/Hy-MT2-30B-A3B"
    [translategemma-27b]="google/translategemma-27b-it"
    [qwen35-27b]="Qwen/Qwen3.5-27B"
)
declare -A BATCH_SIZE=( [hymt2-30b-a3b]=1 [translategemma-27b]=1 [qwen35-27b]=1 )
declare -A NUM_BEAMS=(  [hymt2-30b-a3b]=1 [translategemma-27b]=1 [qwen35-27b]=1 )

run_models_with_ids
