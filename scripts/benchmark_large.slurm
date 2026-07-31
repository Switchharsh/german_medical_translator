#!/usr/bin/env bash
#SBATCH --job-name=medmt-large
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/medmt-large-%j.out
#SBATCH --error=logs/medmt-large-%j.err
#
# Tier 3 — large models (7B–27B, needs a full A100 80GB).
# Submit: sbatch scripts/benchmark_large.slurm
# Resume: RESUME_DIR=results/full_XXXX sbatch scripts/benchmark_large.slurm
#
# TranslateGemma-27B needs ~54GB bf16, only fits on a full A100 (80GB).
# Hy-MT2-7B needs ~14GB bf16, also fits here with generous batch sizes.
#
set -euo pipefail
export HF_HOME="$WORK/.cache/huggingface"
source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/_bench_common.sh"

MODELS="hymt2-7b translategemma-27b"
declare -A BATCH_SIZE=( [hymt2-7b]=4 [translategemma-27b]=2 )
declare -A NUM_BEAMS=(  [hymt2-7b]=1 [translategemma-27b]=1  )
declare -A MODEL_IDS=(
    [hymt2-7b]="tencent/Hy-MT2-7B"
    [translategemma-27b]="google/translategemma-27b-it"
)

run_models_with_ids
