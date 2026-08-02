#!/usr/bin/env bash
#SBATCH --job-name=medmt-hymt2-7b
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/medmt-hymt2-7b-%j.out
#SBATCH --error=logs/medmt-hymt2-7b-%j.err
#
# Hy-MT2-7B (~14GB bf16 weights) — fits the ~20GB a100med MIG slice.
# Submit: sbatch scripts/benchmark_hymt2_7b.slurm
# Resume: RESUME_DIR=results/full_XXXX sbatch scripts/benchmark_hymt2_7b.slurm
#
set -euo pipefail
export HF_HOME="$WORK/.cache/huggingface"
source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/lib/bench_common.sh"

MODELS="hymt2-7b"
declare -A MODEL_IDS=( [hymt2-7b]="tencent/Hy-MT2-7B" )
declare -A BATCH_SIZE=( [hymt2-7b]=8 )
declare -A NUM_BEAMS=(  [hymt2-7b]=1 )

run_models_with_ids
