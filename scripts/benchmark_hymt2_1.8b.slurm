#!/usr/bin/env bash
#SBATCH --job-name=medmt-hymt2-1.8b
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100small:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/medmt-hymt2-1.8b-%j.out
#SBATCH --error=logs/medmt-hymt2-1.8b-%j.err
#
# Hy-MT2-1.8B (~3.6GB bf16 weights) — smallest MIG slice (a100small) is enough.
# Submit: sbatch scripts/benchmark_hymt2_1.8b.slurm
# Resume: RESUME_DIR=results/full_XXXX sbatch scripts/benchmark_hymt2_1.8b.slurm
#
set -euo pipefail
export HF_HOME="$WORK/.cache/huggingface"
source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/_bench_common.sh"

MODELS="hymt2-1.8b"
declare -A MODEL_IDS=( [hymt2-1.8b]="tencent/Hy-MT2-1.8B" )
declare -A BATCH_SIZE=( [hymt2-1.8b]=16 )
declare -A NUM_BEAMS=(  [hymt2-1.8b]=1 )

run_models_with_ids
