#!/usr/bin/env bash
#SBATCH --job-name=medmt-tgemma-27b
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --time=03:00:00
#SBATCH --output=logs/medmt-tgemma-27b-%j.out
#SBATCH --error=logs/medmt-tgemma-27b-%j.err
#
# TranslateGemma-27B (~54GB bf16 weights) — does not fit any a100mig MIG
# slice (max ~20GB). This cluster's a100 nodes are 40GB (SXM4-40GB), NOT
# 80GB — confirmed via a real OOM (job 3914949: "GPU 0 has a total capacity
# of 39.52 GiB"). Requests 2 GPUs; the translategemma adapter uses
# device_map="auto" to shard automatically across both when more than one
# GPU is visible (see models/translategemma_mt.py).
# Submit: sbatch scripts/benchmark_translategemma_27b.slurm
# Resume: RESUME_DIR=results/full_XXXX sbatch scripts/benchmark_translategemma_27b.slurm
#
set -euo pipefail

source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/_bench_common.sh"

MODELS="translategemma-27b"
declare -A MODEL_IDS=( [translategemma-27b]="google/translategemma-27b-it" )
declare -A BATCH_SIZE=( [translategemma-27b]=2 )
declare -A NUM_BEAMS=(  [translategemma-27b]=1 )

run_models_with_ids
