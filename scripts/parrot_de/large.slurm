#!/usr/bin/env bash
#SBATCH --job-name=parrot-large
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --time=04:00:00
#SBATCH --output=logs/parrot-large-%j.out
#SBATCH --error=logs/parrot-large-%j.err
#
# PARROT German radiology reports (296 segments, DE->EN).
# Tier 3: the 27B/30B models.
#
# This cluster's a100 nodes are 40GB (SXM4-40GB), NOT 80GB — confirmed by real
# OOMs on jobs 3914947/3914949/3914951 ("total capacity of 39.52 GiB"). These
# models need ~54-60GB bf16, so 2 GPUs are requested and the adapters shard
# via device_map="auto" whenever more than one GPU is visible.
#
# Submit: sbatch scripts/benchmark_parrot_large.slurm
# Resume: RESUME_DIR=results/parrot_XXXX sbatch scripts/benchmark_parrot_large.slurm
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

MODELS="hymt2-30b-a3b translategemma-27b qwen35-27b"
declare -A MODEL_IDS=(
    [hymt2-30b-a3b]="tencent/Hy-MT2-30B-A3B"
    [translategemma-27b]="google/translategemma-27b-it"
    [qwen35-27b]="Qwen/Qwen3.5-27B"
)
declare -A BATCH_SIZE=( [hymt2-30b-a3b]=2 [translategemma-27b]=1 [qwen35-27b]=1 )
declare -A NUM_BEAMS=(  [hymt2-30b-a3b]=1 [translategemma-27b]=1 [qwen35-27b]=1 )

run_models_with_ids
