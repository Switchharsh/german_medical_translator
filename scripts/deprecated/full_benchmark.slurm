#!/usr/bin/env bash
# DEPRECATED — use the tiered scripts instead for GPU-efficient scheduling:
#
#   # 1. Small models (identity, opus, nllb) — 20GB MIG slice, ~1h
#   sbatch scripts/benchmark_small.slurm
#
#   # 2. Medium models (hymt2-1.8B, translategemma-4B) — 20GB MIG slice, ~2h
#   sbatch scripts/benchmark_medium.slurm
#
#   # 3. Large models (hymt2-7B, translategemma-27B) — full A100 80GB, ~3h
#   sbatch scripts/benchmark_large.slurm
#
#   # 4. Assemble leaderboard (after all tiers finish)
#   RUN_DIR=results/full_XXXX sbatch scripts/assemble_leaderboard.slurm
#
# To resume a previous run (skip already-completed models):
#   RESUME_DIR=results/full_XXXX sbatch scripts/benchmark_small.slurm
#
# This script is kept for backward compatibility — it runs small + medium
# models on a single MIG slice (no large models).
#SBATCH --job-name=medmt-benchmark
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/medmt-benchmark-%j.out
#SBATCH --error=logs/medmt-benchmark-%j.err
set -euo pipefail

source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/lib/bench_common.sh"

MODELS="identity opus nllb hymt2 translategemma"
declare -A BATCH_SIZE=( [identity]=64 [opus]=64 [nllb]=16 [hymt2]=16 [translategemma]=8 )
declare -A NUM_BEAMS=(  [identity]=1  [opus]=4  [nllb]=4  [hymt2]=1  [translategemma]=1 )

run_models
