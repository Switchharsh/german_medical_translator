#!/usr/bin/env bash
#SBATCH --job-name=medmt-hymt2-30b-moe
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/medmt-hymt2-30b-moe-%j.out
#SBATCH --error=logs/medmt-hymt2-30b-moe-%j.err
#
# Hy-MT2-30B-A3B — a Mixture-of-Experts model: only ~3B params are active per
# token (fast compute, comparable to a 3B dense model), but ALL ~30B expert
# weights must be resident in VRAM regardless of how many are active per
# forward pass (~60GB in bf16). This is why it gets its own tier rather than
# being lumped in with the dense 7B/27B "large" models: its compute profile
# is closer to a 3B model, but its memory floor is the largest of any model
# in this benchmark. A single MIG slice (max ~20GB) cannot hold it; needs a
# full A100 80GB.
#
# Submit: sbatch scripts/benchmark_hymt2_30b_moe.slurm
# Resume: RESUME_DIR=results/full_XXXX sbatch scripts/benchmark_hymt2_30b_moe.slurm
#
set -euo pipefail

source "$(dirname "$0")/_bench_common.sh"

MODELS="hymt2-30b-a3b"
declare -A MODEL_IDS=( [hymt2-30b-a3b]="tencent/Hy-MT2-30B-A3B" )
# Small batch size despite the low active-param compute cost — conservative
# until a real run confirms actual peak VRAM with MoE routing overhead.
declare -A BATCH_SIZE=( [hymt2-30b-a3b]=4 )
declare -A NUM_BEAMS=(  [hymt2-30b-a3b]=1 )

run_models_with_ids
