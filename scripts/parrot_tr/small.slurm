#!/usr/bin/env bash
#SBATCH --job-name=parrot-tr-small
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/parrot-tr-small-%j.out
#SBATCH --error=logs/parrot-tr-small-%j.err
#
# PARROT Turkish radiology reports (48 segments, TR->EN).
# See scripts/lib/parrot_tr_common.sh for the two caveats that travel with these
# results: reduced detector coverage (numbers only) and a small, skewed subset.
#
# Submit: sbatch scripts/benchmark_parrot_tr_small.slurm
# Resume: RESUME_DIR=results/parrot_tr_XXXX sbatch scripts/benchmark_parrot_tr_small.slurm
#
set -euo pipefail
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"

# Own results namespace, set BEFORE lib/bench_common.sh so its GPU logger writes
# to the right place.
export RESUME_DIR="${RESUME_DIR:-results/parrot_tr_${SLURM_JOB_ID}}"

source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/lib/bench_common.sh"
source "/home/atuin/b180dc/b180dc50/german_medical_translator/scripts/lib/parrot_tr_common.sh"

MODELS="identity opus nllb"
declare -A BATCH_SIZE=( [identity]=64 [opus]=8 [nllb]=4 )
declare -A NUM_BEAMS=(  [identity]=1  [opus]=4 [nllb]=4 )
# Both models are encoder-limited: Opus max_position_embeddings=512,
# NLLB=1024 (verified from the model configs). Turkish reports have a median
# length of ~2,080 chars (~693 tokens), so Opus truncates on INPUT and NLLB is
# marginal; chunk both on sentence boundaries and reassemble before scoring.
declare -A CHUNK_MAX_TOKENS=( [opus]=400 [nllb]=400 )
# See parrot_de/small.slurm: Opus's decoder position table is 512.
declare -A MODEL_MAX_NEW_TOKENS=( [opus]=480 )

run_models
