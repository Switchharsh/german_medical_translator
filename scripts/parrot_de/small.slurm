#!/usr/bin/env bash
#SBATCH --job-name=parrot-small
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/parrot-small-%j.out
#SBATCH --error=logs/parrot-small-%j.err
#
# PARROT German radiology reports (296 segments, DE->EN).
# Tier 1: identity baseline + the small/generic MT models. All fit a ~20GB MIG slice.
#
# Submit: sbatch scripts/benchmark_parrot_small.slurm
# Resume: RESUME_DIR=results/parrot_XXXX sbatch scripts/benchmark_parrot_small.slurm
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

MODELS="identity opus nllb"
# Chunking raises the number of batches (296 docs -> 358 chunks), so memory
# pressure is higher than the unchunked run. Job 3942873 hit CUDA OOM on
# nllb at batch 8 x 4 beams on a 19.6 GB MIG slice; opus died with a
# device-side assert in the same job, which can follow a corrupted CUDA
# context. Reduced batches and beams to stay well inside the slice.
declare -A BATCH_SIZE=( [identity]=64 [opus]=8 [nllb]=2 )
declare -A NUM_BEAMS=(  [identity]=1  [opus]=1  [nllb]=1 )
# Both encoder-limited: Opus max_position_embeddings=512, NLLB=1024. The
# longest German report is 4,029 chars (~1,343 tokens), so both truncate it
# on INPUT regardless of --max-new-tokens. In the pre-fix run this showed as
# 21/33 (opus) and 28/33 (nllb) long documents coming back short, with
# error rates of 96.97 % and 90.91 % on those docs versus ~18-26 % on short
# ones. Chunk on sentence boundaries and reassemble before scoring.
declare -A CHUNK_MAX_TOKENS=( [opus]=400 [nllb]=400 )
# Marian/Opus has max_position_embeddings=512 on the DECODER as well, so a
# 2048 ceiling lets generation index off embed_positions. 480 leaves margin.
# Both are capped by their own position tables on the DECODER side, so the
# global 2048 ceiling must not be used for either.
declare -A MODEL_MAX_NEW_TOKENS=( [opus]=480 [nllb]=960 )

run_models
