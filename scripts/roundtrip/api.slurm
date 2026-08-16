#!/usr/bin/env bash
#SBATCH --job-name=rt-api
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100small:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/rt-%x-%j.out
#SBATCH --error=logs/rt-%x-%j.err
#
# Round-trip via the hosted API. No GPU is used — a slice is requested only
# because this cluster has no CPU-only partition.
# Verified-honest aliases only: glm-5.2, DeepSeek-V4-Flash, MiniMax-M3.
#
# RATE LIMITS. All three jobs share one API key. Running them at once with 8
# workers each put 24 concurrent requests on the gateway and two of the three
# died on HTTP 429 (jobs 4008963/4008965). The adapter now retries with
# backoff, but the cheaper fix is to not create the burst: submit these
# CHAINED, not in parallel, and keep the worker count modest.
#
# Submit (serial chain):
#   prev=""
#   for m in glm-5.2 DeepSeek-V4-Flash MiniMax-M3; do
#     prev=$(RESUME_DIR=$D MODEL=$m sbatch --parsable ${prev:+--dependency=afterany:$prev} \
#              --job-name=rt-$m scripts/roundtrip/api.slurm)
#   done
set -euo pipefail
: "${MODEL:?Set MODEL, e.g. MODEL=glm-5.2}"
: "${OPENAI_COMPAT_API_KEY:=${DEEPSEEK_API_KEY:-}}"
: "${OPENAI_COMPAT_API_KEY:?Export OPENAI_COMPAT_API_KEY before submitting}"
export OPENAI_COMPAT_API_KEY
export OPENAI_COMPAT_MAX_WORKERS="${OPENAI_COMPAT_MAX_WORKERS:-3}"
source /home/atuin/b180dc/b180dc50/german_medical_translator/scripts/roundtrip/_rt_common.sh
