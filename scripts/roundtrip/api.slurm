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
# Submit: MODEL=glm-5.2 sbatch --job-name=rt-glm-5.2 scripts/roundtrip/api.slurm
set -euo pipefail
: "${MODEL:?Set MODEL, e.g. MODEL=glm-5.2}"
: "${OPENAI_COMPAT_API_KEY:=${DEEPSEEK_API_KEY:-}}"
: "${OPENAI_COMPAT_API_KEY:?Export OPENAI_COMPAT_API_KEY before submitting}"
export OPENAI_COMPAT_API_KEY
source /home/atuin/b180dc/b180dc50/german_medical_translator/scripts/roundtrip/_rt_common.sh
