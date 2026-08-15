#!/usr/bin/env bash
#SBATCH --job-name=rt-mig
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --time=02:30:00
#SBATCH --output=logs/rt-%x-%j.out
#SBATCH --error=logs/rt-%x-%j.err
#
# Round-trip for models that fit a ~20 GB MIG slice:
#   identity opus nllb hymt2-1.8b hymt2-7b translategemma-4b qwen35-4b
#
# Submit: MODEL=hymt2-7b sbatch --job-name=rt-hymt2-7b scripts/roundtrip/mig.slurm
set -euo pipefail
: "${MODEL:?Set MODEL, e.g. MODEL=opus}"
source /home/atuin/b180dc/b180dc50/german_medical_translator/scripts/roundtrip/_rt_common.sh
