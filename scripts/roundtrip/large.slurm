#!/usr/bin/env bash
#SBATCH --job-name=rt-large
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --time=04:00:00
#SBATCH --output=logs/rt-%x-%j.out
#SBATCH --error=logs/rt-%x-%j.err
#
# Round-trip for the 27B/30B models. This cluster's a100 nodes are 40 GB, so
# these need 2 GPUs and shard via device_map="auto".
#   hymt2-30b-a3b translategemma-27b qwen35-27b
#
# One model per job: at 20 passes the slowest (translategemma-27b, ~27.9 s/doc)
# needs ~3.1 h on its own, so three in one job would not fit the wall.
#
# Submit: MODEL=qwen35-27b sbatch --job-name=rt-qwen35-27b scripts/roundtrip/large.slurm
set -euo pipefail
: "${MODEL:?Set MODEL, e.g. MODEL=qwen35-27b}"
source /home/atuin/b180dc/b180dc50/german_medical_translator/scripts/roundtrip/_rt_common.sh
