#!/usr/bin/env bash
#SBATCH --job-name=rt-large80
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/rt-%x-%j.out
#SBATCH --error=logs/rt-%x-%j.err
#
# Round-trip for the 27B/30B models on 80 GB A100s.
#
# The a100 partition is MIXED: 19 nodes carry the a100_40 feature and 18 carry
# a100_80. An unconstrained --gres=gpu:a100 request can land on either, and the
# earlier runs happened to get 40 GB cards — which is where the "these are all
# 40 GB" conclusion in the older comments came from. It was wrong.
# --constraint=a100_80 pins the job to the 80 GB nodes.
#
# ONE 80 GB card is enough for the 27B models: ~54 GB of bf16 weights leaves
# ~26 GB for activations, which supports a much larger batch than the 1 used
# when the weights had to be sharded across two 40 GB cards. A single-GPU
# request also schedules sooner than a two-GPU one on this busy partition.
#
# RT_BATCH defaults to 4 (4x the old setting); raise it if the GPU log in the
# results directory shows spare memory. The 30B MoE needs ~60 GB, leaving only
# ~20 GB, so give that one --gres=gpu:a100:2 and a smaller batch.
#
# Submit: MODEL=translategemma-27b sbatch --job-name=rt-translategemma-27b \
#           scripts/roundtrip/large80.slurm
set -euo pipefail
: "${MODEL:?Set MODEL, e.g. MODEL=translategemma-27b}"
export RT_BATCH="${RT_BATCH:-4}"
source /home/atuin/b180dc/b180dc50/german_medical_translator/scripts/roundtrip/_rt_common.sh
