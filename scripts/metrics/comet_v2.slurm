#!/usr/bin/env bash
#SBATCH --job-name=comet-v2
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/metrics-%x-%j.out
#SBATCH --error=logs/metrics-%x-%j.err
#
# wmt22-comet-da over a directory of per-system outputs. Uses the COMET venv (transformers 4.x).
#   RUN_DIR=results/parrot_de/consolidated_v2_20261007 OUT=results/comet_parrot_de_v2_20261007.json sbatch scripts/metrics/comet_v2.slurm
set -euo pipefail
: "${RUN_DIR:?}"; : "${OUT:?}"
GLOB="${GLOB:-parrot_de_*.jsonl}"; STRIP="${STRIP:-parrot_de_}"
module purge; module load python/3.12-base cuda/12.8.1
cd /home/atuin/b180dc/b180dc50/german_medical_translator
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}" HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
PY=.venv-comet/bin/python
CKPT_FILE=$(find "$HF_HOME/hub/models--Unbabel--wmt22-comet-da/snapshots" -name model.ckpt 2>/dev/null | head -1)
[ -n "$CKPT_FILE" ] || { echo "ERROR: no cached wmt22-comet-da checkpoint" >&2; exit 1; }
[ -e "$OUT" ] && { echo "ERROR: $OUT exists" >&2; exit 1; }
echo "preflight: $(ls $RUN_DIR/$GLOB | wc -l) files in $RUN_DIR | checkpoint $CKPT_FILE | out $OUT"
$PY scripts/score_comet.py "$RUN_DIR" -o "$OUT" --checkpoint "$CKPT_FILE" --glob "$GLOB" --strip "$STRIP" --local-files-only --gpus 1 --batch-size 16
