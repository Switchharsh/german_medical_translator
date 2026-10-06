#!/usr/bin/env bash
#SBATCH --job-name=xcomet
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/metrics-%x-%j.out
#SBATCH --error=logs/metrics-%x-%j.err
#
# xCOMET-XL over metric-input tables (aligned segments, with error spans), reference-based and QE.
# Compute nodes have no internet: the model (Unbabel/XCOMET-XL, gated, CC-BY-NC-SA-4.0) and the
# XLM-R-XL tokenizer must already be in the HF cache (scripts/fetch_metric_models.py).
#
#   IN_DIR=results/metric_inputs_X OUT_DIR=results/xcomet_X SETS="perturb de_en_single" \
#       sbatch scripts/metrics/xcomet.slurm
set -euo pipefail
: "${IN_DIR:?Set IN_DIR}"; : "${OUT_DIR:?Set OUT_DIR}"; : "${SETS:?Set SETS}"
module purge; module load python/3.12-base cuda/12.8.1
cd /home/atuin/b180dc/b180dc50/german_medical_translator
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}" HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
PY=.venv-comet/bin/python
mkdir -p "$OUT_DIR" logs

$PY - <<'PY'
import torch
from pathlib import Path
from huggingface_hub import snapshot_download
d = snapshot_download("Unbabel/XCOMET-XL", local_files_only=True)
ck = next(Path(d).rglob("model.ckpt"), None)
assert ck is not None, "no model.ckpt in the cached XCOMET-XL snapshot"
snapshot_download("facebook/xlm-roberta-xl", local_files_only=True)
print(f"preflight: XCOMET-XL checkpoint {ck.resolve()} ({ck.resolve().stat().st_size/1e9:.1f} GB)")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: no GPU visible")
print(f"preflight: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
PY
for s in $SETS; do [ -f "$IN_DIR/$s.jsonl" ] || { echo "ERROR: $IN_DIR/$s.jsonl missing" >&2; exit 1; }; done
echo "preflight: sets [$SETS] from $IN_DIR -> $OUT_DIR"

for s in $SETS; do
  out="$OUT_DIR/$s.jsonl"
  [ -f "$out" ] && { echo "=== $s already done, skipping ==="; continue; }
  echo "=== xCOMET-XL: $s ==="; _t0=$(date +%s)
  $PY scripts/score_xcomet.py "$IN_DIR/$s.jsonl" -o "$out" --batch-size 16 --local-files-only
  echo "wall_seconds=$(( $(date +%s) - _t0 ))"
done
echo "=== all done ==="
