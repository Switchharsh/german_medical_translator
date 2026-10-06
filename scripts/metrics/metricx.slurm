#!/usr/bin/env bash
#SBATCH --job-name=metricx
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/metrics-%x-%j.out
#SBATCH --error=logs/metrics-%x-%j.err
#
# MetricX-24 (hybrid, large) over metric-input tables, reference-based and QE.
# Compute nodes have no internet, so the model and tokenizer must be pre-fetched
# (scripts/fetch_metric_models.py). Runs from the COMET venv (transformers 4.x).
#
#   IN_DIR=results/metric_inputs_X OUT_DIR=results/metricx_X SETS="de_en_single round_trip perturb" \
#       sbatch scripts/metrics/metricx.slurm
set -euo pipefail
: "${IN_DIR:?Set IN_DIR}"; : "${OUT_DIR:?Set OUT_DIR}"; : "${SETS:?Set SETS, e.g. \"de_en_single round_trip\"}"
module purge; module load python/3.12-base cuda/12.8.1
cd /home/atuin/b180dc/b180dc50/german_medical_translator
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}" HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
PY=.venv-comet/bin/python
mkdir -p "$OUT_DIR" logs

# ── preflight ───────────────────────────────────────────────────────────────
$PY - <<'PY'
import os, torch
from huggingface_hub import snapshot_download
for repo in ("google/metricx-24-hybrid-large-v2p6", "google/mt5-large"):
    p = snapshot_download(repo, local_files_only=True)
    print(f"preflight: {repo} cached at {p}")
print(f"preflight: cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: no GPU visible; refusing to run MetricX on CPU")
PY
for s in $SETS; do [ -f "$IN_DIR/$s.jsonl" ] || { echo "ERROR: $IN_DIR/$s.jsonl missing" >&2; exit 1; }; done
echo "preflight: sets [$SETS] from $IN_DIR -> $OUT_DIR"

for s in $SETS; do
  out="$OUT_DIR/$s.jsonl"
  [ -f "$out" ] && { echo "=== $s already done, skipping ==="; continue; }
  echo "=== MetricX-24: $s ==="; _t0=$(date +%s)
  $PY scripts/score_metricx.py "$IN_DIR/$s.jsonl" -o "$out" --model large --batch-size 32 --local-files-only
  echo "wall_seconds=$(( $(date +%s) - _t0 ))"
done
echo "=== all done ==="
