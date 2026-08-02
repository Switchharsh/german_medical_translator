#!/usr/bin/env bash
# Quick smoke test for an interactive GPU session.
#
# Usage (from a login node):
#   salloc --gres=gpu:a40:1 --time=00:30:00 --partition=a40
#   # once the allocation drops you into a shell on the compute node:
#   bash scripts/quick_test_interactive.sh
#
# Runs only the small, fast Opus-MT checkpoint (~300MB) plus the identity
# baseline against the HimL2015 data, so it finishes in a couple of minutes
# and confirms the environment/GPU/pipeline all work before committing to
# the full NLLB/MADLAD run via sbatch.
set -euo pipefail
# HF_TOKEN must already be exported in your shell/profile (e.g. ~/.bashrc) —
# never hardcode a real token here, this script is committed to git.
: "${HF_TOKEN:?Set HF_TOKEN in your environment before running this script}"
export HF_HOME=$WORK/.cache/huggingface

module purge
module load python/3.12-base cuda/12.8.1

PROJECT=/home/atuin/b180dc/b180dc50/german_medical_translator
cd "$PROJECT"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -e '.[mt,storage,plot,deepl,dev]'

export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
mkdir -p "$HF_HOME"
# Disable the Xet storage backend — it often fails behind HPC firewalls/proxies.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

RUN_DIR="results/smoke_$(date +%Y%m%d_%H%M%S 2>/dev/null || echo run)"
mkdir -p "$RUN_DIR"

echo "=== GPU info ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
echo

# Background GPU usage logger — one line every 5s for the duration of this script.
GPU_LOG="$RUN_DIR/gpu_usage.csv"
echo "timestamp,index,name,mem_used_mib,mem_total_mib,util_pct" > "$GPU_LOG"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader >> "$GPU_LOG"
    sleep 5
  done
) &
GPU_LOG_PID=$!
trap 'kill "$GPU_LOG_PID" 2>/dev/null || true' EXIT

echo "=== Converting HimL2015 (if not already present) ==="
HIML_JSONL="data/derived/himl2015.jsonl"
mkdir -p data/derived
if [ ! -f "$HIML_JSONL" ]; then
  medmt-eval convert himl --input datasets/himl-test-2015.tgz --year 2015 \
    --output "$HIML_JSONL" --src-lang en --tgt-lang de
fi

echo "=== Running identity baseline ==="
medmt-eval run --input "$HIML_JSONL" --model identity \
  --output "$RUN_DIR/identity.jsonl" --summary "$RUN_DIR/identity.summary.json" \
  --term-bank data/term_banks/radiology_en_de_starter.csv

echo "=== Running Opus-MT (small, ~300MB checkpoint, GPU-accelerated) ==="
medmt-eval run --input "$HIML_JSONL" --model opus --device cuda \
  --output "$RUN_DIR/opus.jsonl" --summary "$RUN_DIR/opus.summary.json" \
  --term-bank data/term_banks/radiology_en_de_starter.csv

echo "=== Leaderboard ==="
cat "$RUN_DIR/identity.jsonl" "$RUN_DIR/opus.jsonl" > "$RUN_DIR/combined.jsonl"
medmt-eval leaderboard --input "$RUN_DIR/combined.jsonl" --output-dir "$RUN_DIR/leaderboard" --no-plot
column -s, -t "$RUN_DIR/leaderboard/leaderboard.csv"

echo
echo "GPU usage log written to $GPU_LOG"
echo "Peak memory used (MiB):"
awk -F',' 'NR>1 {gsub(/ MiB/,"",$4); if ($4+0>max) max=$4+0} END {print max}' "$GPU_LOG"
