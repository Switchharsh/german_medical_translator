#!/usr/bin/env bash
#SBATCH --job-name=medmt-assemble
#SBATCH --partition=cpu
#SBATCH --time=00:05:00
#SBATCH --output=logs/medmt-assemble-%j.out
#SBATCH --error=logs/medmt-assemble-%j.err
#
# Assemble results from all benchmark tiers into a single leaderboard.
#
# Usage:
#   RUN_DIR=results/full_XXXX sbatch scripts/assemble_leaderboard.slurm
#
# Or run interactively:
#   RUN_DIR=results/full_3906650 bash scripts/assemble_leaderboard.slurm
#
set -euo pipefail

PROJECT=/home/atuin/b180dc/b180dc50/german_medical_translator
cd "$PROJECT"
source .venv/bin/activate

RUN_DIR="${RUN_DIR:?Set RUN_DIR to the results directory}"
ALL_EVAL_FILES=()

for f in "$RUN_DIR"/*.jsonl; do
  [ -f "$f" ] || continue
  # Skip combined.jsonl from previous runs
  [[ "$f" == *combined* ]] && continue
  ALL_EVAL_FILES+=("$f")
done

if [ ${#ALL_EVAL_FILES[@]} -eq 0 ]; then
  echo "ERROR: No .jsonl files found in $RUN_DIR" >&2
  exit 1
fi

echo "=== Assembling ${#ALL_EVAL_FILES[@]} result files ==="
cat "${ALL_EVAL_FILES[@]}" > "$RUN_DIR/combined.jsonl"

medmt-eval leaderboard \
  --input "$RUN_DIR/combined.jsonl" \
  --output-dir "$RUN_DIR/leaderboard"

echo
echo "=== Leaderboard ==="
column -s, -t "$RUN_DIR/leaderboard/leaderboard.csv"
