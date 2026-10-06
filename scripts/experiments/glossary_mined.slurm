#!/usr/bin/env bash
#SBATCH --job-name=exp-gloss-mined
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/exp-%x-%j.out
#SBATCH --error=logs/exp-%x-%j.err
#
# Experiment 4 — a glossary mined from PARROT's own train split.
#
# Built by scripts/build_glossary_mined.py; see that file's docstring for the
# method and for why reference-based metrics are contaminated here.
#
# Two banks, each run against the standard three arms:
#
#   freq  87 pairs, top-N by frequency — "the top words from the corpus", the
#         straightforward reading of the idea. Mining precision ~92%, but the
#         baselines already produce 68% of these terms unprompted.
#   hard  28 pairs the baselines get right <= 60% of the time. The only subset
#         with room to move a score — and mining precision drops to ~39%,
#         because a term whose English form is context-dependent is exactly the
#         term a context-free glossary entry cannot express.
#
# Evaluated on the HELD-OUT half only (data/derived/parrot_de_val_v1.jsonl, 100
# documents). No document in the val half contributed a mined term.
#
# The terminology detector keeps its own external bank
# (radiology_en_de_starter.csv), disjoint from the injected glossary, so the
# terminology metric is not scored on the terms being injected.
#
# --branches "" is required: corpus-mined pairs carry no MeSH tree code, so any
# branch filter drops all of them.
#
# Submit:
#   BACKEND=local:Qwen/Qwen3.5-4B sbatch scripts/experiments/glossary_mined.slurm
set -euo pipefail
: "${BACKEND:?Set BACKEND, e.g. BACKEND=local:Qwen/Qwen3.5-4B}"

module purge
module load python/3.12-base cuda/12.8.1
PROJECT=/home/atuin/b180dc/b180dc50/german_medical_translator
cd "$PROJECT"
mkdir -p logs results/experiments
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TAG="${TAG:-v1}"
INPUT="data/derived/parrot_de_val_${TAG}.jsonl"
TERM_BANK="data/term_banks/radiology_en_de_starter.csv"
SAMPLE="${SAMPLE:-100}"
SEED="${SEED:-13}"
MAX_TERMS="${MAX_TERMS:-30}"
RUN_DIR="${RESUME_DIR:-results/experiments/${SLURM_JOB_ID:-local}}"
mkdir -p "$RUN_DIR"

# ── preflight ────────────────────────────────────────────────────────────
[ -f "$INPUT" ] || { echo "ERROR: $INPUT missing. Run scripts/build_glossary_mined.py --out-suffix $TAG" >&2; exit 1; }
[ -f "$TERM_BANK" ] || { echo "ERROR: term bank $TERM_BANK missing." >&2; exit 1; }

for BANK in freq hard; do
  G="data/term_banks/parrot_mined_${BANK}_${TAG}.csv"
  [ -f "$G" ] || { echo "ERROR: glossary $G missing." >&2; exit 1; }
done

python - "$INPUT" "$TAG" <<'PY'
import json, sys
from pathlib import Path
from medmt_eval.glossary import Glossary
corpus, tag = sys.argv[1], sys.argv[2]
val_ids = {json.loads(l)["id"] for l in Path(corpus).open()}
split = json.loads(Path(f"data/splits/parrot_de_mine_{tag}.json").read_text())
# The whole point of the split: a mined term must not come from a scored document.
leak = val_ids & set(split["train_ids"])
if leak:
    raise SystemExit(f"ERROR: {len(leak)} val documents are also in the mining half: {sorted(leak)[:5]}")
if val_ids != set(split["val_ids"]):
    raise SystemExit("ERROR: corpus does not match the split manifest's val_ids.")
print(f"preflight: {len(val_ids)} val docs, disjoint from {len(split['train_ids'])} mined docs")
for bank in ("freq", "hard"):
    g = Glossary.from_csv(f"data/term_banks/parrot_mined_{bank}_{tag}.csv").covering("de", "en")
    if len(g) < 20:
        raise SystemExit(f"ERROR: {bank} bank has only {len(g)} usable entries.")
    print(f"preflight: {bank} bank -> {len(g)} entries")
PY

echo "preflight: run dir $RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "preflight: no GPU visible"

# ── run ──────────────────────────────────────────────────────────────────
SLUG=$(echo "$BACKEND" | tr '/:' '__')
for BANK in freq hard; do
  OUT="$RUN_DIR/glossary_mined_${BANK}_${SLUG}.jsonl"
  if [ -f "$OUT" ]; then echo "=== $BANK already done, skipping ==="; continue; fi
  echo "=== mined glossary: bank=$BANK backend=$BACKEND ==="
  _t0=$(date +%s)
  medmt-eval glossary-run \
    --input "$INPUT" --src-lang de --tgt-lang en \
    --backend "$BACKEND" \
    --glossary "data/term_banks/parrot_mined_${BANK}_${TAG}.csv" \
    --branches "" \
    --arms none glossary distractor \
    --sample-size "$SAMPLE" --seed "$SEED" --max-terms "$MAX_TERMS" \
    --term-bank "$TERM_BANK" \
    --output "$OUT" --summary "$RUN_DIR/glossary_mined_${BANK}_${SLUG}.summary.json"
  echo "wall_seconds=$(( $(date +%s) - _t0 ))"
  echo "=== done: $OUT ==="
done
