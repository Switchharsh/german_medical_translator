#!/usr/bin/env bash
# DeepL benchmark on PARROT-DE, run on the LOGIN node.
#
# Compute nodes have no outbound network, so this cannot be an sbatch job. It is
# I/O-bound (an HTTP call per batch); the local work is sacreBLEU + rule-based
# detectors over a few hundred documents.
#
# Runs, in order, the same two things every other system got:
#   1. single pass over all 296 reports          -> parrot_de_deepl.jsonl
#   2. 10-cycle round trip, 20-report sample     -> rt_deepl.jsonl
#
# Preflight refuses to start if the projected character spend exceeds the quota
# still available on the DeepL key. Outputs go to a NEW timestamped directory.
#
#   scripts/deepl/run_deepl.sh
set -euo pipefail

PROJECT=/home/atuin/b180dc/b180dc50/german_medical_translator
cd "$PROJECT"
source .venv/bin/activate
set -a; source /home/atuin/b180dc/b180dc50/.env; set +a
: "${DEEPL_API_KEY:?DEEPL_API_KEY missing from /home/atuin/b180dc/b180dc50/.env}"

INPUT=data/derived/parrot_de.jsonl
TERM_BANK=data/term_banks/radiology_en_de_starter.csv
RT_CYCLES="${RT_CYCLES:-10}"; RT_SAMPLE="${RT_SAMPLE:-20}"; RT_SEED="${RT_SEED:-13}"
RUN_DIR="${RESUME_DIR:-results/deepl_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

# ── preflight ───────────────────────────────────────────────────────────────
python - "$INPUT" "$RT_CYCLES" "$RT_SAMPLE" "$RUN_DIR" <<'PY'
import json, os, sys, requests
inp, cycles, sample, run_dir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
key = os.environ["DEEPL_API_KEY"]
host = "api-free.deepl.com" if key.endswith(":fx") else "api.deepl.com"
u = requests.get(f"https://{host}/v2/usage", headers={"Authorization": f"DeepL-Auth-Key {key}"}, timeout=30)
u.raise_for_status(); u = u.json()
left = u["character_limit"] - u["character_count"]

lens = sorted(len(json.loads(l)["src_text"]) for l in open(inp))
single = sum(lens)
# Round trip: `sample` docs, 2*cycles passes, each pass about the sample's source
# length (translations run a little longer than German; +10% margin). The sample
# is length-stratified, so its mean tracks the corpus mean.
rt = int(sample * (sum(lens) / len(lens)) * 2 * cycles * 1.10)
need = single + rt
print(f"preflight: host {host}")
print(f"preflight: quota used {u['character_count']:,} / {u['character_limit']:,} -> {left:,} left")
print(f"preflight: single pass {single:,} chars; round trip ~{rt:,} chars (estimate); total ~{need:,}")
print(f"preflight: projected use {need/left*100:.0f}% of what is left; run dir {run_dir}")
if need > left * 0.95:
    raise SystemExit(f"ERROR: projected {need:,} chars exceeds 95% of the {left:,} remaining. Not starting.")
PY

# ── 1. single pass ──────────────────────────────────────────────────────────
OUT1="$RUN_DIR/parrot_de_deepl.jsonl"
if [ -f "$OUT1" ]; then echo "=== single pass already done, skipping ==="; else
  echo "=== single pass: deepl, 296 reports ==="; _t0=$(date +%s)
  medmt-eval run --input "$INPUT" --model deepl --batch-size 8 \
    --max-input-tokens 4096 --term-bank "$TERM_BANK" \
    --output "$OUT1" --summary "$RUN_DIR/parrot_de_deepl.summary.json"
  echo "wall_seconds=$(( $(date +%s) - _t0 ))"; echo "=== done: $OUT1 ==="
fi

# ── 2. round trip ───────────────────────────────────────────────────────────
OUT2="$RUN_DIR/rt_deepl.jsonl"
if [ -f "$OUT2" ]; then echo "=== round trip already done, skipping ==="; else
  echo "=== round trip: deepl, $RT_CYCLES cycles, sample $RT_SAMPLE ==="; _t0=$(date +%s)
  medmt-eval roundtrip --input "$INPUT" --model deepl --batch-size 8 \
    --cycles "$RT_CYCLES" --sample-size "$RT_SAMPLE" --seed "$RT_SEED" \
    --max-input-tokens 4096 --term-bank "$TERM_BANK" \
    --output "$OUT2" --summary "$RUN_DIR/rt_deepl.summary.json"
  echo "wall_seconds=$(( $(date +%s) - _t0 ))"; echo "=== done: $OUT2 ==="
fi

python - <<'PY'
import os, requests
key = os.environ["DEEPL_API_KEY"]
host = "api-free.deepl.com" if key.endswith(":fx") else "api.deepl.com"
u = requests.get(f"https://{host}/v2/usage", headers={"Authorization": f"DeepL-Auth-Key {key}"}, timeout=30).json()
print(f"final quota: {u['character_count']:,} / {u['character_limit']:,} characters used")
PY
