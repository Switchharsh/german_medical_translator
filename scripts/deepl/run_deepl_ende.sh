#!/usr/bin/env bash
# DeepL EN->DE single pass on PARROT, on the LOGIN node (compute nodes have no internet).
# Writes into the shared EN->DE results directory. Refuses to start if the projected
# character spend exceeds 95% of the monthly quota still available.
#
#   RESUME_DIR=results/parrot_en_de_20261006 scripts/deepl/run_deepl_ende.sh
set -euo pipefail
cd /home/atuin/b180dc/b180dc50/german_medical_translator
source .venv/bin/activate
set -a; source /home/atuin/b180dc/b180dc50/.env; set +a
: "${DEEPL_API_KEY:?DEEPL_API_KEY missing from /home/atuin/b180dc/b180dc50/.env}"
: "${RESUME_DIR:?Set RESUME_DIR}"
INPUT=data/derived/parrot_en_de.jsonl
OUT="$RESUME_DIR/parrot_en_de_deepl.jsonl"
[ -f "$OUT" ] && { echo "=== already done: $OUT ==="; exit 0; }

python - "$INPUT" <<'PY'
import json, os, sys, requests
key = os.environ["DEEPL_API_KEY"]
host = "api-free.deepl.com" if key.endswith(":fx") else "api.deepl.com"
u = requests.get(f"https://{host}/v2/usage", headers={"Authorization": f"DeepL-Auth-Key {key}"}, timeout=30)
u.raise_for_status(); u = u.json(); left = u["character_limit"] - u["character_count"]
need = sum(len(json.loads(l)["src_text"]) for l in open(sys.argv[1]))
print(f"preflight: quota used {u['character_count']:,} / {u['character_limit']:,} -> {left:,} left")
print(f"preflight: EN->DE single pass needs {need:,} chars ({need/left*100:.0f}% of what is left)")
if need > left * 0.95:
    raise SystemExit(f"ERROR: {need:,} chars exceeds 95% of the {left:,} remaining. Not starting.")
PY
_t0=$(date +%s)
medmt-eval run --input "$INPUT" --model deepl --batch-size 8 --max-input-tokens 4096 \
  --term-bank data/term_banks/radiology_en_de_starter.csv \
  --output "$OUT" --summary "$RESUME_DIR/parrot_en_de_deepl.summary.json"
echo "wall_seconds=$(( $(date +%s) - _t0 ))"; echo "=== done: $OUT ==="
python - <<'PY'
import os, requests
key = os.environ["DEEPL_API_KEY"]; host = "api-free.deepl.com" if key.endswith(":fx") else "api.deepl.com"
u = requests.get(f"https://{host}/v2/usage", headers={"Authorization": f"DeepL-Auth-Key {key}"}, timeout=30).json()
print(f"final quota: {u['character_count']:,} / {u['character_limit']:,}")
PY
