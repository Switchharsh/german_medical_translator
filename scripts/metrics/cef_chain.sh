#!/usr/bin/env bash
# Run CEF over several metric-input sets, one after another, on the login node (the judge is a
# vLLM server reachable over HTTP; the work here is just HTTP calls). Skips sets whose output exists.
#
#   WAIT_PID=<pid of a CEF run to wait for>  IN_DIR=... OUT_DIR=... SETS="de_en_single round_trip" \
#       nohup scripts/metrics/cef_chain.sh > logs/cef-chain.log 2>&1 &
set -uo pipefail
cd /home/atuin/b180dc/b180dc50/german_medical_translator
source .venv/bin/activate
: "${IN_DIR:?}"; : "${OUT_DIR:?}"; : "${SETS:?}"
if [ -n "${WAIT_PID:-}" ]; then
  echo "waiting for pid $WAIT_PID"; while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi
for s in $SETS; do
  out="$OUT_DIR/$s.jsonl"
  [ -f "$out" ] && { echo "=== $s already done ==="; continue; }
  # the judge must be reachable before we start burning a set on it
  python - <<'PY' || { echo "ERROR: judge endpoint down, stopping before $s"; exit 1; }
import requests
env = dict(l.strip().split("=", 1) for l in open("../vllm_serve/endpoint.env") if "=" in l and not l.startswith("#"))
requests.get(env["OPENAI_BASE_URL"] + "/models", headers={"Authorization": "Bearer " + env["OPENAI_API_KEY"]}, timeout=20).raise_for_status()
PY
  echo "=== CEF: $s ($(date +%T)) ==="
  python scripts/score_cef.py "$IN_DIR/$s.jsonl" -o "$out" --workers 96
  echo "=== finished $s ($(date +%T)) ==="
done
echo "=== chain done ==="
