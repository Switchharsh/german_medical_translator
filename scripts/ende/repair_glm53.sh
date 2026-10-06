#!/usr/bin/env bash
# Re-translate the GLM-5.3 EN->DE reports whose answer was the string "None" (reasoning used the whole
# token allowance, content came back null) or was cut off. One report per request, cheapest first,
# the output ceiling escalating on failure. Paced under GLM-5.3's 3,000 output-tokens/minute cap.
#
#   scripts/ende/repair_glm53.sh            (resumes: finished reports are skipped)
set -uo pipefail
cd /home/atuin/b180dc/b180dc50/german_medical_translator
source .venv/bin/activate
RUN=results/parrot_en_de_20261006
W="$RUN/glm53_repair"; mkdir -p "$W"
export OPENAI_COMPAT_BASE_URL=https://llm.scads.ai/v1/chat/completions
export OPENAI_COMPAT_API_KEY=$(grep '^SCADS_API_KEY=' ../.env | cut -d= -f2- | tr -d '"'"'"' \r\n')
export OPENAI_COMPAT_MAX_WORKERS=1 OPENAI_COMPAT_TOKENS_PER_MIN=2700 OPENAI_COMPAT_EXTRA_BODY='{"thinking":{"type":"disabled"}}'

python3 - <<PY > "$W/todo.txt"
import json
rows = [json.loads(l) for l in open("$RUN/parrot_en_de_glm-5.3.jsonl")]
bad = [r for r in rows if r["hyp_text"].strip() in ("", "None") or len(r["hyp_text"]) < 0.6 * len(r["src_text"])]
for r in sorted(bad, key=lambda r: len(r["src_text"])):
    print(r["doc_id"])
PY
echo "to repair: $(wc -l < $W/todo.txt) reports"
while read -r doc; do
  out="$W/out_$doc.jsonl"; [ -s "$out" ] && { echo "$doc already done"; continue; }
  grep -F "\"doc_id\": \"$doc\"" data/derived/parrot_en_de.jsonl | head -1 > "$W/in_$doc.jsonl"
  for cap in 25000 40000 60000; do
    t0=$SECONDS
    medmt-eval run --input "$W/in_$doc.jsonl" --model openai-compat --model-id zai-org/GLM-5.3 --batch-size 1 --num-beams 1 \
      --max-new-tokens $cap --max-input-tokens 4096 --term-bank data/term_banks/radiology_en_de_starter.csv \
      --output "$out" --summary "$out.summary.json" > "$W/log_$doc.txt" 2>&1
    if [ -s "$out" ] && python3 - "$out" <<'PY'
import json, sys
r = json.loads(open(sys.argv[1]).readline())
ok = r["hyp_text"].strip() not in ("", "None") and len(r["hyp_text"]) >= 0.6 * len(r["src_text"])
sys.exit(0 if ok else 1)
PY
    then echo "$doc ok in $((SECONDS-t0))s (ceiling $cap)"; break; fi
    echo "$doc ceiling $cap not enough after $((SECONDS-t0))s: $(grep -a -v -i 'warn\|deprecat' $W/log_$doc.txt | tail -1 | cut -c1-120)"; rm -f "$out"
  done
done < "$W/todo.txt"
echo "ALL DONE"
