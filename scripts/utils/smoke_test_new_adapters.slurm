#!/usr/bin/env bash
#SBATCH --job-name=medmt-smoke-new
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/medmt-smoke-new-%j.out
#SBATCH --error=logs/medmt-smoke-new-%j.err
#
# One real inference per new adapter (hymt2, translategemma, prompted-llm
# with a Qwen3.5 model) on a handful of segments, so any silent
# tokenizer/chat-template/prompt-format bug shows up as visibly garbled text
# BEFORE submitting the full per-model benchmark jobs. This is the same
# class of check that caught MADLAD producing repeated-glyph garbage output
# in the earlier generic-MT run — cheap to run, expensive to skip.
#
# Uses the smallest available checkpoint for each adapter family
# (Hy-MT2-1.8B, TranslateGemma-4B, Qwen3.5-4B) purely to keep this fast;
# it is NOT a substitute for running the actual per-model tier jobs.
#
# Submit: sbatch scripts/smoke_test_new_adapters.slurm
# Read output: logs/medmt-smoke-new-<jobid>.out

set -euo pipefail

module purge
module load python/3.12-base cuda/12.8.1

PROJECT=/home/atuin/b180dc/b180dc50/german_medical_translator
cd "$PROJECT"
mkdir -p logs

source .venv/bin/activate
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
mkdir -p "$HF_HOME"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

TMP_IN="$(mktemp --suffix=.jsonl)"
cat > "$TMP_IN" <<'EOF'
{"id":"smoke-1","src_lang":"en","tgt_lang":"de","src_text":"No pleural effusion.","ref_text":"Kein Pleuraerguss."}
{"id":"smoke-2","src_lang":"en","tgt_lang":"de","src_text":"A 5 mm nodule is present in the left upper lobe.","ref_text":"Im linken Oberlappen besteht ein 5-mm-Knoten."}
{"id":"smoke-3","src_lang":"en","tgt_lang":"de","src_text":"The patient denies chest pain.","ref_text":"Der Patient verneint Brustschmerzen."}
EOF

TMP_OUT="$(mktemp -d)"

run_one() {
  local label="$1" adapter="$2" model_id="$3"
  echo "=================================================================="
  echo "=== $label ($adapter, $model_id) ==="
  echo "=================================================================="
  medmt-eval translate \
    --input "$TMP_IN" --model "$adapter" --model-id "$model_id" --device cuda \
    --batch-size 1 --num-beams 1 --max-new-tokens 128 \
    --output "$TMP_OUT/$label.jsonl" \
    || { echo "!!! $label FAILED to run at all"; return; }
  python3 -c "
import json
for line in open('$TMP_OUT/$label.jsonl'):
    d = json.loads(line)
    print('SRC:', d['src_text'])
    print('REF:', d.get('ref_text', ''))
    print('HYP:', repr(d['hyp_text']))
    if '<think>' in d['hyp_text'].lower():
        print('!!! WARNING: leaked <think> tag survived adapter-level stripping !!!')
    if not d['hyp_text'].strip():
        print('!!! WARNING: empty hyp_text !!!')
    print()
"
}

run_one "hymt2-1.8b"        "hymt2"         "tencent/Hy-MT2-1.8B"
run_one "translategemma-4b" "translategemma" "google/translategemma-4b-it"
run_one "qwen35-4b"         "prompted-llm"  "Qwen/Qwen3.5-4B"

echo "=================================================================="
echo "=== Manually inspect the HYP lines above for garbage output ==="
echo "=== (repeated glyphs, wrong script, empty strings, <think> tags) ==="
echo "=================================================================="
