#!/usr/bin/env bash
# Shared setup for round-trip runs. Sourced, not executed.
#
# Expects MODEL to be set. Resolves the model id, per-model ceilings and
# chunking, then runs `medmt-eval roundtrip`.
#
# SAMPLE SIZE. 10 cycles is 20 translation passes over the corpus. On all 296
# PARROT German reports that is ~168 GPU-hours across the nine models, far past
# the 4 h wall. RT_SAMPLE (default 20) takes a deterministic, length-stratified
# subsample so the slowest model (translategemma-27b, 27.9 s/doc) finishes in
# ~3.1 h. Every model uses the SAME sample so the curves are comparable.

set -euo pipefail

module purge
module load python/3.12-base cuda/12.8.1

PROJECT=/home/atuin/b180dc/b180dc50/german_medical_translator
cd "$PROJECT"
mkdir -p logs data/derived
source .venv/bin/activate

export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RT_CYCLES="${RT_CYCLES:-10}"
RT_SAMPLE="${RT_SAMPLE:-20}"
RT_SEED="${RT_SEED:-13}"
RUN_DIR="${RESUME_DIR:-results/roundtrip_${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"

INPUT="${RT_INPUT:-data/derived/parrot_de.jsonl}"
[ -f "$INPUT" ] || medmt-eval convert parrot \
  --input PARROT_v1.0/data/PARROT_v1_0.jsonl --output "$INPUT" --src-lang de --tgt-lang en

# adapter | model-id | batch | max-new | chunk (0 = off)
case "$MODEL" in
  identity)           A=identity;       ID="";                              B=32; MNT=2048; CH=0   ;;
  # Encoder position limits (Opus 512, NLLB 1024) truncate long reports on
  # INPUT, so both are chunked; Opus's decoder table is 512 too, hence 480.
  opus)               A=opus;           ID="";                              B=8;  MNT=480;  CH=400 ;;
  nllb)               A=nllb;           ID="";                              B=2;  MNT=960;  CH=400 ;;
  hymt2-1.8b)         A=hymt2;          ID=tencent/Hy-MT2-1.8B;             B=4;  MNT=2048; CH=0   ;;
  hymt2-7b)           A=hymt2;          ID=tencent/Hy-MT2-7B;               B=2;  MNT=2048; CH=0   ;;
  hymt2-30b-a3b)      A=hymt2;          ID=tencent/Hy-MT2-30B-A3B;          B=1;  MNT=2048; CH=0   ;;
  translategemma-4b)  A=translategemma; ID=google/translategemma-4b-it;     B=2;  MNT=2048; CH=0   ;;
  translategemma-27b) A=translategemma; ID=google/translategemma-27b-it;    B=1;  MNT=2048; CH=0   ;;
  qwen35-4b)          A=prompted-llm;   ID=Qwen/Qwen3.5-4B;                 B=2;  MNT=2048; CH=0   ;;
  qwen35-27b)         A=prompted-llm;   ID=Qwen/Qwen3.5-27B;                B=1;  MNT=2048; CH=0   ;;
  glm-5.2|DeepSeek-V4-Flash|MiniMax-M3)
                      A=openai-compat;  ID="$MODEL";                        B=4;  MNT=2048; CH=0   ;;
  *) echo "ERROR: unknown MODEL '$MODEL'" >&2; exit 1 ;;
esac

# Allow the job wrapper to override the per-model batch size, e.g. when running
# on an 80 GB card instead of a 40 GB one.
B="${RT_BATCH:-$B}"

# TranslateGemma repos are gated: without a token the run dies mid-job with an
# opaque HTTP 401. Fail here instead. SLURM inherits the submitting shell's
# environment, so HF_TOKEN must be exported there (job 4006927 failed exactly
# this way when submitted from a shell that did not have it).
case "$MODEL" in
  translategemma-*)
    : "${HF_TOKEN:?$MODEL is a gated repo — export HF_TOKEN before submitting}" ;;
esac

OUT="$RUN_DIR/rt_${MODEL}.jsonl"
if [ -f "$OUT" ]; then echo "=== $MODEL already done, skipping ==="; exit 0; fi

echo "=== round-trip: $MODEL ==="
echo "    adapter=$A model_id=${ID:-<default>}"
echo "    cycles=$RT_CYCLES (= $((RT_CYCLES*2)) passes)  sample=$RT_SAMPLE  seed=$RT_SEED"
echo "    batch=$B max_new=$MNT chunk=$CH"
echo "    results -> $RUN_DIR"

_t0=$(date +%s)
medmt-eval roundtrip \
  --input "$INPUT" --model "$A" ${ID:+--model-id "$ID"} --device cuda \
  --cycles "$RT_CYCLES" --sample-size "$RT_SAMPLE" --seed "$RT_SEED" \
  --batch-size "$B" --num-beams 1 \
  --max-new-tokens "$MNT" --max-input-tokens 4096 --chunk-max-tokens "$CH" \
  --term-bank data/term_banks/radiology_en_de_starter.csv \
  --output "$OUT" --summary "$RUN_DIR/rt_${MODEL}.summary.json" \
  2>> "$RUN_DIR/rt_${MODEL}.time.log"
_t1=$(date +%s)
echo "wall_seconds=$(( _t1 - _t0 ))" | tee -a "$RUN_DIR/rt_${MODEL}.time.log"
echo "=== done: $OUT ==="
