#!/usr/bin/env bash
# Shared setup for all benchmark tier scripts. Source this, don't execute.
#
# Expects SLURM_JOB_ID, RESUME_DIR (optional), MODELS, BATCH_SIZE, NUM_BEAMS
# to be set by the caller.  Provides run_models() and run_models_with_ids().

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Resume from a previous run directory or create a new one.
RUN_DIR="${RESUME_DIR:-results/full_${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# Continuous GPU usage logger.
GPU_LOG="$RUN_DIR/gpu_usage_${SLURM_JOB_ID}.csv"
echo "timestamp,index,name,mem_used_mib,mem_total_mib,util_pct,power_draw_w" > "$GPU_LOG"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader >> "$GPU_LOG"
    sleep 10
  done
) &
GPU_LOG_PID=$!
trap 'kill "$GPU_LOG_PID" 2>/dev/null || true' EXIT

# --- Data prep (idempotent) ---
mkdir -p data/derived
[ -f data/derived/himl2015.jsonl ] || medmt-eval convert himl \
  --input datasets/himl-test-2015.tgz --year 2015 --output data/derived/himl2015.jsonl \
  --src-lang en --tgt-lang de
[ -f data/derived/himl2017.jsonl ] || medmt-eval convert himl \
  --input datasets/himl-test-2017.tgz --year 2017 --output data/derived/himl2017.jsonl \
  --src-lang en --tgt-lang de
[ -f data/derived/emea.jsonl ] || medmt-eval convert emea \
  --input datasets/emea-de-en.tmx.gz --output data/derived/emea.jsonl \
  --src-lang de --tgt-lang en --sample-size 400

DATASETS="data/derived/himl2015.jsonl data/derived/himl2017.jsonl data/derived/emea.jsonl"


# --- run_models: uses --model <adapter_name> (default model_id per adapter) ---
run_models() {
    local ALL_EVAL_FILES=()
    for dataset in $DATASETS; do
      local tag
      tag="$(basename "$dataset" .jsonl)"
      for model in $MODELS; do
        local out="$RUN_DIR/${tag}_${model}.jsonl"
        if [ -f "$out" ]; then
          echo "=== [$tag] model=$model — already done, skipping ==="
          ALL_EVAL_FILES+=("$out")
          continue
        fi
        echo "=== [$tag] model=$model (batch_size=${BATCH_SIZE[$model]}, num_beams=${NUM_BEAMS[$model]}) ==="
        local summary="$RUN_DIR/${tag}_${model}.summary.json"
        local time_log="$RUN_DIR/${tag}_${model}.time.log"
        local _t0
        _t0=$(date +%s)
        medmt-eval run \
          --input "$dataset" --model "$model" --device cuda \
          --batch-size "${BATCH_SIZE[$model]}" --num-beams "${NUM_BEAMS[$model]}" \
          --output "$out" --summary "$summary" \
          --term-bank data/term_banks/radiology_en_de_starter.csv \
          2>> "$time_log" || {
            echo "!!! $tag/$model failed, see $time_log" >&2
            continue
          }
        local _t1
        _t1=$(date +%s)
        echo "wall_seconds=$(( _t1 - _t0 ))" >> "$time_log"
        ALL_EVAL_FILES+=("$out")
      done
    done
    echo "=== Tier done. ${#ALL_EVAL_FILES[@]} result files in $RUN_DIR ==="
}


# --- run_models_with_ids: uses MODEL_IDS[name] for --model-id overrides ---
run_models_with_ids() {
    local ALL_EVAL_FILES=()
    for dataset in $DATASETS; do
      local tag
      tag="$(basename "$dataset" .jsonl)"
      for model in $MODELS; do
        local out="$RUN_DIR/${tag}_${model}.jsonl"
        if [ -f "$out" ]; then
          echo "=== [$tag] model=$model — already done, skipping ==="
          ALL_EVAL_FILES+=("$out")
          continue
        fi
        local model_id="${MODEL_IDS[$model]}"
        echo "=== [$tag] model=$model ($model_id, batch_size=${BATCH_SIZE[$model]}, num_beams=${NUM_BEAMS[$model]}) ==="
        local summary="$RUN_DIR/${tag}_${model}.summary.json"
        local time_log="$RUN_DIR/${tag}_${model}.time.log"
        local _t0
        _t0=$(date +%s)
        # For large models, use the prompted-llm adapter with model-id override.
        # Hy-MT2-7B uses hymt2 adapter, TranslateGemma-27B uses translategemma adapter.
        local adapter
        case "$model" in
          hymt2*) adapter="hymt2" ;;
          translategemma*) adapter="translategemma" ;;
          *) adapter="prompted-llm" ;;
        esac
        medmt-eval run \
          --input "$dataset" --model "$adapter" --model-id "$model_id" --device cuda \
          --batch-size "${BATCH_SIZE[$model]}" --num-beams "${NUM_BEAMS[$model]}" \
          --output "$out" --summary "$summary" \
          --term-bank data/term_banks/radiology_en_de_starter.csv \
          2>> "$time_log" || {
            echo "!!! $tag/$model failed, see $time_log" >&2
            continue
          }
        local _t1
        _t1=$(date +%s)
        echo "wall_seconds=$(( _t1 - _t0 ))" >> "$time_log"
        ALL_EVAL_FILES+=("$out")
      done
    done
    echo "=== Tier done. ${#ALL_EVAL_FILES[@]} result files in $RUN_DIR ==="
}
