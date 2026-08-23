#!/usr/bin/env bash
# Shared setup for the glossary and debate experiments. Sourced, not executed.
#
# Preflight is mandatory here for the same reason it is everywhere else in this
# repo: a run that dies twenty minutes in on a missing credential or an absent
# glossary has burned an allocation to learn something checkable in one second.
set -euo pipefail

module purge
module load python/3.12-base cuda/12.8.1

PROJECT=/home/atuin/b180dc/b180dc50/german_medical_translator
cd "$PROJECT"
mkdir -p logs results/experiments
source .venv/bin/activate

export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXP_INPUT="${EXP_INPUT:-data/derived/parrot_de.jsonl}"
EXP_GLOSSARY="${EXP_GLOSSARY:-data/term_banks/wikidata_med.csv}"
EXP_BRANCHES="${EXP_BRANCHES:-ACE}"
EXP_SAMPLE="${EXP_SAMPLE:-40}"
EXP_SEED="${EXP_SEED:-13}"
EXP_MAX_TERMS="${EXP_MAX_TERMS:-30}"
EXP_TERM_BANK="${EXP_TERM_BANK:-data/term_banks/radiology_en_de_starter.csv}"
RUN_DIR="${RESUME_DIR:-results/experiments/${SLURM_JOB_ID:-local}}"
mkdir -p "$RUN_DIR"

# ── preflight ────────────────────────────────────────────────────────────
[ -f "$EXP_INPUT" ] || { echo "ERROR: corpus $EXP_INPUT missing. Run 'medmt-eval convert parrot' first." >&2; exit 1; }
[ -f "$EXP_GLOSSARY" ] || { echo "ERROR: glossary $EXP_GLOSSARY missing. Run scripts/build_glossary_wikidata.py first." >&2; exit 1; }

python - "$EXP_GLOSSARY" "$EXP_BRANCHES" <<'PY'
import sys
from medmt_eval.glossary import Glossary
path, branches = sys.argv[1], sys.argv[2]
g = Glossary.from_csv(path).exclude_mt().in_branches(branches).covering("de", "en")
print(f"preflight: glossary {path} -> {len(g)} entries in branches {branches}")
if len(g) < 100:
    raise SystemExit(f"ERROR: only {len(g)} usable entries; check --branches and the build.")
PY

echo "preflight: run dir  $RUN_DIR"
echo "preflight: corpus   $EXP_INPUT (sample $EXP_SAMPLE, seed $EXP_SEED)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "preflight: no GPU visible"
