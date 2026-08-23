#!/usr/bin/env bash
#SBATCH --job-name=exp-comet
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp-%x-%j.out
#SBATCH --error=logs/exp-%x-%j.err
#
# COMET over the existing single-pass outputs. No retranslation — every system's
# hypotheses are already on disk, so this is the cheapest metric upgrade
# available (thesis/07-metric-roadmap.md step 1).
#
# COMET runs from its OWN venv. unbabel-comet 2.2.7 requires transformers 4.x,
# and installing it into the project venv downgraded transformers from 5.15.1 to
# 4.57.6 — which broke Qwen3.5 loading for every other experiment
# ("model type 'qwen3_5' not recognized") and killed an unrelated cascade job.
# Upgrading back breaks COMET instead: its XLM-R encoder unpacks a three-tuple
# that transformers 5.x no longer returns. They cannot share an environment.
set -euo pipefail
source /home/atuin/b180dc/b180dc50/german_medical_translator/scripts/experiments/_exp_common.sh

COMET_VENV="${COMET_VENV:-$PROJECT/.venv-comet}"
[ -x "$COMET_VENV/bin/python" ] || { echo "ERROR: $COMET_VENV missing. Create it with: python3 -m venv .venv-comet && .venv-comet/bin/pip install unbabel-comet==2.2.7" >&2; exit 1; }
PY_COMET="$COMET_VENV/bin/python"
echo "preflight: COMET venv $($PY_COMET -c 'import transformers,comet; print(\"transformers\", transformers.__version__, \"comet\", comet.__version__)')"

$PY_COMET -c "import comet" 2>/dev/null || { echo "ERROR: pip install unbabel-comet" >&2; exit 1; }

# Compute nodes have no outbound network. The checkpoint must already be in the
# HF cache; fetch it on the LOGIN node first:
#   python -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')"
CKPT="${COMET_CKPT:-$HF_HOME/hub/models--Unbabel--wmt22-comet-da/snapshots}"
CKPT_FILE=$(find "$CKPT" -name model.ckpt 2>/dev/null | head -1)
[ -n "$CKPT_FILE" ] || { echo "ERROR: no cached COMET checkpoint under $CKPT — pre-fetch on the login node." >&2; exit 1; }
echo "preflight: COMET checkpoint $CKPT_FILE"

$PY_COMET scripts/score_comet.py results/parrot_de/consolidated \
  -o results/comet_parrot_de.json --checkpoint "$CKPT_FILE" \
  --local-files-only --gpus 1 --batch-size 16
