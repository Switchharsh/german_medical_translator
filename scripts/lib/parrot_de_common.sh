#!/usr/bin/env bash
# Shared PARROT setup. Source this AFTER lib/bench_common.sh, not instead of it.
#
# lib/bench_common.sh hardcodes DATASETS to the HimL/EMEA set and prepares those
# corpora. This file converts the PARROT German subset (if not already done)
# and then overrides DATASETS so run_models/run_models_with_ids operate on
# PARROT instead. Sourcing order matters — override must come after.
#
# PARROT: 296 German radiology reports, each with a contributor-supplied
# English translation. Scored DE->EN by default (German report as source,
# English translation as reference) — the direction the human translator
# actually worked in.
#
# Provenance caveat carried into any results: PARROT documents only that
# "contributors provided an English translation". No professional translation,
# review, or QA step is stated, so the English side is an unverified-provenance
# human reference, not a certified gold standard.

PARROT_SRC="${PARROT_SRC:-PARROT_v1.0/data/PARROT_v1_0.jsonl}"
PARROT_JSONL="${PARROT_JSONL:-data/derived/parrot_de.jsonl}"

if [ ! -f "$PARROT_SRC" ]; then
  echo "ERROR: PARROT source not found at $PARROT_SRC" >&2
  exit 1
fi

mkdir -p data/derived
[ -f "$PARROT_JSONL" ] || medmt-eval convert parrot \
  --input "$PARROT_SRC" --output "$PARROT_JSONL" --src-lang de --tgt-lang en

echo "=== PARROT dataset: $(wc -l < "$PARROT_JSONL") segments ==="

# Override the HimL/EMEA default from lib/bench_common.sh.
DATASETS="$PARROT_JSONL"

echo "=== PARROT results -> $RUN_DIR ==="
