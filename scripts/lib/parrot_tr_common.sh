#!/usr/bin/env bash
# Shared PARROT-Turkish setup. Source AFTER lib/bench_common.sh, which sets the
# HimL/EMEA defaults this file then overrides.
#
# 48 Turkish radiology reports, scored TR->EN (Turkish report as source, the
# contributor's English translation as reference).
#
# TWO CAVEATS THAT TRAVEL WITH THESE RESULTS:
#
# 1. REDUCED DETECTOR COVERAGE. The clinical detectors have cue lexicons for
#    EN and DE only. For TR->EN the negation, laterality and terminology
#    detectors all skip (they compare a source cue against a target cue and
#    have nothing to match on the Turkish side), leaving only the
#    language-agnostic number/measurement check active. A critical-error rate
#    from this run is therefore NOT comparable to the German one, which has
#    all four detectors running.
#
# 2. THE SUBSET IS SMALL AND SKEWED. 48 reports from just 2 contributors
#    (German: 296 from 10), median 2,080 chars vs 588 for German, and modality
#    is CT 28 / XA 18 / US 2 with no RX or MR at all. Since error rate rises
#    steeply with report length (5% under 250 chars, 70% over 2000), expect
#    higher rates here for reasons unrelated to the language.

# Turkish reports are ~3.5x longer than the German ones, so the default output
# ceiling is raised further still. Runs 3941775-3941778 used the old 512-token
# default and came back truncated; their numbers are invalid, not findings.
if [ "${MAX_NEW_TOKENS_FROM_ENV:-0}" != "1" ]; then MAX_NEW_TOKENS=3072; fi

PARROT_SRC="${PARROT_SRC:-PARROT_v1.0/data/PARROT_v1_0.jsonl}"
PARROT_TR_JSONL="${PARROT_TR_JSONL:-data/derived/parrot_tr.jsonl}"

if [ ! -f "$PARROT_SRC" ]; then
  echo "ERROR: PARROT source not found at $PARROT_SRC" >&2
  exit 1
fi

mkdir -p data/derived
[ -f "$PARROT_TR_JSONL" ] || medmt-eval convert parrot \
  --input "$PARROT_SRC" --output "$PARROT_TR_JSONL" --src-lang tr --tgt-lang en

echo "=== PARROT-Turkish: $(wc -l < "$PARROT_TR_JSONL") segments (TR->EN) ==="
echo "=== NOTE: reduced detector coverage — number checks only ==="
echo "=== results -> $RUN_DIR ==="

DATASETS="$PARROT_TR_JSONL"
