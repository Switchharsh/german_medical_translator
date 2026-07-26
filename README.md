# medmt-eval

`medmt-eval` is a reproducible EN↔DE medical machine-translation evaluation pipeline. It deliberately reports two layers side by side:

1. Standard MT quality: sacreBLEU, chrF++, and TER.
2. Clinical information-loss signals: negation, laterality, and number/measurement preservation, plus optional terminology checks.

The aim is to make the important divergence visible: a fluent translation can score well on surface metrics and still introduce a clinically critical error.

## What is implemented

The Stage-1 MVP includes normalized JSONL/CSV/TSV data loading, Opus-MT and NLLB inference adapters, a generic MADLAD/Tower adapter scaffold, surface metrics with reproducibility signatures, per-segment clinical-rule findings, paired bootstrap comparison, a machine-readable result format, and the BLEU-vs-critical-error plot.

Neural metrics (COMET, COMET-Kiwi, XCOMET) are optional because their checkpoints are large and some are gated. Their scorer is available as an opt-in integration point; the core pipeline never downloads a checkpoint implicitly.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[mt,storage,plot,dev]'
# Optional: pip install -e '.[neural]'
```

## Input schema

Every evaluation example uses this normalized schema. `ref_text` is optional for reference-free safety scoring.

```json
{"id":"ct-001","domain":"radiology-ct","src_lang":"en","tgt_lang":"de","src_text":"No pleural effusion.","ref_text":"Kein Pleuraerguss."}
```

`source_text`/`target_text` and `source`/`reference` are accepted aliases. CSV and TSV use the same field names. A starter synthetic, templated chest-CT fixture is in `data/synthetic/chest_ct_en_de.jsonl`; it is for exercising the pipeline, not a clinical benchmark.

## Quick start

Run a model and score it:

```bash
medmt-eval run \
  --input data/synthetic/chest_ct_en_de.jsonl \
  --model opus --src-lang en --tgt-lang de \
  --output results/opus_ct.jsonl --summary results/opus_ct.summary.json
```

Evaluate predictions already stored in `hyp_text` (no model download):

```bash
medmt-eval evaluate \
  --input my_predictions.jsonl \
  --output results/evaluation.jsonl \
  --summary results/evaluation.summary.json \
  --term-bank data/term_banks/radiology_en_de_starter.csv
```

Create a master table and the headline divergence plot:

```bash
medmt-eval report --input results/evaluation.jsonl --output-dir results/report
```

Compare two systems with paired bootstrap resampling:

```bash
medmt-eval compare --baseline results/opus_predictions.jsonl \
  --candidate results/nllb_predictions.jsonl --metric chrf --resamples 2000
```

## Model choices and licences

`opus` maps to the direction-specific Apache-2.0 Helsinki-NLP checkpoints. `nllb` defaults to `facebook/nllb-200-distilled-1.3B`; `madlad` and `tower` are available as generic adapters. Check each upstream model card and licence before use. In particular, the NLLB and Tower defaults are not appropriate for commercial deployment.

## Important limitations

The built-in safety checks are conservative, lexical heuristics. A detected mismatch is a review candidate, not a clinical conclusion; an absent finding is not proof that the translation is safe. The negation detector currently checks segment-level cue consistency rather than cross-lingual entity-aligned polarity. Use the output to prioritize bilingual clinical review, validate it against experts, and report automatic-label agreement before making safety claims.

Synthetic CT text is intentionally separated from real data. Do not train on WMT biomedical test material, and do not present synthetic-reference metrics as performance on clinical deployment data.
