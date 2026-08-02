# medmt-eval

A reproducible medical machine-translation evaluation pipeline. It deliberately
reports **two layers side by side**:

1. **Surface quality** — sacreBLEU, chrF++, TER against a human reference.
2. **Clinical information loss** — negation, laterality, number/measurement and
   terminology preservation.

The point is to make the divergence visible: a fluent translation can score well
on surface metrics and still invert a negation or drop a measurement. In this
project's own results that divergence is real — on radiology reports the
best-BLEU system ranks ninth of eleven on clinical safety.

**Findings live in [`RESULTS_INFORMATION_LOSS.md`](RESULTS_INFORMATION_LOSS.md).**

## Layout

```
src/medmt_eval/        the package
├── data/                  corpus converters (HimL SGML, EMEA TMX, PARROT JSONL)
├── models/                translation adapters, one per model family
├── taxonomy/clinical.py   the clinical-loss detectors
├── metrics/               surface (sacreBLEU) and optional neural (COMET)
├── stats/                 paired bootstrap, McNemar
└── report/                leaderboard aggregation and plots

scripts/               SLURM jobs, grouped by corpus — see scripts/README.md
data/derived/          converted corpora (gitignored, regenerate with `convert`)
datasets/              raw downloads (gitignored)
results/               run outputs (gitignored)
tests/                 102 tests, no GPU or network needed
```

## Install

```bash
module load python/3.12-base          # on the HPC cluster
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[mt,storage,plot,deepl,dev]'
pytest -q
```

## Corpora

| Corpus | Direction | Segments | Register | Reference provenance |
|---|---|---|---|---|
| HimL 2015/2017 | EN→DE | 472 | Patient-facing | WMT Biomedical, human-translated |
| EMEA (sampled) | DE→EN | 400 | Drug leaflets | Official EMA translations |
| PARROT German | DE→EN | 296 | **Radiology reports** | Contributor-supplied — provenance undocumented |
| PARROT Turkish | TR→EN | 48 | **Radiology reports** | As above; **reduced detector coverage** |

Convert them into the normalised schema:

```bash
medmt-eval convert himl   --input datasets/himl-test-2015.tgz --year 2015 \
                          --output data/derived/himl2015.jsonl --src-lang en --tgt-lang de
medmt-eval convert emea   --input datasets/emea-de-en.tmx.gz \
                          --output data/derived/emea.jsonl --src-lang de --tgt-lang en
medmt-eval convert parrot --input PARROT_v1.0/data/PARROT_v1_0.jsonl \
                          --output data/derived/parrot_de.jsonl --src-lang de --tgt-lang en
```

## Input schema

```json
{"id":"ct-001","domain":"radiology-ct","src_lang":"en","tgt_lang":"de",
 "src_text":"No pleural effusion.","ref_text":"Kein Pleuraerguss."}
```

`source_text`/`target_text` and `source`/`reference` are accepted aliases; CSV,
TSV and Parquet use the same field names. `ref_text` is optional — without it
the clinical detectors still run, only the surface metrics are skipped.

## Usage

```bash
# translate + score in one step
medmt-eval run --input data/derived/parrot_de.jsonl \
  --model hymt2 --model-id tencent/Hy-MT2-7B --device cuda \
  --output results/out.jsonl --term-bank data/term_banks/radiology_en_de_starter.csv

# score translations you already have (in a `hyp_text` field)
medmt-eval evaluate --input predictions.jsonl --output results/eval.jsonl

# leaderboard across several systems
medmt-eval leaderboard --input results/combined.jsonl --output-dir results/leaderboard

# is system B better than A, or is it noise?
medmt-eval compare --baseline a.jsonl --candidate b.jsonl --metric chrf --resamples 2000
```

For cluster runs see **[`scripts/README.md`](scripts/README.md)**.

## Models

| Adapter | Covers |
|---|---|
| `opus` | Helsinki-NLP Marian, direction-specific checkpoints |
| `nllb` | NLLB-200 |
| `hymt2` | Tencent Hy-MT2 1.8B / 7B / 30B-A3B |
| `translategemma` | Google TranslateGemma 4B / 27B (gated) |
| `prompted-llm` | any local causal LM (Qwen3.5, Llama, …) |
| `openai-compat` | any hosted `/v1/chat/completions` endpoint |
| `deepl` | DeepL API |
| `identity` | no-op baseline; a floor for the detectors, not a system |
| `madlad` | **broken** — produces degenerate output, excluded from results |

Check each upstream licence before use; several are non-commercial.

## Limitations

**The detectors are lexical heuristics, not clinical ground truth.** A finding is
a review candidate; its absence is not proof of safety. Known false positives
are quantified in the results document — for example ~38 % of number findings
fire only because *extra* numbers appeared, not because one was lost.

**Detector coverage is language-dependent.** Cue lexicons exist for EN and DE
only. A TR→EN run therefore scores with the number check alone, so its numbers
are not comparable to the German ones. `ClinicalSafetyEvaluator.detector_coverage()`
reports what actually ran for a given pair.

**No clinician has validated any of this.** Establishing that these automatic
labels track expert judgement is the necessary next step before any safety claim.

**Hosted-API results are not reproducible** the way local ones are: model
versions are undisclosed and the gateway has been observed silently substituting
models. The adapter records what actually served each request and aborts on a
mismatch.

Do not train on WMT biomedical test material, and do not present
synthetic-reference metrics as deployment performance.
