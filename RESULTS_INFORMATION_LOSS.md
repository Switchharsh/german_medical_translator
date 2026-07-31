# Clinical Information Loss in EN↔DE Medical Machine Translation

**Date:** 2026-07-29
**Pipeline:** `medmt-eval` (this repository)
**Question:** Do current MT systems lose clinically critical information when translating
medical text between English and German — and if so, enough to justify building or
fine-tuning a specialised model?

---

## 1. What was measured

Two layers are reported side by side for every system:

1. **Surface quality** — sacreBLEU, chrF++, TER against a human reference.
2. **Clinical information loss** — rule-based detectors that flag, per segment:
   - `negation_dropped` / `negation_introduced` — a negation cue present on one side and
     absent on the other ("no effusion" → "effusion").
   - `number_or_measurement_mismatch` — numbers/measurements that do not match after unit
     normalisation (5 mm → 8 mm, 1.5 cm → 15 mm is *not* flagged, since it is equivalent).
   - `laterality_missing_or_flipped` / `laterality_added_or_flipped` — left/right/bilateral
     mismatch.
   - `terminology_not_preserved` — a term-bank concept not rendered as its expected target term.

A segment counts as having a **critical error** if any `critical`-severity finding fires.

**These detectors are conservative lexical heuristics, not a clinical ground truth.**
Section 6 quantifies their false-positive rate, which is material to how the numbers below
should be read.

---

## 2. Evaluation data

| Corpus | Direction | Segments | Register | Provenance |
|---|---|---|---|---|
| HimL 2015 (Cochrane, NHS24, himl) | EN→DE | 353 | Patient-facing plain language | WMT Biomedical shared task, human-translated |
| HimL 2017 (Cochrane, NHS) | EN→DE | 119 | Patient-facing plain language | WMT Biomedical shared task, human-translated |
| EMEA (sampled) | DE→EN | 400 | EU drug-leaflet / regulatory | Official EMA translations (professional, legally mandated) |

Total: **872 segments per system**, 10 systems, **8,720 scored segment-translations**.

EMEA was down-sampled from 364,005 pairs with de-duplication and length-ratio filtering.
Note the direction difference: HimL is scored EN→DE, EMEA DE→EN.

---

## 3. Headline result — critical error rate by system

Percentage of segments containing at least one critical finding, across all 872 segments.

| System | Type | Params | Critical error rate |
|---|---|---|---|
| **hymt2-7b** | specialised MT | 7B | **5.28 %** |
| hymt2-30b-a3b | specialised MT (MoE) | 30B / 3B active | 6.54 % |
| hymt2-1.8b | specialised MT | 1.8B | 6.65 % |
| qwen35-27b | general-purpose LLM | 27B | 6.88 % |
| qwen35-4b | general-purpose LLM | 4B | 7.22 % |
| opus | generic MT | ~74M | 9.06 % |
| translategemma-4b | specialised MT | 4B | 9.52 % |
| translategemma-27b | specialised MT | 27B | 9.98 % |
| nllb | generic MT | 1.3B | 11.47 % |
| *identity (baseline)* | *no translation* | — | *13.07 %* |

`identity` copies the source through untranslated; it is a floor for the detectors, not a system.

### Per-corpus breakdown

| System | HimL2015 (EN→DE) | HimL2017 (EN→DE) | EMEA (DE→EN) |
|---|---|---|---|
| hymt2-7b | 5.67 % | 7.56 % | **4.25 %** |
| hymt2-30b-a3b | 5.67 % | 10.08 % | 6.25 % |
| hymt2-1.8b | 7.08 % | 8.40 % | 5.75 % |
| qwen35-27b | 5.95 % | 10.08 % | 6.75 % |
| qwen35-4b | 6.52 % | 8.40 % | 7.50 % |
| opus | 6.52 % | 9.24 % | 11.25 % |
| translategemma-4b | 8.50 % | 10.08 % | 10.25 % |
| translategemma-27b | 7.65 % | 13.45 % | 11.00 % |
| nllb | 5.67 % | 8.40 % | 17.50 % |
| *identity* | *11.61 %* | *6.72 %* | *16.25 %* |

---

## 4. Where the information is lost

Raw finding counts across all systems (excluding `identity`), all corpora:

| Finding type | Count | Share |
|---|---|---|
| `number_or_measurement_mismatch` | 395 | 59.6 % |
| `negation_introduced` | 185 | 27.9 % |
| `negation_dropped` | 65 | 9.8 % |
| `laterality_added_or_flipped` | 9 | 1.4 % |
| `laterality_missing_or_flipped` | 9 | 1.4 % |
| `terminology_not_preserved` | 0 | 0 % |

**Numbers and measurements are the dominant failure mode**, accounting for ~60 % of all
critical findings — and they concentrate heavily in EMEA (drug-leaflet text, dense with
dosages, concentrations and quantities). This is the single most consequential category
for patient safety and the clearest target for any future work.

`terminology_not_preserved` never fired. The starter term bank contains only 8 radiology
concepts and the corpora are not radiology text, so this detector was effectively untested,
not "passed".

### Surface quality does not predict information loss

The two layers diverge, which is the central motivation for this pipeline:

- **nllb on EMEA**: BLEU **39.7** (near the top) but **17.5 %** critical error rate (the worst
  non-identity result) — driven by a 15 % number-mismatch rate.
- **opus on EMEA**: highest BLEU of any system (**44.3**) but 11.25 % critical errors.
- **hymt2-7b on EMEA**: lower BLEU (37.2) but the best safety result at **4.25 %**.

A system can be fluent, score well, and still drop or corrupt the clinically load-bearing part
of a sentence. BLEU alone would have ranked `opus` and `nllb` at the top.

---

## 5. Concrete examples

**Negation dropped** (`hymt2-1.8b`, EMEA):
> SRC: *Kommt es später erneut zu einem Kontakt mit IBV, so erkranken die Hühner entweder **nicht** oder machen nur eine sehr viel weniger schwere Infektion durch.*
> HYP: *If there is another contact with IBV later on, the chickens either won't get sick or will only suffer a much less severe infection.*

Here the detector fired on the cue `nicht`, but the translation ("won't get sick") is in fact
correct — an example of the false positives discussed below.

**Number mismatch** (`hymt2-1.8b`, EMEA):
> SRC: *… an der **1 367** Patienten teilnahmen …*
> HYP: *… involving **1,367** patients.*

A thousands-separator formatting difference, not information loss. The number parser treats
`1 367` as two numbers.

**Genuine reference misalignment** (EMEA):
> SRC: *Sebivo-Tabletten wurden in einer zweijährigen Studie … verglichen*
> REF: *A response was defined as low levels of viral DNA in the blood …*

The source and reference are different sentences — a corpus artifact, not a model error.

---

## 6. Limitations — read before drawing conclusions

These caveats are large enough to change the interpretation of Section 3.

1. **The detectors produce false positives.** Manual inspection of flagged segments found:
   - Norwegian postal codes (`NO-0401 Oslo`) flagged as `negation_introduced` (~8 occurrences).
   - Valid negation renderings ("not to exceed" → "no more than") flagged as changes.
   - The substring `right` in non-anatomical contexts triggering laterality findings.
   A flagged segment is a **review candidate**, not a confirmed clinical error. The true
   error rates are lower than the headline percentages — likely materially so for negation.

2. **9 % of EMEA reference pairs are misaligned** (source/reference length ratio > 2×), an
   artifact of `pdftotext` extraction and automatic sentence alignment in the OPUS EMEA corpus.
   Misaligned pairs inflate both BLEU penalties and number-mismatch findings. EMEA numbers
   should be treated as the least reliable of the three corpora.

3. **Small samples.** HimL2017 has 119 segments; its per-corpus subsets are as small as 35.
   Differences of a few percentage points there are within noise. No confidence intervals have
   been computed yet — the pipeline supports paired bootstrap (`medmt-eval compare`), which
   should be run before claiming any system is better than another.

4. **Direction is confounded with corpus.** HimL is EN→DE and EMEA is DE→EN, so
   corpus-to-corpus differences mix register *and* direction effects.

5. **No human clinical validation.** No bilingual clinician has reviewed any of these
   translations. Automatic-label agreement with expert judgement is unmeasured.

6. **Not evaluated:** DeepL (the obvious commercial baseline — never run, requires an API key)
   and any dedicated EN↔DE medical MT product. MADLAD-400 was run but produced degenerate
   output from an adapter bug and is excluded entirely.

---

## 7. Answering the question

**Is information lost?** Yes, measurably — but less than the headline figures suggest once
detector false positives and EMEA misalignment are accounted for. The best systems flag
~5 % of segments; a meaningful fraction of those are detector artifacts rather than real errors.

**Does this justify a specialised model?** The evidence does **not** currently support that,
for three reasons:

1. **Off-the-shelf systems already perform comparably.** `hymt2-7b` (an existing open
   specialised MT model) leads at 5.28 %, and a general-purpose LLM (`qwen35-27b`, 6.88 %)
   is close behind. There is no large headroom that a new model would obviously capture.
2. **Bigger is not better here.** `translategemma-27b` (9.98 %) is *worse* than
   `translategemma-4b` (9.52 %), and `hymt2-30b-a3b` is worse than `hymt2-7b`. Scale is not
   the binding constraint.
3. **The dominant failure mode is numbers**, which is plausibly better addressed by
   constrained decoding or a post-hoc numeric-consistency check than by retraining a model.

**Recommended next steps, in order:**

1. Run **DeepL** as a commercial baseline — the most important missing comparison.
2. **Tighten the detectors** (postal-code and thousands-separator false positives are
   straightforward fixes) and re-score; the current numbers overstate the problem.
3. **Validate against clinician judgement** on a sample of flagged segments to establish the
   detectors' true precision.
4. Compute **bootstrap confidence intervals** before ranking systems.
5. Only then revisit fine-tuning — and if pursued, target numeric/measurement fidelity
   specifically rather than general translation quality.

---

## 8. Reproducing

```bash
# consolidated results from all valid runs
results/consolidated/combined.jsonl
results/consolidated/leaderboard/leaderboard.csv

# regenerate the leaderboard
medmt-eval leaderboard \
  --input results/consolidated/combined.jsonl \
  --output-dir results/consolidated/leaderboard
```

**Excluded from consolidation:**
- `madlad` — adapter bug produced repeated-glyph garbage (BLEU ≈ 0.00, TER 650–990).
- `qwen35-4b` / `qwen35-27b` runs from jobs 3914950 / 3916830 — affected by a decoder-only
  right-padding bug that corrupted batched generation. Superseded by jobs 3923919 / 3923920,
  which are the runs reported here.
