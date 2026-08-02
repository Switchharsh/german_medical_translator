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
Section 7 quantifies their false-positive rate, which is material to how the numbers below
should be read.

---

## 2. Evaluation data

| Corpus | Direction | Segments | Register | Provenance |
|---|---|---|---|---|
| HimL 2015 (Cochrane, NHS24, himl) | EN→DE | 353 | Patient-facing plain language | WMT Biomedical shared task, human-translated |
| HimL 2017 (Cochrane, NHS) | EN→DE | 119 | Patient-facing plain language | WMT Biomedical shared task, human-translated |
| EMEA (sampled) | DE→EN | 400 | EU drug-leaflet / regulatory | Official EMA translations (professional, legally mandated) |
| PARROT (German subset) | DE→EN | 296 | **Radiology reports** | Radiologist-authored fictional reports; English translations contributed by the same radiologists |

The first three corpora total **872 segments per system**, 10 systems, **8,720 scored
segment-translations**. PARROT was added later and is reported separately in §4, because it
is the only corpus in the set that consists of actual clinical *reports* rather than
patient-facing or regulatory prose.

EMEA was down-sampled from 364,005 pairs with de-duplication and length-ratio filtering.
Note the direction differences: HimL is scored EN→DE; EMEA and PARROT DE→EN.

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

## 4. PARROT — actual radiology reports (DE→EN)

The three corpora above are patient-facing or regulatory prose. PARROT
([github.com/PARROT-reports/PARROT_v1.0](https://github.com/PARROT-reports/PARROT_v1.0),
[arXiv 2507.22939](https://arxiv.org/abs/2507.22939)) supplies the missing register:
**296 German radiology reports**, each with an English translation, scored DE→EN with the
German report as source. Modalities: CT 104, RX 92, US 45, MR 38, XA 15, MG 2.
12 systems × 296 segments = **3,552 scored segment-translations**.

Systems split into two groups: **local models** run on cluster GPUs, and three
**hosted LLMs** reached over an OpenAI-compatible API (marked ☁). The hosted group is
not size-matched to the local one — their parameter counts are undisclosed — so the
comparison is "what is reachable" rather than a controlled scale study.

### Critical error rate and finding counts

| System | | Critical error rate | Negation | Number | Laterality | Terminology | BLEU | chrF | TER |
|---|---|---|---|---|---|---|---|---|---|
| **glm-5.2** | ☁ | **19.26 %** | 1 | 47 | 13 | 52 | 51.34 | 73.42 | 34.80 |
| DeepSeek-V4-Flash | ☁ | 19.93 % | 1 | 51 | 11 | 50 | **53.75** | 74.70 | **33.26** |
| MiniMax-M3 | ☁ | 21.28 % | 1 | 49 | 17 | 52 | 53.49 | **74.80** | 33.46 |
| translategemma-4b | | 22.97 % | 4 | 53 | 15 | 59 | 44.68 | 68.04 | 45.20 |
| hymt2-7b | | 24.32 % | 7 | 55 | 18 | 54 | 45.99 | 69.17 | 42.83 |
| qwen35-4b | | 24.66 % | 1 | 60 | 34 | 78 | 47.95 | 69.02 | 40.89 |
| hymt2-1.8b | | 25.34 % | 10 | 58 | 12 | 54 | 39.11 | 64.17 | 49.05 |
| opus | | 26.69 % | 22 | 54 | 55 | 93 | 24.62 | 49.05 | 61.40 |
| qwen35-27b | | 28.38 % | 24 | 71 | 37 | 82 | 51.74 | 70.41 | 38.20 |
| hymt2-30b-a3b | | 31.08 % | 7 | 76 | 22 | 57 | 47.71 | 71.16 | 41.78 |
| nllb | | 33.45 % | 13 | 75 | 51 | 104 | 21.76 | 46.59 | 64.96 |
| *identity* | | *95.95 %* | *252* | *0* | *211* | *150* | *3.25* | *25.63* | *94.25* |

(BLEU/chrF/TER are means of per-segment scores. `translategemma-27b` is missing: it failed
with HTTP 401 on the gated `google/translategemma-27b-it` repository, because the Hugging
Face token was absent from the job environment on that run. `translategemma-4b` failed the
same way initially and was re-run successfully once the token was set.)

### Findings specific to report text

**1. Error rates are 3–5× higher than on the other corpora.** The best system on PARROT
flags 19.3 % of segments, versus 5.3 % for the best system across HimL/EMEA. Radiology
reports are denser in exactly the content the detectors watch — measurements, laterality,
named findings — so there is simply more to get wrong per segment. This is the clearest
evidence in the whole study that **results from patient-facing text do not transfer to
clinical reports**, and that the earlier corpora understate the problem for report translation.

**2. Terminology becomes the dominant failure mode.** Findings across all systems
(excluding `identity`), n = 1,760:

| Finding type | Count | Share |
|---|---|---|
| `terminology_not_preserved` | 735 | 41.8 % |
| `number_or_measurement_mismatch` | 649 | 36.9 % |
| `laterality_missing_or_flipped` | 170 | 9.7 % |
| `laterality_added_or_flipped` | 115 | 6.5 % |
| `negation_dropped` | 66 | 3.8 % |
| `negation_introduced` | 25 | 1.4 % |

`terminology_not_preserved` fired **zero times** across all 8,720 segments of HimL/EMEA, and
is the single largest category here. The reason is mundane but important: the term bank is a
radiology term bank, and PARROT is the first corpus containing radiology text. Laterality
likewise jumps from 1.4 % of findings to 16.2 %, since left/right is a routine feature of
imaging reports and rare in drug leaflets. Negation, the dominant category on the earlier
corpora, falls to 5.2 % here.

**3. The three hosted LLMs outperform every local model — on both layers at once.**
`glm-5.2` (19.26 %), `DeepSeek-V4-Flash` (19.93 %) and `MiniMax-M3` (21.28 %) take the top
three safety positions *and* the top three surface positions (BLEU 51–54, chrF 73–75,
TER 33–35). This is the only place in the study where the two layers agree at the top of
the ranking. Note this is a "what is reachable" comparison, not a controlled one — the
hosted models' parameter counts are undisclosed and are probably far larger than anything
run locally here.

**4. Negation is essentially solved by the hosted models.** All three logged exactly **one**
negation error across 296 reports, against 4–24 for the local models (`opus` 22,
`qwen35-27b` 24). Their remaining failures are almost entirely lexical — terminology and
numbers — rather than semantic. That is a meaningful safety distinction: a dropped negation
inverts clinical meaning, whereas an unpreferred synonym usually does not.

**5. Surface metrics still misorder the local models.** Among local systems `qwen35-27b`
has the best BLEU (51.74) and TER (38.20) yet ranks **ninth of eleven** on critical errors
(28.38 %); the safest local system, `translategemma-4b` (22.97 %), is only sixth on BLEU
(44.68). Ranking by BLEU alone would materially misorder them on the dimension that matters
clinically. `opus` and `nllb` are the exception — weakest on both layers, so there the two
agree.

**6. Model scale does not help within a family.** The *smaller* model is safer in both
comparable pairs: `hymt2-7b` (24.32 %) beats `hymt2-30b-a3b` (31.08 %), and `qwen35-4b`
(24.66 %) beats `qwen35-27b` (28.38 %). The TranslateGemma pair cannot be compared because
the 27B run failed on repository gating. This mirrors the pattern already seen on
HimL/EMEA — within a family, scale is not the binding constraint for clinical fidelity.

### Per-modality variation

Error rates are far from uniform across exam types (`hymt2-7b` / `nllb`):

| Modality | n | hymt2-7b | nllb |
|---|---|---|---|
| XA (angiography) | 15 | 66.7 % | 86.7 % |
| CT | 104 | 30.8 % | 47.1 % |
| MR | 38 | 26.3 % | 42.1 % |
| US | 45 | 15.6 % | 24.4 % |
| RX (radiography) | 92 | 14.1 % | 10.9 % |
| MG | 2 | 0.0 % | 0.0 % |

Angiography reports are the hardest by a wide margin, plain radiography the easiest. The XA
and MG cells rest on 15 and 2 reports respectively and should be treated as indicative only.

### PARROT-specific caveats

- **Translation provenance is not documented.** PARROT states only that "contributors
  provided an English translation"; no professional translation, review, or QA step is
  described anywhere in the paper or repository, and the paper's own limitations section does
  not discuss translation quality. The English side is therefore an **unverified-provenance
  human reference, not a certified gold standard** — contributors may have used machine or
  LLM assistance, and nothing would have caught it. Any claim resting on these references
  should say so.
- **Reports are fictional**, authored by radiologists rather than drawn from real practice.
  The PARROT authors report that participants distinguished PARROT reports from GPT-generated
  ones at only 53.9 % accuracy, which argues the register is authentic — but fictional reports
  still lack the dictation errors, inconsistent abbreviations and copy-paste artifacts of real
  clinical documents, so results here likely remain optimistic.
- **Licence: CC BY-NC-SA 4.0**, and the README states the dataset "should not be used for
  training." Evaluation only; derivatives inherit the ShareAlike terms.
- **Single direction.** Only DE→EN was run, matching the direction the human translator
  worked in. EN→DE would invert the pair and reverse the translationese.

### Caveats specific to the hosted (☁) models

- **The gateway substitutes models for some aliases.** Requests went through
  `api.hcnsec.cn`, a third-party gateway rather than each vendor's own API. Asking for
  `DeepSeek-V4-Pro` was served `nvidia/nemotron-3-ultra-550b-a55b`, and `Kimi-K2.6` was
  served `thinkingmachines/inkling` — silently, with no error. The `/v1/models` catalogue is
  no guide either: every entry there claims `"owned_by": "openai"`. **Only the three aliases
  verified to route honestly were benchmarked**, and the adapter re-checks the served model
  on every request, aborting on a mismatch. Each result file records `served_models` in its
  `generation_config`; all three report `model_substitution: false`.
- **Undisclosed size and version.** Parameter counts, quantisation and checkpoint dates for
  the hosted models are unknown, and the endpoint could change what it serves at any time
  without notice. These rows are **not reproducible in the way the local ones are** — a rerun
  months later may measure a different model under the same name.
- **Not size-matched.** The hosted models are probably far larger than anything run locally,
  so §4 finding 3 says "hosted LLMs beat local models here", not "API models are inherently
  better than local ones at equal scale". No such claim is supported by this data.
- **Data left the cluster.** PARROT is fictional, CC BY-NC-SA licensed text, so sending it to
  a third-party endpoint raises no confidentiality issue. **That would not hold for real
  clinical reports**, which must not be sent to an external API without an appropriate legal
  basis.

---

## 5. Where the information is lost

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

## 6. Concrete examples

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

## 7. Limitations — read before drawing conclusions

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

## 8. Answering the question

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

## 9. Reproducing

**HimL / EMEA (§3, §5–§6):**

```bash
# consolidated results from all valid runs
results/consolidated/combined.jsonl
results/consolidated/leaderboard/leaderboard.csv

# regenerate the leaderboard
medmt-eval leaderboard \
  --input results/consolidated/combined.jsonl \
  --output-dir results/consolidated/leaderboard
```

**PARROT (§4):**

```bash
# convert the German subset (296 of 2,738 records; filters on `language`,
# not `country`, which contains "German" as a value for 50 records)
medmt-eval convert parrot \
  --input PARROT_v1.0/data/PARROT_v1_0.jsonl \
  --output data/derived/parrot_de.jsonl --src-lang de --tgt-lang en

# local models (HF_TOKEN must be exported — TranslateGemma repos are gated)
sbatch scripts/benchmark_parrot_small.slurm    # identity, opus, nllb
sbatch scripts/benchmark_parrot_medium.slurm   # hymt2-1.8b, translategemma-4b, qwen35-4b
sbatch scripts/benchmark_parrot_7b.slurm       # hymt2-7b
sbatch scripts/benchmark_parrot_large.slurm    # hymt2-30b-a3b, translategemma-27b, qwen35-27b

# hosted models (OPENAI_COMPAT_API_KEY must be exported).
# Only these three aliases are verified to route honestly — see the caveats above.
API_MODEL=glm-5.2            sbatch scripts/benchmark_parrot_api.slurm
API_MODEL=DeepSeek-V4-Flash  sbatch scripts/benchmark_parrot_api.slurm
API_MODEL=MiniMax-M3         sbatch scripts/benchmark_parrot_api.slurm

# confirm a compute node can reach the endpoint before a long run
sbatch scripts/check_api_connectivity.slurm

# results
results/parrot_consolidated/combined.jsonl
results/parrot_consolidated/leaderboard/leaderboard.csv
```

Source jobs: 3933278 (small), 3933279 + 3935437 (medium, incl. the translategemma-4b
re-run), 3933280 (7b), 3933281 (large), 3935933 (glm-5.2), 3935934 (DeepSeek-V4-Flash),
3935935 (MiniMax-M3).

**Excluded from consolidation:**
- `madlad` — adapter bug produced repeated-glyph garbage (BLEU ≈ 0.00, TER 650–990).
- `qwen35-4b` / `qwen35-27b` runs from jobs 3914950 / 3916830 — affected by a decoder-only
  right-padding bug that corrupted batched generation. Superseded by jobs 3923919 / 3923920,
  which are the runs reported here.
- `translategemma-27b` on PARROT — never produced output; the run failed on repository
  gating (HTTP 401) rather than any pipeline fault, and has not yet been re-run.
