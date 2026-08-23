# Datasets

## The corpora

| Corpus | Direction | Segments | Register | Provenance |
|---|---|---|---|---|
| **PARROT** (German subset) | DE→EN | 296 | **Radiology reports** | Radiologist-authored fictional reports; English translations by the same radiologists |
| EMEA (sampled) | DE→EN | 400 | EU drug leaflets / regulatory | Official EMA translations — professional, legally mandated |
| HimL 2015 (Cochrane, NHS24) | EN→DE | 353 | Patient-facing plain language | WMT Biomedical shared task, human-translated |
| HimL 2017 (Cochrane, NHS) | EN→DE | 119 | Patient-facing plain language | WMT Biomedical shared task, human-translated |
| PARROT (Turkish subset) | TR→EN | 48 | Radiology reports | Same corpus, other language — see caveat below |

All references are **human translations**, not synthetic. That is a hard requirement:
scoring MT against MT-generated references measures agreement between systems, not
quality.

## Why PARROT carries the argument

The first three corpora total 872 segments and were the original benchmark. PARROT was
added later and is reported separately, because it is the only corpus in the set made of
**actual clinical reports** rather than patient-facing or regulatory prose.

That distinction decides the question this thesis asks. Drug leaflets and Cochrane
summaries are edited, well-formed, full-sentence prose. Radiology reports are not: they
are telegraphic, abbreviation-dense, full of measurements and laterality, and written
for another clinician. If a specialised model is needed anywhere, it is here.

PARROT composition (German subset, 296 reports):

| Modality | Reports |
|---|---|
| CT | 104 |
| Radiography (RX) | 92 |
| Ultrasound (US) | 45 |
| MR | 38 |
| Angiography (XA) | 15 |
| Mammography (MG) | 2 |

Source length: mean 773 characters, median 588, max 4029. The long tail matters — it is
what forces the chunking machinery described in [03-methods.md](03-methods.md).

## Report structure — and the medical-text-only corpus

PARROT stores each report as a **single free-text string**; there are no section
tags. Sections appear as inline headers, and they are not tidy: 89 distinct
line-initial headers across the 296 German reports.

The regular ones are:

| German header | reports |
|---|---|
| `Beurteilung` (impression) | 88 |
| `Befund` (findings) | 83 |
| `Klinik, Fragestellung, Rechtfertigende Indikation` | 68 |
| `Technik` / `Untersuchungstechnik` | 52 |
| `Fragestellung` | 36 |
| `Befund und Beurteilung` (combined) | 19 |

**Why this matters for the metrics.** The material before the findings —
clinical question, justifying indication, acquisition protocol, consent — is
about the *examination*, not the patient. It contributes numbers that are
protocol parameters: slice thickness, T1/T2 weighting, kV, contrast volume. The
number/measurement detector cannot tell those from a lesion diameter.

[`data/sections.py`](../src/medmt_eval/data/sections.py) extracts sections by
role, and the converter exposes it:

```bash
medmt-eval convert parrot --input … --sections findings impression \
    --sections-mode strict     # 109 reports, medical text only
```

Two constraints, both measured rather than assumed:

- **Only 109 of 296 reports (37%) carry a findings/impression header on *both*
  sides.** A further 35 have one in the English translation only. Extraction is
  impossible for the remaining 152, so `--sections-mode` makes the choice
  explicit: `strict` drops them (109 reports), `lenient` keeps them whole and
  flags `metadata.sectioned` so the two groups are never silently mixed.
- **Headers do not map one-to-one across languages.** One report's German has
  clinical-question, findings and impression sections while its English has only
  findings and impression — the translator dropped a section. Roles are therefore
  matched independently on each side, never by position or count.

On the 109 extractable reports the source shrinks from a mean of 821 to 488
characters — **40% of the text is not the medical content.**

### What removing it actually changes

Less than the character count suggests:

| System | BLEU whole → medical | crit% whole → medical |
|---|---|---|
| DeepSeek-V4-Flash | 57.04 → 52.05 | 19.3% → 18.3% |
| qwen35-27b | 59.26 → 56.05 | 18.8% → 18.8% |
| hymt2-30b-a3b | 52.68 → 52.65 | 29.0% → 28.0% |

BLEU falls a few points — boilerplate is formulaic and easy, so removing it
removes cheap matches. The **critical-error rate moves by 0 to 1 point.**

An earlier estimate put the preamble's share of critical findings at 30%, from
locating each finding's evidence within the source. That estimate was based on 23
findings from one system with 7 unlocated, and the corpus-scale effect is about a
point. The direction was right; the magnitude was overstated by the small sample,
and the corpus-scale number is the one to quote.

So the medical-text corpus is worth having — it answers "can the medical content
be translated" rather than "can the document be reproduced" — but it does not
rescue the headline result. Systems still corrupt clinical content in 18–28% of
reports when only findings and impressions are scored.

## Conversion notes

- **Filter on `language`, not `country`.** The `country` field is dirty: it contains the
  value `"German"` for 50 records, which is not a country. Filtering on it silently
  produces the wrong subset. The converter
  ([`data/parrot.py`](../src/medmt_eval/data/parrot.py)) uses `language`.
- **Area normalisation.** `normalise_area()` folds free-text anatomical area labels into
  a consistent set for the per-modality breakdown.
- **Round-trip sample.** The ten-cycle experiment uses a deterministic,
  length-stratified 20-report subsample (`RT_SAMPLE=20`, `RT_SEED=13`). All twelve
  systems see the *same* twenty reports, so the curves are directly comparable. The full
  296 would have cost roughly 168 GPU-hours across the model set — past every wall-clock
  limit available.

## EMEA

Down-sampled from 364,005 pairs with de-duplication and length-ratio filtering. Official
EMA translations are professional and legally mandated, which makes them a good
reference but an easy register: highly repetitive, heavily templated, and consequently
flattering to any MT system.

## The Turkish subset — a negative control that did not work

The Turkish–English PARROT pairs were run through the same pipeline. The results are
**not comparable** to the German ones and are reported only as a limitation.

The reason is detector coverage. Negation, laterality and terminology detectors each
need a cue lexicon on *both* sides; only the number/measurement check is
language-agnostic. For TR→EN only that one detector is active, so the "critical error
rate" measures numeric fidelity alone.

The `identity` control makes this concrete: passing Turkish through untranslated scores
**0%** critical errors on the Turkish set, against **95.95%** on the German set. On
German the control correctly identifies untranslated text as catastrophic; on Turkish it
sees nothing wrong. Without a working floor, the Turkish numbers have no scale.

Recorded in [`results/parrot_tr/INVALID_RUNS.md`](../results/parrot_tr/INVALID_RUNS.md).

## Terminology bank

A nine-entry starter bank of EN↔DE radiology terms
([`data/term_banks/radiology_en_de_starter.csv`](../data/term_banks/radiology_en_de_starter.csv)).
It is deliberately small and its findings are `major`, not `critical` — it exists to
demonstrate the mechanism, not to be authoritative. Matching is inflection-tolerant
(`term_surface_pattern()`); adding that tolerance cut terminology findings from 735 to
293, i.e. roughly two thirds of the original findings were German inflection, not
translation errors.
