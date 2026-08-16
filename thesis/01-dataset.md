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
