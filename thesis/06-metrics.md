# Metrics: what we compute, how to read it, and why

Two independent layers are computed over every translation. They answer different
questions and they disagree, which is the point.

| Layer | Metric | Range | Direction | Needs a reference? |
|---|---|---|---|---|
| Surface | BLEU | 0–100 | higher better | yes |
| Surface | chrF++ | 0–100 | higher better | yes |
| Surface | TER | 0–∞ (usually 0–100) | **lower** better | yes |
| Clinical | critical-error rate | 0–100% | **lower** better | no — compares source to output |
| Clinical | finding counts by code | integer | lower better | no |

---

## 0. The one table that motivates everything

The same reference sentence, five candidate translations, scored both ways:

| candidate | BLEU | chrF++ | TER | clinical detectors |
|---|---|---|---|---|
| exact match | 100.00 | 100.00 | 0.00 | — |
| **negation dropped** ("No pleural effusion" → "Pleural effusion") | 84.46 | 92.19 | 8.33 | `negation_dropped` |
| **laterality flipped** (left → right) | 78.25 | 87.50 | 8.33 | `laterality_*_or_flipped` |
| **measurement wrong** (5 mm → 15 mm) | 78.25 | 92.54 | 8.33 | `number_or_measurement_mismatch` |
| harmless paraphrase ("pleural effusion" → "pleural fluid collection") | 53.04 | 75.34 | 25.00 | — |

Read the last two rows together. The **harmless paraphrase scores worst on all three
surface metrics** — 53 BLEU — while the three translations that would change patient
management all score 78–85. A ranking built on BLEU prefers a report that says the
patient *has* an effusion when they do not, over a report that says the same true thing
in different words.

This is not a flaw in BLEU. BLEU measures string overlap and reports string overlap
faithfully. It is a flaw in using BLEU alone to decide whether a medical translation is
safe. Hence the second layer.

---

## 1. Surface metrics

All three come from **sacreBLEU 2.6.0**, which exists precisely so that scores are
comparable between papers — it fixes tokenisation and prints a signature recording
every setting. The signatures are persisted with every result:

```
BLEU  |nrefs:1|case:mixed|eff:yes|tok:13a|smooth:exp|version:2.6.0
chrF++|nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0
TER   |nrefs:1|case:lc|tok:tercom|norm:no|punct:yes|asian:no|version:2.6.0
```

A BLEU number without a signature is not reproducible — different tokenisers move BLEU
by several points, which is why detokenised comparisons across papers are usually
meaningless.

### 1.1 BLEU

**What it counts.** How many word n-grams (n = 1, 2, 3, 4) of the candidate also appear
in the reference, with a penalty for being too short.

**How it is calculated.**

For each n, *modified precision* pₙ = (matching n-grams, clipped so a candidate cannot
get credit for repeating an n-gram more often than the reference contains it) divided by
(total n-grams in the candidate). The four precisions are combined as a geometric mean,
then multiplied by a brevity penalty:

```
BP   = 1                    if c > r
     = exp(1 − r/c)         if c ≤ r          c = candidate length, r = reference length

BLEU = BP · exp( Σ_{n=1..4} ¼ · log pₙ ) · 100
```

The geometric mean is unforgiving: if any pₙ is zero the whole score is zero. Two
settings in use here soften that for short segments — `smooth:exp` (exponential
smoothing of zero counts) and `eff:yes` (effective order: for a segment shorter than
four tokens, average over the orders that actually exist). Both matter because we score
per-segment as well as per-corpus.

**How to read it.** Roughly: <15 useless, 15–30 gist only, 30–40 understandable,
40–50 good, 50–60 very good, >60 approaching a second human reference. These bands
are folklore, not a standard — treat differences under ~1 BLEU as noise, and never
compare BLEU across different test sets.

**Why we use it.** Comparability. Every MT paper reports it, so it locates our systems
in the literature. It is the *baseline* metric, not the deciding one.

**What it cannot do.** It has no notion of meaning. Every word is equally important, so
"no" and "the" carry the same weight. It cannot see that a number changed, only that
*a token* changed. Precision-based n-gram overlap is exactly the wrong instrument for
"did the clinically load-bearing content survive".

### 1.2 chrF++

**What it counts.** The same idea at the character level: F-score over character
n-grams up to order 6, plus word n-grams up to order 2 (that "++").

**How it is calculated.**

```
chrP = matched n-grams / n-grams in the candidate      (precision)
chrR = matched n-grams / n-grams in the reference      (recall)

chrF++ = (1 + β²) · chrP · chrR / (β² · chrP + chrR) · 100      with β = 2
```

β = 2 weights **recall twice as heavily as precision** — omitting reference content is
penalised harder than adding content.

**How to read it.** Runs 15–25 points higher than BLEU on the same output; do not
compare the two numbers directly. It is more stable on small test sets and much fairer
to German, where compounding (`Pleuraerguss`) means a single wrong morpheme destroys a
word-level match but only dents a character-level one.

**Why we use it.** It is the better surface metric for this language pair, and its
recall weighting partially aligns with the concern about dropped content. It is also
more reliable than BLEU at our sample sizes (20–296 segments).

### 1.3 TER — Translation Edit Rate

**What it counts.** The minimum number of edits to turn the candidate into the
reference, normalised by reference length.

**How it is calculated.**

```
TER = (insertions + deletions + substitutions + shifts) / average reference length × 100
```

A *shift* — moving a contiguous block — costs 1, the same as a single-word
substitution. That is what distinguishes TER from plain edit distance and makes it
tolerant of word-order differences, which matters for German verb placement.

**How to read it.** **Lower is better**, and it is the only one of the three that runs
that way — a frequent source of misread tables. 0 is perfect; values above 100 are
possible when the candidate is much longer than the reference.

**Why we use it.** It is a post-editing-effort proxy: it approximates how much work a
human would do to fix the output. That is a different and practically useful question
from "how similar is this string", and it is the metric a clinic would care about if MT
were used as a first draft.

---

## 2. Clinical information loss

The second layer ignores the reference translation and compares the **source** against
the **output** directly, looking for the specific ways a radiology report can become
dangerous. Four detectors, implemented in
[`taxonomy/clinical.py`](../src/medmt_eval/taxonomy/clinical.py).

### 2.1 The detectors

| Detector | Code(s) | Severity | Method |
|---|---|---|---|
| Negation | `negation_dropped`, `negation_introduced` | critical | Per-language cue lexicon (`kein`, `nicht`, `ohne` / `no`, `not`, `without`, …). Fires when the source has negation cues and the output has none, or vice versa. |
| Laterality | `laterality_missing_or_flipped`, `laterality_added_or_flipped` | critical | Lexicon maps to the set {left, right, bilateral}. Fires on any set difference between source and output. |
| Number / measurement | `number_or_measurement_mismatch` | critical | Parses every number and unit, normalises decimals (German `1,5` = English `1.5`) and units, then compares multisets. |
| Terminology | `terminology_not_preserved` | **major** | For each term-bank concept found in the source, checks the expected target term appears in the output, tolerating inflection. |

**Severity matters and is easy to misread.** Only the first three are `critical`.
Terminology findings are `major` and contribute **zero** to the critical-error rate.
A model can have many terminology findings and a low critical-error rate.

### 2.2 Critical-error rate — the headline number

```
critical-error rate = documents with ≥ 1 critical finding / total documents × 100
```

It is a **document-level rate, not a count**. A report with six critical findings and a
report with one both count once. With 20 documents in the round-trip sample, the metric
moves in 5-point steps and nothing finer than that is meaningful.

**How to read it.** "In what fraction of reports did at least one clinically
load-bearing fact fail to survive translation?" A rate of 45% means nine of twenty
reports were damaged somewhere.

**Why we use it.** It is the closest available proxy for the question that actually
matters, it needs no reference translation, and it is *interpretable* — every finding
carries the source evidence and target evidence that triggered it, so any number can be
audited back to a specific sentence.

### 2.3 Known limitations — read before quoting these numbers

- **Precision is imperfect.** Measured against manual review: roughly 38% of number
  findings and (before an inflection-tolerance fix that cut findings from 735 to 293)
  67% of terminology findings were false positives. Number false positives come mostly
  from legitimate reformatting (`5-10 mm` → `5 to 10 mm`).
- **Recall is worse, and unmeasured.** The detectors find only what they are written to
  look for. In `parrot-1093`, three models rendered `BWK 12` (twelfth *thoracic*
  vertebra) as `L12` (twelfth *lumbar*) — a different bone, plainly wrong, and **no
  detector fired**. There is no anatomical-abbreviation detector. The true error rate
  is higher than the reported one, by an unknown margin.
- **Under-translation gets rewarded.** A detector cannot flag a measurement that was
  never emitted. `opus` posts the lowest critical-error rate (10% on a single pass)
  while its output *shrinks* from 599 to 506 characters over the cycles. Its low score
  is partly a shorter answer, not a safer one — which is why
  [figure 5](../figures/fig5_output_length.png) exists and must be read alongside
  [figure 3](../figures/fig3_clinical_errors.png).
- **Coverage varies by language pair.** `detector_coverage()` reports which detectors
  can run. Negation, laterality and terminology need a lexicon on *both* sides; for
  Turkish–English only the language-agnostic number check is active, so Turkish
  critical-error rates are **not comparable** to German ones.

These are the reasons [07-metric-roadmap.md](07-metric-roadmap.md) exists.

---

## 3. Round-trip measurements

The round-trip protocol reuses the metrics above; what changes is the *anchor*.

- Odd steps (DE→EN) are scored against the **human English reference**.
- Even steps (EN→DE) are scored against the **original German source**.

Both anchors are fixed for all twenty passes. Step 1 is therefore identical to the
ordinary single-pass evaluation, which makes the first point of every curve directly
comparable to the main benchmark.

**This is not back-translation-as-reference.** That practice — translating output back
and comparing to the source without a human reference — is discredited because it
measures a model's self-consistency and rewards a model that makes the same mistake in
both directions. Here no output is ever used as a reference for another output.

Two evaluations are recorded per step:

- `clinical_vs_origin` — original source vs current output. **Cumulative drift.** This
  is what the reported `crit_rate` uses.
- `clinical_vs_input` — that hop's input vs that hop's output. **Isolates which step
  introduced an error.** Reported as `hop_crit_rate`.

Derived quantities used in the results:

```
Δ BLEU (c1→c10)     = single-pass BLEU − BLEU after ten cycles     (total quality lost)
front-loaded share  = (BLEU_c1 − BLEU_c2) / (BLEU_c1 − BLEU_c10)   (how much went in cycle 1)
```

The front-loaded share averages 77% across the twelve systems, which is the evidence for
the convergence claim in [05-results.md](05-results.md).

---

## 4. What is deliberately *not* used

**Reference-free quality estimation (COMET-Kiwi, XCOMET-QE) as a safety signal.**
Documented as failing on exactly this task: Mehandru et al. (EMNLP 2023) ran a physician
study on Emergency Department discharge instructions and found that QE improved
*appropriate reliance* on MT but that **back-translation helped physicians detect more
clinically harmful errors that QE alone missed**. A single scalar "quality" score does
not separate a fluent paraphrase from a fluent lie, and it is the second that harms
patients. That finding is also the strongest external support for the round-trip
protocol used here.

**BLEU as the deciding metric.** See §0.

**A single aggregate "quality score" combining both layers.** Any weighting would be
invented, and the disagreement between the layers *is* the result — collapsing it would
destroy the finding.

---

## References

- Papineni et al. (2002), *BLEU: a Method for Automatic Evaluation of Machine Translation*, ACL.
- Popović (2017), *chrF++: words helping character n-grams*, WMT.
- Snover et al. (2006), *A Study of Translation Edit Rate with Targeted Human Annotation*, AMTA.
- Post (2018), *A Call for Clarity in Reporting BLEU Scores*, WMT. (sacreBLEU)
- Lommel et al., *Multidimensional Quality Metrics (MQM)*, [themqm.org](https://themqm.org/error-types-2/the-mqm-scoring-models/).
- Mehandru, Agrawal, Xiao, Khoong, Gao, Carpuat & Salehi (2023), *Physician Detection of
  Clinical Harm in Machine Translation: Quality Estimation Aids in Reliance and
  Backtranslation Identifies Critical Errors*, EMNLP. [arXiv:2310.16924](https://arxiv.org/abs/2310.16924)
