# Experiments: dictionary injection, debate, and a review cascade

Two interventions on top of the benchmark in [05-results.md](05-results.md), both
aimed at the same finding: every system corrupts clinical content in 25–50% of
reports, and surface metrics do not see it. The question is whether either
intervention moves the clinical layer.

The glossary both experiments use, and why it is the one it is, is
[08-terminology.md](08-terminology.md).

---

## Experiment 1 — does a dictionary help?

**Command:** `medmt-eval glossary-run`
**Code:** [`inference/glossary_mt.py`](../src/medmt_eval/inference/glossary_mt.py)

### Design

One model, one prompt template, three arms. Only the terminology block differs.

| Arm | Block contents |
|---|---|
| `none` | *(nothing — the prompt has no trace of a glossary)* |
| `glossary` | the terms that occur in **this** document, source = target |
| `distractor` | the **same number** of terms, from the same glossary, that do *not* occur in this document |

The distractor arm is what makes this an experiment rather than a demonstration.
A glossary block makes the prompt longer, more structured and more
domain-flavoured, and any of those could improve output on its own. Comparing
`glossary` against `none` measures *the block*. Comparing `glossary` against
`distractor` measures *the terms*. Only the second comparison supports the claim
that a dictionary helps, and the two comparisons can disagree.

### The circularity guard

If the terms injected are the terms later scored, the terminology detector
improves by construction and measures copying, not translation. `--holdout F`
splits the glossary by concept into disjoint inject/score halves
(`Glossary.split`), so a gain on the scored half is generalisation.

The other three detectors — negation, laterality, number/measurement — are
independent of the glossary and need no such guard. They are the interesting
ones anyway: they carry every `critical` finding, while terminology findings are
`major` and contribute zero to the critical-error rate.

### What is recorded

Per (segment, arm): the hypothesis, all three surface metrics, every clinical
finding, and **which terms were actually injected** (`glossary_terms`), so any
result can be traced to the exact block the model saw.

### Reading the outcome

- `glossary` > `distractor` on the clinical layer → the dictionary helps.
- `glossary` ≈ `distractor` > `none` → the *prompt shape* helps; the terms are
  decorative. This is a real and reportable outcome and would not be visible
  without the control arm.
- No arm separates → the glossary is too thin (mean 3.5 terms/report — see
  [08-terminology.md](08-terminology.md)), and the honest conclusion is about
  the glossary, not about dictionaries.

---

## Experiment 2 — three specialists argue

**Command:** `medmt-eval debate`
**Code:** [`inference/debate.py`](../src/medmt_eval/inference/debate.py)

### Design

Three agents, each with a **different persona** and a **disjoint slice of the
glossary**, so they disagree for substantive reasons rather than by sampling
noise. The personas map one-to-one onto this project's critical detectors, so a
disagreement between agents is about something the evaluation actually measures:

| Agent | Responsible for | Corresponding detector |
|---|---|---|
| `anatomist` | laterality, structures, vertebral levels, modality convention | `laterality_*` |
| `safety` | negation, numbers, measurements, hedging strength | `negation_*`, `number_or_measurement_mismatch` |
| `linguist` | register, idiom, standard target terminology | `terminology_not_preserved` |

### Protocol

```
round 0   each agent translates independently, seeing only its own persona + glossary slice
round 1+  each agent sees every current proposal and revises its own,
          replying   CRITIQUE: … / TRANSLATION: …
final     a synthesis step produces the translation to sign off
```

Convergence is checked after every round on whitespace- and case-normalised
text; identical proposals end the debate early rather than burning turns on
agreement. An agent that returns an empty `TRANSLATION` keeps its previous
proposal — a formatting slip must not blank a good translation.

### Scoring

The debate output **and every agent's round-0 solo proposal** are scored on the
same scale. This is the comparison that matters: a debate that lands below its
own best participant is a real outcome, and without the solo baselines it would
look like a success.

Recorded per document: the full transcript (every critique and every revision),
the round at which it converged, and the served model behind each agent.

### The caution that belongs in the design

**Consensus is not correctness.** Three models sharing a blind spot will agree
confidently and wrongly, and the protocol will report convergence. The
`BWK 12` → `L12` error from [05-results.md](05-results.md) — three independent
models rendering a thoracic vertebra as lumbar, with no detector firing — is
precisely the shape of failure this method cannot fix and will actively
disguise. The transcript and per-round scores exist so that can be checked
rather than assumed.

A related risk is cost: three agents × (1 + N rounds) + synthesis is roughly
`3N + 4` model calls per document, against one for a plain translation.

### Model choice, and a constraint discovered while building

The debate defaults to **local** models — `Qwen/Qwen3.5-4B`,
`tencent/Hy-MT2-7B`, `google/translategemma-4b-it`: three different families,
~32 GB of bf16 weights, one 80 GB card.

The hosted gateway is not usable for this today. Its catalogue changed between
the round-trip runs and now:

| Requested | Served | Usable |
|---|---|---|
| `DeepSeek-V4-Flash` | `deepseek-ai/deepseek-v4-flash-0731` | ✅ |
| `Qwen3.8-27B` | `meta/muse-glimmer-30b` | ❌ substituted |
| `Qwen3.8-35B-A3B` | `nvidia/nemotron-3.5-lightning-30b-a3b` | ❌ substituted |
| `Kimi-K2.6` | `thinkingmachines/inkling` | ❌ substituted |
| `glm-5.2`, `MiniMax-M3` | — | ❌ withdrawn |

Only one hosted model still routes honestly, so a three-model hosted debate
cannot be attributed to named participants. The substitution guard caught every
case; this is what it is for.

---

## Experiment 3 — an asymmetric review cascade

**Command:** `medmt-eval cascade`
**Code:** [`inference/cascade.py`](../src/medmt_eval/inference/cascade.py)

### Why not another debate

Experiment 2 was symmetric: three peers proposed in parallel and revised each
other. It moved BLEU, left the clinical error rate untouched, and — because every
agent revised simultaneously — could not say which participant was responsible
for anything.

A cascade fixes the attribution problem. Each stage sees the work of the stage
before it, is asked for something different, and is **scored separately**, so
stage *n* is credited only with what it changed relative to stage *n−1*.

### Protocol

```
1  translate       Hy-MT2-7B drafts the translation from the source
2  terminologise   Qwen3.5-4B WITH the RadLex glossary sees source + draft
                   and corrects what it believes is wrong
3  arbitrate       Qwen3.5-4B as a German radiologist, WITHOUT the glossary,
                   weighs the disagreement and issues the final text
```

**Only stage 2 is shown the dictionary, and that asymmetry is the experiment.**
Stage 3 exists to catch corrections the glossary got *wrong* — and the rerun of
Experiment 1 established that it has something to catch: 21% of glossary terms
the model adopted were terms the human reference does not use. A reviewer with
the same glossary would inherit the same bias.

### What the design has to answer for

- **Later stages can make things worse.** An arbiter free to rewrite can undo a
  correct fix. Scoring every stage makes that visible instead of hiding it behind
  a final number; `bleu_delta` and `critical_delta` report it per stage.
- **A reviewer shown a draft anchors on it.** Stages 2 and 3 are shown the source
  *first* and the draft second, and are asked to justify changes against the
  source rather than to prefer the draft.
- **A stage that changes nothing looks identical to a stage that changes
  everything** in the score table alone, so `changed_fraction` records how often
  each stage actually edited its input.
- **An empty `TRANSLATION:` means "no change", not "delete the report"** — a
  formatting slip must not blank a good translation, so the previous stage's text
  is carried forward.

### Results — first run retracted

The first run (job 4077545) produced a spectacular-looking result:

| stage | BLEU | Δ | crit% | Δ |
|---|---|---|---|---|
| 1 translate (Hy-MT2-7B) | 47.31 | — | 32.5 | — |
| 2 terminologise (+glossary) | 29.41 | −17.90 | 55.0 | +22.5 |
| 3 arbitrate (no glossary) | 50.31 | +20.90 | 35.0 | −20.0 |

Read naively that is a perfect confirmation of Experiment 1 — the glossary
devastates the translation, the glossary-free arbiter rescues it. **It is an
artefact and is retracted.**

Stage 2's mean output was **3,446 characters against 769** for the other stages,
because 17 of 40 replies were scored as their own commentary:

```
stage 2 : ISSUES:
- Changed "spondyloarthritis" to "spondylarthrosis" to match the source term …
```

Only **1 reply in 40** contained a `TRANSLATION:` marker at all. The 4B model
wrote its issues list and never reached the translation, and the parser's
fallback — "no marker, so the whole reply is the translation" — returned the
prose. The entire −17.9 BLEU is that.

Three fixes, in order of importance:

1. **The required reply format was reordered so `TRANSLATION:` comes first**, with
   the note second. Truncation now costs the optional field rather than the
   translation. The parser fix alone would have left the format fragile.
2. **The parser distinguishes three cases, not two**: marker present; note
   present but no translation (an explicit *failure*, carrying the previous
   stage's text forward); and bare reply. The note pattern also no longer
   swallows a translation that follows it.
3. **`parse_failures` is reported per stage**, so a stage scored on
   carried-forward text can never again pass as a measurement of that stage.

The lesson generalises beyond this bug. The same guard — count the replies you
could not parse, never treat them as clean — had been written into the LLM judge
([`metrics/llm_judge.py`](../src/medmt_eval/metrics/llm_judge.py)) an hour
earlier and was not carried across. And the evidence was already in the output:
a 4.5× jump in mean output length is not subtle. A structured-output stage needs
a parse-rate check before its scores are read at all.

### Results — rerun with the parser fixed

40 whole reports, 0 parse failures at every stage:

| stage | BLEU | Δ | crit% | Δ | changed |
|---|---|---|---|---|---|
| 1 translate (Hy-MT2-7B) | 47.31 | — | 32.5 | — | — |
| 2 terminologise (+glossary) | 46.78 | −0.52 | 37.5 | **+5.0** | 85% |
| 3 arbitrate (no glossary) | **51.02** | **+4.23** | 37.5 | **0.0** | 88% |

The retracted run's *direction* partly survives and its *magnitude* does not. The
glossary stage does raise the critical-error rate — by 5.0 points, not 22.5 — and
it costs 0.5 BLEU, not 17.9. Everything beyond that was the parser.

Two things the corrected run shows clearly:

- **The arbiter improves fluency and not safety.** It gains 4.23 BLEU, finishing
  above the MT baseline it was given, and fixes exactly zero clinical errors. It
  rewrote 88% of the documents to do it.
- **Net, the pipeline is not better than its first stage.** Against Hy-MT2 alone
  the cascade ends +3.71 BLEU and +5.0 points *worse* on critical errors. Three
  models and roughly three times the compute buy fluency and cost clinical
  fidelity.

This is the third protocol to show the same split: surface metrics move, the
clinical layer does not.

## Experiments 1 and 2 on the medical-text corpus

Both were rerun on the 109 reports reduced to findings and impression
([01-dataset.md](01-dataset.md)), which removes the acquisition preamble and with
it every `Beurteilung → assessment`-style report-component term.

### Dictionary — precision helps, but nothing reaches significance

109 reports, `qwen35-4b`, two banks:

| bank | arm | BLEU | chrF++ | crit% | terms/doc |
|---|---|---|---|---|---|
| — | `none` | **53.28** | **72.60** | 14.7 | 0.0 |
| merged (47,997) | `glossary` | 48.42 | 69.62 | 13.8 | 11.5 |
| merged | `distractor` | 51.09 | 71.29 | 13.8 | 11.5 |
| findings-only (2,777) | `glossary` | 50.47 | 70.99 | **11.9** | 1.6 |
| findings-only | `distractor` | 50.92 | 71.13 | 12.8 | 1.6 |

McNemar, paired (n=109):

| bank | comparison | discordant | p |
|---|---|---|---|
| merged | `distractor` → `glossary` | 1 / 1 | 1.000 |
| findings-only | `distractor` → `glossary` | 1 / 0 | 1.000 |
| findings-only | `none` → `glossary` | 3 / 0 | 0.250 |

**Precision beats coverage, on the evidence available.** The BLEU penalty against
the distractor shrinks from 4.01 (whole reports, merged bank) to 2.67 (medical
text, merged) to 0.45 (medical text, findings-only). Injecting 1.6 well-chosen
terms costs almost nothing; injecting 11.5 mixed ones costs 2.7 BLEU.

**And for the first time the relevant glossary is not behind the control on the
clinical layer** — findings-only scores 11.9% against the distractor's 12.8%, and
`none` → `glossary` fixes 3 documents and breaks 0. That is the only directional
evidence in this project that a dictionary helps clinically.

It is also **not significant** (p = 0.250, three documents out of 109), and it
should not be reported as a positive result. What it justifies is a properly
powered rerun of exactly this arm, not a conclusion.

### Debate — worse than its best participant

40 medical-text reports:

| variant | BLEU | chrF++ | crit% |
|---|---|---|---|
| debate | 49.13 | 70.04 | 17.5 |
| `solo:anatomist` (Qwen3.5-4B) | **49.32** | **70.11** | 17.5 |
| `solo:safety` (Hy-MT2-7B) | 47.97 | 69.44 | **15.0** |
| `solo:linguist` (medgemma-1.5-4b-it) | 41.70 | 64.55 | 35.0 |

On whole reports the debate beat every participant on every surface metric. On
medical text it beats none of them: it is 0.19 BLEU *below* the best solo agent
and 2.5 points *worse* on critical errors than `solo:safety`.

The likely reason is visible in the table. `solo:linguist` collapses on this
corpus — 41.70 BLEU and a 35% error rate against 15% for the best agent — and a
consensus mechanism has no way to discount a participant that is reliably wrong.
Removing the boilerplate removed the easy, formulaic text that was masking the
weakest model, and the debate inherited its errors.

**Consensus is not correctness**, which was written into the design before this
run. It is now measured: three models averaging toward the worst of them.

Again 0% of documents converged in two rounds.

## Experiment 4 — a glossary mined from the corpus itself

Experiments 1 and 3 both failed in the same place, and
[08-terminology.md](08-terminology.md) identified the mechanism: 21% of the
injected terms the model adopted are terms the human reference does not use,
because **an ontology's preferred label is not report register**. RadLex has the
right concepts and the wrong wording.

A glossary read off the reports themselves cannot have that defect. Its terms are
report register by construction. This experiment tests whether that is enough.

### The cost, stated first

Mining terminology from PARROT **invalidates every reference-based metric on this
corpus**. BLEU, chrF++, TER and COMET score against the same English
translations the terms were mined from, so injecting mined wording and then
measuring agreement with it measures leakage. Splitting by document does not
repair it — the same radiologists wrote both halves, so house style crosses the
split.

So the split here buys something narrower than it looks: it guarantees no scored
document contributed a term, which removes *direct* memorisation, and leaves
register leakage untouched. The primary read is therefore the source-referenced
layer — the negation, laterality, number and measurement detectors, which never
look at the reference. Reference-based scores are reported and labelled
contaminated. This is the rule in
[03-methods.md](03-methods.md#contamination-discipline) applied rather than
broken: the corpus measures terminology and does not generate it, *except* in
this one experiment, which exists to measure what that exception costs.

### Method

196 train / 100 val documents, stratified by modality, seed 20260928
(`data/splits/parrot_de_mine_v1.json`). Terms come from the train half; the
experiment runs on the val half. Built by
[`scripts/build_glossary_mined.py`](../scripts/build_glossary_mined.py).

Three decisions were forced by measurement rather than chosen up front, and each
one is the reason the output is usable at all.

**Align at the sentence level, not the document level.** The first attempt scored
Dice coefficients over whole documents and produced `Herz -> mediastinum`,
`Milz -> kidneys` and `Leber -> pancreas`. The cause is structural, not
statistical: radiology anatomy terms *systematically co-occur* — heart,
mediastinum, pleura and lungs appear in every chest CT — so a document-wide
window cannot separate "is the translation of" from "appears in the same report
as". Narrowing the window to a sentence removed every such swap. Sentences are
paired by index in the 49% of train documents whose sentence counts match on both
sides, with a 0.5–2.0 length-ratio guard. This is much cruder than `fast_align`
or `eflomal`; it is adequate here only because PARROT's two sides are direct
translations, so order is preserved.

**Let the English side be a phrase.** One German compound routinely maps to an
English multiword. Unigram-only alignment truncated `Pleuraerguss` to `pleural`
and `Perikarderguss` to `pericardial`, and injecting a truncation is worse than
injecting nothing. Candidates are 1–3 grams and the longest within 10% of the
best Dice wins.

**Require a mutual best match.** Without it, `available for comparison` was
simultaneously the best match for four different German words. A German word's
best English candidate must also have that word as its own best German candidate.

That yields 88 pairs. The last step is the one that turned the experiment into a
different finding.

### Screening against what the models already do

For each mined pair, count how often three baseline systems' *existing*
train-split translations already contain the English side. A pair at 97% is
correct and useless — the glossary slot is spent telling the model something it
already knows, which is precisely the failure mode of the frequency-weighted
RadLex matches in Experiment 1, where the most-matched term was `indium`.

Splitting the 87 screened pairs at 60% separates them cleanly, and the two halves
are cleanly *opposed*:

| bank | terms | models already produce it | mining precision (hand audit) |
|---|---|---|---|
| `freq` ∖ `hard` (already > 60%) | 59 | 68% | ~92% (54/59) |
| `hard` (already ≤ 60%) | 28 | 29% | ~39% (11/28) |

Precision and headroom are **anti-correlated**, and the mechanism is not a tuning
problem. A German term with a stable one-to-one English rendering is easy for the
aligner *and* easy for the model, for the same reason — the mapping is
context-free. A term the models get wrong is one whose English form depends on
context, and a context-free glossary entry cannot express it:

- `frei -> well aerated` is correct inside *Mastoidzellen frei* and wrong
  everywhere else.
- `Verschattung -> lung field` is simply wrong (*Verschattung* is an opacity).
- `groß -> normal in size`, `Frei -> cells are well` — fragments of collocations.

Against which, the correct entries in the hard bank are exactly the register
differences an ontology could never supply:

| German | mined English | models produce it | what it is |
|---|---|---|---|
| `Beurteilung` | conclusion | 0% | models write "Assessment"; radiologists write "Conclusion" |
| `Ebenen` | two views | 9% | *in zwei Ebenen* is an idiom, not compositional |
| `belüftet` | aerated | 22% | register |
| `Abklärung` | evaluation | 37% | register |
| `Raumforderung` | mass | 41% | register |
| `Kontrastierung` | enhancement | 62% | register |

Both banks are shipped **exactly as mined, with no hand editing**, so the
precision of the mining is part of what the experiment tests rather than
something corrected out of it.

Each bank runs against the standard three arms — `none`, `glossary`,
`distractor` — on the 100 held-out documents, with `qwen35-4b`, the same backend
as Experiments 1 and 3. The terminology detector keeps its own external bank
(`radiology_en_de_starter.csv`), disjoint from the injected glossary, so it is
not scored on the terms being injected.

### The ceiling, computed before reading the result

The overview asserts that this project's failures are "concentrated in numbers
and measurements, where terminology and committee methods cannot reach by
construction". That was a qualitative claim. On the val split it can be made
exact, and it should be, because it bounds what Experiment 4 could possibly
achieve before any score is read.

`qwen35-4b` on the 100 held-out documents: 28 carry a critical error, from 35
critical findings.

| detector | findings | share |
|---|---|---|
| `number_unit_parser` | 23 | 65.7% |
| `laterality_lexicon` | 11 | 31.4% |
| `segment_negation_cues` | 1 | 2.9% |

**16 of the 28 error documents fail on numbers alone.** No terminology
intervention can reach them — a bilingual term pair says nothing about whether
`7646,06 µGym²` survived. That caps any glossary at the remaining 12
documents, or 12 points of the 28% rate.

The 12 do not survive inspection either. Their non-numeric failures are 5 *dropped*
laterality findings (the expected side is absent from the output entirely, and a
glossary cannot supply a word the model declined to emit), 6 flipped-or-other,
and 1 negation. And the lexical mapping is not what is failing: `rechts → right`
and `links → left` are in the mined bank precisely because they are frequent, and
the screening step measures that the baselines already produce them 98% and 96%
of the time. The laterality errors are not mistranslations of *rechts*; they are
omissions elsewhere in a long document.

So the honest expectation is a **ceiling near zero on the clinical layer** — at
most 6 reachable findings across 100 documents, where one document is one point,
which is inside the noise this design can resolve. Experiment 4 is therefore not
a test of whether a mined glossary fixes clinical errors on PARROT; the ceiling
analysis answers that, and the answer is that it cannot, whatever the bank
contains. What the run can still establish is narrower and worth having:

1. whether mined terms are **adopted** at all, and at what rate against RadLex's
   80.8%;
2. whether the adopted-but-absent-from-reference rate falls below RadLex's 20.8%,
   which is the direct test of the register hypothesis and the reason this
   experiment exists;
3. whether the glossary block does **harm** — the distractor arm, and the 39%
   mining precision of the hard bank, make injected error a live possibility
   rather than a hypothetical;
4. how much reference-based score moves purely from leakage, which is a
   measurement of the contamination itself and useful as a caution for anyone
   reading corpus-mined glossary results elsewhere.

Recording this before the numbers arrive is the point. A null result that was
predicted from a computed ceiling is evidence about the corpus; the same null
reported afterwards reads as a failed experiment.

## Running them

```bash
# Experiment 1 — one arm-triple per model
BACKEND=local:Qwen/Qwen3.5-4B  sbatch scripts/experiments/glossary.slurm
BACKEND=api:DeepSeek-V4-Flash  sbatch scripts/experiments/glossary.slurm

# Experiment 2 — three local personas, two rounds
sbatch scripts/experiments/debate.slurm

# Experiment 3 — the review cascade
sbatch scripts/experiments/cascade.slurm

# with the anti-circularity split active
EXP_HOLDOUT=0.5 BACKEND=local:Qwen/Qwen3.5-4B sbatch scripts/experiments/glossary.slurm

# Experiment 4 — mine the banks from the train split, then run both on val.
# The builder refuses to overwrite, so --out-suffix is the version.
python scripts/build_glossary_mined.py --out-suffix v1
BACKEND=local:Qwen/Qwen3.5-4B sbatch scripts/experiments/glossary_mined.slurm
```

Every job script preflights the corpus, the glossary and any credential the
chosen backend needs before requesting work. Experiment 1 fails if fewer than 100
usable entries survive filtering; Experiment 4 uses a floor of 20, since the hard
bank is 28 entries by construction, and adds one check the others do not need —
it intersects the val document ids with the mining half and aborts if they
overlap. That assertion is the entire methodological basis for the run, so it is
verified at submission rather than assumed.

## Results

Run 2026-08-21, `results/experiments/run_20260821_221738/`. Experiment 1:
`qwen35-4b`, 40 reports, all three arms. Experiment 2: three local agents,
20 reports, 2 rounds.

### Experiment 1 — the dictionary does nothing; the *block* might

| arm | BLEU | chrF++ | TER | crit% | terms/doc |
|---|---|---|---|---|---|
| `none` | **51.41** | **72.70** | **36.09** | 35.0 | 0.0 |
| `glossary` | 48.42 | 70.76 | 38.52 | **27.5** | 3.0 |
| `distractor` | 48.67 | 70.91 | 37.69 | **27.5** | 3.0 |

McNemar on critical errors, paired by document (n=40):

| comparison | discordant | p |
|---|---|---|
| `none` → `glossary` | 4 / 1 | 0.375 |
| `none` → `distractor` | 3 / 0 | 0.250 |
| **`distractor` → `glossary`** | **1 / 1** | **1.000** |

**Nothing here is significant, and the one comparison that matters is exactly
null.** `glossary` and `distractor` differ by 0.25 BLEU and by a single
discordant document in each direction. The terms the model was shown made no
measurable difference; an equally long block of terms drawn from the *same
glossary* that do not occur in the document produced the same result.

What did change is the same in both arms: adding any glossary block cost about
3 BLEU and coincided with a 7.5-point drop in critical-error rate. That drop is
three documents and p = 0.375 — suggestive at best. If it is real, the mechanism
is the prompt block, not the terminology.

Ten of the fourteen `none`-arm failures fail in all three arms.

**This is the outcome the control arm exists to detect.** Without a distractor
arm, `none` → `glossary` (35.0% → 27.5%) reads as a 21% relative reduction in
clinical errors from adding a dictionary, and would have been reported as such.

Why it is null is not yet established, and the two candidate explanations have
very different implications:

1. **The glossary is too thin.** 3.0 terms per document, and every term matched
   is one a competent model already translates correctly (`Lymphknoten`,
   `Pleuraerguss`, `Pneumothorax`). The dictionary supplies no information the
   model lacked.
2. **Injected terminology does not reach the failure modes.** Every critical
   finding in this run is a number, a negation or a laterality error. A glossary
   is the wrong instrument for all three.

Explanation 2 is supported by the finding counts: `number_or_measurement_mismatch`
accounts for 12 of 14 findings in the `none` arm, 8 of 11 with the glossary.
A term bank cannot fix a dropped measurement.

### Experiment 1, rerun — a *bigger* glossary made it worse

The first run's null was confounded: at 3.0 terms per document it could not
separate "dictionaries do not help" from "that dictionary was too thin". The
RadLex OWL ([08-terminology.md](08-terminology.md)) raised the working bank to
47,997 entries and 13.6 injected terms per document — 4.5× more — so the rerun
can separate them.

Same model, same 40 reports, same three arms:

| arm | BLEU | chrF++ | TER | crit% | terms/doc |
|---|---|---|---|---|---|
| `none` | **51.41** | **72.70** | **36.09** | 35.0 | 0.0 |
| `glossary` | 43.00 | 67.18 | 43.10 | 32.5 | 13.6 |
| `distractor` | 47.01 | 69.62 | 38.93 | **25.0** | 13.6 |

**The relevant glossary is now worse than the irrelevant one on every metric** —
4.0 BLEU below the distractor and 7.5 points worse on critical errors. More
terminology did not rescue the intervention; it inverted it.

McNemar, `distractor` → `glossary`: 0 documents fixed, 3 broken, p = 0.250. Not
significant at n = 40, but the direction is unambiguous and it is consistent
across all four metrics.

#### Why: the ontology's preferred label is not report register

The mechanism is visible in single documents. `parrot-1124`, MRI lumbar spine:

| | text |
|---|---|
| reference | MRI of the **lumbar spine** |
| `none` | MRI of the **lumbar spine** ✅ |
| `glossary` | MRI of the **lumbar vertebral column** ❌ |

The injected term was `Lendenwirbelsäule → lumbar vertebral column`, which is
RadLex's preferred label and is not wrong — it is simply not what a radiologist
writes. The no-glossary arm already had it right, and the glossary overrode a
correct translation with a formally correct one.

Measured across the run: of 542 injected terms, the model adopted 438, and
**91 of those 438 (20.8%) are terms the human reference does not use.** Counting
every injection, matching case- and punctuation-insensitively; the rate is
16-28% across all three Experiment-1 runs and every counting variant tried
(per-injection or deduplicated, raw or normalised match), so the conclusion does
not depend on the definition. An earlier draft of this chapter reported
"386 adopted, 100 (26%)", which does not reproduce from the stored results under
any of those variants and has been corrected; the per-term detail below was
verified against them and reproduces exactly.

The most frequent offenders are exactly the high-frequency report vocabulary --
`Beurteilung → assessment` (15), `Indikation → indication` (5),
`Hinweis auf → suggestive` (4).

`Beurteilung` also shows *why* a single preferred label cannot win here. It is a
section header, so its rendering can be counted exactly by aligning the German and
English `impression`-role headers: across the 88 reports carrying both, the
English side is *conclusion* 50, *impression* 32, *assessment* 6. RadLex's label
is `assessment` -- the **rarest** of the three, 7% of usage -- which is the
register failure in one number. But no label would have been safe: even
*conclusion*, the modal form, is wrong on 43% of reports. (An earlier draft of
this paragraph asserted the reference says *evaluation*, and a later one gave a
five-way split including *findings*; both came from searching reference text
rather than aligning headers, and neither is correct. See
[the measurement error recorded in Experiment 4](#experiment-4--the-corpus-cannot-agree-with-itself).)
[Experiment 4](#experiment-4--a-glossary-mined-from-the-corpus-itself) mines
`Beurteilung → conclusion` from the corpus instead and finds the models produce
it 0% of the time -- the same disagreement reached from the other side.

So the glossary is doing precisely what it was told to do, and that is the
problem. "Where the source uses the term on the left, the translation must use
the term on the right" is the wrong instruction when the right-hand side is an
ontology label chosen for conceptual precision rather than for how reports are
written.

#### What this changes

- **Prompt-injected terminology is not a free improvement.** It has a real cost,
  it scales with how much you inject, and the cost is invisible without a
  distractor arm — `none` → `glossary` alone still reads as a 2.5-point
  improvement in critical errors.
- **The failure is in the term selection, not the mechanism.** A glossary
  restricted to terms the model actually gets *wrong*, or one whose target side
  is report register rather than ontology label, is untested and might well help.
- **This is the sharpest possible argument for the DeepL comparison**
  ([07-metric-roadmap.md](07-metric-roadmap.md)): DeepL applies a glossary inside
  the decoder rather than as a prompt instruction, so it separates "dictionaries
  do not help" from "*prompt-injected* dictionaries do not help". After this
  result that is no longer a nice-to-have.

### Experiment 2 — the debate improves fluency and not safety

| variant | BLEU | chrF++ | TER | crit% |
|---|---|---|---|---|
| **debate** | **51.62** | **73.26** | **34.31** | 40.0 |
| `solo:anatomist` (Qwen3.5-4B) | 50.59 | 72.80 | 35.41 | 40.0 |
| `solo:safety` (Hy-MT2-7B) | 48.87 | 70.48 | 37.87 | 40.0 |
| `solo:linguist` (medgemma-1.5-4b-it) | 46.61 | 68.61 | 39.07 | 40.0 |

The debate beats every participant on every surface metric — +1.03 BLEU over the
best solo agent, +5.01 over the worst. On the clinical layer it beats none of
them: **all four variants sit at exactly 40.0%**, and McNemar against each solo
agent gives p = 1.000 (0/0, 2/2 and 1/1 discordant pairs). Against the strongest
agent the failing document set is *identical*.

Two rounds of argument between three specialists, each explicitly briefed on one
class of clinical error, moved the clinical error rate by zero.

**All eight debate failures are `number_or_measurement_mismatch`.** Not one is a
negation or laterality error. The `safety` persona's entire brief is numbers and
measurements, and the failures that survive are all numbers.

**No document converged.** In 20 of 20 the agents were still disagreeing after
two rounds, so the synthesis step is doing the real work rather than a consensus
emerging.

### What the transcripts show, and why it matters more than the table

The agents are not going through the motions. From the first document:

> **[anatomist]** The [safety] translator incorrectly translated "NBKS"
> (Nierenbecken) as "lymph node" and "LWK" (Lumbalwirbelsäule) as "L5", creating
> dangerous anatomi…
>
> **[safety]** [anatomist] incorrectly translated "NBKS" as "renal pelvis"
> instead of "renal pelvic sinus" and changed "suspect vergrössert" to "possibly
> intermitten…

These are real, specific, German-radiology-abbreviation disputes — `NBKS`
(Nierenbeckenkelchsystem) and `LWK` (Lendenwirbelkörper) are exactly the class of
error that produced the `BWK 12` → `L12` failure in
[05-results.md](05-results.md), and **exactly the class no detector can see.**

So the debate is generating clinically substantive review that the evaluation
cannot score, while failing to move the one thing the evaluation *can* score.
That is a statement about the instrument as much as about the method, and it is
the strongest argument yet for the metric work in
[07-metric-roadmap.md](07-metric-roadmap.md): an LLM-as-judge with a clinical
rubric would read these critiques as findings.

### Experiment 4 — the corpus cannot agree with itself

`qwen35-4b`, 100 held-out documents, 87-term frequency bank
(`results/experiments/4399511/`). The hard-bank arms are reported below it.

| arm | crit% | BLEU* | chrF* | TER* | terms/doc | adopted | adopted but absent from reference |
|---|---|---|---|---|---|---|---|
| `none` | 22.0% | 49.01 | 71.19 | 37.20 | 0 | — | — |
| `glossary` | 19.0% | 46.85 | 69.69 | 38.42 | 9.3 | **90.6%** | 17.6% |
| `distractor` | 17.0% | 45.75 | 68.57 | 39.80 | 9.3 | 8.5% | 48.1% |

\* reference-based and contaminated: the terms were mined from this corpus'
English side. See [the cost, stated first](#the-cost-stated-first).

**The clinical layer is null, as the ceiling required.** Glossary moves 22 → 19
documents, distractor moves it to 17, and the distractor is again the best arm.
Paired on the same documents, glossary-only errors 4 against distractor-only 2,
McNemar exact p = 0.688. Computed on this run's own baseline the reachable
ceiling was 7 documents of 100 — 15 of the 22 error documents fail on numbers
alone — so a 3-document movement inside a 7-document ceiling is exactly the
"predicted null" the previous section registered, not a new finding.

**Adoption was not the problem.** At 90.6% this is the highest adoption rate in
the project, against RadLex's 80.8% and Wikidata's 86.0%. Mined terms *are*
report register and the model takes them up more readily than ontology labels.
That closes the loophole that would otherwise make every null here
uninterpretable: the terms were used, and using them did not help.

**And the register hypothesis turns out to explain almost none of it.** This was
the experiment's actual question. If Experiment 1 failed because an ontology
label is not the wording a radiologist writes, then a bank read off the reports
themselves should drive the adopted-but-absent-from-reference rate toward zero.
It went from 20.8% to **17.6%** — about three points. Roughly seventeen points
survive mining the terminology out of the corpus that is scoring it.

Those seventeen points are not a defect of the mining. They are the corpus
disagreeing with itself. The largest single contributor is the term this
experiment was most pleased to find:

`Beurteilung` is a section header, so its rendering can be measured exactly
rather than estimated: take the German section the classifier assigns the
`impression` role, take the English section it assigns the same role, and read
both literal headers. Across the 92 reports carrying that role on both sides:

| `Beurteilung` is rendered | whole corpus (n=88) | held-out half (n=31) |
|---|---|---|
| *conclusion* | 50 (**57%**) | 16 (**52%**) |
| *impression* | 32 (36%) | 14 (45%) |
| *assessment* | 6 (7%) | 1 (3%) |

`Beurteilung → conclusion` accounts for 39 of the orphaned adoptions on its own,
and the mining chose correctly: *conclusion* is the modal form. It is still
orphaned on **roughly half** the documents, because the corpus splits almost
evenly between *conclusion* and *impression*. **No single term pair can exceed
about 57% on this term**, whichever is chosen, so the mining did not err — a
single right answer does not exist.

Two other mined terms show the same pattern more mildly, counted by searching the
aligned val sentence (these are not headers, so the exact method above does not
apply): `Raumforderung → mass` against *mass* 8 / *lesion* 3, and
`Thorax → thorax` against *thorax* 14 / *chest* 4.

This also sharpens what RadLex got wrong. Its label for this concept was
*assessment* — 7% of actual usage, the rarest of the three forms in play — while
mining from the corpus yields *conclusion* at 57%. That gap is the register
effect, and it is large per-term. What it buys in aggregate is small (20.8% →
17.6%) only because both banks agree on the many terms that were never in doubt.

This is not sloppiness in PARROT either. English radiology has no settled word
for that heading, and **this project already encodes that fact**:
[`data/sections.py`](../src/medmt_eval/data/sections.py) classifies
`Impression | Assessment | Conclusion | Summary | Interpretation` as five surface
forms of one section role, and `Beurteilung | Zusammenfassung | Fazit |
Schlussfolgerung` as its German counterparts. One module in this repository
treats them as interchangeable while the metric layer scores choosing among them
as a terminology error. The section classifier was right.

**What this does to the reference-based metrics.** Two readings, and the second
matters more.

The narrow one: injecting the mined glossary *lowered* BLEU against no glossary
at all, 49.01 → 46.85, −2.16 sentence-level, paired bootstrap p = 0.018. That is
in the teeth of the contamination, which biases this comparison *toward* the
glossary — the terms were lifted from the same English text BLEU scores against,
and the leakage still did not cover the cost of the prompt block. Against the
distractor the mined terms are worth +1.10 BLEU (p = 0.11), so relevant terms do
beat irrelevant ones, within noise. Compare Experiment 1's −4.01 against
distractor: mining removes most of the ontology's surface penalty without buying
anything clinical.

The broad one is the more useful result of this experiment, and it is about the
instrument rather than the intervention. If the corpus renders its most frequent
term four ways, then BLEU, chrF++ and TER **cannot distinguish "wrong term" from
"a correct synonym this particular reference did not use"** — and on the
vocabulary a glossary targets, that ambiguity is the common case, not the edge
case. A system writing *impression* is penalised against a reference writing
*conclusion*, and both are what a radiologist writes. That is a measured argument
for the direction in [07-metric-roadmap.md](07-metric-roadmap.md) — multiple
references, or source-referenced judging that never needs the reference's word
choice — and it is stronger than the arguments already recorded there because it
is a number rather than a concern.

#### The hard bank closes the loop

The 28-term hard bank is the arm that was supposed to matter: the only terms with
measurable headroom, at 29% baseline agreement against the freq bank's 68%.

| arm | crit% | BLEU* | chrF* | TER* | terms/doc | adopted | adopted but absent from reference |
|---|---|---|---|---|---|---|---|
| `none` | 22.0% | 49.01 | 71.19 | 37.20 | 0 | — | — |
| `glossary` | 20.0% | 46.78 | 69.26 | 38.95 | 3.7 | 83.4% | **43.6%** |
| `distractor` | 19.0% | 46.90 | 69.58 | 39.53 | 3.7 | 10.4% | 82.1% |

Two numbers finish the argument.

**43.6% of adopted hard-bank terms are absent from the reference**, against 17.6%
for the freq bank. The hand audit of this bank put its mining precision near 39%;
the metric independently reports 56% of adoptions landing in the reference. Two
unrelated estimates of the same defect agree, which is the strongest evidence
available here that the audit was not just pessimism.

**The hard terms are worth nothing over random ones: −0.12 BLEU against the
distractor, paired bootstrap p = 0.45.** Compare +1.10 (p = 0.11) for the freq
bank. The subset with headroom carries no measurable signal at all, while the
subset that carries signal had no headroom. The anti-correlation predicted from
the screening step is therefore not an artefact of how the banks were split — it
survives end to end, in the arm comparison that was designed to isolate the terms
from the prompt block.

The clinical layer is null for a third time (22 → 20 documents, McNemar exact
p = 1.000 against the distractor), as the ceiling required, and the distractor is
again nominally the best arm.

So the two banks bracket a trade-off with no useful point on it:

| | freq bank | hard bank |
|---|---|---|
| terms | 87 | 28 |
| baselines already produce them | 68% | 29% |
| mining precision (audit) | ~92% | ~39% |
| adopted but absent from reference | 17.6% | 43.6% |
| BLEU* vs distractor | +1.10 (p = 0.11) | −0.12 (p = 0.45) |
| critical errors vs distractor | p = 0.688 | p = 1.000 |

**A determinism check, obtained for free.** Each bank re-translated the same 100
documents for its own `none` arm, so the run contains two independent executions
of an identical configuration at temperature 0. All **100 of 100 outputs are
byte-identical**. That was not designed as a check and is the reason to record it:
the paired tests above assume the pipeline is deterministic, and this run
demonstrates it rather than assuming it. The redundant arm cost 67 GPU-minutes;
next time `--arms glossary distractor` plus one shared baseline would save that,
at the cost of this evidence.

**A measurement error worth recording, because it inverted a conclusion.** The
`Beurteilung` row above was first computed by searching each aligned val sentence
for candidate words, which returned *impression* 13, *conclusion* 10, *findings*
9, *assessment* 7 and a "33% ceiling". That was wrong in a specific way: a report
contains a `Findings:` header *and* an impression header, so scanning the whole
sentence neighbourhood counted another section's header as a rendering of this
one. *findings* and *evaluation* are not renderings of `Beurteilung` at all. The
header-aligned count above supersedes it, and it moves the ceiling from 33% to
about 57% — the argument survives, the number did not. The general lesson is the
one already in [04-experiments.md](04-experiments.md): a loose proxy that happens
to support the expected conclusion is the most dangerous kind, and this one was
caught only by measuring it a second way on purpose.

### Limits

- **n = 40 and n = 20.** Critical-error rate moves in 2.5- and 5-point steps.
  These runs can detect a large effect and nothing smaller. The null results are
  "no large effect", not "no effect".
- **One model in Experiment 1** (`qwen35-4b`). A stronger model may use a
  glossary differently.
- **`--holdout` was not used** in this run, so the terminology detector was free
  to reward injected terms. It did not — terminology findings are `major` and
  contribute zero to `crit%` — but a run that reports terminology should set it.
- The debate cost roughly `3N + 4` model calls per document against 1 for a
  plain translation: 43 minutes for 20 documents versus 21 minutes for 120
  translations in Experiment 1.
