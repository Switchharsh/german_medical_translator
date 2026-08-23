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
Experiment 1 established that it has something to catch: 26% of glossary terms
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
```

Both job scripts preflight the corpus, the glossary (failing if fewer than 100
usable entries survive filtering), and any credential the chosen backend needs,
before requesting any work.

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

Measured across the run: of **386 injected terms the model adopted, 100 (26%)
are terms the human reference does not use.** The most frequent offenders are
exactly the high-frequency report vocabulary — `Beurteilung → assessment` (15,
where the reference says *evaluation*), `Indikation → indication` (5),
`Hinweis auf → suggestive` (4).

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
