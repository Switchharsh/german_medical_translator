# Overview

## Question

**Does German↔English medical translation need a specialised model?**

The honest way to answer that is not to fine-tune something and report that it beat a
generic baseline. It is to benchmark what already exists — generic MT, specialised
medical MT, small and large LLMs, hosted frontier APIs — and only build if a real gap
survives measurement. This project is that benchmark.

The answer turns on *what you measure*. Ranked by BLEU, several systems look
interchangeable. Ranked by whether clinical facts survive translation, they do not.

## Approach

Two layers of evaluation over the same outputs:

1. **Surface quality** — sacreBLEU, chrF++, TER against a human reference.
   Standard, comparable to the literature, and by itself misleading here.
2. **Clinical information loss** — rule-based detectors for the four failure modes
   that change patient meaning: dropped/introduced negation, missing or flipped
   laterality, altered numbers and measurements, and unpreserved terminology.

On top of both, an **iterative round-trip protocol**: DE→EN→DE→EN for ten cycles,
scoring every one of the twenty passes against a *fixed* anchor. This is not
back-translation used as a stand-in for a reference (a discredited practice) — English
outputs are always scored against the human English reference, German outputs always
against the untouched German source. What accumulates is measured drift from ground
truth, not a model agreeing with itself.

## What was found

### From the benchmark

- **Surface quality does not predict clinical safety.** `qwen35-27b` scores 57.6
  BLEU and corrupts clinical content in 45% of reports; `MiniMax-M3` scores 56.2
  and corrupts 30%. Ranking by BLEU picks the less safe system. See
  [figure 4](../figures/fig4_bleu_vs_clinical.png).
- **A learned semantic metric does not rescue this.** COMET produces a *third*
  ordering, disagreeing with both BLEU and the clinical layer, and ranks
  `hymt2-30b-a3b` first — a system that is sixth on BLEU and second-worst of
  eleven on clinical errors. The divergence is therefore not an artefact of
  n-gram matching. See [07-metric-roadmap.md](07-metric-roadmap.md).
- **Degradation is front-loaded and converges.** 77% of all BLEU lost across ten
  round trips is lost in the *first* one; after cycle 2 the text reaches a fixed
  point rather than decaying without bound. True of all twelve systems in the figure;
  DeepL, added later, is the slowest to settle (63% in the first cycle, and 4.0 BLEU more
  lost after cycle 2 than any other system).
- **Round-trip stability is a separate axis from single-pass quality.**
  `translategemma-27b` ties `qwen35-27b` on one pass (57.1 vs 57.6) and loses
  12.3 BLEU round-tripping against qwen's 5.0.
- **The commercial baseline does not change this.** DeepL, on the free API plan, is
  sixth of thirteen on negation and laterality (6.8% of documents), fourth on BLEU, and
  less stable under ten round trips than any of the three leading hosted LLMs (−10.8
  BLEU against −3.8 to −5.7). Ranked on the words alone the order differs from the
  standard one: `opus` is the worst system, not a middling one, because it emits fewer
  numbers to get wrong. See [05-results.md](05-results.md).
- **Every system fails often.** The best single-pass critical-error rate among
  high-quality systems is 30% of reports. That is the finding that matters for
  the original question.

### From the interventions

- **Injecting a dictionary does not help, and a *bigger* dictionary hurts.** With
  a 47,997-entry RadLex bank the relevant glossary scored *below* an
  equally-sized block of irrelevant terms on every metric. The mechanism is
  measurable: 21% of injected terms the model adopted are terms the human
  reference does not use, because an ontology's preferred label is not report
  register. Only the distractor control makes this visible.
- **A three-model debate improved fluency and not safety** — and on the
  medical-text corpus it was *worse* than its best participant (17.5% against
  15.0%), because a consensus mechanism cannot discount a participant that is
  reliably wrong.
- **A three-stage review cascade behaved the same way.** The arbiter gained 4.23
  BLEU and fixed zero clinical errors; the glossary-armed stage before it added
  5.0 points of critical error. Net, the pipeline was worse than its own first
  stage on the clinical layer.
- **Precision beats coverage for terminology.** The BLEU penalty for injecting a
  glossary fell from 4.01 to 0.45 as the bank went from 47,997 mixed entries to
  2,777 validated clinical findings — and that smallest bank produced the only
  directional clinical improvement seen in the project (11.9% vs a 12.8% control,
  p = 0.250, not significant).
- **A glossary mined from the corpus itself does not rescue it either, and shows
  why.** Terms mined from PARROT's own training half are adopted at 90.6%, the
  highest rate in the project — so the terms are report register and the model
  does take them. But the share of adopted terms the reference does not use falls
  only from 20.8% to 17.6%. The residue is **the corpus disagreeing with
  itself**: `Beurteilung` is rendered *conclusion* 57% of the time, *impression*
  36% and *assessment* 7%, so even the best possible single term pair is wrong on
  four reports in ten. Register mismatch explains about three points of the
  original failure in aggregate; roughly seventeen are irreducible reference
  variance.
- **Useful terms and improvable terms are disjoint sets.** Screening the mined
  pairs against what the baselines already produce splits them cleanly and
  oppositely: the 59 terms the models get right anyway are ~92% correctly mined,
  while the 28 with real headroom are only ~39% correctly mined and inject
  non-reference wording 43.6% of the time. In the arm comparison that isolates
  the terms from the prompt block, the headroom bank is worth **−0.12 BLEU
  against randomly chosen irrelevant terms** (p = 0.45) — no signal at all. A
  term with a stable one-to-one rendering is easy for the aligner *and* easy for
  the model, for the same reason; a term the model gets wrong is context-dependent,
  and a context-free glossary entry cannot express it.
- **Most of the failures are unreachable by construction, and this is now
  computed rather than asserted.** On the held-out 100 documents, 15 of the 22
  documents carrying a critical error fail on numbers alone, and the rest are
  mostly *dropped* laterality — an omission a term pair cannot repair. The
  reachable ceiling for any terminology intervention is 7 documents in 100. The
  run of null results is what that ceiling predicts.

Full numbers in [05-results.md](05-results.md).

## Chapter summaries

| File | Contents |
|---|---|
| [01-dataset.md](01-dataset.md) | Corpora, why PARROT, licensing and provenance |
| [02-models.md](02-models.md) | The fourteen entries (thirteen systems and a control) and how each is run |
| [03-methods.md](03-methods.md) | Two-layer evaluation, chunking, round-trip design |
| [04-experiments.md](04-experiments.md) | What was run, on what hardware, what failed |
| [05-results.md](05-results.md) | Scores, figures, interpretation |
| [06-metrics.md](06-metrics.md) | **Every metric: how it is computed, how to read it, why it is here** |
| [07-metric-roadmap.md](07-metric-roadmap.md) | Metrics *not* used, assessed for this use case |
| [08-terminology.md](08-terminology.md) | Which bilingual dictionary, and the hygiene a general-purpose one needs |
| [09-experiments-glossary-debate.md](09-experiments-glossary-debate.md) | Dictionary injection and multi-agent debate |

## Status

**2026-09-29.** All thirteen systems have completed the single-pass benchmark and
the ten-cycle round-trip. Figures generated. Beyond that:

| Work | State |
|---|---|
| Terminology: RadLex OWL + Wikidata, 47,997-entry bank | done — [08](08-terminology.md) |
| Experiment 1, dictionary injection (2 glossary sizes, 3 arms) | done — [09](09-experiments-glossary-debate.md) |
| Experiment 2, symmetric three-agent debate | done — [09](09-experiments-glossary-debate.md) |
| Medical-text-only corpus (section extraction) | done — [01](01-dataset.md) |
| RadLex terminology validated against radiologists | done — [08](08-terminology.md) |
| Experiment 3, asymmetric review cascade | done — [09](09-experiments-glossary-debate.md) |
| Experiments 1 and 2 on the medical-text corpus | done — [09](09-experiments-glossary-debate.md) |
| COMET over all systems | done — [07](07-metric-roadmap.md) |
| LLM-as-judge with a clinical rubric | implemented, not yet run at scale |
| Experiment 4, glossary mined from the corpus (train/val split) | done — [09](09-experiments-glossary-debate.md) |
| DeepL baseline (single pass + round trip) | done 2026-09-29 — [05](05-results.md), [07](07-metric-roadmap.md) |
| DeepL glossary arm | **not started**; free-plan glossary creation untested — [07](07-metric-roadmap.md) |
| Human MQM validation | not started; the binding constraint is clinician time |

**No model has been fine-tuned, and on the current evidence none should be yet.**
Seven interventions have now been tried — a small dictionary, a large one, a
validated small one, one mined from the corpus' own training half, a three-agent
debate, a three-stage cascade, and restricting the corpus to medical text only.
**Not one produced a statistically significant improvement in clinical error
rate.** Several made it worse.

That is not evidence the problem is unfixable, and as of Experiment 4 it is no
longer even a surprise. Three things are now measured rather than suspected:

- **The reachable headroom is 7 documents in 100.** Two thirds of the critical
  errors are numeric, which no terminology or committee method touches; most of
  the remainder are omissions. A null was the arithmetically predicted outcome.
- **The reference cannot adjudicate the interventions that remain.** Where the
  corpus renders one German term four ways, a single-reference lexical metric
  cannot separate a wrong term from a correct synonym, so part of the measured
  "cost" of every glossary was the metric, not the translation.
- **The instrument cannot see the fixes that do exist** — the debate transcripts
  contain correct, specific catches (`NBKS`, `LWK`) that no detector can score.

The first of those bounds the value of further intervention work on this corpus;
the second and third say the next real progress is in measurement.

The honest next step remains the metric work in
[07-metric-roadmap.md](07-metric-roadmap.md), and the single most informative
missing experiment is now the **DeepL glossary arm** — because DeepL applies glossaries
inside the decoder rather than as a prompt instruction, and so separates "dictionaries do
not help" from "*prompt-injected* dictionaries do not help". The DeepL *benchmark* is
done; this is the half of the DeepL question that remains.
