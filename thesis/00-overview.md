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

- **Surface quality does not predict clinical safety.** `qwen35-27b` scores 57.6 BLEU
  and corrupts clinical content in 45% of reports; `MiniMax-M3` scores 56.2 and
  corrupts 30%. Ranking by BLEU picks the less safe system. See
  [figure 4](../figures/fig4_bleu_vs_clinical.png).
- **Degradation is front-loaded and converges.** 77% of all BLEU lost across ten round
  trips is lost in the *first* one; after cycle 2 the text reaches a fixed point rather
  than decaying without bound. This holds for all twelve systems regardless of size.
- **Round-trip stability is a separate axis from single-pass quality.**
  `translategemma-27b` ties `qwen35-27b` on one pass (57.1 vs 57.6) and loses 12.3 BLEU
  round-tripping against qwen's 5.0. Single-pass benchmarking calls them equivalent.
- **The hosted APIs lead on clinical safety**, taking the three most stable positions
  and the two lowest genuine error rates.
- **Every system fails often.** The best single-pass critical-error rate among
  high-quality systems is 30% of reports. That is the finding that matters for the
  original question.

Full numbers in [05-results.md](05-results.md).

## Reading order

| File | Contents |
|---|---|
| [01-dataset.md](01-dataset.md) | Corpora, why PARROT, licensing and provenance |
| [02-models.md](02-models.md) | The thirteen systems and how each is run |
| [03-methods.md](03-methods.md) | Two-layer evaluation, chunking, round-trip design |
| [04-experiments.md](04-experiments.md) | What was run, on what hardware, what failed |
| [05-results.md](05-results.md) | Scores, figures, interpretation |
| [06-metrics.md](06-metrics.md) | **Every metric: how it is computed, how to read it, why it is here** |
| [07-metric-roadmap.md](07-metric-roadmap.md) | Metrics *not* used, assessed for this use case |

## Status

2026-08-15 — All thirteen systems have completed both the single-pass benchmark and the
ten-cycle round-trip. Figures generated. The remaining work is the metric upgrade
described in [07-metric-roadmap.md](07-metric-roadmap.md): the current clinical layer is
high-precision but narrow, and neither it nor BLEU is the right instrument for the final
claim.

**No model has been fine-tuned, and on the current evidence none should be yet** — see
[05-results.md](05-results.md) §"Answering the question".
