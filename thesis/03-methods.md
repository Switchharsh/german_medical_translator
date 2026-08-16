# Methods

The metrics themselves are specified in [06-metrics.md](06-metrics.md). This chapter
covers the experimental machinery around them.

## Two-layer evaluation

Every translation is scored twice, independently:

- **Surface** — sacreBLEU / chrF++ / TER against the human reference.
- **Clinical** — rule-based detectors comparing the *source* to the *output*, with no
  reference involved.

They are kept separate and never combined into an aggregate. The disagreement between
them is the result of this work; a weighted sum would destroy it.

## Chunking: translate in pieces, score whole

Radiology reports run to 4029 characters, past the encoder position limit of both Opus
(512) and NLLB (1024). Feeding a whole report silently truncates it, and a truncated
translation scores as information loss that the *harness* caused.

The protocol is **translate-chunked, score-whole**: split at sentence boundaries,
translate each chunk, reassemble, then score the reassembled document. Scoring is never
done per chunk, so segmentation cannot influence the metric.

Sentence splitting ([`data/chunking.py`](../src/medmt_eval/data/chunking.py)) is a
boundary-*matching* regex, not a lookbehind:

```python
_SENTENCE_BOUNDARY = re.compile(r"(?<=[^\sA-ZÇĞİÖŞÜ0-9])([.!?])\s+")
```

The original lookbehind formulation silently never fired under `re.split` — a bug that
produced correct-looking output while doing nothing. The character class protects two
things verified by hand against the corpus: **decimals** (83 instances in the Turkish
corpus, where `1.5` must not split) and **lone capitals** (11 instances, all `A.` for
Latin *arteria* — none were personal initials, so treating them as non-boundaries is
safe here and would not be in a corpus with author names).

Four separate truncation causes were found and each needed its own fix: the generation
output ceiling, the encoder position limit, the decoder position limit, and the input
character ceiling. They present identically — a short translation — which is why they
were resolved one at a time rather than by one change.

**A calibration failure worth recording.** The NLLB chunk budget was initially set to
800 tokens ≈ 2400 characters, which is longer than most documents, so chunking never
engaged at all. Correcting it to 400 changed 25 of 33 documents from unsplit to split.
A parameter that looks conservative can silently disable the mechanism it configures.

## Round-trip protocol

`DE → EN → DE → EN …` for ten cycles = **twenty translation passes** per system.
Implemented in [`inference/roundtrip.py`](../src/medmt_eval/inference/roundtrip.py).

**Fixed anchors.** English outputs are always scored against the human English
reference; German outputs always against the untouched German source. Neither anchor
ever changes, so step 1 is identical to the ordinary single-pass evaluation and the
first point of every curve is directly comparable to the main benchmark.

**This is not back-translation-as-reference.** That practice compares an output to its
own back-translation with no human reference, and is discredited because it rewards a
model that makes the same mistake in both directions. Here no model output is ever used
as the reference for another model output.

**Two evaluations per step.**

| Field | Compares | Answers |
|---|---|---|
| `clinical_vs_origin` | original source ↔ current output | cumulative drift from ground truth |
| `clinical_vs_input` | this hop's input ↔ this hop's output | which step introduced the error |

**Independent support.** Mehandru et al. (EMNLP 2023) ran a physician study on Emergency
Department discharge instructions and found that back-translation helped physicians
detect more clinically harmful errors than quality estimation alone, which "QE often
misses". That is direct evidence that round-tripping surfaces exactly the class of error
this project is trying to measure.

## Sampling

The round-trip uses a deterministic, length-stratified 20-report subsample
(`RT_SEED=13`). Every system sees the same twenty reports. Ten cycles over all 296
reports would have cost ~168 GPU-hours across the model set, past every wall-clock
limit available.

Consequence for reading the results: with n=20 the critical-error rate moves in 5-point
steps, and **nothing finer than 5 points is meaningful**. Differences of one step should
not be interpreted.

## Statistical treatment

Corpus-level surface scores use sacreBLEU's corpus scorers, not the mean of
sentence-level scores — those are different quantities and the difference is not small
for BLEU. Sentence-level scores are computed separately (with `effective_order`) for
per-segment analysis and bootstrap resampling.

## Reproducibility

- Every result row persists the adapter's full `generation_config` (model id, beams,
  token limits, batch size, device).
- Every surface score persists its sacreBLEU signature.
- Hosted-model rows additionally persist the **served** model string and a
  `model_substitution` flag.
- Corpus conversion, sampling seed and chunk budget are all recorded in the run
  directory.

## Infrastructure discipline

All jobs run under SLURM and **preflight before consuming resources**: credentials are
checked, gated repositories are probed, and API endpoints are contacted with a single
request before a long run is launched. This was adopted after a run failed 55 seconds in
on a gated-repository 401, and again after two of three API jobs died on HTTP 429 from
sharing one key. Both classes of failure are now caught at submission time; the
gated-repo check is a hard `:?` guard in the job script itself.
