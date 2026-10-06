# Metric roadmap: what else should be measured, and in what order

The current instrument ([06-metrics.md](06-metrics.md)) is high-precision and narrow.
BLEU/chrF++/TER measure string similarity; the clinical detectors measure four specific
failure modes with hand-written lexicons. Neither is what the field now considers
state of the art, and neither is sufficient for the final claim of this work.

This chapter surveys the alternatives and commits to an order of adoption.

---

## The gap, stated precisely

Three things are missing.

1. **A learned semantic metric.** Every surface metric here is lexical. None can tell
   that "pleural fluid collection" and "pleural effusion" mean the same thing — as
   §0 of [06-metrics.md](06-metrics.md) shows, the harmless paraphrase is punished
   *harder* than the dangerous negation flip. Neural metrics fix exactly this.
2. **Recall over clinical errors.** The detectors find what they were written to find.
   The `BWK 12` → `L12` miss (thoracic vertebra rendered as lumbar, no detector fired)
   is not a bug to patch — it is the signature of a closed-class approach. Something
   open-class is needed to bound the true error rate.
3. **More than one reference — or none.** This was added after
   [Experiment 4](09-experiments-glossary-debate.md#experiment-4--the-corpus-cannot-agree-with-itself)
   measured the size of the problem, and it is the most concrete of the three.

   PARROT gives one English translation per report, and on the vocabulary that
   matters the corpus does not agree with itself about what that translation is.
   `Beurteilung` is rendered *conclusion* 50 times, *impression* 32 and
   *assessment* 6 across the reports where both sides carry the header, so **the
   most frequent form accounts for only 57% of occurrences** — 52% on the
   held-out half, where it is nearly a coin flip between *conclusion* and
   *impression*. All of those forms are what radiologists write — this project's
   own section classifier in
   [`data/sections.py`](../src/medmt_eval/data/sections.py) lists
   `Impression | Assessment | Conclusion | Summary | Interpretation` as one role
   precisely because they are interchangeable.

   A single-reference lexical metric therefore cannot separate *wrong term* from
   *correct synonym this reference happens not to use*, and for high-frequency
   report vocabulary the second case is the common one. This is not the same
   complaint as item 1: a learned metric scores meaning but still scores it
   against one reference, so COMET inherits the problem wherever the reference's
   word choice was arbitrary. The fixes are multiple references (expensive — it
   is retranslation, and by clinicians) or metrics that never consult a reference
   at all, which is what makes the source-referenced clinical detectors and the
   LLM judge in §C structurally more interesting than their current accuracy
   suggests.

   Experiment 4 also gives the practical warning for anyone tempted to shortcut
   this: it mined its glossary from the corpus' own English side, which biases
   every reference-based comparison *in the glossary's favour*, and BLEU still
   fell 2.16 points against no glossary at all (p = 0.018). Contamination did not
   even buy a spurious win.

---

## Candidates

### A. COMET-22 / COMET-Kiwi — learned regression on human judgements

**What it is.** Source, hypothesis and reference are encoded with XLM-R-large; the
embeddings and their combinations feed a feed-forward regressor trained to predict human
MQM scores. `wmt22-comet-da` is the reference-based standard; `wmt22-cometkiwi-da` is
the reference-free (QE) variant.

**Why it would help.** It is the single biggest upgrade over BLEU for semantic
adequacy, it correlates far better with human judgement, and it stops punishing
legitimate paraphrase. `pip install unbabel-comet` (2.2.7) works in this environment;
it needs a GPU and roughly the same per-segment cost as a small translation model.

**Why it is not sufficient alone.** Two documented problems, both of which bite here:

- **Domain shift.** *Fine-Tuned Machine Translation Metrics Struggle in Unseen Domains*
  ([arXiv:2402.18747](https://arxiv.org/abs/2402.18747)) shows learned metrics degrade
  off their training distribution — and radiology reports are far from WMT news. The
  same line of work shows that including **Bio-MQM** annotations in training materially
  improves COMET on biomedical test sets, so the fix is domain-specific training data,
  not the stock checkpoint.
- **It is a scalar.** *Pitfalls and Outlooks in Using COMET*
  ([arXiv:2408.15366](https://arxiv.org/abs/2408.15366)) catalogues the failure modes.
  A single number cannot say *what* went wrong, so it cannot replace the clinical layer
  — a fluent mistranslation of a laterality can score well.

**Verdict: adopt as a third surface metric, not as the safety metric.**

### B. XCOMET / MetricX-25 / GemSpanEval — error-span prediction

**What they are.** Metrics that output *error spans with severities*, not just a score.
XCOMET frames it as per-token classification. Google's WMT25 submission
([arXiv:2510.24707](https://arxiv.org/abs/2510.24707)) pairs **MetricX-25** (Gemma-3
adapted to an encoder with a regression head, trained to predict both MQM and ESA
scores, hybrid reference-based/reference-free) with **GemSpanEval**, which emits MQM
error spans as JSON with severity and category.

**Why it would help.** This is the closest published analogue to what the clinical
detector layer does by hand — localised, categorised, severity-weighted errors — but
learned and open-class. It could catch the `BWK 12` → `L12` class of error that no
hand-written detector anticipates.

**Caveat.** Its severity labels are MQM-generic (accuracy/fluency/terminology), not
clinical. A mistranslated vertebra level and a mistranslated adjective may both come
back as "accuracy/major". It bounds recall; it does not rank clinical risk.

**Verdict: adopt for recall estimation — use it to find what the detectors miss.**

### C. LLM-as-judge — GEMBA-MQM and successors

**What it is.** Prompt a frontier LLM to annotate MQM error spans directly.
GEMBA-MQM ([arXiv:2310.13988](https://arxiv.org/abs/2310.13988)) uses a fixed
three-shot prompt and is reference-free. **GEMBA V2**
([WMT 2025](https://aclanthology.org/2025.wmt-1.67/)) ranks first by average correlation
on the WMT24 MQM test sets.

**Why it fits this project unusually well.** We already have a working
OpenAI-compatible client, three verified frontier models, and a batching/retry layer.
The marginal engineering cost is a prompt and a parser. More importantly, an LLM judge
can be given a **clinical** rubric — "flag anything that changes the patient's diagnosis,
laterality, dosage, measurement, or urgency" — which is precisely the open-class
judgement the detectors cannot make.

**Caveats, and they are serious.**
- Known label biases and an inability to discriminate near-perfect translations
  (RUBRIC-MQM, [ACL 2025 Industry](https://aclanthology.org/2025.acl-industry.12/));
  moving from a bare GEMBA prompt to a rubric-style one lifted correlation with humans
  from 0.09 to 0.35 — a warning about how much the prompt determines the result.
- **Self-preference.** Three of our fourteen systems are hosted LLMs. Using an LLM to
  judge LLM translations invites a conflict of interest, and the judge must not be one
  of the systems under test.

**Verdict: adopt as the clinical-recall instrument, with a rubric, a non-competing
judge, and a human-validated subset.**

### D. MQM proper — the human ceiling

**What it is.** The framework the above metrics all approximate. Annotators mark error
spans with a category and a severity; severities carry exponential weights — typically
minor 1, major 5, **critical 25** — and penalties are summed and normalised per segment.

**Why it matters here.** It is the only way to get a *trustworthy* number, and it is
what a thesis claim about clinical safety ultimately rests on. **Bio-MQM** (ACL 2024)
already provides biomedical MQM annotations including EN↔DE, produced by 46 annotators
under ISO 17100 — so a comparable protocol exists and need not be invented.

**Cost.** Bilingual clinician time. This is the binding constraint, not compute.

**Verdict: required for the final claim, on a small stratified subset (~50 reports),
used to validate every automatic metric above.**

### E. Considered and rejected

| Metric | Why not |
|---|---|
| METEOR | Superseded by chrF++ and neural metrics; weak German support. |
| ROUGE | Built for summarisation; no advantage over chrF++ here. |
| BERTScore | Not trained on translation judgements; COMET dominates it for MT. |
| Reference-free QE **as a safety signal** | Directly documented to fail this task — see [06-metrics.md](06-metrics.md) §4. Keep as a triage signal only. |

---

## Result — COMET disagrees with BLEU, and with the clinical layer

Run 2026-08-22 over the existing single-pass PARROT outputs (296 reports, no
retranslation), `Unbabel/wmt22-comet-da`, results in
[`results/comet_parrot_de.json`](../results/comet_parrot_de.json).

| System | COMET | BLEU | crit% |
|---|---|---|---|
| hymt2-30b-a3b | **0.8284** | 47.71 | 31.1 |
| hymt2-7b | 0.8239 | 45.99 | 24.3 |
| MiniMax-M3 ☁ | 0.8207 | 53.49 | 21.3 |
| DeepSeek-V4-Flash ☁ | 0.8186 | **53.75** | 19.9 |
| glm-5.2 ☁ | 0.8152 | 51.34 | **19.3** |
| translategemma-4b | 0.8145 | 44.68 | 23.0 |
| hymt2-1.8b | 0.8067 | 39.11 | 25.3 |
| qwen35-4b | 0.7937 | 47.95 | 24.7 |
| qwen35-27b | 0.7894 | 51.74 | 28.4 |
| nllb | 0.7039 | 21.76 | 33.4 |
| opus | 0.6704 | 24.62 | 26.7 |
| *identity (control)* | *0.6099* | *3.25* | *95.9* |

**Three metrics, three different orderings.** BLEU and COMET do not agree
(`by_bleu == by_comet` is `False`), and neither agrees with the clinical layer:

| | 1st | 2nd | 3rd |
|---|---|---|---|
| BLEU | DeepSeek-V4-Flash | MiniMax-M3 | qwen35-27b |
| COMET | **hymt2-30b-a3b** | hymt2-7b | MiniMax-M3 |
| crit% | glm-5.2 | DeepSeek-V4-Flash | MiniMax-M3 |

The sharpest case is `hymt2-30b-a3b`: **first on COMET, sixth on BLEU, and
second-worst of eleven real systems on clinical errors (31.1%).** The metric
built to capture semantic adequacy ranks first the system that damages clinical
content nearly most. `qwen35-27b` is the mirror image — third on BLEU, ninth on
COMET.

This settles the question the run was designed to ask. The divergence between
surface quality and clinical safety is **not an artefact of n-gram matching**: a
learned semantic metric trained on human adequacy judgements produces a third
ordering, and still does not track clinical risk. The clinical layer is measuring
something neither surface nor learned semantic metrics capture.

Two caveats that bound the claim:

- **COMET is off-distribution here.** It is trained on WMT news-domain human
  judgements and is documented to degrade outside that
  ([arXiv:2402.18747](https://arxiv.org/abs/2402.18747)). Its ordering is
  evidence of disagreement, not adjudication of who is right.
- **The control behaves correctly**, which is what makes the rest readable:
  `identity` scores 0.6099, far below every real system. COMET's scale is
  compressed — 0.67 to 0.83 spans the entire real field — so small COMET
  differences should not be over-read.

### Environment note: COMET cannot share this project's transformers

`unbabel-comet` 2.2.7 requires `transformers` 4.x — installing it silently
downgraded the shared venv from 5.15.1 to 4.57.6, which broke `Qwen3.5` loading
everywhere (`model type 'qwen3_5' not recognized`) and failed a cascade job that
had nothing to do with COMET. Upgrading back breaks COMET in turn: its XLM-R
encoder unpacks a three-tuple that transformers 5.x no longer returns
(`not enough values to unpack (expected 3, got 2)`).

They are mutually exclusive in one environment. COMET therefore lives in its own
`.venv-comet`; the main venv stays on transformers 5.x for the translation
models.

## Result — DeepL benchmarked; the glossary arm is still open

**Done 2026-09-29.** DeepL now has a single pass over PARROT-DE and the ten-cycle
round trip. Full numbers in
[05-results.md](05-results.md#finding-6--deepl-is-competitive-on-the-words-and-unstable-under-repetition).
In short: sixth of thirteen on negation and laterality (6.8% of documents), fourth on
BLEU, and the least stable of the strong systems under repetition (−10.8 BLEU against
−3.8 to −5.7 for the best hosted LLMs). It does not change the conclusion, and it removes
the objection that every claim in the thesis was relative to open models and one
hosted gateway.

It was kept separate from German MeSH as planned: DeepL was scored against the human
references on PARROT, never against an MT-derived term bank, because German MeSH is
itself a DeepL first pass ([08-terminology.md](08-terminology.md)).

**Practical notes.** It runs on the login node (the GPU nodes have no internet). The free
plan allows **1,000,000 characters a month**, not the 500,000 an earlier version of this
section stated; the run used 530,862 in total (226,785 for the single pass), leaving
roughly 469,000 unspent. See [02-models.md](02-models.md#deepl) for the adapter fixes.

**Still open: the glossary arm, and it is the more informative half.** DeepL applies a
glossary inside the translation engine rather than as a prompt instruction, so a DeepL
glossary run would separate "dictionaries do not help" from "*prompt-injected*
dictionaries do not help". `/v3/glossaries` is reachable on the project's key (HTTP 200,
no glossaries yet), but **whether the free plan allows creating one has not been tested**.
The cheapest first step is a single tiny glossary. The ceiling analysis in
[09](09-experiments-glossary-debate.md) predicts a small effect on the clinical layer
whatever the glossary contains, since two thirds of critical errors are numeric, so this
is a test of mechanism, not an expected fix.

## Order of adoption

1. **COMET-22 + COMET-Kiwi over the existing outputs.** Cheapest, no new translations
   needed — all fourteen systems' outputs are already on disk. Answers immediately
   whether the BLEU ranking survives a semantic metric. If COMET reorders the systems,
   that strengthens the central finding; if it reproduces the BLEU order, the clinical
   layer is carrying the whole argument and must be hardened first.
2. **LLM-as-judge with a clinical rubric**, on the same outputs, judge held out from the
   systems under test. Produces the open-class error list the detectors cannot.
3. **Detector recall audit.** Diff the LLM judge's findings against the detectors'.
   Every error the judge finds and the detectors miss is a named gap; fix the ones that
   generalise (anatomical abbreviations first — that is a known miss).
4. **Human MQM on ~50 stratified reports** with a clinician, using the Bio-MQM protocol.
   Validates 1–3 and becomes the number the thesis actually claims.
5. **Only then** consider fine-tuning — with a metric trustworthy enough to detect
   whether it helped.

Steps 1–3 are compute-only and can run on the existing outputs. Step 4 is the one that
needs a person.

---

## References

- Rei et al. (2022), *COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task*, WMT. [aclanthology](https://aclanthology.org/2022.wmt-1.52/)
- Guerreiro et al. (2024), *xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection*, TACL. [MIT Press](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00683/124263/)
- Juraska et al. (2025), *MetricX-25 and GemSpanEval: Google Translate Submissions to the WMT25 Evaluation Shared Task*. [arXiv:2510.24707](https://arxiv.org/abs/2510.24707)
- Kocmi & Federmann (2023), *GEMBA-MQM: Detecting Translation Quality Error Spans with GPT-4*. [arXiv:2310.13988](https://arxiv.org/abs/2310.13988)
- *GEMBA V2: Ten Judgments Are Better Than One*, WMT 2025. [aclanthology](https://aclanthology.org/2025.wmt-1.67/)
- *RUBRIC-MQM: Span-Level LLM-as-judge in Machine Translation For High-End Models*, ACL 2025 Industry. [aclanthology](https://aclanthology.org/2025.acl-industry.12/)
- Zouhar et al. (2024), *Fine-Tuned Machine Translation Metrics Struggle in Unseen Domains*. [arXiv:2402.18747](https://arxiv.org/abs/2402.18747)
- Zouhar et al. (2024), *Pitfalls and Outlooks in Using COMET*, WMT. [arXiv:2408.15366](https://arxiv.org/abs/2408.15366)
- Mehandru et al. (2023), *Physician Detection of Clinical Harm in Machine Translation*, EMNLP. [arXiv:2310.16924](https://arxiv.org/abs/2310.16924)
