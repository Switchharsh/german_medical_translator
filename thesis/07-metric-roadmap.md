# Metric roadmap: what else should be measured, and in what order

The current instrument ([06-metrics.md](06-metrics.md)) is high-precision and narrow.
BLEU/chrF++/TER measure string similarity; the clinical detectors measure four specific
failure modes with hand-written lexicons. Neither is what the field now considers
state of the art, and neither is sufficient for the final claim of this work.

This chapter surveys the alternatives and commits to an order of adoption.

---

## The gap, stated precisely

Two things are missing.

1. **A learned semantic metric.** Every surface metric here is lexical. None can tell
   that "pleural fluid collection" and "pleural effusion" mean the same thing — as
   §0 of [06-metrics.md](06-metrics.md) shows, the harmless paraphrase is punished
   *harder* than the dangerous negation flip. Neural metrics fix exactly this.
2. **Recall over clinical errors.** The detectors find what they were written to find.
   The `BWK 12` → `L12` miss (thoracic vertebra rendered as lumbar, no detector fired)
   is not a bug to patch — it is the signature of a closed-class approach. Something
   open-class is needed to bound the true error rate.

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
- **Self-preference.** Three of our thirteen systems are hosted LLMs. Using an LLM to
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

## Order of adoption

1. **COMET-22 + COMET-Kiwi over the existing outputs.** Cheapest, no new translations
   needed — all thirteen systems' outputs are already on disk. Answers immediately
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
