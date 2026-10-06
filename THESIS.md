# Clinical Information Loss in German↔English Medical Machine Translation

*Does German↔English medical translation need a specialised model?* — a benchmark of thirteen systems on radiology reports, and what it takes to measure the answer.

---

## Contents

1. [Overview](#1-overview)
2. [Datasets](#2-datasets)
3. [Systems under test](#3-systems-under-test)
4. [Methods](#4-methods)
5. [Experiments](#5-experiments)
6. [Results](#6-results)
7. [Metrics: what we compute, how to read it, and why](#7-metrics-what-we-compute-how-to-read-it-and-why)
8. [Metric roadmap: what else should be measured, and in what order](#8-metric-roadmap-what-else-should-be-measured-and-in-what-order)
9. [Terminology: which dictionary, and what it costs to get it wrong](#9-terminology-which-dictionary-and-what-it-costs-to-get-it-wrong)
10. [Experiments: dictionary injection, debate, and a review cascade](#10-experiments-dictionary-injection-debate-and-a-review-cascade)
11. [Configurations and protocols](#11-configurations-and-protocols)

---

## 1. Overview

### Question

**Does German↔English medical translation need a specialised model?**

The honest way to answer that is not to fine-tune something and report that it beat a
generic baseline. It is to benchmark what already exists — generic MT, specialised
medical MT, small and large LLMs, hosted frontier APIs — and only build if a real gap
survives measurement. This project is that benchmark.

The answer turns on *what you measure*. Ranked by BLEU, several systems look
interchangeable. Ranked by whether clinical facts survive translation, they do not.

### Approach

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

### What was found

#### From the benchmark

- **Surface quality does not predict clinical safety.** `qwen35-27b` scores 57.6
  BLEU and corrupts clinical content in 45% of reports; `MiniMax-M3` scores 56.2
  and corrupts 30%. Ranking by BLEU picks the less safe system. See
  [figure 4](figures/fig4_bleu_vs_clinical.png).
- **A learned semantic metric does not rescue this.** COMET produces a *third*
  ordering, disagreeing with both BLEU and the clinical layer, and ranks
  `hymt2-30b-a3b` first — a system that is sixth on BLEU and second-worst of
  eleven on clinical errors. The divergence is therefore not an artefact of
  n-gram matching. See [8. Metric roadmap: what else should be measured, and in what order](#8-metric-roadmap-what-else-should-be-measured-and-in-what-order).
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
  numbers to get wrong. See [6. Results](#6-results).
- **Every system fails often.** The best single-pass critical-error rate among
  high-quality systems is 30% of reports. That is the finding that matters for
  the original question.

#### From the interventions

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

Full numbers in [6. Results](#6-results).

### Chapter summaries

| File | Contents |
|---|---|
| [2. Datasets](#2-datasets) | Corpora, why PARROT, licensing and provenance |
| [3. Systems under test](#3-systems-under-test) | The fourteen entries (thirteen systems and a control) and how each is run |
| [4. Methods](#4-methods) | Two-layer evaluation, chunking, round-trip design |
| [5. Experiments](#5-experiments) | What was run, on what hardware, what failed |
| [6. Results](#6-results) | Scores, figures, interpretation |
| [7. Metrics: what we compute, how to read it, and why](#7-metrics-what-we-compute-how-to-read-it-and-why) | **Every metric: how it is computed, how to read it, why it is here** |
| [8. Metric roadmap: what else should be measured, and in what order](#8-metric-roadmap-what-else-should-be-measured-and-in-what-order) | Metrics *not* used, assessed for this use case |
| [9. Terminology: which dictionary, and what it costs to get it wrong](#9-terminology-which-dictionary-and-what-it-costs-to-get-it-wrong) | Which bilingual dictionary, and the hygiene a general-purpose one needs |
| [10. Experiments: dictionary injection, debate, and a review cascade](#10-experiments-dictionary-injection-debate-and-a-review-cascade) | Dictionary injection and multi-agent debate |

### Status

**2026-09-29.** All thirteen systems have completed the single-pass benchmark and
the ten-cycle round-trip. Figures generated. Beyond that:

| Work | State |
|---|---|
| Terminology: RadLex OWL + Wikidata, 47,997-entry bank | done — [08](#9-terminology-which-dictionary-and-what-it-costs-to-get-it-wrong) |
| Experiment 1, dictionary injection (2 glossary sizes, 3 arms) | done — [09](#10-experiments-dictionary-injection-debate-and-a-review-cascade) |
| Experiment 2, symmetric three-agent debate | done — [09](#10-experiments-dictionary-injection-debate-and-a-review-cascade) |
| Medical-text-only corpus (section extraction) | done — [01](#2-datasets) |
| RadLex terminology validated against radiologists | done — [08](#9-terminology-which-dictionary-and-what-it-costs-to-get-it-wrong) |
| Experiment 3, asymmetric review cascade | done — [09](#10-experiments-dictionary-injection-debate-and-a-review-cascade) |
| Experiments 1 and 2 on the medical-text corpus | done — [09](#10-experiments-dictionary-injection-debate-and-a-review-cascade) |
| COMET over all systems | done — [07](#8-metric-roadmap-what-else-should-be-measured-and-in-what-order) |
| LLM-as-judge with a clinical rubric | implemented, not yet run at scale |
| Experiment 4, glossary mined from the corpus (train/val split) | done — [09](#10-experiments-dictionary-injection-debate-and-a-review-cascade) |
| DeepL baseline (single pass + round trip) | done 2026-09-29 — [05](#6-results), [07](#8-metric-roadmap-what-else-should-be-measured-and-in-what-order) |
| DeepL glossary arm | **not started**; free-plan glossary creation untested — [07](#8-metric-roadmap-what-else-should-be-measured-and-in-what-order) |
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
[8. Metric roadmap: what else should be measured, and in what order](#8-metric-roadmap-what-else-should-be-measured-and-in-what-order), and the single most informative
missing experiment is now the **DeepL glossary arm** — because DeepL applies glossaries
inside the decoder rather than as a prompt instruction, and so separates "dictionaries do
not help" from "*prompt-injected* dictionaries do not help". The DeepL *benchmark* is
done; this is the half of the DeepL question that remains.

## 2. Datasets

### The corpora

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

### Why PARROT carries the argument

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
what forces the chunking machinery described in [4. Methods](#4-methods).

### Report structure — and the medical-text-only corpus

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

[`data/sections.py`](src/medmt_eval/data/sections.py) extracts sections by
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

#### What removing it actually changes

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

### Conversion notes

- **Filter on `language`, not `country`.** The `country` field is dirty: it contains the
  value `"German"` for 50 records, which is not a country. Filtering on it silently
  produces the wrong subset. The converter
  ([`data/parrot.py`](src/medmt_eval/data/parrot.py)) uses `language`.
- **Area normalisation.** `normalise_area()` folds free-text anatomical area labels into
  a consistent set for the per-modality breakdown.
- **Round-trip sample.** The ten-cycle experiment uses a deterministic,
  length-stratified 20-report subsample (`RT_SAMPLE=20`, `RT_SEED=13`). All twelve
  systems see the *same* twenty reports, so the curves are directly comparable. The full
  296 would have cost roughly 168 GPU-hours across the model set — past every wall-clock
  limit available.

### EMEA

Down-sampled from 364,005 pairs with de-duplication and length-ratio filtering. Official
EMA translations are professional and legally mandated, which makes them a good
reference but an easy register: highly repetitive, heavily templated, and consequently
flattering to any MT system.

### The Turkish subset — a negative control that did not work

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

Recorded in [`results/parrot_tr/INVALID_RUNS.md`](results/parrot_tr/INVALID_RUNS.md).

### Terminology bank

A nine-entry starter bank of EN↔DE radiology terms
([`data/term_banks/radiology_en_de_starter.csv`](data/term_banks/radiology_en_de_starter.csv)).
It is deliberately small and its findings are `major`, not `critical` — it exists to
demonstrate the mechanism, not to be authoritative. Matching is inflection-tolerant
(`term_surface_pattern()`); adding that tolerance cut terminology findings from 735 to
293, i.e. roughly two thirds of the original findings were German inflection, not
translation errors.

## 3. Systems under test

Fourteen entries: thirteen translation systems plus a control.

| Name | Model ID | Class | Params | Adapter |
|---|---|---|---|---|
| `identity` | — | **control** | — | `identity` |
| `opus` | `Helsinki-NLP/opus-mt-{de-en,en-de}` | Dedicated bilingual MT | 74 M | `opus` |
| `nllb` | `facebook/nllb-200-distilled-1.3B` | Massively multilingual MT | 1.3 B | `nllb` |
| `hymt2-1.8b` | `tencent/Hy-MT2-1.8B` | Specialised MT | 1.8 B | `hymt2` |
| `hymt2-7b` | `tencent/Hy-MT2-7B` | Specialised MT | 7 B | `hymt2` |
| `hymt2-30b-a3b` | `tencent/Hy-MT2-30B-A3B` | Specialised MT (MoE) | 30 B / 3 B active | `hymt2` |
| `translategemma-4b` | `google/translategemma-4b-it` | Specialised MT | 4 B | `translategemma` |
| `translategemma-27b` | `google/translategemma-27b-it` | Specialised MT | 27 B | `translategemma` |
| `qwen35-4b` | `Qwen/Qwen3.5-4B` | General LLM, prompted | 4 B | `prompted-llm` |
| `qwen35-27b` | `Qwen/Qwen3.5-27B` | General LLM, prompted | 27 B | `prompted-llm` |
| `glm-5.2` | `z-ai/glm-5.2` | Hosted frontier LLM | — | `openai-compat` |
| `DeepSeek-V4-Flash` | `deepseek-ai/deepseek-v4-flash-0731` | Hosted frontier LLM | — | `openai-compat` |
| `MiniMax-M3` | `minimaxai/minimax-m3` | Hosted frontier LLM | — | `openai-compat` |
| `deepl` | none disclosed | Commercial neural MT, free API plan | — | `deepl` |

The set spans the four hypotheses worth testing: a small dedicated bilingual model, a
massively multilingual one, purpose-built translation LLMs, and general-purpose LLMs
prompted to translate — local and hosted. DeepL, added last (2026-09-29), is the fifth: the
commercial system a German hospital would actually reach for.

### The control

`identity` returns the source unchanged. It is not a translation system; it is the
floor. Its scores establish what "no translation at all" looks like under every metric,
which is the only way to know a metric has a working scale. On German it scores BLEU
3.39 and a 95% critical-error rate — correctly catastrophic. On Turkish it scores 0%
critical errors, which is how the Turkish coverage problem was found
([2. Datasets](#2-datasets)).

It is excluded from every figure and reported separately in tables.

### Adapter notes

Each family needed something specific. These are the ones that changed results, not just
plumbing.

**`opus` is direction-specific.** Helsinki-NLP ships one checkpoint per language pair,
and an instance pins itself to the first direction used. It is the only adapter with
`direction_specific = True`, so the round-trip runner builds two instances for it and
exactly one for everything else. Building two of everything doubled GPU memory and
OOM'd a 14 GB model on a 20 GB slice before this was distinguished.

**Prompted LLMs must use the chat template.** `PromptedLLMTranslator` originally fed raw
text to the model. Fixed to call `apply_chat_template` with `enable_thinking=False`,
plus a `strip_thinking()` regex as a second line of defence for templates that emit
`<think>` blocks anyway. Without this, reasoning traces land in the translation output.

**Left padding for batched decoder-only generation.** Right padding corrupts batched
generation for causal LMs. Setting `padding_side="left"` moved `qwen35-4b`'s critical
error rate from 24.66% to 18.92% — a change in the *result*, not just in speed.

**Position limits are a per-model property.** Opus has 512 encoder *and* decoder
positions; NLLB has 1024. Both truncate radiology reports, so both are chunked at
sentence boundaries. Opus additionally needs `max_new_tokens = 480`: exceeding its
decoder table raises a CUDA device-side assert (`marian/modeling_marian.py:596`), which
took three attempts to attribute correctly — the first fix reduced batch size, which
addressed the wrong cause.

**MoE weight storage ≠ compute.** `Hy-MT2-30B-A3B` activates ~3 B parameters but must
*hold* ~60 GB of weights, so it needs 2×80 GB despite being cheap to run. Sizing it by
active parameters would have failed to load.

### DeepL

DeepL is the one system here with no model identifier. The service does not disclose
which model answers, so nothing can be pinned and the result is a dated measurement
(2026-09-29), not a reproducible artefact. Rerunning it next year may give different
numbers.

Four things the adapter
([`models/deepl_mt.py`](src/medmt_eval/models/deepl_mt.py)) has to get right, three
of which it originally got wrong:

- **Where it runs.** DeepL is reachable only over the internet, and the GPU nodes have no
  outbound network, so it runs on the login node
  ([`scripts/deepl/run_deepl.sh`](scripts/deepl/run_deepl.sh)). The work is I/O-bound:
  the whole 296-report benchmark took 254 s.
- **The key variable.** The adapter read `DEEPL_AUTH_KEY`; the project's `.env` defines
  `DEEPL_API_KEY`. The adapter would have refused to start with a valid key. It now
  accepts either.
- **The endpoint follows the key.** Free-plan keys end in `:fx` and are valid only on
  `api-free.deepl.com`. The adapter previously chose the host from a flag, and a wrong
  flag produces a 403 that reads like a bad key.
- **The English target is pinned to `EN-US`.** Bare `EN` is deprecated. It currently
  returns US spelling, checked directly and identical to `EN-US`, and PARROT's references
  are US-spelled (78 US tokens, 0 UK). But what a deprecated alias means is DeepL's to
  change.

Failure handling follows the lesson of the hosted gateway: HTTP 429 and 5xx back off and
retry; **456 (monthly quota exhausted) is fatal**, because retrying cannot help. The free
plan allows 1,000,000 characters a month (read from `/v2/usage`; an earlier draft of the
roadmap said 500,000). The run script refuses to start if the projected spend exceeds
95% of what remains.

### Hosted models: the substitution guard

The gateway used for the hosted models has been observed **serving a different model
than requested, with no error**: `DeepSeek-V4-Pro` was answered by
`nvidia/nemotron-3-ultra-550b-a55b`, and `Kimi-K2.6` by `thinkingmachines/inkling`.
The `/v1/models` catalogue is not trustworthy either — every entry claims
`"owned_by": "openai"`.

The only reliable signal is the `model` field of an actual response, which
`_check_served_model()` verifies on **every call**. Results are never attributed to a
model that did not produce them; a mismatch fails the run rather than mislabelling data.

Only three aliases are verified to route honestly and they are the only hosted models
reported:

```
glm-5.2            → z-ai/glm-5.2
DeepSeek-V4-Flash  → deepseek-ai/deepseek-v4-flash-0731
MiniMax-M3         → minimaxai/minimax-m3
```

The guard needed two corrections, both of which blocked legitimate runs rather than
admitting bad ones: first for namespacing (`glm-5.2` → `z-ai/glm-5.2`), then for dated
snapshots (`DeepSeek-V4-Flash` → `…-flash-0731`). Version-tag stripping is deliberately
narrow — digits only, optionally `v`-prefixed — so a differing *name* can never be
excused as a differing version.

### The gateway degraded mid-project

The hosted models were verified honest at the time of the round-trip runs. They
did not stay that way. Re-checked on 2026-08-21:

| Requested | Served | Status |
|---|---|---|
| `DeepSeek-V4-Flash` | `deepseek-ai/deepseek-v4-flash-0731` | ✅ honest |
| `glm-5.2` | — | ❌ withdrawn from the catalogue |
| `MiniMax-M3` | — | ❌ withdrawn |
| `Qwen3.8-27B` | `meta/muse-glimmer-30b` | ❌ substituted |
| `Qwen3.8-35B-A3B` | `nvidia/nemotron-3.5-lightning-30b-a3b` | ❌ substituted |
| `Kimi-K2.6` | `thinkingmachines/inkling` | ❌ substituted |

Two of the three systems in the published benchmark can no longer be re-run at
all. The guard caught every substitution, so no result was mislabelled — but this
is a limit on reproducibility that no amount of care in this repository can fix,
and it is the reason the later experiments default to local models.

### Decoding

Greedy/beam settings are fixed across systems and persisted with every result
(`generation_config`). Round-trip runs use `num_beams=1` for tractability; the
single-pass benchmark uses the per-adapter default. Temperature is 0 for the hosted
models so runs are reproducible, matching the local models.

## 4. Methods

The metrics themselves are specified in [7. Metrics: what we compute, how to read it, and why](#7-metrics-what-we-compute-how-to-read-it-and-why). This chapter
covers the experimental machinery around them.

### Two-layer evaluation

Every translation is scored twice, independently:

- **Surface** — sacreBLEU / chrF++ / TER against the human reference.
- **Clinical** — rule-based detectors comparing the *source* to the *output*, with no
  reference involved.

They are kept separate and never combined into an aggregate. The disagreement between
them is the result of this work; a weighted sum would destroy it.

### Chunking: translate in pieces, score whole

Radiology reports run to 4029 characters, past the encoder position limit of both Opus
(512) and NLLB (1024). Feeding a whole report silently truncates it, and a truncated
translation scores as information loss that the *harness* caused.

The protocol is **translate-chunked, score-whole**: split at sentence boundaries,
translate each chunk, reassemble, then score the reassembled document. Scoring is never
done per chunk, so segmentation cannot influence the metric.

Sentence splitting ([`data/chunking.py`](src/medmt_eval/data/chunking.py)) is a
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

### Round-trip protocol

`DE → EN → DE → EN …` for ten cycles = **twenty translation passes** per system.
Implemented in [`inference/roundtrip.py`](src/medmt_eval/inference/roundtrip.py).

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

### Sampling

The round-trip uses a deterministic, length-stratified 20-report subsample
(`RT_SEED=13`). Every system sees the same twenty reports. Ten cycles over all 296
reports would have cost ~168 GPU-hours across the model set, past every wall-clock
limit available.

Consequence for reading the results: with n=20 the critical-error rate moves in 5-point
steps, and **nothing finer than 5 points is meaningful**. Differences of one step should
not be interpreted.

### Contamination discipline

PARROT is simultaneously the only report-register parallel corpus available here
**and** the evaluation set. That dual role constrains what may be built from it,
and the constraint is easy to violate without noticing.

**Mining a glossary from PARROT would invalidate every reference-based metric.**
BLEU, chrF++, TER and COMET all score against the same English translations the
terms would be mined from, so injecting mined wording and then scoring against it
measures leakage, not translation quality. Given that
[Experiment 1](#10-experiments-dictionary-injection-debate-and-a-review-cascade) found a BLEU *drop* caused by
wording mismatch, mining would trivially reverse that finding while proving
nothing. Splitting by document does not fix it either: the same radiologists
wrote both halves, so house style leaks across the split.

What survives such contamination is precisely the set of metrics that never touch
the reference — the negation, laterality and number detectors, and the LLM judge.
What does not survive is everything reference-based, plus the terminology
detector if it shares a bank with the injected glossary.

The rule adopted here: **terminology comes from sources external to the
evaluation corpus** (RadLex, Wikidata), and the corpus is used to *measure*
terminology, never to *generate* it.

**One deliberate exception, run as an experiment.**
[Experiment 4](#10-experiments-dictionary-injection-debate-and-a-review-cascade)
mines a glossary from PARROT on purpose, because Experiment 1's failure was
traced to register — an ontology's preferred label is not the wording a
radiologist writes — and the only available source of report register is the
reports. It is run under the constraints this section implies rather than in
spite of them: a document-level train/val split so no scored document
contributed a term, the source-referenced detectors as the primary read, and
every reference-based score labelled contaminated. What the split buys is
narrower than it looks — it removes direct memorisation and leaves house-style
leakage intact — so the experiment's reference-based numbers are reported as an
upper bound that cannot be cleaned up, not as a result.

Two places where that line was approached and should be stated plainly:

- **Branch selection.** RadLex branches were first filtered by a ≥70% measured
  agreement threshold — a threshold computed on the evaluation set, and one that
  falls exactly where a branch changes sides between corpus halves. The selection
  was redone on a priori grounds (findings and procedures carry clinical meaning;
  report components and descriptors are not terminology), with the measurement
  reported as confirmation rather than as the criterion. See
  [9. Terminology: which dictionary, and what it costs to get it wrong](#9-terminology-which-dictionary-and-what-it-costs-to-get-it-wrong).
- **Section patterns.** The header patterns in
  [`data/sections.py`](src/medmt_eval/data/sections.py) were written by
  inspecting PARROT's headers. This is much weaker than lexical leakage —
  structure, not wording, and the extracted text is unchanged — but it is the
  same family and is disclosed for the same reason.

The structural fix for both is a **second German report corpus**: one to develop
against, one held for evaluation. Nothing else makes corpus-derived register
testable. Failing that, a frozen held-out slice of PARROT that no glossary or
section work ever touches is the minimum defensible arrangement.

### Statistical treatment

Corpus-level surface scores use sacreBLEU's corpus scorers, not the mean of
sentence-level scores — those are different quantities and the difference is not small
for BLEU. Sentence-level scores are computed separately (with `effective_order`) for
per-segment analysis and bootstrap resampling.

### Reproducibility

- Every result row persists the adapter's full `generation_config` (model id, beams,
  token limits, batch size, device).
- Every surface score persists its sacreBLEU signature.
- Hosted-model rows additionally persist the **served** model string and a
  `model_substitution` flag.
- Corpus conversion, sampling seed and chunk budget are all recorded in the run
  directory.

### Infrastructure discipline

All jobs run under SLURM and **preflight before consuming resources**: credentials are
checked, gated repositories are probed, and API endpoints are contacted with a single
request before a long run is launched. This was adopted after a run failed 55 seconds in
on a gated-repository 401, and again after two of three API jobs died on HTTP 429 from
sharing one key. Both classes of failure are now caught at submission time; the
gated-repo check is a hard `:?` guard in the job script itself.

## 5. Experiments

### What was run

**Single-pass benchmark** — all thirteen systems over PARROT-DE (296 reports), EMEA
(400), HimL 2015 (353) and HimL 2017 (119). Both evaluation layers.

**Round-trip** — all thirteen systems, ten cycles (twenty passes) over the 20-report
stratified PARROT-DE subsample. Results in
`results/roundtrip_20260814_132452/`.

**Turkish** — the same pipeline on the 48 TR→EN PARROT pairs. Reported as a limitation
only; see [2. Datasets](#2-datasets).

**Dictionary injection** — three arms (`none` / `glossary` / `distractor`) on 40
PARROT-DE reports, run twice: once on a 2,834-entry Wikidata bank (3.0 terms per
document) and once on a 47,997-entry RadLex+Wikidata bank (13.6 terms per
document). See [09](#10-experiments-dictionary-injection-debate-and-a-review-cascade).

**Multi-agent debate** — three local instruction-following models with distinct
personas and disjoint glossary slices, two rounds plus synthesis, 20 reports.

**Review cascade** — an asymmetric pipeline (MT drafts → terminologist with the
glossary corrects → radiologist without the glossary arbitrates), every stage
scored separately, 40 reports.

**Mined glossary** — PARROT split 196 train / 100 val by document, stratified by
modality (seed 20260928). Two banks mined from the train half by sentence-level
Dice alignment with mutual-best filtering — 87 terms by frequency, and the 28 of
them the baselines get right at most 60% of the time — each run against the same
three arms on the held-out half. `results/experiments/4399511/`. See
[09](#10-experiments-dictionary-injection-debate-and-a-review-cascade).

**DeepL** — the commercial baseline, added 2026-09-29. Single pass over PARROT-DE
(296 reports) and the same ten-cycle, 20-report round trip as every other system.
EMEA and HimL were not run. `results/deepl_20260929_134803/`. See
[05](#6-results).

**COMET** — `Unbabel/wmt22-comet-da` over all thirteen systems' existing
single-pass outputs. No retranslation. See [07](#8-metric-roadmap-what-else-should-be-measured-and-in-what-order).

### Hardware and cost

SLURM, mixed A100 partition. The nodes are **not** uniform: 19 carry the `a100_40`
feature and 18 carry `a100_80`, and an unconstrained `--gres=gpu:a100` request lands on
either. Early runs happened to get 40 GB cards, which produced a wrong conclusion —
recorded here because it shaped several sizing decisions before it was corrected — that
all the cluster's A100s were 40 GB. `--constraint=a100_80` pins the large jobs.

Round-trip wall time (20 passes × 20 reports):

| System | Wall | Placement |
|---|---|---|
| `identity` | <1 min | MIG slice |
| `opus` | 5 min | MIG slice |
| `glm-5.2` | 13 min | API (no GPU) |
| `MiniMax-M3` | 13 min | API (no GPU) |
| `nllb` | 27 min | MIG slice |
| `qwen35-4b` | 39 min | MIG slice |
| `hymt2-1.8b` | 44 min | MIG slice |
| `qwen35-27b` | 46 min | 1×80 GB, batch 4 |
| `translategemma-4b` | 48 min | MIG slice |
| `hymt2-7b` | 54 min | MIG slice, batch 4 |
| `translategemma-27b` | 92 min | 1×80 GB, batch 4 |
| `hymt2-30b-a3b` | 137 min | 2×80 GB, batch 8 |
| `DeepSeek-V4-Flash` | 189 min | API (no GPU) |

DeepL used no GPU: 254 s for the single pass and 252 s for the round trip on the login
node, 530,862 of the free plan's 1,000,000 monthly characters.

Moving the 27 B models from 2×40 GB (batch 1, weights sharded) to 1×80 GB (batch 4) cut
`qwen35-27b` from an estimated 2.4 h to 46 min, and a single-GPU request also schedules
sooner on a busy partition.

### Failures worth recording

Each of these changed either the code or the conclusions.

**Gated repositories (jobs 4006927, 4006946).** Both TranslateGemma runs died on an
opaque HTTP 401 after ~55 s. Cause: SLURM inherits the submitting shell's environment,
and `HF_TOKEN` was not exported in it. Fixed with a hard `:?` guard in `_rt_common.sh`
that fails at submission with a readable message.

There is a second lesson here that cost a full round of work. The token *was* present in
the user's `~/.bashrc`, but a non-interactive shell does not source `.bashrc` — bash
skips it unless the shell is interactive or `BASH_ENV` is set, and there was no
`~/.bash_profile` to bridge it. The environment looked empty when it was not. Sourcing
it explicitly resolved it.

**API rate limiting (jobs 4008963, 4008965).** Three hosted-model jobs submitted in
parallel, 8 workers each, one shared key → 24 concurrent requests → HTTP 429. Two of
three died. Two fixes: exponential backoff with jitter honouring `Retry-After`, and —
more importantly — the per-item fallback was *amplifying* the problem, turning one
rate-limited batch of 8 into 8 more requests. The jobs now run as a **serial chain** at
3 workers.

**Origin timeout (job 4009014).** After the 429 fix, `DeepSeek-V4-Flash` failed at 78
minutes with Cloudflare `524 Response Timeout by Origin Server`. Two distinct errors:
524 was not in the retry set, and the no-fanout rule introduced for 429 had been applied
to *all* HTTP errors, which made this case worse. A 429 means "you are sending too
much" — do not send more. A 524 means "this request did not come back" — smaller
requests are precisely the cure. Only 429 now propagates; timeouts fan out as before.
Re-run at batch 2, completed in 189 minutes.

**Round-trip OOM (job 4006929).** `hymt2-7b` (14 GB) OOM'd on a 20 GB MIG slice because
the runner built one translator instance per direction. Only Opus needs that; every
other adapter takes the direction as a call argument. Fixed with the
`direction_specific` flag — one instance unless the adapter is pinned.

**A destructive mistake of my own.** An earlier refactor used `$(dirname "$0")` in a
SLURM script, which does not resolve as expected under SLURM. Correcting it to absolute
paths killed two running jobs (3942019, 3942020). The check performed before editing
looked for writers in `results/` but not in `scripts/`.

**Compute nodes have no outbound network (job 4077499).** COMET's checkpoint
fetch works interactively and fails inside a job — and COMET swallows the real
error, re-raising it as `Model 'Unbabel/wmt22-comet-da' not supported by COMET`,
which points at the checkpoint name rather than at the network. Fixed by
pre-fetching on the login node and loading from a local `.ckpt` path, with a
preflight that fails at submission if the cache is empty.

**Installing a metric broke the models (job 4077516).** `unbabel-comet` 2.2.7
requires `transformers` 4.x, so installing it silently downgraded the shared venv
from 5.15.1 to 4.57.6. That broke `Qwen3.5` loading everywhere
(`model type 'qwen3_5' not recognized`) and killed a cascade job that had nothing
to do with COMET. Upgrading back breaks COMET instead — its XLM-R encoder unpacks
a three-tuple that transformers 5.x no longer returns. They are mutually
exclusive; COMET now lives in `.venv-comet`.

Isolating it surfaced one more layer: the fresh venv's `setuptools` 84 no longer
ships `pkg_resources`, which COMET still imports, and `pip install setuptools`
reported the requirement already satisfied. Pinning `setuptools<81` fixed it. The
diagnostic lesson is the same as the transformers one — a package manager saying
"satisfied" is a statement about versions, not about whether an import works.

This one is worth dwelling on because the misdiagnosis was instructive. The
symptom was a model failing *only* in the cascade, where Hy-MT2 loads before
Qwen, and not in the debate, where Qwen loads first. That is a perfectly coherent
load-order hypothesis, and it was wrong. What settled it was the dist-info
timestamp on `transformers-4.57.6` matching the COMET install to the minute — a
fact about the environment, not about the code.

**Tokenizer output the model rejects (job 4077495).** `Hy-MT2-7B`'s tokenizer
returns `token_type_ids`; its `generate()` refuses them
(`model_kwargs are not used by the model`). Qwen's and Gemma's tokenizers do not
return the field, so this is per-model and cannot be assumed away. Fixed by
filtering the encoding against the model's actual `forward` signature rather than
a hardcoded deny-list, so the next tokenizer with an extra field does not break
the same way.

**A translation-only model cannot join a debate (job 4077236).**
`translategemma-4b-it`'s chat template requires each message's content to be a
structured mapping carrying `source_lang_code`/`target_lang_code`. It can express
"translate this" and nothing else — no persona, no critique. The job died twelve
minutes in, after loading three models, on a Jinja `TemplateError`. Fixed with a
tokenizer-only capability probe that runs in the preflight, before any weights
load.

**The hosted gateway degraded mid-project.** Between the round-trip runs and the
experiments, `glm-5.2` and `MiniMax-M3` were withdrawn entirely, and of what
remained only `DeepSeek-V4-Flash` still routed honestly — `Qwen3.8-27B` was
served as `meta/muse-glimmer-30b`, `Qwen3.8-35B-A3B` as
`nvidia/nemotron-3.5-lightning-30b-a3b`, `Kimi-K2.6` as
`thinkingmachines/inkling`. The substitution guard caught every case. This is why
the debate and cascade experiments default to local models: a participant that
cannot be named is not reproducible.

**A parser turned commentary into a finding (job 4077545).** The cascade's
terminologist stage appeared to destroy translation quality — BLEU −17.9,
critical errors +22.5. It had not. A 4B model asked for a two-field structured
reply wrote its `ISSUES:` list and never reached `TRANSLATION:` (1 reply in 40
contained the marker), and the parser's fallback returned the whole reply as the
translation on 17 of 40 documents.

Three properties of this failure make it worth recording. It produced a
*plausible* result — one that confirmed an existing finding, which is exactly
when a result gets least scrutiny. The evidence was already in hand: mean output
length jumped from 769 to 3,446 characters. And the guard that would have caught
it had been written into a different module an hour earlier and not carried
across. The fix reorders the reply format so truncation costs the optional field,
and reports `parse_failures` per stage.

**Two score sets that could not be compared (DeepL, 2026-09-29).** The first
comparison table put DeepL's terminology-failure rate at 2.0% against 17–24% for every
other system, which looked like a large win. It was an artefact. The stored findings for
the twelve older systems were produced by an earlier version of the terminology detector;
DeepL had just been scored by the current one. One term, `Lymphknoten`, accounted for
almost all of it: about 50 flags for every older system and none for DeepL, even though
DeepL wrote "lymph nodes" as often as they did. Rescoring `glm-5.2`'s stored outputs with
current code took its terminology flags from 52 documents to 4, and `qwen35-4b`'s from 55
to 22.

What made this tractable was checking the *critical* rate too: it is identical under
both detector versions for all thirteen systems, so nothing published on negation,
laterality or numbers moves. Only the terminology findings, which are `major` and add
nothing to the critical rate, changed. All systems were rescored into a new directory,
`results/parrot_de/rescored_20260929/`, and every comparison in this chapter reads from
it. The `terminology_error_rate` column of the stored leaderboard is stale and was left
in place. **A fresh score set must never be compared to stored findings.**

**`consolidated/` is missing a system.** `results/parrot_de/consolidated/` holds twelve
files including the control, but the round-trip and published tables have thirteen.
`translategemma-27b` is in `results/parrot_3942876/`. A comparison built from
`consolidated/` alone silently left it out and mis-ranked DeepL by one place.

**A claim I made without measuring it.** After the DeepL run I described laterality as
its weak spot, "the failure the other systems mostly avoid". The per-system breakdown
says otherwise: DeepL flips laterality in 5.7% of documents, tied with `MiniMax-M3` and
mid-table (`DeepSeek-V4-Flash` 3.7%, `qwen35-27b` 12.5%). The only real difference from
the leaders is negation (1.4% against 0.3%), which is four documents against one and
cannot be interpreted.

### Verification practices adopted

- **Preflight everything.** Credentials, gated-repo access, API reachability,
  chat-template capability, glossary size and cached checkpoints are all probed
  before a long job is submitted. Five separate failure classes in this project
  presented as an opaque error minutes into a job and were each one cheap check
  at submission time.
- **Check the parse rate before reading any score** from a stage that asks a
  model for structured output, and put the critical field first in the required
  format so truncation damages the optional one.
- **Never transcribe numbers.** Figures and published tables are generated
  programmatically from `roundtrip_steps.csv`. This was adopted after several
  intermediate values in a hand-written chart data block turned out to be wrong — they
  had been typed from memory rather than read from the file. Published output is now
  re-parsed and diffed against the source CSV before release.
- **Look at the rendered figure.** Three separate layout defects — colliding scatter
  labels, a legend sitting on top of the data, and a mislabelled series — were only
  visible on inspection, not from the code.

## 6. Results

All numbers below are generated from
[`results/roundtrip_20260814_132452/roundtrip_steps.csv`](results/roundtrip_20260814_132452/roundtrip_steps.csv)
by [`scripts/make_figures.py`](scripts/make_figures.py) and the collection script —
none are transcribed by hand.

☁ marks hosted API systems. `c1` = cycle 1 = the ordinary single-pass benchmark.

### Master table

| System | BLEU c1 | BLEU c2 | BLEU c10 | Δ | chrF++ c1 | chrF++ c10 | crit% DE→EN | crit% EN→DE |
|---|---|---|---|---|---|---|---|---|
| `DeepSeek-V4-Flash` ☁ | 58.45 | 55.14 | 54.07 | −4.38 | 77.80 | 74.89 | 35 → 40 | 35 → 35 |
| `qwen35-27b` | 57.61 | 54.53 | 52.58 | −5.03 | 77.96 | 74.55 | 45 → 45 | 45 → 45 |
| `translategemma-27b` | 57.14 | 47.64 | 44.89 | −12.25 | 77.93 | 70.84 | 40 → 50 | 45 → 45 |
| `qwen35-4b` | 56.30 | 49.04 | 47.77 | −8.53 | 75.82 | 70.90 | 40 → 45 | 55 → 55 |
| `MiniMax-M3` ☁ | 56.17 | 52.81 | 52.41 | −3.76 | 75.82 | 74.18 | **30** → 40 | **30** → 35 |
| `glm-5.2` ☁ | 55.07 | 52.50 | 49.40 | −5.67 | 75.39 | 72.55 | 40 → 40 | **30** → **30** |
| `deepl` ☁ | 52.73 | 45.97 | 41.95 | −10.78 | 72.93 | 67.34 | 30 → 45 | 35 → 45 |
| `hymt2-30b-a3b` | 50.08 | 35.75 | 33.54 | −16.54 | 72.90 | 63.30 | 45 → 50 | 40 → 45 |
| `translategemma-4b` | 47.90 | 35.40 | 32.18 | −15.72 | 69.53 | 59.24 | 25 → 35 | 40 → 45 |
| `hymt2-7b` | 45.39 | 36.79 | 35.79 | −9.60 | 69.65 | 62.71 | 45 → 50 | 45 → 50 |
| `hymt2-1.8b` | 41.15 | 33.84 | 32.65 | −8.50 | 65.39 | 59.81 | 50 → 55 | 65 → 65 |
| `opus` | 28.88 | 25.77 | 24.15 | −4.73 | 54.69 | 49.12 | 10 → 30 | 30 → 35 |
| `nllb` | 22.88 | 16.04 | 14.27 | −8.61 | 50.31 | 38.56 | 35 → 50 | 50 → 65 |
| `identity` *(control)* | 3.39 | 3.39 | 3.39 | 0.00 | 25.33 | 25.33 | 95 → 95 | 0 → 0 |

With n=20 the critical-error rate moves in 5-point steps; **single-step differences are
not interpretable**.

---

### Finding 1 — surface quality does not predict clinical safety

![BLEU vs clinical error rate](figures/fig4_bleu_vs_clinical.png)

If BLEU predicted safety the points would lie on a downward line. They do not.

- `qwen35-27b` beats `MiniMax-M3` by 1.4 BLEU and damages **45%** of reports against
  MiniMax's **30%**. Ranked by BLEU you pick the less safe system.
- `translategemma-4b` scores 47.9 BLEU — nearly ten points below `qwen35-27b` — and has
  the lowest genuine single-pass error rate in the set at 25%.
- The three highest-BLEU systems span 35–45% critical errors, a spread as wide as the
  whole rest of the table.

The mechanism is visible in a single constructed example
([7. Metrics: what we compute, how to read it, and why](#7-metrics-what-we-compute-how-to-read-it-and-why) §0): dropping the word "No" from "No pleural effusion"
costs only 15 BLEU points, while an entirely harmless paraphrase costs 47. The metric is
working exactly as designed; the design is wrong for this question.

**This is not an artefact of n-gram matching.** COMET — a learned metric trained
on human adequacy judgements, which does not penalise paraphrase — produces a
*third* ordering, agreeing with neither BLEU nor the clinical layer, and ranks
`hymt2-30b-a3b` first while that system sits second-worst of eleven on clinical
errors. Full table in [8. Metric roadmap: what else should be measured, and in what order](#8-metric-roadmap-what-else-should-be-measured-and-in-what-order).

### Finding 2 — degradation is front-loaded and converges

![Round-trip curves](figures/fig1_roundtrip_curves.png)

The shape is the same in all twelve panels: a cliff between cycle 1 and cycle 2, then a
plateau. (The figure was generated before DeepL was run and shows twelve systems; DeepL
is in the tables.) Averaged over those twelve, **77% of all BLEU lost across ten round
trips is lost in the first one**; with DeepL it is 76%. By cycle 5 most systems have stopped changing entirely —
`hymt2-7b` is bit-identical from cycle 7 onward.

Translation converges to a fixed point rather than decaying without bound. The
practical implication is that round-trip degradation is a *property measurable in one
cycle*; ten cycles were needed to establish that, but not to use it.

Critical errors behave the same way. Seven of the twelve systems in the figure gain ≤5
points on DE→EN across all ten cycles (an earlier draft said five; it does not
reproduce from the data), and the systems that move most (`opus` +20, `nllb` +15,
DeepL +15) are mostly the weakest translators. The level, not the slope, is the story.

**DeepL is the one exception to "converges" in degree.** It loses 10.8 BLEU over ten
cycles, and only 63% of that in the first; it loses **4.0 more after cycle 2**, the most
of any system (next: `translategemma-4b`, 3.2). It does flatten (42.2, 41.8, 41.9 over
the last three cycles), so "fixed point" still holds, but it is reached late.

### Finding 3 — round-trip stability is an independent axis

![Quality lost](figures/fig2_quality_lost.png)

| System | Δ BLEU over 10 cycles | front-loaded share |
|---|---|---|
| `MiniMax-M3` ☁ | −3.76 | 89% |
| `DeepSeek-V4-Flash` ☁ | −4.38 | 76% |
| `opus` | −4.73 | 66% |
| `qwen35-27b` | −5.03 | 61% |
| `glm-5.2` ☁ | −5.67 | 45% |
| `hymt2-1.8b` | −8.50 | 86% |
| `qwen35-4b` | −8.53 | 85% |
| `nllb` | −8.61 | 79% |
| `hymt2-7b` | −9.60 | 90% |
| `deepl` ☁ | −10.78 | 63% |
| `translategemma-27b` | −12.25 | 78% |
| `translategemma-4b` | −15.72 | 80% |
| `hymt2-30b-a3b` | −16.54 | 87% |

`translategemma-27b` and `qwen35-27b` are indistinguishable on a single pass — 57.1 vs
57.6, well inside noise — and differ by a factor of 2.4 in round-trip loss. A benchmark
that reports only single-pass BLEU calls these two systems equivalent. They are not.

`hymt2-30b-a3b` is the sharpest case: third-best single-pass score, worst stability in
the set, finishing below the 1.8 B model of its own family.

Note that `opus` appears stable only because it has little left to lose — see Finding 5.

### Finding 4 — the hosted APIs lead on clinical safety

![Clinical errors](figures/fig3_clinical_errors.png)

The three hosted models take the three most stable positions on round-trip loss, and
`MiniMax-M3` and `glm-5.2` post the lowest genuine critical-error rates among
high-quality systems (30%). `glm-5.2` is the only system whose EN→DE rate does not move
at all across ten cycles (30 → 30).

This is a finding about *capability*, not deployability. Sending German radiology
reports to a third-party API is a data-protection decision, not a benchmark result, and
nothing here should be read as recommending it.

### Finding 5 — under-translation flatters the safety metric

![Output length](figures/fig5_output_length.png)

`opus` posts the lowest single-pass critical-error rate in the set (10%). It is also the
only system whose output *shrinks* — 599 → 506 characters, −16% — while every other
system's grows or holds.

A detector cannot flag a measurement that was never emitted. `opus`'s low score is
substantially an artefact of producing less text, which is consistent with its
second-from-bottom BLEU. **Figure 3 must not be read without figure 5.**

This is a general hazard for any source-vs-output safety metric, and it is one of the
reasons the metric roadmap ([8. Metric roadmap: what else should be measured, and in what order](#8-metric-roadmap-what-else-should-be-measured-and-in-what-order)) prioritises an
open-class recall instrument.

---

### Finding 6 — DeepL is competitive on the words and unstable under repetition

The commercial baseline, added 2026-09-29 and run on the free API plan
([3. Systems under test](#3-systems-under-test)). All 296 reports, scored with **one** detector
version across all thirteen systems (see [5. Experiments](#5-experiments) for why
that matters). **words%** is the share of documents with a critical *negation* or
*laterality* finding; number and measurement findings, which are two thirds of all
critical findings on this corpus, are shown separately and excluded from it.

| System | words% | numbers% | all-crit% | term% | BLEU | chrF++ | TER |
|---|---|---|---|---|---|---|---|
| `DeepSeek-V4-Flash` ☁ | 4.1 | 17.2 | 19.9 | 1.4 | 53.7 | 74.7 | 33.3 |
| `glm-5.2` ☁ | 4.7 | 15.9 | 19.3 | 1.4 | 51.3 | 73.4 | 34.8 |
| `MiniMax-M3` ☁ | 6.1 | 16.6 | 21.3 | 1.7 | 53.5 | 74.8 | 33.5 |
| `translategemma-4b` | 6.1 | 17.9 | 23.0 | 2.7 | 44.7 | 68.0 | 45.2 |
| `translategemma-27b` | 6.4 | 18.9 | 23.6 | 1.4 | 55.0 | 75.8 | 34.9 |
| **`deepl`** ☁ | **6.8** | 20.6 | 25.7 | 2.0 | 52.7 | 73.8 | 36.2 |
| `hymt2-1.8b` | 7.4 | 19.6 | 25.3 | 2.0 | 39.1 | 64.2 | 49.0 |
| `hymt2-7b` | 8.1 | 18.6 | 24.3 | 2.7 | 46.0 | 69.2 | 42.8 |
| `hymt2-30b-a3b` | 9.8 | 25.7 | 31.1 | 3.0 | 47.7 | 71.2 | 41.8 |
| `qwen35-4b` | 11.8 | 20.3 | 24.7 | 7.4 | 48.0 | 69.0 | 40.9 |
| `qwen35-27b` | 13.9 | 24.0 | 28.4 | 8.1 | 51.7 | 70.4 | 38.2 |
| `nllb` | 19.9 | 25.3 | 33.4 | 17.9 | 21.8 | 46.6 | 65.0 |
| `opus` | 20.6 | 18.2 | 26.7 | 12.2 | 24.6 | 49.0 | 61.4 |

DeepL is **sixth of thirteen on the words** and fourth on BLEU (per-document mean).
Ranked by all-crit it drops to ninth, because its number-finding rate is the fifth
highest; that is only partly a typographic artefact (below), which is why the words-only
column exists.

- **The words-only ranking differs from the standard one, and in a useful direction.**
  `opus` looks middling on all-crit (26.7%) but is the *worst* system on the words
  (20.6%). It emits fewer numbers, so it has fewer to get wrong — the under-translation
  bias of Finding 5, visible from the other side.
- **DeepL is unremarkable on laterality and negation.** Laterality: 5.7% of documents,
  tied with `MiniMax-M3`, better than the seven weakest systems. Negation: 1.4%
  (four documents) against 0.3% (one) for the three hosted LLMs. At this sample size that
  is not a difference.
- **Terminology is not a DeepL strength.** After the detector-version correction its
  2.0% sits in the ordinary 1.4–3.0% band of the good systems.
- **Round trip: the weakness.** −10.78 BLEU over ten cycles against −3.8 to −5.7 for the
  best hosted LLMs, and the slowest to settle (Finding 2). Its critical rate goes
  30% → 45% DE→EN, which at n = 20 is three documents.
- **It has no COMET score**, nor does `translategemma-27b`; the COMET run predates both.

**The `5-mm` artefact, measured.** DeepL writes a hyphenated unit for compound
modifiers ("a 5-mm nodule"), and the number detector's measurement pattern requires
whitespace between value and unit, so it reads the hyphenated form as a mismatch
against the source's `5 mm` — the same code it gives a genuine `15 mm` error.
Normalising the hyphen and re-scoring: DeepL emits it in **5 of 296 reports**, and
doing so clears **2** critical documents (25.68% → 25.00%). `hymt2-30b-a3b` is the only
other system affected (−0.34 points). The effect is real but small; it was described
as "probably inflated" before being measured, which overstated it. The detector
was **not** changed, because altering the shared instrument would revalue the whole
benchmark.

**What this does to the founding question.** The strongest commercial baseline lands in
the middle of the field on the words, well behind the leading hosted LLMs, and is less
stable than any of them under repetition. That neither answers "does it need a
specialised model" nor supports "commercial MT is enough". It does mean the ~1-in-5
document-level failure rate is not a property of open models: it is what the best
available systems, commercial or not, produce on this corpus.

### Answering the question

**Does German↔English medical translation need a specialised model?**

On the current evidence, **the case for fine-tuning is not yet made — but the case for
better evaluation is overwhelming.**

Three things point that way.

1. **The best available systems already score well on surface metrics.** 58 BLEU on
   radiology reports is good. A fine-tune would be chasing a few points on a metric
   already shown not to track the thing that matters.
2. **Every system fails clinically, at a rate no fine-tune plausibly closes.** The best
   genuine single-pass critical-error rate among high-quality systems is 30% of reports.
   That is not a gap a domain adapter closes; it is a different problem.
3. **The measurement is not yet trustworthy enough to detect success.** The detectors
   have known false positives (~38% on numbers), unmeasured false negatives (the
   `BWK 12` → `L12` miss), and a documented bias toward rewarding under-translation.
   Fine-tuning against an instrument with these properties risks optimising the
   instrument rather than the translation.
4. **Three interventions that should have helped did not.** Injecting a
   dictionary, injecting a 16× larger dictionary, and running a three-model
   debate each moved the clinical error rate by zero or made it worse
   ([10. Experiments: dictionary injection, debate, and a review cascade](#10-experiments-dictionary-injection-debate-and-a-review-cascade)).
   Since every one of those is cheaper than fine-tuning and none of them worked,
   there is no reason to expect the expensive intervention to succeed where they
   failed — *unless* the reason they failed is that the instrument cannot see
   their effect, which is exactly what needs establishing first.

The failures also point somewhere specific. Across both experiments the surviving
critical findings are overwhelmingly **numbers and measurements** — every one of
the debate's eight failures, and 12 of 14 in the dictionary baseline. Terminology
interventions cannot reach that category by construction. If a targeted fix
exists, it is constrained decoding or a numeric post-check, not a glossary and not
a committee.

The honest next step is the metric work in
[8. Metric roadmap: what else should be measured, and in what order](#8-metric-roadmap-what-else-should-be-measured-and-in-what-order) — a learned semantic metric, an open-class
error finder, and a small human-MQM validation set — and only then a decision about
model building. Steps 1–3 of that roadmap need no new translations and no new
annotation; all fourteen systems' outputs are already on disk.

### Threats to validity

- **n=20 for the round-trip.** Resolution is 5 points on the error rate. Single-step
  differences mean nothing.
- **One corpus.** Findings 1–5 are established on PARROT-DE radiology reports. They may
  not transfer to patient-facing or regulatory text, where the single-pass benchmark
  used three other corpora.
- **Fictional reports.** PARROT reports are radiologist-authored but not real patient
  data, and may be cleaner than production dictation.
- **Detector precision and recall** — quantified where possible, unbounded where not.
  See [7. Metrics: what we compute, how to read it, and why](#7-metrics-what-we-compute-how-to-read-it-and-why) §2.3.
- **DeepL is a single dated run on an unversioned service.** No model id is exposed, so
  it cannot be re-run to the same result later, and it was run once.
- **Score sets from different dates are not comparable.** The terminology detector
  changed after the July runs; see [5. Experiments](#5-experiments). The published
  critical rates are unaffected, but any terminology figure from the stored leaderboard
  is stale.
- **No human validation yet.** Nothing in this chapter has been checked by a clinician.
  That is the single largest gap.

## 7. Metrics: what we compute, how to read it, and why

Four instruments across three layers. They answer different questions and they
disagree with each other, which is the point — §4 shows BLEU, COMET and the
clinical detectors producing three different rankings of the same systems.

| Layer | Metric | Range | Direction | Needs a reference? |
|---|---|---|---|---|
| Surface | BLEU | 0–100 | higher better | yes |
| Surface | chrF++ | 0–100 | higher better | yes |
| Surface | TER | 0–∞ (usually 0–100) | **lower** better | yes |
| Clinical | critical-error rate | 0–100% | **lower** better | no — compares source to output |
| Clinical | finding counts by code | integer | lower better | no |
| Semantic | COMET | ~0–1 | higher better | yes |
| Clinical (open-class) | LLM judge, critical-error rate | 0–100% | **lower** better | no — compares source to output |

---

### 0. The one table that motivates everything

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

### 1. Surface metrics

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

#### 1.1 BLEU

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

#### 1.2 chrF++

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

#### 1.3 TER — Translation Edit Rate

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

### 2. Clinical information loss

The second layer ignores the reference translation and compares the **source** against
the **output** directly, looking for the specific ways a radiology report can become
dangerous. Four detectors, implemented in
[`taxonomy/clinical.py`](src/medmt_eval/taxonomy/clinical.py).

#### 2.1 The detectors

| Detector | Code(s) | Severity | Method |
|---|---|---|---|
| Negation | `negation_dropped`, `negation_introduced` | critical | Per-language cue lexicon (`kein`, `nicht`, `ohne` / `no`, `not`, `without`, …). Fires when the source has negation cues and the output has none, or vice versa. |
| Laterality | `laterality_missing_or_flipped`, `laterality_added_or_flipped` | critical | Lexicon maps to the set {left, right, bilateral}. Fires on any set difference between source and output. |
| Number / measurement | `number_or_measurement_mismatch` | critical | Parses every number and unit, normalises decimals (German `1,5` = English `1.5`) and units, then compares multisets. |
| Terminology | `terminology_not_preserved` | **major** | For each term-bank concept found in the source, checks the expected target term appears in the output, tolerating inflection. |

**Severity matters and is easy to misread.** Only the first three are `critical`.
Terminology findings are `major` and contribute **zero** to the critical-error rate.
A model can have many terminology findings and a low critical-error rate.

#### 2.2 Critical-error rate — the headline number

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

#### 2.3 Known limitations — read before quoting these numbers

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
  [figure 5](figures/fig5_output_length.png) exists and must be read alongside
  [figure 3](figures/fig3_clinical_errors.png).
- **Coverage varies by language pair.** `detector_coverage()` reports which detectors
  can run. Negation, laterality and terminology need a lexicon on *both* sides; for
  Turkish–English only the language-agnostic number check is active, so Turkish
  critical-error rates are **not comparable** to German ones.

These are the reasons [8. Metric roadmap: what else should be measured, and in what order](#8-metric-roadmap-what-else-should-be-measured-and-in-what-order) exists.

---

### 3. Round-trip measurements

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

The front-loaded share averages 77% across the twelve systems (76% with DeepL, the slowest
to settle at 63%), which is the evidence for the convergence claim in
[6. Results](#6-results).

---

### 4. COMET — the learned semantic metric

**What it is.** Source, hypothesis and reference are each encoded with
XLM-R-large; the embeddings and their combinations feed a feed-forward regressor
trained to predict human MQM scores. Checkpoint in use:
`Unbabel/wmt22-comet-da` (reference-based).

**How to read it.** Roughly 0–1, higher better, but the scale is compressed and
not calibrated across language pairs — on this corpus the entire real field spans
0.67 to 0.83, and the `identity` control sits at 0.61. **Differences under ~0.01
should not be read as meaningful**, and a COMET score is never comparable across
test sets.

**Why it is here.** It fixes the specific failure demonstrated in §0: BLEU
punishes a harmless paraphrase (53.04) harder than a dropped negation (84.46)
because it counts n-grams. COMET reasons over representations, so paraphrase
stops being penalised. It is the standard modern answer to "BLEU is not enough".

**What it cannot do, and this is measured rather than assumed.** It still does not
track clinical risk. On this corpus it ranks `hymt2-30b-a3b` **first** — a system
sixth on BLEU and second-worst of eleven on critical clinical errors. See
[8. Metric roadmap: what else should be measured, and in what order](#8-metric-roadmap-what-else-should-be-measured-and-in-what-order) for the full table. It is also
off-distribution here: trained on WMT news, applied to radiology reports, and
learned metrics are documented to degrade outside their training domain
([arXiv:2402.18747](https://arxiv.org/abs/2402.18747)).

The failure is visible at the level of a single segment, using the same sentence
as §0:

| source → candidate | COMET |
|---|---|
| "Kein Pleuraerguss." → "No pleural effusion." ✅ | 0.9809 |
| "Ein 5 mm Knoten links." → "A 5 mm nodule on the left." ✅ | 0.9702 |
| **"Kein Pleuraerguss." → "Pleural effusion."** ❌ negation dropped | **0.9110** |

Inverting the finding — the patient now *has* the effusion the source rules out —
costs COMET seven hundredths and still scores 0.91. BLEU charged the identical
error 15.5 points (§0). The metric designed to capture semantic adequacy
reproduces BLEU's blind spot rather than fixing it, which is the cleanest single
demonstration that the problem is not n-gram matching.

**Operational note.** `unbabel-comet` 2.2.7 requires `transformers` 4.x and
cannot share this project's environment; it lives in `.venv-comet`. Installing it
into the main venv downgraded transformers and broke `Qwen3.5` loading for
unrelated jobs. The isolated venv also needs `setuptools<81`: setuptools 81+
**removed** `pkg_resources`, which COMET 2.2.7 still imports — so `pip install
setuptools` reports "already satisfied" while the import keeps failing.

### 5. LLM-as-judge with a clinical rubric

**What it is.** A strong model is prompted to annotate error spans in the
translation, given only the source. Implemented in
[`metrics/llm_judge.py`](src/medmt_eval/metrics/llm_judge.py), following
GEMBA-MQM ([arXiv:2310.13988](https://arxiv.org/abs/2310.13988)) with two
deliberate departures:

- **The rubric is clinical, not generic.** GEMBA's categories are
  accuracy/fluency/terminology, which cannot distinguish a mistranslated
  adjective from a mistranslated vertebra level. The categories here are
  `negation`, `laterality`, `measurement`, `anatomy`, `finding`, `certainty`,
  `terminology`, `omission`, and severity is defined by effect on patient
  management rather than by linguistic magnitude.
- **It scores against the source, never a reference**, so legitimate paraphrase
  is not penalised.

**Why it is here.** It is the *open-class* counterpart to the rule detectors.
The rules find only what someone wrote a rule for; their recall is unknown and
demonstrably imperfect — the `BWK 12` → `L12` miss had no rule. A judge can flag
errors nobody anticipated. `agreement_with_detectors()` reports the confusion
between the two layers directly, and the `judge_only` cell **is** the recall gap.

**How to read it — carefully.**

- **Prompt sensitivity is severe.** Moving GEMBA from a bare prompt to a
  rubric-style one moved correlation with human judgement from 0.09 to 0.35
  ([RUBRIC-MQM, ACL 2025 Industry](https://aclanthology.org/2025.acl-industry.12/)).
  This rubric is **unvalidated against clinicians**. Treat findings as candidates
  for human review, not as ground truth.
- **Unparseable replies are counted separately.** A model that returns prose
  instead of JSON contributes zero findings; silently treating that as "clean"
  would bias every system toward looking good, so `unparseable_replies` is
  reported and a run with many of them is not a clean run.
- **Self-preference is guarded in code.** `run_llm_judge` raises if the judge is
  the system under test, matching on the leaf name so `z-ai/glm-5.2` is caught
  when testing `glm-5.2`.

### 6. What is deliberately *not* used

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

### References

- Papineni et al. (2002), *BLEU: a Method for Automatic Evaluation of Machine Translation*, ACL.
- Popović (2017), *chrF++: words helping character n-grams*, WMT.
- Snover et al. (2006), *A Study of Translation Edit Rate with Targeted Human Annotation*, AMTA.
- Post (2018), *A Call for Clarity in Reporting BLEU Scores*, WMT. (sacreBLEU)
- Lommel et al., *Multidimensional Quality Metrics (MQM)*, [themqm.org](https://themqm.org/error-types-2/the-mqm-scoring-models/).
- Mehandru, Agrawal, Xiao, Khoong, Gao, Carpuat & Salehi (2023), *Physician Detection of
  Clinical Harm in Machine Translation: Quality Estimation Aids in Reliance and
  Backtranslation Identifies Critical Errors*, EMNLP. [arXiv:2310.16924](https://arxiv.org/abs/2310.16924)

## 8. Metric roadmap: what else should be measured, and in what order

The current instrument ([7. Metrics: what we compute, how to read it, and why](#7-metrics-what-we-compute-how-to-read-it-and-why)) is high-precision and narrow.
BLEU/chrF++/TER measure string similarity; the clinical detectors measure four specific
failure modes with hand-written lexicons. Neither is what the field now considers
state of the art, and neither is sufficient for the final claim of this work.

This chapter surveys the alternatives and commits to an order of adoption.

---

### The gap, stated precisely

Three things are missing.

1. **A learned semantic metric.** Every surface metric here is lexical. None can tell
   that "pleural fluid collection" and "pleural effusion" mean the same thing — as
   §0 of [7. Metrics: what we compute, how to read it, and why](#7-metrics-what-we-compute-how-to-read-it-and-why) shows, the harmless paraphrase is punished
   *harder* than the dangerous negation flip. Neural metrics fix exactly this.
2. **Recall over clinical errors.** The detectors find what they were written to find.
   The `BWK 12` → `L12` miss (thoracic vertebra rendered as lumbar, no detector fired)
   is not a bug to patch — it is the signature of a closed-class approach. Something
   open-class is needed to bound the true error rate.
3. **More than one reference — or none.** This was added after
   [Experiment 4](#10-experiments-dictionary-injection-debate-and-a-review-cascade)
   measured the size of the problem, and it is the most concrete of the three.

   PARROT gives one English translation per report, and on the vocabulary that
   matters the corpus does not agree with itself about what that translation is.
   `Beurteilung` is rendered *conclusion* 50 times, *impression* 32 and
   *assessment* 6 across the reports where both sides carry the header, so **the
   most frequent form accounts for only 57% of occurrences** — 52% on the
   held-out half, where it is nearly a coin flip between *conclusion* and
   *impression*. All of those forms are what radiologists write — this project's
   own section classifier in
   [`data/sections.py`](src/medmt_eval/data/sections.py) lists
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

### Candidates

#### A. COMET-22 / COMET-Kiwi — learned regression on human judgements

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

#### B. XCOMET / MetricX-25 / GemSpanEval — error-span prediction

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

#### C. LLM-as-judge — GEMBA-MQM and successors

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

#### D. MQM proper — the human ceiling

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

#### E. Considered and rejected

| Metric | Why not |
|---|---|
| METEOR | Superseded by chrF++ and neural metrics; weak German support. |
| ROUGE | Built for summarisation; no advantage over chrF++ here. |
| BERTScore | Not trained on translation judgements; COMET dominates it for MT. |
| Reference-free QE **as a safety signal** | Directly documented to fail this task — see [7. Metrics: what we compute, how to read it, and why](#7-metrics-what-we-compute-how-to-read-it-and-why) §4. Keep as a triage signal only. |

---

### Result — COMET disagrees with BLEU, and with the clinical layer

Run 2026-08-22 over the existing single-pass PARROT outputs (296 reports, no
retranslation), `Unbabel/wmt22-comet-da`, results in
[`results/comet_parrot_de.json`](results/comet_parrot_de.json).

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

#### Environment note: COMET cannot share this project's transformers

`unbabel-comet` 2.2.7 requires `transformers` 4.x — installing it silently
downgraded the shared venv from 5.15.1 to 4.57.6, which broke `Qwen3.5` loading
everywhere (`model type 'qwen3_5' not recognized`) and failed a cascade job that
had nothing to do with COMET. Upgrading back breaks COMET in turn: its XLM-R
encoder unpacks a three-tuple that transformers 5.x no longer returns
(`not enough values to unpack (expected 3, got 2)`).

They are mutually exclusive in one environment. COMET therefore lives in its own
`.venv-comet`; the main venv stays on transformers 5.x for the translation
models.

### Result — DeepL benchmarked; the glossary arm is still open

**Done 2026-09-29.** DeepL now has a single pass over PARROT-DE and the ten-cycle
round trip. Full numbers in
[6. Results](#6-results).
In short: sixth of thirteen on negation and laterality (6.8% of documents), fourth on
BLEU, and the least stable of the strong systems under repetition (−10.8 BLEU against
−3.8 to −5.7 for the best hosted LLMs). It does not change the conclusion, and it removes
the objection that every claim in the thesis was relative to open models and one
hosted gateway.

It was kept separate from German MeSH as planned: DeepL was scored against the human
references on PARROT, never against an MT-derived term bank, because German MeSH is
itself a DeepL first pass ([9. Terminology: which dictionary, and what it costs to get it wrong](#9-terminology-which-dictionary-and-what-it-costs-to-get-it-wrong)).

**Practical notes.** It runs on the login node (the GPU nodes have no internet). The free
plan allows **1,000,000 characters a month**, not the 500,000 an earlier version of this
section stated; the run used 530,862 in total (226,785 for the single pass), leaving
roughly 469,000 unspent. See [3. Systems under test](#3-systems-under-test) for the adapter fixes.

**Still open: the glossary arm, and it is the more informative half.** DeepL applies a
glossary inside the translation engine rather than as a prompt instruction, so a DeepL
glossary run would separate "dictionaries do not help" from "*prompt-injected*
dictionaries do not help". `/v3/glossaries` is reachable on the project's key (HTTP 200,
no glossaries yet), but **whether the free plan allows creating one has not been tested**.
The cheapest first step is a single tiny glossary. The ceiling analysis in
[09](#10-experiments-dictionary-injection-debate-and-a-review-cascade) predicts a small effect on the clinical layer
whatever the glossary contains, since two thirds of critical errors are numeric, so this
is a test of mechanism, not an expected fix.

### Order of adoption

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

### References

- Rei et al. (2022), *COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task*, WMT. [aclanthology](https://aclanthology.org/2022.wmt-1.52/)
- Guerreiro et al. (2024), *xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection*, TACL. [MIT Press](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00683/124263/)
- Juraska et al. (2025), *MetricX-25 and GemSpanEval: Google Translate Submissions to the WMT25 Evaluation Shared Task*. [arXiv:2510.24707](https://arxiv.org/abs/2510.24707)
- Kocmi & Federmann (2023), *GEMBA-MQM: Detecting Translation Quality Error Spans with GPT-4*. [arXiv:2310.13988](https://arxiv.org/abs/2310.13988)
- *GEMBA V2: Ten Judgments Are Better Than One*, WMT 2025. [aclanthology](https://aclanthology.org/2025.wmt-1.67/)
- *RUBRIC-MQM: Span-Level LLM-as-judge in Machine Translation For High-End Models*, ACL 2025 Industry. [aclanthology](https://aclanthology.org/2025.acl-industry.12/)
- Zouhar et al. (2024), *Fine-Tuned Machine Translation Metrics Struggle in Unseen Domains*. [arXiv:2402.18747](https://arxiv.org/abs/2402.18747)
- Zouhar et al. (2024), *Pitfalls and Outlooks in Using COMET*, WMT. [arXiv:2408.15366](https://arxiv.org/abs/2408.15366)
- Mehandru et al. (2023), *Physician Detection of Clinical Harm in Machine Translation*, EMNLP. [arXiv:2310.16924](https://arxiv.org/abs/2310.16924)

## 9. Terminology: which dictionary, and what it costs to get it wrong

Both experiments in [10. Experiments: dictionary injection, debate, and a review cascade](#10-experiments-dictionary-injection-debate-and-a-review-cascade)
depend on a bilingual glossary. This chapter is the evaluation of the candidates and
the reasoning for the one in use.

### Recommendation

| Rank | Source | Verdict |
|---|---|---|
| **1** | **RadLex** (`RadLex.owl`) | **Obtained and in use.** 45,163 EN-DE pairs from explicit `xml:lang` tags. Radiology-native, so none of the cross-domain polysemy below. Needs the function-word filter. |
| 2 | **Wikidata**, MeSH/UMLS-anchored, filtered to MeSH branches A/C/E | **Merged in as a supplement.** CC0, no registration. Adds 3.5 terms/report of general medical vocabulary RadLex lacks; needs all the hygiene below or it is actively harmful. |
| 3 | UMLS | The join key (CUI) that makes everything else composable. Worth starting the licence now; not needed to run the experiments. |
| 4 | SNOMED CT DE, ICD-10-GM / ICD-11 | Diagnoses and findings backbone. Registration-gated, valuable later. |
| — | **German MeSH (ZB MED)** | **Excluded on purpose.** See "the circularity problem". |
| — | RadReport.org | Does not do what we need. See "what I checked and rejected". |

### The circularity problem — why German MeSH is excluded

German MeSH is a DeepL first pass with human curation on top. That is fine for many
uses and disqualifying for this one.

Both experiments show a model a term and then measure whether the output used it. If
the term's German side was produced by a machine translator, the measurement is
"did this model agree with DeepL", not "was this correct". Any model whose training
distribution resembles DeepL's output scores well for the wrong reason, and the
terminology detector — which is the metric most directly affected — reports an
improvement that means nothing.

This is enforced in code rather than left to discipline:
`Glossary.exclude_mt()` drops any entry whose `source` is in `MT_DERIVED_SOURCES`, and
it runs by default. Overriding it takes an explicit `--allow-mt-derived`.

German MeSH is still worth having as a **recall aid** for finding candidate terms, and
as a comparison target once a non-MT gold set exists. It is not a gold standard.

### What I checked and rejected

#### RadReport.org — the API is live, the German corpus is not there

Checked directly against `api3.rsna.org/radreport/v1`:

| Language | Templates |
|---|---|
| English | 269 |
| Turkish | 23 |
| **German** | **9** |

Two findings, both of which undercut using it as the primary source:

1. **German coverage is nine templates.** Not enough to derive a glossary from.
2. **Template IDs do not align across languages.** The intersection of German and
   English template IDs is *empty*, as is Turkish ∩ English. Each translation is a
   separate template with its own id, so template ID is not a parallel-corpus key.

What it *is* good for: the nine German templates carry bilingual titles in the source
data (`"CT Lungenembolie (CT pulmonary embolism)"`), so they yield a small,
register-perfect seed list, and the licence is genuinely unrestricted. Worth mining for
those nine, not worth prioritising.

#### RadLex — obtained, and the OWL is the file that matters

Two RadLex distributions were supplied on 2026-08-21/22. They are not equivalent,
and the difference is the whole ballgame.

| File | German content | EN-DE pairs recovered |
|---|---|---|
| `RADLEX.csv.gz` (BioPortal export) | `Preferred_name_German` populated for **1 concept of 46,900**; some untagged German inside the `Synonyms` field | **389**, by rule-based language ID |
| **`RadLex.owl`** | **47,414 explicit `xml:lang="de"` tags** on `rdfs:label` and `RID:Synonym` | **45,163**, read directly |

**Use the OWL.** The DRG translation is there as proper language tags; nothing has
to be inferred. The BioPortal CSV path in
[`scripts/build_glossary_radlex.py`](scripts/build_glossary_radlex.py) is kept
only as a fallback, and its rule-based German detector exists because langdetect
is unusable on single medical terms — it called *Epiduralhämatom* Estonian and
*posteriore Schallverstärkung* Swedish.

Three things the OWL parser has to get right:

1. **Concepts are serialised twice.** Most labels sit in an `owl:Class` block, but
   a later `rdf:Description` block carries more of them. Of the 47,414 German
   tags, only 21,810 are in the class section. Parsing classes alone silently
   loses more than half the translation, so every element with an `rdf:about`
   under `/RID/` is read and merged on the RID.
2. **The German label is often Latin.** RID10012's `rdfs:label@de` is *Aponeurosis
   palatina*; its `RID:Synonym@de` is *Gaumenaponeurose*. Latin is correct German
   anatomical nomenclature and is not what a dictated report says, so the most
   German-looking tagged string becomes the preferred term and the rest — Latin
   included — become aliases.
3. **RadLex ships function words as concepts.** RID28454 is literally
   `kein → none` and RID28475 is `nicht → no`. See the hygiene section below;
   this one is a safety problem, not a tidiness problem.

### Coverage — the number that decides whether Experiment 1 can work

| Glossary | Entries | Terms/report | Zero-hit reports |
|---|---|---|---|
| RadLex from the CSV | 389 | 0.3 | 228 / 296 |
| Wikidata A/C/E | 2,834 | 3.5 | 26 / 296 |
| **RadLex from the OWL** | **45,163** | **14.5** | **2 / 296** |
| **merged (RadLex + Wikidata)** | **47,997** | **18.0** | **2 / 296** |

This is a five-fold increase in the amount of terminology a model is actually
shown, and it removes the strongest confound in
[10. Experiments: dictionary injection, debate, and a review cascade](#10-experiments-dictionary-injection-debate-and-a-review-cascade): the first
run of Experiment 1 returned a null result on a glossary supplying 3.0 terms per
document, which could not distinguish "dictionaries do not help" from "this
dictionary was too thin to test the question".

### Is RadLex's terminology what a clinician actually writes?

Coverage is not correctness. RadLex is radiology-native, but that guarantees it
is *about* radiology, not that its preferred terms are the ones a radiologist
puts in a report. The corpus answers this directly: when RadLex's German term
appears in a source report, does its English term appear in the radiologist's own
English translation of that same report?

**Over 4,294 occurrences across 296 report pairs, agreement is 66%.**

By RadLex's own taxonomy:

| branch | occurrences | agrees with the radiologist |
|---|---|---|
| **clinical finding** | 631 | **86%** |
| procedure | 34 | 79% |
| object | 26 | 69% |
| **anatomical entity** (84% of RadLex) | 1,429 | **66%** |
| RadLex descriptor | 1,463 | 64% |
| property | 227 | 55% |
| imaging observation | 355 | 55% |
| non-anatomical substance | 37 | 43% |
| **report component** | 62 | **0%** |

The failures are systematic, and each has a different cause:

1. **Report components: zero agreement.** `Beurteilung → assessment`,
   `Indikation → indication`, `Vergleich → comparison section`. RadLex's names
   for report sections are never what the radiologist writes. `Beurteilung`
   alone accounted for 125 misses — the single largest source of bad injections
   in the whole bank.
2. **Anatomy is systematically too formal.** `Unterlappen → lower lobe of lung`
   (clinician: "lower lobe"), `Harnblase → urinary bladder` ("bladder"),
   `Kontrastmittel → contrast agent` ("contrast"), `basal → basilar`. Ontological
   precision, report brevity.
3. **Descriptors and properties are not terminology.** `frei → clear`,
   `gering → minor`, `groß → large`, `niedrig → low`.
4. **Ambiguity is unresolved.** `Flüssigkeit` has two RIDs — `→ fluid` (43 hits)
   and `→ liquid` (43 misses). The ontology ships both and injecting both is
   worse than injecting neither.

There is a German-side symptom of the same problem: **only 1.8% of RadLex's
45,163 German terms ever appear in 296 real reports** (0.9% for anatomy). That is
not a coverage complaint — it is evidence that the German side is also written in
ontology register rather than dictation register.

#### How much of this branch ranking is real?

Splitting the 296 reports in half and measuring each independently:

| branch | first half | second half |
|---|---|---|
| clinical finding | 88% (n=260) | 84% (n=371) |
| RadLex descriptor | 65% (n=639) | 64% (n=824) |
| anatomical entity | 70% (n=523) | 64% (n=906) |
| imaging observation | 61% (n=155) | 50% (n=200) |
| **property** | **37% (n=106)** | **71% (n=121)** |
| report component | 0% (n=27) | 0% (n=35) |

The extremes replicate; the middle does not. `property` swings by 34 points
between halves.

**This matters methodologically, not just descriptively.** A first pass selected
branches by a ≥70% agreement threshold — and that threshold falls exactly where
`anatomical entity` changes sides between halves, so it was fitted to the
evaluation set. The defensible selection is the one that would have been made
without looking: keep clinical findings and procedures because they carry
clinical meaning, drop report components and descriptors because they are not
terminology. The measurement then *confirms* that choice at the stable extremes
rather than generating it.

#### The precision-first bank

[`radlex_findings_de_en.csv`](data/term_banks/radlex_findings_de_en.csv) —
clinical findings and procedures only, **2,777 entries, 85% agreement**, 2.2
terms per report, 59 of 296 reports with no match.

That is a deliberate trade of coverage for trust, and it is the right direction
given that [Experiment 1](#10-experiments-dictionary-injection-debate-and-a-review-cascade) found *more*
terminology made translation worse.

### The hygiene problem — what a general-purpose source does to clinical German

Wikidata linked to MeSH/UMLS gives 22,200 DE/EN concepts for free. Used naively it is
worse than no glossary at all. Three distinct failure modes showed up on PARROT, all
found by measurement rather than inspection.

#### 1. Short aliases are catastrophic

Wikidata stores chemical symbols and initialisms as aliases. Matched
case-insensitively against German clinical text:

| Alias | Concept | What it actually matched |
|---|---|---|
| `Cm` | curium | **cm** — the centimetre unit, in nearly every report |
| `CT` | *circuit training* | **CT** — the modality |
| `In` | indium | German *in* — 125 spurious hits |
| `Es` | einsteinium | German *es* |
| `Am` | americium | German *am* |

Before the fix, the most frequently "matched" term in the whole corpus was **indium**.

**Fix:** aliases below 4 characters are dropped (`MIN_ALIAS_LENGTH`), and preferred
labels below 3 (`MIN_TERM_LENGTH`) — "Es" is MeSH's preferred label for the
psychoanalytic *id*.

#### 2. General terminologies carry the whole tree

MeSH is not a radiology terminology. Its branches include Z (Geographicals) and
L (Information Science), which is where "Deutsche Demokratische Republik" and
"maschinelles Lernen" came from.

**Fix:** `Glossary.in_branches("ACE")` — A (Anatomy), C (Diseases), E (Diagnostic and
Therapeutic Techniques and Equipment). This is what a radiology report is made of.
22,200 concepts → 2,834.

#### 3. A radiology terminology ships function words as concepts

RadLex is radiology-native, which removes the cross-domain polysemy above — and
introduces a different problem. It models report *language*, not only report
*content*, so its concept list includes closed-class words:

| RID | Entry | Why it is dangerous |
|---|---|---|
| **RID28454** | **`kein → none`** | **"Kein Erguss" is "No effusion", never "None effusion".** Negation is one of the three critical error classes measured here, so injecting this gloss could manufacture the exact failure under study. |
| RID28475 | `nicht → no` | Useless; no model needs it. |
| RID49853/4 | `mit → with`, `ohne → without` | Useless. |

Before filtering, `nicht` (212 matches) and `kein` (197) were the two most
frequently injected terms in the entire corpus.

**Fix:** `FUNCTION_WORD_STOPLIST` removes closed-class words on both the source
and the target side — an entry can be selected on a legitimate source term and
still gloss it to a function word, so `render()` guards the target independently.
Coverage drops 17.2 → 14.5 terms per report, which is the right trade.

Content words that merely look generic are kept: `Beurteilung → assessment`,
`Indikation → indication` and `Vergleich → comparison section` are real report
section headers and useful.

One residual, not fixed: `Flüssigkeit` maps to both `fluid` and `liquid` via two
distinct RIDs, so both are injected. Genuine ontological ambiguity rather than an
extraction error.

#### 4. Polysemy survives both filters

Even inside branch E, a correct alias can be a false friend in a report:

| Surface form | Wikidata concept | Meaning in a radiology report |
|---|---|---|
| `Darstellung` | chemical synthesis | depiction, visualisation |
| `Technik` | technology | imaging technique |
| `Höhe` | altitude | level |

**Fix:** `DEFAULT_STOPLIST`, a short explicit list. This is the mode that does *not*
have a principled fix within a general-purpose source, and it is the reason RadLex
ranks first: a radiology terminology cannot contain "Darstellung = chemical synthesis"
because it does not contain chemical synthesis.

### Where that leaves the working glossary

| Stage | Concepts |
|---|---|
| RadLex OWL, EN+DE tagged | 45,163 |
| Wikidata (MeSH + UMLS anchored) | 22,200 |
| Wikidata after `in_branches("ACE")` | 2,834 |
| **merged working bank** | **47,997** |

Coverage on the 296 PARROT German reports: **mean 18.0 terms per report, median
15, only 2 reports with no match at all.** The most frequently injected terms are
`Beurteilung → assessment`, `Indikation → indication`, `Läsion → lesion`,
`Pleuraerguss → pleural effusion`, `Fraktur → fracture`, `Pneumothorax`,
`Niere → kidney`.

That is a real glossary, and it is what makes the dictionary question answerable.
The first run of Experiment 1 was conducted on the 2,834-entry Wikidata bank at
3.0 terms per document and returned a null result that could not be distinguished
from "the glossary was too thin"; the rerun on this bank can.

### Reproducing

```bash
# RadLex — 45,163 pairs, seconds, needs datasets/RadLex.owl
python3 scripts/build_glossary_radlex.py datasets/RadLex.owl \
    -o data/term_banks/radlex_de_en.csv

# Wikidata — 22,200 concepts, ~20 min, no credentials needed
python3 scripts/build_glossary_wikidata.py -o data/term_banks/wikidata_med.csv

# inspect what a filter does before trusting it
python3 -c "
from medmt_eval.glossary import Glossary
g = Glossary.from_csv('data/term_banks/wikidata_med.csv')
print(len(g), '->', len(g.exclude_mt().in_branches('ACE')))"
```

### Still to obtain

| Source | Blocker | Value |
|---|---|---|
| MuchMore (DE-EN Springer abstracts, ~29K) | registration | mine gaps the ontologies miss |
| UMLS | licence, few days | CUI joins every other source |
| SNOMED CT DE | MLDS affiliate licence | synonyms, findings, procedures |

Alignment tooling (eflomal / fast_align → phrase extraction with a frequency filter) is
only worth standing up once MuchMore is in hand; there is nothing to align until then.

## 10. Experiments: dictionary injection, debate, and a review cascade

Two interventions on top of the benchmark in [6. Results](#6-results), both
aimed at the same finding: every system corrupts clinical content in 25–50% of
reports, and surface metrics do not see it. The question is whether either
intervention moves the clinical layer.

The glossary both experiments use, and why it is the one it is, is
[9. Terminology: which dictionary, and what it costs to get it wrong](#9-terminology-which-dictionary-and-what-it-costs-to-get-it-wrong).

---

### Experiment 1 — does a dictionary help?

**Command:** `medmt-eval glossary-run`
**Code:** [`inference/glossary_mt.py`](src/medmt_eval/inference/glossary_mt.py)

#### Design

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

#### The circularity guard

If the terms injected are the terms later scored, the terminology detector
improves by construction and measures copying, not translation. `--holdout F`
splits the glossary by concept into disjoint inject/score halves
(`Glossary.split`), so a gain on the scored half is generalisation.

The other three detectors — negation, laterality, number/measurement — are
independent of the glossary and need no such guard. They are the interesting
ones anyway: they carry every `critical` finding, while terminology findings are
`major` and contribute zero to the critical-error rate.

#### What is recorded

Per (segment, arm): the hypothesis, all three surface metrics, every clinical
finding, and **which terms were actually injected** (`glossary_terms`), so any
result can be traced to the exact block the model saw.

#### Reading the outcome

- `glossary` > `distractor` on the clinical layer → the dictionary helps.
- `glossary` ≈ `distractor` > `none` → the *prompt shape* helps; the terms are
  decorative. This is a real and reportable outcome and would not be visible
  without the control arm.
- No arm separates → the glossary is too thin (mean 3.5 terms/report — see
  [9. Terminology: which dictionary, and what it costs to get it wrong](#9-terminology-which-dictionary-and-what-it-costs-to-get-it-wrong)), and the honest conclusion is about
  the glossary, not about dictionaries.

---

### Experiment 2 — three specialists argue

**Command:** `medmt-eval debate`
**Code:** [`inference/debate.py`](src/medmt_eval/inference/debate.py)

#### Design

Three agents, each with a **different persona** and a **disjoint slice of the
glossary**, so they disagree for substantive reasons rather than by sampling
noise. The personas map one-to-one onto this project's critical detectors, so a
disagreement between agents is about something the evaluation actually measures:

| Agent | Responsible for | Corresponding detector |
|---|---|---|
| `anatomist` | laterality, structures, vertebral levels, modality convention | `laterality_*` |
| `safety` | negation, numbers, measurements, hedging strength | `negation_*`, `number_or_measurement_mismatch` |
| `linguist` | register, idiom, standard target terminology | `terminology_not_preserved` |

#### Protocol

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

#### Scoring

The debate output **and every agent's round-0 solo proposal** are scored on the
same scale. This is the comparison that matters: a debate that lands below its
own best participant is a real outcome, and without the solo baselines it would
look like a success.

Recorded per document: the full transcript (every critique and every revision),
the round at which it converged, and the served model behind each agent.

#### The caution that belongs in the design

**Consensus is not correctness.** Three models sharing a blind spot will agree
confidently and wrongly, and the protocol will report convergence. The
`BWK 12` → `L12` error from [6. Results](#6-results) — three independent
models rendering a thoracic vertebra as lumbar, with no detector firing — is
precisely the shape of failure this method cannot fix and will actively
disguise. The transcript and per-round scores exist so that can be checked
rather than assumed.

A related risk is cost: three agents × (1 + N rounds) + synthesis is roughly
`3N + 4` model calls per document, against one for a plain translation.

#### Model choice, and a constraint discovered while building

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

### Experiment 3 — an asymmetric review cascade

**Command:** `medmt-eval cascade`
**Code:** [`inference/cascade.py`](src/medmt_eval/inference/cascade.py)

#### Why not another debate

Experiment 2 was symmetric: three peers proposed in parallel and revised each
other. It moved BLEU, left the clinical error rate untouched, and — because every
agent revised simultaneously — could not say which participant was responsible
for anything.

A cascade fixes the attribution problem. Each stage sees the work of the stage
before it, is asked for something different, and is **scored separately**, so
stage *n* is credited only with what it changed relative to stage *n−1*.

#### Protocol

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

#### What the design has to answer for

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

#### Results — first run retracted

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
([`metrics/llm_judge.py`](src/medmt_eval/metrics/llm_judge.py)) an hour
earlier and was not carried across. And the evidence was already in the output:
a 4.5× jump in mean output length is not subtle. A structured-output stage needs
a parse-rate check before its scores are read at all.

#### Results — rerun with the parser fixed

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

### Experiments 1 and 2 on the medical-text corpus

Both were rerun on the 109 reports reduced to findings and impression
([2. Datasets](#2-datasets)), which removes the acquisition preamble and with
it every `Beurteilung → assessment`-style report-component term.

#### Dictionary — precision helps, but nothing reaches significance

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

#### Debate — worse than its best participant

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

### Experiment 4 — a glossary mined from the corpus itself

Experiments 1 and 3 both failed in the same place, and
[9. Terminology: which dictionary, and what it costs to get it wrong](#9-terminology-which-dictionary-and-what-it-costs-to-get-it-wrong) identified the mechanism: 21% of the
injected terms the model adopted are terms the human reference does not use,
because **an ontology's preferred label is not report register**. RadLex has the
right concepts and the wrong wording.

A glossary read off the reports themselves cannot have that defect. Its terms are
report register by construction. This experiment tests whether that is enough.

#### The cost, stated first

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
[4. Methods](#4-methods) applied rather than
broken: the corpus measures terminology and does not generate it, *except* in
this one experiment, which exists to measure what that exception costs.

#### Method

196 train / 100 val documents, stratified by modality, seed 20260928
(`data/splits/parrot_de_mine_v1.json`). Terms come from the train half; the
experiment runs on the val half. Built by
[`scripts/build_glossary_mined.py`](scripts/build_glossary_mined.py).

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

#### Screening against what the models already do

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

#### The ceiling, computed before reading the result

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

### Running them

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

### Results

Run 2026-08-21, `results/experiments/run_20260821_221738/`. Experiment 1:
`qwen35-4b`, 40 reports, all three arms. Experiment 2: three local agents,
20 reports, 2 rounds.

#### Experiment 1 — the dictionary does nothing; the *block* might

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

#### Experiment 1, rerun — a *bigger* glossary made it worse

The first run's null was confounded: at 3.0 terms per document it could not
separate "dictionaries do not help" from "that dictionary was too thin". The
RadLex OWL ([9. Terminology: which dictionary, and what it costs to get it wrong](#9-terminology-which-dictionary-and-what-it-costs-to-get-it-wrong)) raised the working bank to
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

##### Why: the ontology's preferred label is not report register

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

##### What this changes

- **Prompt-injected terminology is not a free improvement.** It has a real cost,
  it scales with how much you inject, and the cost is invisible without a
  distractor arm — `none` → `glossary` alone still reads as a 2.5-point
  improvement in critical errors.
- **The failure is in the term selection, not the mechanism.** A glossary
  restricted to terms the model actually gets *wrong*, or one whose target side
  is report register rather than ontology label, is untested and might well help.
- **This is the sharpest possible argument for the DeepL comparison**
  ([8. Metric roadmap: what else should be measured, and in what order](#8-metric-roadmap-what-else-should-be-measured-and-in-what-order)): DeepL applies a glossary inside
  the decoder rather than as a prompt instruction, so it separates "dictionaries
  do not help" from "*prompt-injected* dictionaries do not help". After this
  result that is no longer a nice-to-have.

#### Experiment 2 — the debate improves fluency and not safety

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

#### What the transcripts show, and why it matters more than the table

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
[6. Results](#6-results), and **exactly the class no detector can see.**

So the debate is generating clinically substantive review that the evaluation
cannot score, while failing to move the one thing the evaluation *can* score.
That is a statement about the instrument as much as about the method, and it is
the strongest argument yet for the metric work in
[8. Metric roadmap: what else should be measured, and in what order](#8-metric-roadmap-what-else-should-be-measured-and-in-what-order): an LLM-as-judge with a clinical
rubric would read these critiques as findings.

#### Experiment 4 — the corpus cannot agree with itself

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
[`data/sections.py`](src/medmt_eval/data/sections.py) classifies
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
for the direction in [8. Metric roadmap: what else should be measured, and in what order](#8-metric-roadmap-what-else-should-be-measured-and-in-what-order) — multiple
references, or source-referenced judging that never needs the reference's word
choice — and it is stronger than the arguments already recorded there because it
is a number rather than a concern.

##### The hard bank closes the loop

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
one already in [5. Experiments](#5-experiments): a loose proxy that happens
to support the expected conclusion is the most dangerous kind, and this one was
caught only by measuring it a second way on purpose.

#### Limits

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

## 11. Configurations and protocols

This chapter is the settings reference: every model with its exact runtime
configuration, every experiment with its exact protocol, and what was recorded
and selected on. It exists because the other chapters record *verdicts* and
*reasoning*, and a reader reproducing the work needs the numbers.

Everything below was read out of the code and launchers rather than from memory,
and cites the file it came from.

### 0. There are no training recipes, because nothing was trained

**No model in this project was fine-tuned, adapted or otherwise updated.** The
repository contains no optimiser, no backward pass and no PEFT configuration —
`src/medmt_eval/` is an inference and evaluation harness. Every system was run at
its published weights.

That is a deliberate finding rather than an omission: the project asked whether a
specialised German↔English medical model is *needed*, and concluded that the case
for fine-tuning is not yet made ([06-results](#6-results) §"Answering the
question"). So the analogue of a training recipe here is the **decoding
configuration** and the **experimental protocol**, and those are what follow.

### 1. Model inventory and runtime configuration

Per-model settings are resolved in
[`scripts/roundtrip/_rt_common.sh`](scripts/roundtrip/_rt_common.sh). `chunk`
is the sentence-chunking budget in tokens (0 = disabled).

| Name | Adapter | Model id | batch | max-new | chunk |
|---|---|---|---|---|---|
| `identity` | `identity` | — | 32 | 2048 | 0 |
| `opus` | `opus` | `Helsinki-NLP/opus-mt-{de-en,en-de}` | 8 | **480** | **400** |
| `nllb` | `nllb` | `facebook/nllb-200-distilled-1.3B` | 2 | 960 | **400** |
| `hymt2-1.8b` | `hymt2` | `tencent/Hy-MT2-1.8B` | 4 | 2048 | 0 |
| `hymt2-7b` | `hymt2` | `tencent/Hy-MT2-7B` | 2 | 2048 | 0 |
| `hymt2-30b-a3b` | `hymt2` | `tencent/Hy-MT2-30B-A3B` | 1 | 2048 | 0 |
| `translategemma-4b` | `translategemma` | `google/translategemma-4b-it` | 2 | 2048 | 0 |
| `translategemma-27b` | `translategemma` | `google/translategemma-27b-it` | 1 | 2048 | 0 |
| `qwen35-4b` | `prompted-llm` | `Qwen/Qwen3.5-4B` | 2 | 2048 | 0 |
| `qwen35-27b` | `prompted-llm` | `Qwen/Qwen3.5-27B` | 1 | 2048 | 0 |
| `glm-5.2` ☁ | `openai-compat` | served as `z-ai/glm-5.2` | 4 | 2048 | 0 |
| `DeepSeek-V4-Flash` ☁ | `openai-compat` | served as `deepseek-ai/deepseek-v4-flash-0731` | 4 | 2048 | 0 |
| `MiniMax-M3` ☁ | `openai-compat` | served as `minimaxai/minimax-m3` | 4 | 2048 | 0 |
| `deepl` ☁ | `deepl` | none disclosed | 8 | — | — |

`deepl` differs in kind: no GPU, no model id, no generation parameters. Settings that
matter are the target variant (`EN-US`), a batch of 8 documents per request, and the
login-node run ([`scripts/deepl/run_deepl.sh`](scripts/deepl/run_deepl.sh)). Batch
sizes for the other systems are memory-driven, not tuned for quality: they are the largest that
fit the allocation each model was given ([05-experiments](#5-experiments)).

#### Why three models differ from the rest

- **`opus` — 480 max-new, chunked at 400.** Marian has 512 encoder *and* decoder
  positions. Exceeding the decoder table raises a CUDA device-side assert
  (`marian/modeling_marian.py:596`), so generation is capped below it.
- **`nllb` — chunked at 400.** 1024 encoder positions, still short of a full
  report. The budget was originally 800 tokens (≈2400 characters), longer than
  most documents, so chunking never engaged at all; 400 was the correction.
- **`identity` — batch 32.** It performs no computation; the batch size only
  affects loop overhead.

### 2. Generation defaults

`GenerationConfig` ([`models/base.py`](src/medmt_eval/models/base.py)):

| Field | Default |
|---|---|
| `batch_size` | 8 |
| `num_beams` | 4 |
| `max_input_tokens` | 512 |
| `max_new_tokens` | 512 |
| `device` | `None` (auto) |

The round-trip runs override these: **`--num-beams 1`** (greedy, for tractability
over 20 passes × 13 systems) and **`--max-input-tokens 4096`**. Hosted models run
at **temperature 0** so results are reproducible, matching the local models.

Adapter-specific behaviour that changes outputs, not just speed:

- **Chat templates, not raw text.** `PromptedLLMTranslator` calls
  `apply_chat_template(..., enable_thinking=False)`, with a `strip_thinking()`
  regex as a second line of defence.
- **Left padding.** `padding_side="left"` for batched decoder-only generation;
  right padding corrupts it, and fixing this moved `qwen35-4b` from 24.66% to
  18.92% critical errors.
- **`direction_specific = True` for Opus only.** One checkpoint per language
  pair, so the round-trip builds two instances for Opus and exactly one for
  every other adapter.
- **Served-model verification** on every hosted call (`_check_served_model`).

### 3. Prompt templates

Verbatim, because a prompt is a setting.

**Plain translation** — local (`models/llm_mt.py`) and hosted
(`models/openai_compat_mt.py`), which additionally forbids quotation marks:

```
Translate the following medical text from {source_lang} to {target_lang}.
Return only the translation, with no explanation[, no preamble, and no quotation marks].

{text}
```

**Glossary arms** ([`inference/glossary_mt.py`](src/medmt_eval/inference/glossary_mt.py)) —
the block is empty in the `none` arm, so that prompt carries no trace of a
glossary having been considered:

```
You are translating a medical document from {source_lang} to {target_lang}.
{glossary_block}
Return only the translation. No preamble, no commentary, no quotation marks.

{text}
```

The injected block is rendered as `- <source term> = <target term>` lines under
the header *"Approved terminology for this text. Where the source uses the term
on the left, the translation must use the term on the right."*

**Debate and cascade personas** are in
[`inference/debate.py`](src/medmt_eval/inference/debate.py) and
[`inference/cascade.py`](src/medmt_eval/inference/cascade.py); the clinical
judge rubric is in [`metrics/llm_judge.py`](src/medmt_eval/metrics/llm_judge.py).

### 4. Experiment protocols

#### 4.1 Single-pass benchmark

All 13 systems over PARROT-DE (296), EMEA (400), HimL 2015 (353), HimL 2017 (119).
Per-adapter defaults; both evaluation layers. DeepL: PARROT-DE only.

#### 4.2 Round-trip

| Setting | Value |
|---|---|
| Cycles | 10 (= **20 translation passes**) |
| Sample | 20 reports, deterministic length-stratified, **seed 13** |
| Beams | 1 |
| Term bank | `radiology_en_de_starter.csv` (9 entries) |
| Anchors | English steps → human English reference; German steps → original German |

Every system sees the **same** 20 reports. Full corpus × 10 cycles was ~168
GPU-hours, past every wall-clock limit. Recorded per step: `clinical_vs_origin`
(cumulative drift) and `clinical_vs_input` (which hop introduced the error).

#### 4.3 Dictionary injection

| Setting | Value |
|---|---|
| Arms | `none` / `glossary` / `distractor` |
| Sample | 40 (whole-report runs), 109 (medical-text runs), seed 13 |
| Max terms/doc | 30 |
| Glossary | `wikidata_med.csv` (default) or `merged_de_en.csv` / `radlex_findings_de_en.csv` |
| Branch filter | `ACE` (MeSH anatomy / disease / diagnostics) |
| Model | `local:Qwen/Qwen3.5-4B` |

The `distractor` arm injects the **same number** of terms that do *not* occur in
the document — the control that distinguishes "the terms helped" from "the block
helped". `--holdout F` splits the glossary by concept so injected and scored
terms are disjoint.

#### 4.4 Multi-agent debate

Three agents, disjoint glossary slices, **2 rounds** plus synthesis, 20–40 reports.
Default backends: `Qwen/Qwen3.5-4B`, `tencent/Hy-MT2-7B`,
`google/medgemma-1.5-4b-it`. Personas map one-to-one onto the critical detectors
(anatomist → laterality, safety → negation/numbers, linguist → terminology).
Convergence is checked on whitespace- and case-normalised text. **Every agent's
round-0 solo proposal is scored alongside the debate output.**

#### 4.5 Review cascade

Three stages, scored separately: `translate` (`Hy-MT2-7B`) → `terminologise`
(`Qwen3.5-4B` **with** `merged_de_en.csv`) → `arbitrate` (`Qwen3.5-4B`
**without** the glossary). Only the terminologist sees the dictionary; the
arbiter exists to catch corrections the glossary got wrong.

Reply format requires **`TRANSLATION:` first**, note second, so truncation costs
the optional field rather than the translation — and `parse_failures` is reported
per stage. Both properties are consequences of a run where 17 of 40 replies were
scored as their own commentary.

#### 4.6 COMET

`Unbabel/wmt22-comet-da`, reference-based, over existing outputs — no
retranslation. Runs from an isolated `.venv-comet` (transformers 4.57.6,
`setuptools<81`); it cannot share the project environment.

#### 4.7 LLM judge

Clinical rubric with eight categories (negation, laterality, measurement,
anatomy, finding, certainty, terminology, omission), severity by effect on
patient management, scored against the **source** only. Refuses to grade a model
against itself; counts unparseable replies separately. Implemented but **not yet
run at scale**.

### 5. Evaluation configuration

**Surface** — sacreBLEU 2.6.0, signatures persisted with every result:

```
BLEU  |nrefs:1|case:mixed|eff:yes|tok:13a|smooth:exp|version:2.6.0
chrF++|nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0
TER   |nrefs:1|case:lc|tok:tercom|norm:no|punct:yes|asian:no|version:2.6.0
```

Corpus scores use sacreBLEU's corpus scorers, not the mean of sentence scores.

**Clinical** — source-vs-output detectors
([`taxonomy/clinical.py`](src/medmt_eval/taxonomy/clinical.py)):

| Detector | Severity |
|---|---|
| `negation_dropped`, `negation_introduced` | critical |
| `laterality_missing_or_flipped`, `laterality_added_or_flipped` | critical |
| `number_or_measurement_mismatch` | critical |
| `terminology_not_preserved` | **major** — contributes **zero** to `crit%` |

Headline metric = share of **documents** with ≥1 critical finding.

**Statistics** — exact McNemar on paired critical-error incidence; paired
bootstrap (2,000 resamples, seed 13) for BLEU deltas. At n=20 the error rate
moves in 5-point steps and at n=40 in 2.5-point steps; finer differences are not
interpretable.

**Selection** — there are no checkpoints to select. What is "selected" is an
experimental arm or variant, and every arm is reported rather than only the best.

### 6. Environment and reproducibility

| | |
|---|---|
| Main venv | transformers **5.15.1** (required for `qwen3_5`) |
| COMET venv | `.venv-comet`, transformers 4.57.6, `setuptools<81` |
| Offline | `HF_HUB_DISABLE_XET=1`; compute nodes have no outbound network, so checkpoints are pre-fetched on the login node |
| Allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Seeds | `RT_SEED=13`, `EXP_SEED=13` |
| Persisted per row | full `generation_config`, sacreBLEU signature, served model string, `model_substitution` flag |

Every launcher preflights before requesting work: credentials, gated-repo access,
API reachability, chat-template capability, glossary size, cached checkpoints.
