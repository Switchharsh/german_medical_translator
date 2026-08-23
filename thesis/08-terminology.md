# Terminology: which dictionary, and what it costs to get it wrong

Both experiments in [09-experiments-glossary-debate.md](09-experiments-glossary-debate.md)
depend on a bilingual glossary. This chapter is the evaluation of the candidates and
the reasoning for the one in use.

## Recommendation

| Rank | Source | Verdict |
|---|---|---|
| **1** | **RadLex** (`RadLex.owl`) | **Obtained and in use.** 45,163 EN-DE pairs from explicit `xml:lang` tags. Radiology-native, so none of the cross-domain polysemy below. Needs the function-word filter. |
| 2 | **Wikidata**, MeSH/UMLS-anchored, filtered to MeSH branches A/C/E | **Merged in as a supplement.** CC0, no registration. Adds 3.5 terms/report of general medical vocabulary RadLex lacks; needs all the hygiene below or it is actively harmful. |
| 3 | UMLS | The join key (CUI) that makes everything else composable. Worth starting the licence now; not needed to run the experiments. |
| 4 | SNOMED CT DE, ICD-10-GM / ICD-11 | Diagnoses and findings backbone. Registration-gated, valuable later. |
| — | **German MeSH (ZB MED)** | **Excluded on purpose.** See "the circularity problem". |
| — | RadReport.org | Does not do what we need. See "what I checked and rejected". |

## The circularity problem — why German MeSH is excluded

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

## What I checked and rejected

### RadReport.org — the API is live, the German corpus is not there

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

### RadLex — obtained, and the OWL is the file that matters

Two RadLex distributions were supplied on 2026-08-21/22. They are not equivalent,
and the difference is the whole ballgame.

| File | German content | EN-DE pairs recovered |
|---|---|---|
| `RADLEX.csv.gz` (BioPortal export) | `Preferred_name_German` populated for **1 concept of 46,900**; some untagged German inside the `Synonyms` field | **389**, by rule-based language ID |
| **`RadLex.owl`** | **47,414 explicit `xml:lang="de"` tags** on `rdfs:label` and `RID:Synonym` | **45,163**, read directly |

**Use the OWL.** The DRG translation is there as proper language tags; nothing has
to be inferred. The BioPortal CSV path in
[`scripts/build_glossary_radlex.py`](../scripts/build_glossary_radlex.py) is kept
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

## Coverage — the number that decides whether Experiment 1 can work

| Glossary | Entries | Terms/report | Zero-hit reports |
|---|---|---|---|
| RadLex from the CSV | 389 | 0.3 | 228 / 296 |
| Wikidata A/C/E | 2,834 | 3.5 | 26 / 296 |
| **RadLex from the OWL** | **45,163** | **14.5** | **2 / 296** |
| **merged (RadLex + Wikidata)** | **47,997** | **18.0** | **2 / 296** |

This is a five-fold increase in the amount of terminology a model is actually
shown, and it removes the strongest confound in
[09-experiments-glossary-debate.md](09-experiments-glossary-debate.md): the first
run of Experiment 1 returned a null result on a glossary supplying 3.0 terms per
document, which could not distinguish "dictionaries do not help" from "this
dictionary was too thin to test the question".

## Is RadLex's terminology what a clinician actually writes?

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

### How much of this branch ranking is real?

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

### The precision-first bank

[`radlex_findings_de_en.csv`](../data/term_banks/radlex_findings_de_en.csv) —
clinical findings and procedures only, **2,777 entries, 85% agreement**, 2.2
terms per report, 59 of 296 reports with no match.

That is a deliberate trade of coverage for trust, and it is the right direction
given that [Experiment 1](09-experiments-glossary-debate.md) found *more*
terminology made translation worse.

## The hygiene problem — what a general-purpose source does to clinical German

Wikidata linked to MeSH/UMLS gives 22,200 DE/EN concepts for free. Used naively it is
worse than no glossary at all. Three distinct failure modes showed up on PARROT, all
found by measurement rather than inspection.

### 1. Short aliases are catastrophic

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

### 2. General terminologies carry the whole tree

MeSH is not a radiology terminology. Its branches include Z (Geographicals) and
L (Information Science), which is where "Deutsche Demokratische Republik" and
"maschinelles Lernen" came from.

**Fix:** `Glossary.in_branches("ACE")` — A (Anatomy), C (Diseases), E (Diagnostic and
Therapeutic Techniques and Equipment). This is what a radiology report is made of.
22,200 concepts → 2,834.

### 3. A radiology terminology ships function words as concepts

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

### 4. Polysemy survives both filters

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

## Where that leaves the working glossary

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

## Reproducing

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

## Still to obtain

| Source | Blocker | Value |
|---|---|---|
| MuchMore (DE-EN Springer abstracts, ~29K) | registration | mine gaps the ontologies miss |
| UMLS | licence, few days | CUI joins every other source |
| SNOMED CT DE | MLDS affiliate licence | synonyms, findings, procedures |

Alignment tooling (eflomal / fast_align → phrase extraction with a frequency filter) is
only worth standing up once MuchMore is in hand; there is nothing to align until then.
