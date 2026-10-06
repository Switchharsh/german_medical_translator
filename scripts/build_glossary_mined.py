#!/usr/bin/env python
"""Mine a bilingual glossary from PARROT's own training split.

Motivation
----------
Experiment 1 found that an *ontology* glossary (RadLex, Wikidata) scores below an
equally-sized block of irrelevant terms: 26% of the injected terms the model
adopted are terms the human reference does not use, because an ontology's
preferred label is not report register. A frequency-mined glossary cannot have
that defect — its terms are report register by construction, since they were
read off reports.

What it costs is testability, and the cost is unavoidable. See "Reading the
results" at the bottom of this docstring, and ``thesis/03-methods.md``.

Method
------
1. **Split by document**, stratified by modality, fixed seed. Terms are mined
   from the train half only; the experiment runs on the held-out half.

2. **Align at the sentence level, not the document level.** This is the one
   decision that makes the output usable, and it was arrived at by measurement.
   Document-level co-occurrence produced ``Herz -> mediastinum``,
   ``Milz -> kidneys`` and ``Leber -> pancreas`` — because radiology anatomy
   terms systematically co-occur (heart, mediastinum, pleura and lungs appear
   together in every chest CT), so a document-wide window cannot separate "is
   the translation of" from "appears in the same report as". Shrinking the
   window to a sentence removed every such swap.

   Sentences are paired by index in documents whose sentence counts match on
   both sides (49% of the train half), with a 0.5–2.0 length-ratio guard. This
   is far cruder than fast_align or eflomal; it is sufficient here only because
   PARROT's two sides are direct translations, so order is preserved.

3. **English candidates are 1–3 grams.** One German compound routinely maps to
   an English multiword: unigram-only alignment truncates ``Pleuraerguss`` to
   ``pleural`` and ``Perikarderguss`` to ``pericardial``, and injecting those
   truncations is worse than injecting nothing. Among candidates within 10% of
   the best Dice, the longest wins.

4. **Mutual best match.** A German word's best English candidate must also have
   that word as *its* best German candidate. Without this, ``available for
   comparison`` was simultaneously the best match for four different German
   words.

5. **Screen against what the models already do.** For each mined pair, count how
   often three baseline systems' existing train-split translations already
   contain the English side. A pair at 97% is correct and useless — the glossary
   slot is spent telling the model something it knows. This yields the two banks:

   ``freq``  top N by document frequency — the straightforward "top words from
             the corpus" bank.
   ``hard``  only pairs the baselines get right at most ``--max-already`` of the
             time — the subset with headroom to move a score at all.

   Both are written exactly as mined, with no hand editing, so the precision of
   the mining itself is what the experiment tests.

Reading the results
-------------------
A glossary mined from the evaluation corpus **invalidates every reference-based
metric**. BLEU, chrF++, TER and COMET score against the same English
translations the terms were mined from, so injecting mined wording and then
measuring agreement with it measures leakage. Splitting by document does not
repair this: the same radiologists wrote both halves, so house style crosses the
split.

What survives is the source-referenced layer — the negation, laterality, number
and measurement detectors, which never look at the reference, and the LLM judge.
Those are the primary read. Reference-based scores are reported and must be
labelled contaminated.

Usage
-----
    python scripts/build_glossary_mined.py --out-suffix v1
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medmt_eval.data.chunking import split_sentences  # noqa: E402

# Function words carry no terminology and align to everything.
DE_STOP = set(
    "der die das den dem des ein eine einer eines einem einen oder und aber auch"
    " nicht kein keine mit ohne nach vor bei von vom fuer für zur zum beim sich"
    " sind ist wird werden wurde wurden dass diese dieser dieses durch sowie"
    " unter ueber über zwischen noch mehr sehr als am im in an auf aus zu es sie"
    " er wir man soweit".split()
)
EN_STOP = set(
    "the a an of and or with without in on at to for from is are was were be been"
    " no not any all also more than into onto over under both such same other have"
    " has had can may this that these those there which where when it its as by".split()
)
#: Words from PARROT's boilerplate section headers. They appear in most
#: documents, so they co-occur with everything and align to whatever is frequent
#: — the source of ``Klinik -> clinical information question``. Note that
#: "conclusion" is deliberately absent: ``Beurteilung -> conclusion`` is a
#: genuine register find (the models write "Assessment"), not an artefact.
HEADER_WORDS = frozenset({"question", "information", "justification", "indication"})

WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def de_terms(text: str) -> set[str]:
    return {w for w in WORD.findall(text) if len(w) >= 4 and w.lower() not in DE_STOP}


def en_ngrams(text: str, max_n: int = 3) -> set[str]:
    words = [w.lower() for w in WORD.findall(text)]
    out: set[str] = set()
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            gram = words[i : i + n]
            if gram[0] in EN_STOP or gram[-1] in EN_STOP:
                continue
            if n == 1 and len(gram[0]) < 4:
                continue
            if any(w in HEADER_WORDS for w in gram):
                continue
            out.add(" ".join(gram))
    return out


def stratified_split(rows, *, seed: int, train_frac: float):
    by_domain = collections.defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    rng = random.Random(seed)
    train, val = [], []
    for _, items in sorted(by_domain.items()):
        items = sorted(items, key=lambda r: r["id"])
        rng.shuffle(items)
        cut = int(len(items) * train_frac)
        train += items[:cut]
        val += items[cut:]
    return train, val


def sentence_pairs(rows, *, min_chars: int = 15):
    """Index-align sentences in documents whose counts match on both sides."""
    pairs, matched = [], 0
    for row in rows:
        src = [s for s in split_sentences(row["src_text"]) if len(s) > min_chars]
        ref = [s for s in split_sentences(row["ref_text"]) if len(s) > min_chars]
        if len(src) != len(ref) or len(src) < 2:
            continue
        matched += 1
        for a, b in zip(src, ref):
            if 0.5 <= len(b) / max(1, len(a)) <= 2.0:
                pairs.append((a, b))
    return pairs, matched


def mine(pairs, *, min_df: int, min_co: int, min_dice: float, tolerance: float):
    src_sets = [de_terms(a) for a, _ in pairs]
    tgt_sets = [en_ngrams(b) for _, b in pairs]
    df_src, df_tgt = collections.Counter(), collections.Counter()
    for s in src_sets:
        df_src.update(s)
    for s in tgt_sets:
        df_tgt.update(s)

    # Inverted index: scanning every term against every sentence is O(n*m) and
    # took minutes on the full bank earlier in this project.
    postings = collections.defaultdict(list)
    for i, s in enumerate(src_sets):
        for w in s:
            postings[w].append(i)

    forward = {}
    for word, count in df_src.items():
        if count < min_df:
            continue
        co = collections.Counter()
        for i in postings[word]:
            co.update(tgt_sets[i])
        cands = []
        for gram, k in co.items():
            if k < min_co:
                continue
            dice = 2 * k / (count + df_tgt[gram])
            if dice >= min_dice:
                cands.append((gram, dice, k))
        if not cands:
            continue
        best = max(c[1] for c in cands)
        near = [c for c in cands if c[1] >= best * tolerance]
        # Longest surface form among the near-best: prefer the full multiword.
        forward[word] = max(near, key=lambda c: (c[0].count(" "), c[1]))

    reverse: dict[str, tuple[str, float]] = {}
    for word, (gram, dice, _) in forward.items():
        if gram in reverse and reverse[gram][1] >= dice:
            continue
        reverse[gram] = (word, dice)

    return [
        {"de": w, "en": g, "dice": round(d, 3), "df": df_src[w]}
        for w, (g, d, _) in forward.items()
        if reverse[g][0] == w
    ]


def screen(pairs, train_ids, src_by_id, systems, *, min_obs: int):
    """Fraction of baseline outputs that already contain the mined English side."""
    hyps: dict[str, dict[str, str]] = {}
    for system in systems:
        path = Path(f"results/parrot_de/consolidated/parrot_de_{system}.jsonl")
        if not path.exists():
            print(f"  warning: no baseline outputs for {system}, skipping", file=sys.stderr)
            continue
        for line in path.open():
            row = json.loads(line)
            if row["doc_id"] in train_ids:
                hyps.setdefault(row["doc_id"], {})[system] = row["hyp_text"].lower()

    kept = []
    for pair in pairs:
        hits = total = 0
        for doc_id, per_system in hyps.items():
            if pair["de"] not in de_terms(src_by_id[doc_id]):
                continue
            for text in per_system.values():
                total += 1
                hits += pair["en"] in text
        if total >= min_obs:
            pair["already"] = round(hits / total, 3)
            pair["n_obs"] = total
            kept.append(pair)
    return kept


def write_bank(path: Path, entries, *, source: str) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}; bump --out-suffix")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["concept_id", "source", "en", "de", "tree"])
        for i, e in enumerate(entries):
            # No tree code: these are corpus-derived, not ontology concepts, so
            # they are unclassifiable by construction. Run with --branches "".
            writer.writerow([f"{source}-{i:04d}", source, e["en"], e["de"], ""])
    print(f"  wrote {path} ({len(entries)} entries)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/derived/parrot_de.jsonl")
    ap.add_argument("--out-suffix", required=True, help="version tag; outputs are never overwritten")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--train-frac", type=float, default=0.67)
    ap.add_argument("--top-n", type=int, default=100, help="size of the frequency bank")
    ap.add_argument("--max-already", type=float, default=0.60,
                    help="hard bank keeps pairs baselines get right at most this often")
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--min-co", type=int, default=4)
    ap.add_argument("--min-dice", type=float, default=0.45)
    ap.add_argument("--tolerance", type=float, default=0.90)
    ap.add_argument("--min-obs", type=int, default=12)
    ap.add_argument("--systems", nargs="+",
                    default=["hymt2-7b", "qwen35-4b", "translategemma-4b"])
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.input).open()]
    train, val = stratified_split(rows, seed=args.seed, train_frac=args.train_frac)
    print(f"split: {len(train)} train / {len(val)} val (seed {args.seed}, stratified by modality)")

    pairs, matched = sentence_pairs(train)
    print(f"sentence alignment: {matched}/{len(train)} docs count-matched, {len(pairs)} sentence pairs")

    mined = mine(pairs, min_df=args.min_df, min_co=args.min_co,
                 min_dice=args.min_dice, tolerance=args.tolerance)
    print(f"mutual-best pairs: {len(mined)}")

    src_by_id = {r["id"]: r["src_text"] for r in train}
    mined = screen(mined, {r["id"] for r in train}, src_by_id, args.systems, min_obs=args.min_obs)
    mined.sort(key=lambda p: -p["df"])
    print(f"screened (>= {args.min_obs} observations): {len(mined)}")

    freq = mined[: args.top_n]
    hard = [p for p in mined if p["already"] <= args.max_already]
    tag = args.out_suffix

    print(f"\nfreq bank: {len(freq)} terms, mean already-correct "
          f"{sum(p['already'] for p in freq) / max(1, len(freq)) * 100:.0f}%")
    print(f"hard bank: {len(hard)} terms, mean already-correct "
          f"{sum(p['already'] for p in hard) / max(1, len(hard)) * 100:.0f}%")

    write_bank(Path(f"data/term_banks/parrot_mined_freq_{tag}.csv"), freq, source="parrot-mined-freq")
    write_bank(Path(f"data/term_banks/parrot_mined_hard_{tag}.csv"), hard, source="parrot-mined-hard")

    # Val-only corpus: the experiment must never see a mined document.
    val_path = Path(f"data/derived/parrot_de_val_{tag}.jsonl")
    if val_path.exists():
        raise SystemExit(f"refusing to overwrite {val_path}; bump --out-suffix")
    with val_path.open("w", encoding="utf-8") as handle:
        for row in val:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {val_path} ({len(val)} documents)")

    manifest = Path(f"data/splits/parrot_de_mine_{tag}.json")
    if manifest.exists():
        raise SystemExit(f"refusing to overwrite {manifest}; bump --out-suffix")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "seed": args.seed, "train_frac": args.train_frac,
        "train_ids": [r["id"] for r in train], "val_ids": [r["id"] for r in val],
        "sentence_pairs": len(pairs), "docs_count_matched": matched,
        "screened_systems": args.systems, "pairs": mined,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
