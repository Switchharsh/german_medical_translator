#!/usr/bin/env python
"""Rank systems on the *words*: negation, laterality and terminology, plus surface metrics.

The standard critical-error rate counts ``number_or_measurement_mismatch`` findings, which
are two thirds of all critical findings on PARROT. This report drops them, so a system is
judged on whether it got the words right. The detector itself is untouched, and the number
findings are still shown in their own column so nothing is hidden.

Every system is scored from its stored findings, so the comparison is like for like.

    python scripts/words_only_compare.py results/deepl_*/parrot_de_deepl.jsonl
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

WORD_CRITICAL = {"negation_dropped", "negation_introduced",
                 "laterality_missing_or_flipped", "laterality_added_or_flipped"}
NUMBER = "number_or_measurement_mismatch"


def load(paths):
    by = collections.defaultdict(dict)
    for p in paths:
        for line in Path(p).open():
            r = json.loads(line)
            by[r["model"] if "model" in r else Path(p).stem][r["doc_id"]] = r
    return by


def summarise(rows):
    n = len(rows)
    word = num = allc = terms = 0
    bleu = chrf = ter = 0.0
    for r in rows.values():
        codes = {f["code"] for f in r["findings"]}
        crit = {f["code"] for f in r["findings"] if f.get("severity") == "critical"}
        word += bool(crit & WORD_CRITICAL)
        num += NUMBER in crit
        allc += bool(crit)
        terms += "terminology_not_preserved" in codes
        m = r["metrics"]; bleu += m["bleu"]; chrf += m["chrf"]; ter += m["ter"]
    return dict(n=n, word=100 * word / n, num=100 * num / n, allc=100 * allc / n,
                terms=100 * terms / n, bleu=bleu / n, chrf=chrf / n, ter=ter / n)


def main():
    # One directory, one detector version. Mixing stored findings from different
    # runs is not safe: the terminology detector changed after the July runs, so
    # a fresh score set beside stored findings mis-ranked DeepL's terminology by 8x.
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/parrot_de/rescored_20260929")
    by = {}
    for p in sorted(d.glob("parrot_de_*.jsonl")):
        rows = {}
        for line in p.open():
            r = json.loads(line); rows[r["doc_id"]] = r
        by[p.stem.removeprefix("parrot_de_")] = rows
    table = sorted(((m, summarise(r)) for m, r in by.items() if m != "identity"),
                   key=lambda kv: kv[1]["word"])
    print(f"detector scores from {d}")
    print(f"{'system':20}{'n':>5}{'words-crit%':>13}{'numbers%':>11}{'all-crit%':>11}"
          f"{'term%':>7}{'BLEU':>7}{'chrF++':>8}{'TER':>7}")
    for m, s in table:
        mark = "  <== DeepL" if m == "deepl" else ""
        print(f"{m:20}{s['n']:>5}{s['word']:>12.1f}%{s['num']:>10.1f}%{s['allc']:>10.1f}%"
              f"{s['terms']:>6.1f}%{s['bleu']:>7.1f}{s['chrf']:>8.1f}{s['ter']:>7.1f}{mark}")


if __name__ == "__main__":
    main()
