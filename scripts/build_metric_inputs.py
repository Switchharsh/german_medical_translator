#!/usr/bin/env python
"""One canonical table per experiment for the learned and LLM metrics to read.

    python scripts/build_metric_inputs.py --tag 20261006 \\
        --en-de results/parrot_en_de_20261006

Rows: set, system, doc_id, step, direction, src, hyp, ref.

* de_en_single  results/translations_20261005/single_pass/*.jsonl
                src = German original, hyp = system English, ref = human English.
* round_trip    results/translations_20261005/round_trip/*.jsonl, 20 steps per report.
                Fixed anchors, as in the thesis: an English step is scored against the human
                English (src = the ORIGINAL German); a German step against the ORIGINAL German
                (src = the English it was translated from, i.e. the previous hop).
* en_de_single  <en-de dir>/parrot_en_de_<system>.jsonl
                src = human English, hyp = system German, ref = original German.

System codes are taken from file names, never from the rows' ``model`` field, which holds
an adapter name for many systems. Never overwrites an existing output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows_of(path: Path):
    for line in path.open(encoding="utf-8"):
        if line.strip():
            yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--de-en", default="results/translations_20261005")
    ap.add_argument("--en-de", default=None, help="directory holding parrot_en_de_<system>.jsonl")
    ap.add_argument("--single-dir", default=None,
                    help="directory of parrot_de_<system>.jsonl to use for de_en_single instead of the translations folder "
                         "(e.g. results/parrot_de/consolidated_v2_20261007)")
    ap.add_argument("--sets", default="de_en_single,round_trip,en_de_single",
                    help="comma list; build the DE->EN sets now and EN->DE once those runs finish")
    args = ap.parse_args()
    sets = set(args.sets.split(","))
    if "en_de_single" in sets and not args.en_de:
        raise SystemExit("--en-de is required when en_de_single is among --sets")
    out = Path(f"results/metric_inputs_{args.tag}")
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    n = {}

    def write(name, rows):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as w:
            for r in rows:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        n[name] = len(rows)

    # 1. DE -> EN single pass
    rows = []
    if args.single_dir:
        files = [(p.stem.removeprefix("parrot_de_"), p) for p in sorted(Path(args.single_dir).glob("parrot_de_*.jsonl"), key=lambda p: p.stem.lower())]
    else:
        files = [(p.stem, p) for p in sorted((Path(args.de_en) / "single_pass").glob("*.jsonl"), key=lambda p: p.stem.lower())]
    for name, p in ([] if "de_en_single" not in sets else files):
        for r in rows_of(p):
            rows.append(dict(set="de_en_single", system=name, doc_id=r["doc_id"], step=1, direction="de->en",
                             src=r["src_text"], hyp=r["hyp_text"], ref=r["ref_text"]))
    if "de_en_single" in sets:
        write("de_en_single", rows)

    # 2. round trip, fixed anchors
    rows = []
    for p in ([] if "round_trip" not in sets else sorted((Path(args.de_en) / "round_trip").glob("*.jsonl"), key=lambda p: p.stem.lower())):
        for r in rows_of(p):
            if r["direction"] == "de->en":
                src, ref = r["src_text"], r["ref_text"]
            else:
                src, ref = r["input_text"], r["src_text"]
            rows.append(dict(set="round_trip", system=p.stem, doc_id=r["id"], step=r["step"],
                             direction=r["direction"], src=src, hyp=r["hyp_text"], ref=ref))
    if "round_trip" in sets:
        write("round_trip", rows)

    # 3. EN -> DE single pass
    rows = []
    files = sorted(Path(args.en_de).glob("parrot_en_de_*.jsonl"), key=lambda p: p.stem.lower()) if "en_de_single" in sets else []
    for p in files:
        system = p.stem.removeprefix("parrot_en_de_")
        for r in rows_of(p):
            rows.append(dict(set="en_de_single", system=system, doc_id=r["doc_id"], step=1, direction="en->de",
                             src=r["src_text"], hyp=r["hyp_text"], ref=r["ref_text"]))
    if "en_de_single" in sets:
        write("en_de_single", rows)

    print(f"wrote {out}: " + ", ".join(f"{k} {v} rows" for k, v in n.items()))
    if "en_de_single" in sets:
        print(f"EN->DE systems present: {sorted({r['system'] for r in rows})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
