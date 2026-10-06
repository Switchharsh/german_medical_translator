#!/usr/bin/env python
"""Gather every stored translation into one folder, one file per system code.

    translations_<tag>/
        single_pass/<system>.jsonl    all 296 PARROT-DE reports, DE -> EN
        round_trip/<system>.jsonl     20 reports x 20 passes (10 cycles)
        combined/single_pass_all.jsonl   every system in one file, explicit `system` column
        combined/round_trip_all.jsonl    same, with step / cycle / direction
        MANIFEST.csv                  system, mode, rows, docs, sha256, origin
        README.md

Files are COPIED byte for byte, never rewritten: the thesis cites the original paths
and the rows carry findings and metrics that a reshaping would lose. The system code is
the file name, and also the ``model`` field inside each row.

Never overwrites: the output directory must not already exist.

    python scripts/collect_translations.py --tag 20261005
    python scripts/collect_translations.py --tag 20261005 --add-combined   # onto an existing folder

Why a combined file: the ``model`` field inside the rows is NOT a reliable system code.
Round-trip rows of glm-5.2, DeepSeek-V4-Flash and MiniMax-M3 all say ``openai-compat``;
the three Hy-MT2 sizes all say ``hymt2``; both Qwens say ``prompted-llm``; and
translategemma-27b's single-pass rows say ``translategemma``, the same as the 4B. The
file name is the code, so the combined files take it from there.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

SINGLE = {  # system code -> stored single-pass file
    **{p.stem.removeprefix("parrot_de_"): p
       for p in Path("results/parrot_de/consolidated").glob("parrot_de_*.jsonl")
       if p.name != "combined.jsonl"},
    # Not in consolidated/. Two runs exist (3942876, 3944005) with identical
    # translations (296/296); the earlier is the one the thesis cites.
    "translategemma-27b": Path("results/parrot_3942876/parrot_de_translategemma-27b.jsonl"),
    "deepl": Path("results/deepl_20260929_134803/parrot_de_deepl.jsonl"),
}
ROUND = {
    **{p.stem.removeprefix("rt_"): p
       for p in Path("results/roundtrip_20260814_132452").glob("rt_*.jsonl")},
    "deepl": Path("results/deepl_20260929_134803/rt_deepl.jsonl"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(mode: str, system: str, path: Path) -> tuple[int, int]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    key = "doc_id" if mode == "single_pass" else "id"
    docs = {r[key] for r in rows}
    empty = sum(1 for r in rows if not r["hyp_text"].strip())
    if mode == "single_pass" and system != "identity" and empty:
        raise SystemExit(f"{system} {mode}: {empty} empty hypotheses")
    if any(r.get("model", system) != system and system != "translategemma-27b" for r in rows[:1]):
        print(f"  note: {system} {mode}: model field is {rows[0].get('model')!r}", file=sys.stderr)
    return len(rows), len(docs)


def write_combined(out: Path) -> None:
    """One file per mode with an explicit system column, taken from the file name."""
    comb = out / "combined"
    comb.mkdir(exist_ok=True)
    spec = {
        "single_pass": ("doc_id", ("domain", "src_text", "hyp_text", "ref_text")),
        "round_trip": ("id", ("domain", "step", "cycle", "direction", "input_text", "hyp_text", "ref_text")),
    }
    for mode, (key, fields) in spec.items():
        dest = comb / f"{mode}_all.jsonl"
        if dest.exists():
            raise SystemExit(f"refusing to overwrite {dest}")
        n = 0
        with dest.open("w", encoding="utf-8") as w:
            for src in sorted((out / mode).glob("*.jsonl"), key=lambda p: p.stem.lower()):
                for line in src.open(encoding="utf-8"):
                    r = json.loads(line)
                    row = {"system": src.stem, "doc_id": r[key]}
                    row.update({f: r.get(f) for f in fields})
                    w.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
        print(f"  wrote {dest} ({n} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--add-combined", action="store_true",
                    help="only write combined/ onto an existing folder")
    args = ap.parse_args()
    out = Path(f"results/translations_{args.tag}")
    if args.add_combined:
        if not out.exists():
            raise SystemExit(f"{out} does not exist")
        write_combined(out)
        return 0
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    for mode, table in (("single_pass", SINGLE), ("round_trip", ROUND)):
        (out / mode).mkdir(parents=True)

    manifest = []
    for mode, table in (("single_pass", SINGLE), ("round_trip", ROUND)):
        for system, src in sorted(table.items(), key=lambda kv: kv[0].lower()):
            if not src.exists():
                raise SystemExit(f"missing {src}")
            rows, docs = check(mode, system, src)
            dest = out / mode / f"{system}.jsonl"
            shutil.copyfile(src, dest)
            if sha(dest) != sha(src):
                raise SystemExit(f"copy mismatch for {dest}")
            manifest.append({"system": system, "mode": mode, "rows": rows, "docs": docs,
                             "sha256": sha(dest), "origin": str(src)})

    # Cross-checks: every system must cover the same documents.
    by_mode = collections.defaultdict(dict)
    for m in manifest:
        by_mode[m["mode"]][m["system"]] = m
    for mode, expect_docs in (("single_pass", 296), ("round_trip", 20)):
        bad = {s: m["docs"] for s, m in by_mode[mode].items() if m["docs"] != expect_docs}
        if bad:
            raise SystemExit(f"{mode}: wrong document count {bad}")
    with (out / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader(); w.writerows(manifest)
    write_combined(out)
    print(f"wrote {out}: {len(by_mode['single_pass'])} single-pass + {len(by_mode['round_trip'])} round-trip files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
