#!/usr/bin/env python
"""Rebuild the DE->EN single-pass set from each system's newest correctly-configured run.

    python scripts/build_consolidated_v2.py --tag 20261007

WHY. results/parrot_de/consolidated/ (the files the thesis and the first metrics page read) holds
the July-31 runs, which were generated with the CLI defaults max_input_tokens=512, max_new_tokens=512.
A German report longer than ~1,500 characters was cut to its last 512 tokens before translation, so
the model translated only a fragment from the END of the source (qwen35-27b returned 12 characters
for a 1,894-character report). The thesis records the fix (max_input_tokens=4096), and the corrected
August-3 runs sit in results/parrot_3944003/4/5 and parrot_3967601, but they were never consolidated.

RULE. For each system take the newest run of 296 reports whose recorded generation config has
max_input_tokens >= 4096; hosted-API systems have no local input ceiling and have a single run
each, so that run is used. DeepL is added from its own run. Every choice is written to MANIFEST.csv
with the file's sha256 and its short-output count, so it can be audited.

Outputs (never overwritten):
  results/parrot_de/consolidated_v2_<tag>/parrot_de_<system>.jsonl      the selected runs, byte for byte
  results/parrot_de/rescored_<tag>/parrot_de_<system>.jsonl             findings recomputed with ONE detector version
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from medmt_eval.taxonomy.clinical import ClinicalSafetyEvaluator, TerminologyBank  # noqa: E402

API = {"DeepSeek-V4-Flash", "MiniMax-M3", "glm-5.2"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    cons = ROOT / f"results/parrot_de/consolidated_v2_{a.tag}"
    resc = ROOT / f"results/parrot_de/rescored_{a.tag}"
    for d in (cons, resc):
        if d.exists():
            raise SystemExit(f"refusing to overwrite {d}")
        d.mkdir(parents=True)

    cands = {}
    for f in glob.glob(str(ROOT / "results/**/parrot_de_*.jsonl"), recursive=True):
        if any(x in f for x in ("rescored", "consolidated", "translations_", "deepl_2")):
            continue
        m = re.search(r"parrot_de_(.+)\.jsonl$", f)
        if not m or m.group(1) == "combined":
            continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if len(rows) != 296 or "generation" not in rows[0]:
            continue
        g = rows[0]["generation"]
        ok = (g.get("max_input_tokens") or 0) >= 4096 or m.group(1) in API
        if ok:
            cands.setdefault(m.group(1), []).append((os.path.getmtime(f), f, g))
    chosen = {s: sorted(v)[-1] for s, v in cands.items()}
    chosen["deepl"] = (0, str(ROOT / "results/deepl_20260929_134803/parrot_de_deepl.jsonl"), {})

    ev = ClinicalSafetyEvaluator(term_bank=TerminologyBank.from_csv(str(ROOT / "data/term_banks/radiology_en_de_starter.csv")))
    src = {json.loads(l)["id"]: json.loads(l)["src_text"] for l in open(ROOT / "data/derived/parrot_de.jsonl")}
    man = []
    for s in sorted(chosen):
        f = chosen[s][1]
        dest = cons / f"parrot_de_{s}.jsonl"
        shutil.copyfile(f, dest)
        assert sha(dest) == sha(f)
        rows = [json.loads(l) for l in open(dest)]
        short = sum(len(r["hyp_text"]) < 0.5 * len(r["src_text"]) for r in rows) if s != "identity" else 0
        with open(resc / f"parrot_de_{s}.jsonl", "w", encoding="utf-8") as w:
            for r in rows:
                fs = ev.evaluate(src[r["doc_id"]], r["hyp_text"], "de", "en")
                r2 = dict(r)
                r2["findings"] = [{"code": x.code, "severity": x.severity, "detector": getattr(x, "detector", None),
                                   "details": getattr(x, "details", None)} for x in fs]
                r2["has_critical_error"] = any(x.severity == "critical" for x in fs)
                w.write(json.dumps(r2, ensure_ascii=False) + "\n")
        g = rows[0].get("generation", {})
        man.append({"system": s, "origin": os.path.relpath(f, ROOT), "rows": len(rows), "max_input_tokens": g.get("max_input_tokens"),
                    "max_new_tokens": g.get("max_new_tokens"), "short_outputs_lt_half_source": short, "sha256": sha(dest)})
        print(f"  {s:20} <- {os.path.relpath(f, ROOT):55} in={g.get('max_input_tokens')!s:>5} short={short}", flush=True)
    with open(cons / "MANIFEST.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0])); w.writeheader(); w.writerows(man)
    print(f"wrote {cons} and {resc} ({len(man)} systems)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
