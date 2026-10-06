#!/usr/bin/env python
"""Rule-based detector recall on single-fact perturbations, split by how many facts of that kind the report has.

    python scripts/detector_recall_by_density.py results/perturbation_20261006 --json OUT.json

Negation and laterality detectors compare whole-document PRESENCE (any cue / the set of sides), not
individual statements. So an edit that changes one negation in a report with several should go
unseen, while the same edit in a one-negation report should be caught. This measures exactly that.
"""
from __future__ import annotations
import argparse, collections, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from medmt_eval.taxonomy.clinical import ClinicalSafetyEvaluator, _NEGATION_PATTERNS  # noqa: E402
import build_perturbations as bp  # noqa: E402

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir", type=Path); ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--base", default="ref@de->en")
    a = ap.parse_args()
    if a.json.exists():
        raise SystemExit(f"refusing to overwrite {a.json}")
    ev = ClinicalSafetyEvaluator()
    by = collections.defaultdict(dict)
    for l in (a.dir / "perturbed.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        if r["base"] == a.base:
            by[r["kind"]][r["doc_id"]] = r
    codes = lambda r: {f.code for f in ev.evaluate(r["src"], r["hyp"], *r["direction"].split("->"))}
    count = {"negation_drop": lambda t: len(_NEGATION_PATTERNS["en"].findall(t)),
             "laterality_flip": lambda t: len({m.group(1).lower() for m in bp.EN_LAT.finditer(t)}),
             "number_change": lambda t: len(bp.NUM_UNIT.findall(t))}
    prefix = {"negation_drop": "negation", "laterality_flip": "laterality", "number_change": "number_or_measurement"}
    out = {"base": a.base, "kinds": {}}
    for kind, fn in count.items():
        b = collections.defaultdict(lambda: [0, 0])
        for doc, r in by[kind].items():
            o = by["none"][doc]; new = codes(r) - codes(o)
            k = fn(o["hyp"]); k = k if k < 4 else 4
            b[k][1] += 1; b[k][0] += any(c.startswith(prefix[kind]) for c in new)
        out["kinds"][kind] = {str(k): {"caught": v[0], "n": v[1]} for k, v in sorted(b.items())}
    a.json.write_text(json.dumps(out, indent=1), encoding="utf-8"); print("wrote", a.json)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
