#!/usr/bin/env python
"""Metric scores by report modality and by report length, pooled and per system.

    python scripts/analyse_by_group.py --json OUT.json

Length is the character count of the German SOURCE report, bucketed <=500, 501-1000, 1001-2000,
>2000. Modality is PARROT's own code (CT, RX = X-ray, US, MR, XA = angiography, MG = mammography).

For every group and metric: the mean per system, and a pooled value (each report averaged over the
13 real systems first, then averaged over reports) with a 95% bootstrap interval over REPORTS, so
a small group is not over-read. The control (`identity`) is excluded. COMET is system-level only in
this project (no per-report file), so it is not broken down.
"""
from __future__ import annotations
import argparse, collections, json, random, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODALITY = {"CT": "CT", "RX": "X-ray", "US": "Ultrasound", "MR": "MRI", "XA": "Angiography", "MG": "Mammography"}
BUCKETS = [("<=500", 0, 500), ("501-1000", 501, 1000), ("1001-2000", 1001, 2000), (">2000", 2001, 10**9)]
WORD = {"negation_dropped", "negation_introduced", "laterality_missing_or_flipped", "laterality_added_or_flipped"}
STD = ["words", "numbers", "crit", "terms", "bleu", "chrf", "ter"]
LEARNED = {"xcomet": "xcomet", "xcomet_qe": "xcomet", "metricx_ref": "metricx", "metricx_qe": "metricx",
           "cef_conformity": "cef", "cef_coverage": "cef", "cef_consistency": "cef"}


def bucket(n):
    return next(name for name, lo, hi in BUCKETS if lo <= n <= hi)


def boot(vals, n=1000, seed=13):
    rng = random.Random(seed); m = len(vals)
    ms = sorted(statistics.mean(vals[rng.randrange(m)] for _ in range(m)) for _ in range(n))
    return ms[int(n * .025)], ms[int(n * .975)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--xcomet", default="results/xcomet_20261006_deen/de_en_single.jsonl")
    ap.add_argument("--metricx", default="results/metricx_20261006_deen/de_en_single.jsonl")
    ap.add_argument("--cef", default="results/cef_20261006_deen/de_en_single.jsonl")
    a = ap.parse_args()
    if a.json.exists():
        raise SystemExit(f"refusing to overwrite {a.json}")

    docs = {}
    for l in open(ROOT / "data/derived/parrot_de.jsonl", encoding="utf-8"):
        r = json.loads(l)
        docs[r["doc_id"]] = {"mod": MODALITY[r["metadata"]["modality"]], "len": len(r["src_text"]), "bucket": bucket(len(r["src_text"])), "all": "All reports"}

    val = collections.defaultdict(lambda: collections.defaultdict(dict))        # metric -> system -> doc -> value
    for f in sorted((ROOT / "results/parrot_de/rescored_20260929").glob("parrot_de_*.jsonl")):
        s = f.stem.removeprefix("parrot_de_")
        if s == "identity":
            continue
        for l in open(f, encoding="utf-8"):
            x = json.loads(l); d = x["doc_id"]
            crit = {y["code"] for y in x["findings"] if y.get("severity") == "critical"}
            codes = {y["code"] for y in x["findings"]}
            val["words"][s][d] = 100.0 * bool(crit & WORD)
            val["numbers"][s][d] = 100.0 * ("number_or_measurement_mismatch" in crit)
            val["crit"][s][d] = 100.0 * bool(crit)
            val["terms"][s][d] = 100.0 * ("terminology_not_preserved" in codes)
            for k in ("bleu", "chrf", "ter"):
                val[k][s][d] = x["metrics"][k]
    for path, cols in ((a.xcomet, ("xcomet", "xcomet_qe")), (a.metricx, ("metricx_ref", "metricx_qe")),
                       (a.cef, ("cef_conformity", "cef_coverage", "cef_consistency"))):
        for l in open(ROOT / path, encoding="utf-8"):
            x = json.loads(l)
            if x["system"] == "identity":
                continue
            for c in cols:
                if x.get(c) is not None:
                    val[c][x["system"]][x["doc_id"]] = x[c]

    systems = sorted(val["bleu"])
    metrics = STD + list(LEARNED)
    groupings = {"modality": [m for m, _ in collections.Counter(d["mod"] for d in docs.values()).most_common()],
                 "length": [b[0] for b in BUCKETS], "all": ["All reports"]}
    key = {"modality": "mod", "length": "bucket", "all": "all"}
    out = {"systems": systems, "metrics": metrics, "groupings": groupings, "n_docs": len(docs),
           "length_note": "characters of the German source report", "groups": {}}
    for g, names in groupings.items():
        out["groups"][g] = {}
        for name in names:
            ids = [d for d, v in docs.items() if v[key[g]] == name]
            entry = {"n": len(ids), "median_chars": statistics.median(docs[d]["len"] for d in ids) if ids else None,
                     "pooled": {}, "system": {}}
            for m in metrics:
                per_doc = []
                for d in ids:
                    vs = [val[m][s][d] for s in systems if d in val[m][s]]
                    if vs:
                        per_doc.append(statistics.mean(vs))
                if per_doc:
                    lo, hi = boot(per_doc) if len(per_doc) >= 2 else (per_doc[0], per_doc[0])
                    entry["pooled"][m] = {"mean": statistics.mean(per_doc), "lo": lo, "hi": hi}
                for s in systems:
                    vs = [val[m][s][d] for d in ids if d in val[m][s]]
                    if vs:
                        entry["system"].setdefault(s, {})[m] = statistics.mean(vs)
            out["groups"][g][name] = entry
    cross = collections.Counter((d["mod"], d["bucket"]) for d in docs.values())
    out["cross"] = {m: {b[0]: cross.get((m, b[0]), 0) for b in BUCKETS} for m in groupings["modality"]}
    a.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("modality x length (reports):")
    print(f"  {'':14}" + "".join(f"{b[0]:>11}" for b in BUCKETS) + f"{'all':>8}")
    for m in groupings["modality"]:
        print(f"  {m:14}" + "".join(f"{out['cross'][m][b[0]]:>11}" for b in BUCKETS) + f"{sum(out['cross'][m].values()):>8}")
    print(f"  {'all':14}" + "".join(f"{out['groups']['length'][b[0]]['n']:>11}" for b in BUCKETS) + f"{len(docs):>8}")
    print("\npooled by LENGTH (13 systems, per-report average; [95% CI over reports])")
    print(f"  {'bucket':11}{'n':>4}{'words%':>20}{'crit%':>20}{'BLEU':>20}{'xCOMET':>20}")
    for b in groupings["length"]:
        e = out["groups"]["length"][b]; p = e["pooled"]
        print(f"  {b:11}{e['n']:>4}" + "".join(f"{p[m]['mean']:>9.2f} [{p[m]['lo']:5.2f},{p[m]['hi']:5.2f}]" for m in ("words", "crit", "bleu", "xcomet")))
    print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
