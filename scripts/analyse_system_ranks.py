#!/usr/bin/env python
"""Do the metrics rank the systems the same way?

    python scripts/analyse_system_ranks.py \\
        --std "results/parrot_de/rescored_20260929/parrot_de_*.jsonl" --strip parrot_de_ \\
        --xcomet results/xcomet_X/de_en_single.jsonl --metricx results/metricx_X/de_en_single.jsonl \\
        --cef results/cef_X/de_en_single.jsonl --comet results/comet_parrot_de.json

System-level means over the 296 reports, converted so that larger is always better, then ranked.
Reports the full table, Spearman and Kendall correlations between metrics, and which systems
move most between families. The control (`identity`) is shown but excluded from every
correlation: it would dominate them, since every metric agrees it is the worst.

`words%` is the share of reports with a critical negation or laterality finding (numbers
excluded); `crit%` is any critical finding. Both come from the same detector run, so the
comparison is like for like.
"""
from __future__ import annotations

import argparse
import collections
import glob
import itertools
import json
import statistics
from pathlib import Path

WORD = ("negation_dropped", "negation_introduced", "laterality_missing_or_flipped", "laterality_added_or_flipped")
# metric -> (label, higher_is_better, family)
META = {
    "bleu": ("BLEU", True, "surface"), "chrf": ("chrF++", True, "surface"), "ter": ("TER", False, "surface"),
    "comet": ("COMET", True, "learned"), "xcomet": ("xCOMET", True, "learned"), "xcomet_qe": ("xCOMET-QE", True, "learned"),
    "metricx_ref": ("MetricX", False, "learned"), "metricx_qe": ("MetricX-QE", False, "learned"),
    "cef_conformity": ("CEF-conform", True, "cef"), "cef_coverage": ("CEF-cover", True, "cef"),
    "cef_consistency": ("CEF-consist", True, "cef"),
    "crit": ("crit%", False, "detectors"), "words": ("words%", False, "detectors"),
}


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def kendall(a, b):
    c = d = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        s = (a[i] - a[j]) * (b[i] - b[j])
        c += s > 0; d += s < 0
    return (c - d) / (c + d) if c + d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--std", required=True, help="glob of per-system standard-metric files")
    ap.add_argument("--strip", default="", help="file-name prefix to strip to get the system code")
    ap.add_argument("--xcomet"); ap.add_argument("--metricx"); ap.add_argument("--cef")
    ap.add_argument("--comet", help="results/comet_*.json: {system: {comet: x}}")
    ap.add_argument("--json", type=Path, help="also write the tables as JSON (never overwrites)")
    args = ap.parse_args()

    per_doc = collections.defaultdict(lambda: collections.defaultdict(list))   # system -> metric -> [values]
    for f in sorted(glob.glob(args.std)):
        sysname = Path(f).stem.removeprefix(args.strip)
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            for k in ("bleu", "chrf", "ter"):
                per_doc[sysname][k].append(r["metrics"][k])
            crit = {x["code"] for x in r["findings"] if x.get("severity") == "critical"}
            per_doc[sysname]["crit"].append(100.0 * bool(crit))
            per_doc[sysname]["words"].append(100.0 * bool(crit & set(WORD)))
    for path, cols in ((args.xcomet, ("xcomet", "xcomet_qe")), (args.metricx, ("metricx_ref", "metricx_qe")),
                       (args.cef, ("cef_coverage", "cef_conformity", "cef_consistency"))):
        if not path:
            continue
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            for c in cols:
                if r.get(c) is not None:
                    per_doc[r["system"]][c].append(r[c])
    mean = {s: {m: statistics.mean(v) for m, v in d.items()} for s, d in per_doc.items()}
    if args.comet:
        blob = json.load(open(args.comet))
        for s, v in (blob.get("systems") or blob).items():     # file is {"checkpoint": ..., "systems": {...}}
            if s in mean and isinstance(v, dict) and "comet" in v:
                mean[s]["comet"] = v["comet"]

    systems = sorted(s for s in mean if s != "identity")
    metrics = [m for m in META if any(m in mean[s] for s in systems)]
    print(f"{len(systems)} systems (control excluded); metrics: {', '.join(META[m][0] for m in metrics)}\n")
    print(f"{'system':20}" + "".join(f"{META[m][0]:>13}" for m in metrics))
    for s in sorted(systems, key=lambda s: -mean[s].get("bleu", 0)):
        print(f"{s:20}" + "".join(f"{mean[s][m]:>13.3f}" if m in mean[s] else f"{'--':>13}" for m in metrics))
    if "identity" in mean:
        print(f"{'identity (control)':20}" + "".join(f"{mean['identity'][m]:>13.3f}" if m in mean["identity"] else f"{'--':>13}" for m in metrics))

    good = {m: {s: (mean[s][m] if META[m][1] else -mean[s][m]) for s in systems if m in mean[s]} for m in metrics}
    rank = {}
    for m in metrics:
        ss = sorted(good[m], key=lambda s: -good[m][s])
        rank[m] = {s: i + 1 for i, s in enumerate(ss)}
    print("\nRANK (1 = best) by metric")
    print(f"{'system':20}" + "".join(f"{META[m][0]:>13}" for m in metrics) + f"{'spread':>8}")
    for s in sorted(systems, key=lambda s: sum(rank[m].get(s, 0) for m in metrics)):
        rs = [rank[m][s] for m in metrics if s in rank[m]]
        print(f"{s:20}" + "".join(f"{rank[m][s]:>13}" if s in rank[m] else f"{'--':>13}" for m in metrics) + f"{max(rs)-min(rs):>8}")
    print("\nBEST system per metric: " + "; ".join(f"{META[m][0]}={min(rank[m], key=rank[m].get)}" for m in metrics))

    for name, fn in (("SPEARMAN rho", spearman), ("KENDALL tau", kendall)):
        print(f"\n{name} between metrics (over systems scored by both; sign-aligned so + = agree)")
        print(f"{'':13}" + "".join(f"{META[m][0]:>13}" for m in metrics))
        for a in metrics:
            row = []
            for b in metrics:
                common = [s for s in systems if s in good[a] and s in good[b]]
                row.append(f"{fn([good[a][s] for s in common], [good[b][s] for s in common]):>13.2f}" if len(common) >= 4 else f"{'--':>13}")
            print(f"{META[a][0]:13}" + "".join(row))
    fam = collections.defaultdict(list)
    for a, b in itertools.combinations(metrics, 2):
        common = [s for s in systems if s in good[a] and s in good[b]]
        if len(common) >= 4:
            fam[tuple(sorted((META[a][2], META[b][2])))].append(spearman([good[a][s] for s in common], [good[b][s] for s in common]))
    print("\nMEAN Spearman rho within and between families")
    for k, v in sorted(fam.items()):
        print(f"  {k[0]:>9} - {k[1]:<9} {statistics.mean(v):5.2f}  (n pairs {len(v)})")
    if args.json:
        if args.json.exists():
            raise SystemExit(f"refusing to overwrite {args.json}")
        sp = {}
        for a in metrics:
            sp[a] = {}
            for b in metrics:
                common = [s for s in systems if s in good[a] and s in good[b]]
                sp[a][b] = spearman([good[a][s] for s in common], [good[b][s] for s in common]) if len(common) >= 4 else None
        args.json.write_text(json.dumps({
            "systems": systems, "metrics": [{"key": m, "label": META[m][0], "higher_better": META[m][1], "family": META[m][2]} for m in metrics],
            "mean": {s: {m: mean[s].get(m) for m in metrics} for s in mean}, "rank": rank,
            "best": {m: min(rank[m], key=rank[m].get) for m in metrics}, "spearman": sp,
            "family_mean_rho": {f"{k[0]}|{k[1]}": statistics.mean(v) for k, v in fam.items()},
            "family_pairs": {f"{k[0]}|{k[1]}": len(v) for k, v in fam.items()}}, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
