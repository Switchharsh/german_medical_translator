#!/usr/bin/env python
"""Assemble every number the metrics page shows into one JSON file.

    python scripts/build_metrics_page_data.py --out results/page_data_X/page_data.json \\
        [--perturb results/perturbation_X/analysis_full.txt]

Nothing on the page is typed by hand: each section's data is read from the file that produced it
(or, for the older sections, from the previous page's embedded block, which was itself generated).
Never overwrites.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORD = {"negation_dropped", "negation_introduced", "laterality_missing_or_flipped", "laterality_added_or_flipped"}
KINDS = ("negation_drop", "laterality_flip", "number_change", "harmless")


def jl(p):
    return [json.loads(l) for l in open(ROOT / p, encoding="utf-8") if l.strip()]


def std_means(path):
    rows = jl(path); n = len(rows)
    agg = collections.defaultdict(float)
    for r in rows:
        crit = {y["code"] for y in r["findings"] if y.get("severity") == "critical"}
        codes = {y["code"] for y in r["findings"]}
        agg["words"] += 100.0 * bool(crit & WORD); agg["numbers"] += 100.0 * ("number_or_measurement_mismatch" in crit)
        agg["crit"] += 100.0 * bool(crit); agg["terms"] += 100.0 * ("terminology_not_preserved" in codes)
        for k in ("bleu", "chrf", "ter"):
            agg[k] += r["metrics"][k]
    return {k: v / n for k, v in agg.items()} | {"n": n, "empty": sum(1 for r in rows if not r["hyp_text"].strip())}


def parse_perturbation(path):
    out, base, mode = {}, None, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^BASE (\S+)\s+\((\d+) reports", line)
        if m:
            base = m.group(1); out[base] = {"n": int(m.group(2)), "kinds": [], "summary": [], "detectors": {}}; mode = "kinds"; continue
        if base is None:
            continue
        if line.startswith("DANGEROUS vs HARMLESS"):
            mode = "summary"; continue
        if line.startswith("CLINICAL DETECTORS"):
            mode = "det"; continue
        if mode == "kinds":
            m = re.match(r"^(?P<metric>[a-z_]+?)\s*(?P<kind>negation_drop|laterality_flip|number_change|harmless)\s+(?P<n>\d+)\s+(?P<w>-?\d+\.\d+)\s+(?P<z>-?\d+\.\d+|nan)\s+(?P<c>\d+)%", line)
            if m:
                out[base]["kinds"].append({"metric": m["metric"], "kind": m["kind"], "n": int(m["n"]), "worse": float(m["w"]),
                                           "z": None if m["z"] == "nan" else float(m["z"]), "caught": int(m["c"])})
        elif mode == "summary":
            m = re.match(r"^(?P<metric>[a-z_]+)\s+(?P<zd>-?\d+\.\d+)\s+(?P<zh>-?\d+\.\d+)\s+(?P<r>inf|-?\d+\.\d+)\s+(?P<auc>-?\d+\.\d+|nan)\s*$", line)
            if m:
                out[base]["summary"].append({"metric": m["metric"], "z_dangerous": float(m["zd"]), "z_harmless": float(m["zh"]),
                                             "ratio": None if m["r"] == "inf" else float(m["r"]), "auc": None if m["auc"] == "nan" else float(m["auc"])})
        elif mode == "det":
            m = re.match(r"^\s+(\w+)\s+(?:caught|false alarms)\s+(\d+)/(\d+)", line)
            if m:
                out[base]["detectors"][m.group(1)] = {"hit": int(m.group(2)), "n": int(m.group(3))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--perturb", default="results/perturbation_20261006/analysis_full.txt")
    ap.add_argument("--data-dir", default="results/page_data_20261006")
    ap.add_argument("--old-page", default="metrics_reference.html")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    dd = ROOT / a.data_dir

    html = (ROOT / a.old_page).read_text(encoding="utf-8")
    tag = '<script id="data" type="application/json">'
    s = html.index(tag) + len(tag)
    old = json.loads(html[s:html.index("</script>", s)])
    D = {k: old[k] for k in ("roundtrip", "experiments", "worked_example", "signatures", "comet") if k in old}

    ranks = json.load(open(dd / "system_ranks.json")); groups = json.load(open(dd / "by_group_v2.json"))
    D["ranks"], D["groups"] = ranks, groups
    D["types"] = json.load(open(dd / "corpus_types.json")); D["recall"] = json.load(open(dd / "detector_recall.json"))

    # ---- single pass DE->EN: every metric, every system, one table ----
    allg = groups["groups"]["all"]["All reports"]
    sp = {}
    for sysn, vals in allg["system"].items():
        sp[sysn] = dict(vals) | {"n": 296}
        if sysn in D["comet"]:
            sp[sysn]["comet"] = D["comet"][sysn]["comet"]
    idm = std_means("results/parrot_de/rescored_20260929/parrot_de_identity.jsonl")
    ident = {k: idm[k] for k in ("words", "numbers", "crit", "terms", "bleu", "chrf", "ter")} | {"n": 296}
    for m in ranks["metrics"]:
        v = ranks["mean"].get("identity", {}).get(m["key"])
        if v is not None:
            ident[m["key"]] = v
    if "identity" in D["comet"]:
        ident["comet"] = D["comet"]["identity"]["comet"]
    sp["identity"] = ident
    spans = collections.defaultdict(lambda: collections.Counter()); cnt = collections.Counter()
    for r in jl("results/xcomet_20261006_deen/de_en_single.jsonl"):
        cnt[r["system"]] += 1
        for k in ("minor", "major", "critical"):
            spans[r["system"]][k] += r[f"spans_{k}"]
    for sysn in sp:
        if cnt[sysn]:
            sp[sysn]["spans"] = {k: spans[sysn][k] / cnt[sysn] for k in ("minor", "major", "critical")}
    D["singlepass"] = sp
    D["pooled"] = allg["pooled"]

    # ---- learned metrics along the round trip (English steps, cycles 1..10) ----
    rt = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for path, cols in (("results/xcomet_20261006_deen/round_trip.jsonl", ("xcomet", "xcomet_qe")),
                       ("results/metricx_20261006_deen/round_trip.jsonl", ("metricx_ref", "metricx_qe")),
                       ("results/cef_20261006_deen/round_trip.jsonl", ("cef_conformity", "cef_coverage", "cef_consistency"))):
        for r in jl(path):
            if r["direction"] != "de->en":
                continue
            for c in cols:
                if r.get(c) is not None:
                    rt[r["system"]][c][(r["step"] + 1) // 2].append(r[c])
    D["rt_learned"] = {s: {c: [statistics.mean(v[k]) for k in sorted(v)] for c, v in cols.items()} for s, cols in rt.items()}

    # ---- EN->DE: standard metrics only, for the systems whose translations are complete ----
    ende, excluded = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results/parrot_en_de_20261006/parrot_en_de_*.jsonl"))):
        sysn = os.path.basename(f)[len("parrot_en_de_"):-6]
        m = std_means(os.path.relpath(f, ROOT))
        if m["empty"]:
            excluded[sysn] = f"{m['empty']} of 296 translations are empty (repair in progress)"
        else:
            ende[sysn] = m
    D["ende"] = {"systems": ende, "excluded": excluded,
                 "pending": {"glm-5.3": "running (SCADS API, one report per request, rate-limited)"}}

    # ---- perturbation ----
    pp = ROOT / a.perturb
    D["perturb"] = parse_perturbation(pp) if pp.exists() else {}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(D, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {a.out} ({a.out.stat().st_size/1e3:.0f} KB) | systems {len(sp)} | rt_learned {len(D['rt_learned'])} | "
          f"EN->DE {len(ende)} complete, {len(excluded)} excluded | perturbation bases {list(D['perturb'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
