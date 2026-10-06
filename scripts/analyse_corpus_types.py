#!/usr/bin/env python
"""What kinds of reports are in PARROT-DE, and does the kind change how well systems do?

    python scripts/analyse_corpus_types.py > results/corpus_types_YYYYMMDD.txt

Part 1 describes the corpus (modality, body region, subspecialty, country, length) and the
content features that create exposure to the failure modes this project measures: how many
measurements, negations and side mentions a report carries. Part 2 pools the 13 real systems'
DE->EN scores by modality, with a bootstrap interval over REPORTS so small groups are not
over-read. The control (`identity`) is excluded.

PARROT modality codes: CT, RX = conventional radiography (X-ray), US = ultrasound, MR = MRI,
XA = X-ray angiography / interventional (DICOM code XA), MG = mammography.
Raw metadata is messy ('German' vs 'Germany'; 'CH, MSK' vs 'CH,MSK'); it is normalised below
and the rules are printed so nothing is hidden.
"""
from __future__ import annotations

import collections
import json
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from medmt_eval.data.sections import split_sections  # noqa: E402
from medmt_eval.metrics.segmentation import units  # noqa: E402
from medmt_eval.taxonomy.clinical import _NEGATION_PATTERNS  # noqa: E402
import build_perturbations as bp  # noqa: E402

MODALITY = {"CT": "CT", "RX": "X-ray (RX)", "US": "Ultrasound", "MR": "MRI", "XA": "Angiography (XA)", "MG": "Mammography"}
SUBSPEC = {"MSK": "Musculoskeletal", "CH": "Chest", "GI": "Gastrointestinal", "NR": "Neuroradiology",
           "GU": "Genitourinary", "CV": "Cardiovascular", "BR": "Breast"}
REGION_RULES = [  # first match wins; keyword -> coarse region
    ("Breast", ("breast", "mamma")), ("Spine", ("spine", "lumbar", "cervical", "thoracic spine", "wirbel", "sacr")),
    ("Head & neck", ("head", "brain", "neck", "skull", "sinus", "orbit", "face", "cranial", "jaw", "temporal")),
    ("Chest", ("chest", "thorax", "lung", "heart", "rib", "cardiac", "mediastin")),
    ("Abdomen/pelvis", ("abdomen", "pelvis", "liver", "kidney", "renal", "pancrea", "bowel", "bladder", "uterus", "prostate", "gallbladder")),
    ("Limbs & joints", ("knee", "shoulder", "hip", "hand", "foot", "ankle", "elbow", "wrist", "limb", "extremit", "femur", "tibia", "humer", "joint", "thigh", "arm", "leg", "finger", "toe")),
]


def region(area: str) -> str:
    parts = [p.strip().lower() for p in re.split(r"[,;/]| and ", area) if p.strip()]
    found = []
    for p in parts:
        for name, kws in REGION_RULES:
            if any(k in p for k in kws):
                found.append(name); break
        else:
            found.append("Other")
    uniq = sorted(set(found))
    return uniq[0] if len(uniq) == 1 else "Multi-region"


def boot(vals, n=1000, seed=13):
    rng = random.Random(seed); m = len(vals)
    ms = sorted(statistics.mean(vals[rng.randrange(m)] for _ in range(m)) for _ in range(n))
    return ms[int(n * .025)], ms[int(n * .975)]


def table(counter, total, title, top=None):
    print(f"\n{title}")
    for k, v in counter.most_common(top):
        print(f"  {str(k):28}{v:>5}  {v / total * 100:5.1f}%  " + "#" * round(v / total * 40))


def main() -> int:
    json_out = Path(sys.argv[sys.argv.index("--json") + 1]) if "--json" in sys.argv else None
    if json_out and json_out.exists():
        raise SystemExit(f"refusing to overwrite {json_out}")
    rows = [json.loads(l) for l in open(ROOT / "data/derived/parrot_de.jsonl", encoding="utf-8")]
    n = len(rows)
    for r in rows:
        md = r["metadata"]
        r["mod"] = MODALITY[md["modality"]]
        r["country"] = "Germany" if md["country"].lower().startswith("german") else md["country"]
        r["region"] = region(md["area"])
        toks = [t.strip().upper() for t in re.split(r"[,;/ ]+", md["subspecialty"]) if t.strip()]
        r["subs"] = [SUBSPEC.get(t, t) for t in toks]
        r["multi_sub"] = len(set(toks)) > 1
        s = r["src_text"]
        r["chars"] = len(s); r["words"] = len(s.split()); r["units"] = len(units(s))
        r["n_meas"] = len(bp.NUM_UNIT.findall(s)); r["n_neg"] = len(_NEGATION_PATTERNS["de"].findall(s))
        r["n_lat"] = len(bp.DE_LAT.findall(s))
        r["n_num"] = len(re.findall(r"\d+(?:[.,]\d+)?", s))           # any numeric token (doses, dates, kV ...)
        low = " ".join(x.lower() for x in bp.DE_LAT.findall(s))
        r["both_sides"] = bool(re.search(r"\blink", low)) and bool(re.search(r"\brecht", low))
        roles = {x.role for x in split_sections(s, "de")}
        r["has_prea"] = bool(roles & {"indication", "technique"}); r["has_find"] = "findings" in roles
        r["has_impr"] = "impression" in roles

    print(f"PARROT-DE: {n} reports\n" + "=" * 78)
    table(collections.Counter(r["mod"] for r in rows), n, "MODALITY")
    table(collections.Counter(r["region"] for r in rows), n, "BODY REGION (coarse; 48 raw area labels grouped by keyword)")
    prim = collections.Counter(r["subs"][0] for r in rows)
    table(prim, n, "PRIMARY SUBSPECIALTY (first listed)")
    print(f"  reports listing more than one subspecialty: {sum(r['multi_sub'] for r in rows)} ({sum(r['multi_sub'] for r in rows)/n*100:.0f}%)")
    table(collections.Counter(r["country"] for r in rows), n, "COUNTRY (German/Germany merged)")
    unk = sorted({r["metadata"]["area"] for r in rows if r["region"] == "Other"})
    print(f"\n  area labels that fell to 'Other': {unk[:12]}")

    print("\nMODALITY x BODY REGION")
    regs = [k for k, _ in collections.Counter(r["region"] for r in rows).most_common()]
    print(f"  {'':18}" + "".join(f"{x[:11]:>12}" for x in regs))
    for m in [k for k, _ in collections.Counter(r["mod"] for r in rows).most_common()]:
        c = collections.Counter(r["region"] for r in rows if r["mod"] == m)
        print(f"  {m:18}" + "".join(f"{c.get(x, 0):>12}" for x in regs))

    print("\nCONTENT PROFILE BY MODALITY (German source; medians unless noted)")
    print(f"  {'modality':18}{'n':>4}{'chars':>7}{'words':>7}{'units':>7}{'meas/rep':>9}{'nums/rep':>9}{'neg/rep':>8}{'sides/rep':>10}"
          f"{'>=2 neg':>9}{'both-sides':>11}{'has prea':>9}{'impress.':>9}")
    order = [k for k, _ in collections.Counter(r["mod"] for r in rows).most_common()]
    groups = {m: [r for r in rows if r["mod"] == m] for m in order}
    groups["ALL"] = rows
    for m, g in groups.items():
        k = len(g)
        both = sum(r["both_sides"] for r in g)
        print(f"  {m:18}{k:>4}{statistics.median(r['chars'] for r in g):>7.0f}{statistics.median(r['words'] for r in g):>7.0f}"
              f"{statistics.median(r['units'] for r in g):>7.0f}{statistics.mean(r['n_meas'] for r in g):>9.1f}{statistics.mean(r['n_num'] for r in g):>9.1f}"
              f"{statistics.mean(r['n_neg'] for r in g):>8.1f}{statistics.mean(r['n_lat'] for r in g):>10.1f}"
              f"{sum(r['n_neg'] >= 2 for r in g) / k * 100:>8.0f}%{both / k * 100:>10.0f}%"
              f"{sum(r['has_prea'] for r in g) / k * 100:>8.0f}%{sum(r['has_impr'] for r in g) / k * 100:>8.0f}%")
    print("  (meas/rep counts values with mm/cm/ml/mg/HU only; nums/rep counts every numeric token, including doses,\n"
          "   kV and dates, so it is the fairer exposure measure for angiography)")
    print("  (>=2 neg: share with two or more negations, where the rule-based negation detector is blind;\n"
          "   both-sides: left AND right both mentioned, where the laterality detector is mostly blind)")

    # ---- Part 2: performance by modality ----
    print("\n" + "=" * 78 + "\nPART 2  DO THE REPORT TYPES DIFFER IN HOW WELL THEY TRANSLATE? (DE->EN, 13 real systems pooled)")
    mod_of = {r["doc_id"]: r["mod"] for r in rows}
    WORD = {"negation_dropped", "negation_introduced", "laterality_missing_or_flipped", "laterality_added_or_flipped"}
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))  # mod->metric->doc->[vals over systems]
    sysn = set()
    for f in sorted((ROOT / "results/parrot_de/rescored_20260929").glob("parrot_de_*.jsonl")):
        s = f.stem.removeprefix("parrot_de_")
        if s == "identity":
            continue
        sysn.add(s)
        for line in f.open(encoding="utf-8"):
            x = json.loads(line); d = x["doc_id"]; m = mod_of[d]
            crit = {y["code"] for y in x["findings"] if y.get("severity") == "critical"}
            per[m]["words%"][d].append(100.0 * bool(crit & WORD))
            per[m]["numbers%"][d].append(100.0 * ("number_or_measurement_mismatch" in crit))
            per[m]["crit%"][d].append(100.0 * bool(crit))
            per[m]["BLEU"][d].append(x["metrics"]["bleu"])
    for path, cols in (("results/xcomet_20261006_deen/de_en_single.jsonl", ("xcomet",)),
                       ("results/metricx_20261006_deen/de_en_single.jsonl", ("metricx_ref",)),
                       ("results/cef_20261006_deen/de_en_single.jsonl", ("cef_conformity", "cef_coverage"))):
        for line in open(ROOT / path, encoding="utf-8"):
            x = json.loads(line)
            if x["system"] == "identity":
                continue
            for c in cols:
                if x.get(c) is not None:
                    per[mod_of[x["doc_id"]]][c][x["doc_id"]].append(x[c])
    metrics = ["words%", "numbers%", "crit%", "BLEU", "xcomet", "metricx_ref", "cef_conformity", "cef_coverage"]
    arrow = {"words%": "lower", "numbers%": "lower", "crit%": "lower", "BLEU": "higher", "xcomet": "higher",
             "metricx_ref": "lower", "cef_conformity": "higher", "cef_coverage": "higher"}
    print(f"  {len(sysn)} systems; each report is averaged over systems first, then groups are compared. [lo, hi] = 95% bootstrap over reports.\n")
    print(f"  {'modality':18}{'n':>4}" + "".join(f"{m:>26}" for m in metrics[:4]))
    for m in order:
        cells = []
        for k in metrics[:4]:
            per_doc = [statistics.mean(v) for v in per[m][k].values()]
            lo, hi = boot(per_doc)
            cells.append(f"{statistics.mean(per_doc):7.1f} [{lo:5.1f},{hi:5.1f}]")
        print(f"  {m:18}{len(groups[m]):>4}" + "".join(f"{c:>26}" for c in cells))
    print(f"\n  {'modality':18}{'n':>4}" + "".join(f"{m:>26}" for m in metrics[4:]))
    for m in order:
        cells = []
        for k in metrics[4:]:
            per_doc = [statistics.mean(v) for v in per[m][k].values()]
            lo, hi = boot(per_doc)
            cells.append(f"{statistics.mean(per_doc):7.2f} [{lo:5.2f},{hi:5.2f}]")
        print(f"  {m:18}{len(groups[m]):>4}" + "".join(f"{c:>26}" for c in cells))
    print("  direction: " + ", ".join(f"{k} {v}" for k, v in arrow.items()) + " is better")
    print("  CAUTION: Mammography n=2 and Angiography n=15 cannot support a conclusion; read them as descriptive.")

    # what kind of critical error dominates in each modality
    print("\n  SHARE OF CRITICAL FINDINGS BY KIND (pooled over systems and reports)")
    kinds = collections.defaultdict(collections.Counter)
    for f in sorted((ROOT / "results/parrot_de/rescored_20260929").glob("parrot_de_*.jsonl")):
        if f.stem.endswith("identity"):
            continue
        for line in f.open(encoding="utf-8"):
            x = json.loads(line)
            for y in x["findings"]:
                if y.get("severity") == "critical":
                    c = y["code"]; c = "number" if c.startswith("number") else ("negation" if c.startswith("negation") else "laterality")
                    kinds[mod_of[x["doc_id"]]][c] += 1
    print(f"  {'modality':18}{'findings':>9}{'number':>9}{'laterality':>12}{'negation':>10}")
    for m in order:
        t = sum(kinds[m].values()) or 1
        print(f"  {m:18}{t:>9}{kinds[m]['number'] / t * 100:>8.0f}%{kinds[m]['laterality'] / t * 100:>11.0f}%{kinds[m]['negation'] / t * 100:>9.0f}%")
    if json_out:
        out = {"n": n, "modality": collections.Counter(r["mod"] for r in rows),
               "region": collections.Counter(r["region"] for r in rows), "subspecialty": prim,
               "country": collections.Counter(r["country"] for r in rows),
               "multi_subspecialty": sum(r["multi_sub"] for r in rows), "order": order, "profile": {}, "performance": {}, "kinds": {}}
        for m, g in groups.items():
            k = len(g)
            out["profile"][m] = {"n": k, "chars": statistics.median(r["chars"] for r in g), "words": statistics.median(r["words"] for r in g),
                                 "units": statistics.median(r["units"] for r in g), "meas": statistics.mean(r["n_meas"] for r in g),
                                 "nums": statistics.mean(r["n_num"] for r in g), "neg": statistics.mean(r["n_neg"] for r in g),
                                 "sides": statistics.mean(r["n_lat"] for r in g), "neg2": sum(r["n_neg"] >= 2 for r in g) / k,
                                 "both_sides": sum(r["both_sides"] for r in g) / k, "preamble": sum(r["has_prea"] for r in g) / k,
                                 "impression": sum(r["has_impr"] for r in g) / k}
        for m in order:
            out["performance"][m] = {}
            for k in metrics:
                per_doc = [statistics.mean(v) for v in per[m][k].values()]
                lo, hi = boot(per_doc)
                out["performance"][m][k] = {"mean": statistics.mean(per_doc), "lo": lo, "hi": hi}
            t_ = sum(kinds[m].values()) or 1
            out["kinds"][m] = {k: kinds[m][k] / t_ for k in ("number", "laterality", "negation")} | {"findings": t_}
        json_out.write_text(json.dumps(out, indent=1, default=dict), encoding="utf-8"); print(f"\nwrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
