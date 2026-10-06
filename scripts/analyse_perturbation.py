#!/usr/bin/env python
"""How far does each metric move when a single clinical fact is corrupted?

    python scripts/analyse_perturbation.py results/perturbation_20261006 \\
        [--scores xcomet=results/xcomet_X/perturb.jsonl:xcomet,xcomet_qe ...]

For every (base, kind) the unedited ``none`` row is the control. Per metric we report

  worse      mean score change in the "got worse" direction, in the metric's own units
             (sign-flipped for lower-is-better metrics so that positive = penalised)
  z          the same, in standard deviations of that metric over the real systems' per-report
             scores, which makes BLEU, chrF++, xCOMET and MetricX comparable
  caught%    share of edits the metric scored strictly worse
  AUC        P(a dangerous edit is penalised more than a harmless one); 0.5 = cannot tell them apart
  d/h        mean z of the dangerous kinds divided by mean z of the harmless one

A metric that tracks meaning has d/h well above 1 and AUC near 1. One that is mostly
lexical has d/h near or below 1, because a rewording changes more n-grams than a flipped
negation does.

The clinical detectors are listed separately: they are scored as hit / no hit on the edit's
own category, and see ``build_perturbations.py`` for why that is partly circular.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sacrebleu.metrics import BLEU, CHRF, TER  # noqa: E402

from medmt_eval.taxonomy.clinical import ClinicalSafetyEvaluator, TerminologyBank  # noqa: E402

HIGHER_BETTER = {"bleu": True, "chrf": True, "ter": False, "xcomet": True, "xcomet_qe": True, "xcomet_mqm": True,
                 "comet": True, "metricx_ref": False, "metricx_qe": False,
                 "cef_coverage": True, "cef_conformity": True, "cef_consistency": True}
DANGEROUS = ("negation_drop", "laterality_flip", "number_change")
TARGET_CODE = {"negation_drop": "negation", "laterality_flip": "laterality", "number_change": "number_or_measurement"}


def auc(pos, neg):
    """P(pos > neg) with ties as 1/2, by ranking."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    r_pos = sum(ranks[k] for k, (_, lab) in enumerate(allv) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (r_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--scores", nargs="*", default=[],
                    help="name=path:col1,col2 -- per-row score files keyed by (system, doc_id)")
    ap.add_argument("--scale", nargs="*", default=[],
                    help="col=path -- score file of REAL systems used as the SD yardstick for that column")
    ap.add_argument("--base", default=None, help="restrict to one base, e.g. ref@de->en")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (args.dir / "perturbed.jsonl").open(encoding="utf-8")]
    bleu, chrf, ter = BLEU(effective_order=True), CHRF(word_order=2), TER()
    ev = ClinicalSafetyEvaluator(term_bank=TerminologyBank.from_csv("data/term_banks/radiology_en_de_starter.csv"))

    metric = collections.defaultdict(dict)   # name -> (system, doc_id) -> value
    findings = {}                            # (system, doc_id) -> set of codes
    for r in rows:
        key = (r["system"], r["doc_id"])
        metric["bleu"][key] = bleu.sentence_score(r["hyp"], [r["ref"]]).score
        metric["chrf"][key] = chrf.sentence_score(r["hyp"], [r["ref"]]).score
        metric["ter"][key] = ter.sentence_score(r["hyp"], [r["ref"]]).score
        s, t = r["direction"].split("->")
        findings[key] = {f.code for f in ev.evaluate(r["src"], r["hyp"], s, t)}

    for spec in args.scores:
        name, _, rest = spec.partition("=")
        path, _, cols = rest.partition(":")
        for line in open(path, encoding="utf-8"):
            x = json.loads(line)
            for c in cols.split(","):
                if x.get(c) is not None:
                    metric[c][(x["system"], x["doc_id"])] = x[c]

    # SD yardstick: per-report scores of the real systems, DE->EN single pass.
    scale = {}
    stored = collections.defaultdict(list)
    for p in sorted((ROOT / "results/translations_20261005/single_pass").glob("*.jsonl")):
        if p.stem == "identity":
            continue
        for line in p.open(encoding="utf-8"):
            m = json.loads(line)["metrics"]
            for k in ("bleu", "chrf", "ter"):
                stored[k].append(m[k])
    for k, v in stored.items():
        scale[k] = statistics.pstdev(v)
    for spec in args.scale:
        col, _, path = spec.partition("=")
        vals = []
        for l in open(path, encoding="utf-8"):
            x = json.loads(l)
            # a row can carry null (e.g. a CEF question-generation failure); skip it
            if x.get("system") != "identity" and x.get(col) is not None:
                vals.append(x[col])
        if vals:
            scale[col] = statistics.pstdev(vals)

    by_base = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        by_base[r["base"]][r["kind"]][r["doc_id"]] = r

    names = [n for n in HIGHER_BETTER if n in metric]
    out = {}
    for base, kinds in sorted(by_base.items()):
        if args.base and base != args.base:
            continue
        print(f"\n{'=' * 100}\nBASE {base}   ({len(kinds['none'])} reports; one edit per report per kind)\n{'=' * 100}")
        res = {}
        for n in names:
            hb = HIGHER_BETTER[n]
            sd = scale.get(n)
            for kind in ("negation_drop", "laterality_flip", "number_change", "harmless"):
                deltas = []
                for doc, r in kinds[kind].items():
                    o = kinds["none"].get(doc)
                    if o is None:
                        continue
                    a, b = metric[n].get((r["system"], doc)), metric[n].get((o["system"], doc))
                    if a is None or b is None:
                        continue
                    deltas.append((b - a) if hb else (a - b))     # positive = penalised
                if deltas:
                    res[(n, kind)] = deltas
        print(f"{'metric':13}{'kind':17}{'n':>5}{'worse':>10}{'z (SD)':>9}{'caught%':>9}")
        summary = []
        for n in names:
            sd = scale.get(n)
            for kind in ("negation_drop", "laterality_flip", "number_change", "harmless"):
                d = res.get((n, kind))
                if not d:
                    continue
                z = statistics.mean(d) / sd if sd else float("nan")
                print(f"{n:13}{kind:17}{len(d):>5}{statistics.mean(d):>10.3f}{z:>9.3f}{sum(x > 1e-9 for x in d) / len(d) * 100:>8.0f}%")
            dang = [x for k in DANGEROUS for x in res.get((n, k), [])]
            harm = res.get((n, "harmless"), [])
            if dang and harm and sd:
                zd = statistics.mean(dang) / sd
                zh = statistics.mean(harm) / sd
                summary.append((n, zd, zh, zd / zh if zh > 1e-9 else float("inf"), auc(dang, harm)))
            print()
        print(f"{'-' * 100}\nDANGEROUS vs HARMLESS on {base}   (d/h > 1 and AUC -> 1 means the metric ranks real errors as worse)")
        print(f"{'metric':13}{'z dangerous':>13}{'z harmless':>12}{'d/h ratio':>11}{'AUC':>8}")
        for n, zd, zh, ratio, a in summary:
            print(f"{n:13}{zd:>13.3f}{zh:>12.3f}{ratio:>11.2f}{a:>8.3f}")
        out[base] = [dict(metric=n, z_dangerous=zd, z_harmless=zh, ratio=ratio, auc=a) for n, zd, zh, ratio, a in summary]

        print(f"\nCLINICAL DETECTORS on {base} (hit = a NEW finding of the edit's own category; partly circular):")
        for kind in DANGEROUS + ("harmless",):
            hits = n_ = 0
            for doc, r in kinds[kind].items():
                o = kinds["none"].get(doc)
                if not o:
                    continue
                n_ += 1
                new = findings[(r["system"], doc)] - findings[(o["system"], doc)]
                if kind == "harmless":
                    hits += bool(new)                 # any NEW critical-class finding on a harmless edit = false alarm
                else:
                    hits += any(c.startswith(TARGET_CODE[kind]) for c in new)
            label = "false alarms" if kind == "harmless" else "caught"
            print(f"  {kind:16} {label:13} {hits}/{n_} = {hits / max(1, n_) * 100:.0f}%")
    (args.dir / "analysis.json").write_text(json.dumps({"scale": scale, "by_base": out}, indent=1), encoding="utf-8") \
        if not (args.dir / "analysis.json").exists() else print("\n(analysis.json exists, not overwritten)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
