#!/usr/bin/env python
"""Controlled single-edit corruptions of translations, for the metric sensitivity test.

    python scripts/build_perturbations.py --tag 20261006

For each report and each applicable kind, ONE edit is made to a hypothesis:

  negation_drop     delete one negation cue          "No pleural effusion" -> "Pleural effusion"
  laterality_flip   swap one left<->right            "left upper lobe"      -> "right upper lobe"
  number_change     alter one measurement's last digit   "5 mm" -> "8 mm"
  harmless          swap a phrase for a clinically equivalent one  "shows" -> "demonstrates"

The first three change what a clinician would conclude. The fourth does not, and exists to
show which metrics punish a rewording as hard as a real error (the worked example in the
metrics page does this on one sentence; this does it across the corpus).

Two bases: ``ref`` (the human reference used as the hypothesis, so the starting point is a
perfect translation and any change is purely the edit) and ``DeepSeek-V4-Flash`` (a real
strong system, so the test is not only run on a flawless baseline). Every perturbed row is
paired with an unedited ``none`` row of the same base and report.

CIRCULARITY: the cue lists below overlap the clinical detectors' lexicons, so the detectors
will flag these edits partly by construction. Their detection rate here is a ceiling check
on the plumbing, not evidence of recall on real errors. Learned and n-gram metrics have no
such lexicon, so for them the test is a fair one.

Deterministic: the edit site is chosen with a seed derived from (seed, report, kind).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

EN_NEG = re.compile(r"\b(?:[Nn]o|[Nn]ot|[Ww]ithout|[Nn]one)\s+(?=[A-Za-z])")
DE_NEG = re.compile(r"\b(?:[Kk]ein(?:e|en|er|em|es)?|[Nn]icht|[Oo]hne)\s+(?=[A-Za-zÄÖÜäöü])")
EN_LAT = re.compile(r"\b(left|right)\b", re.I)
DE_LAT = re.compile(r"\b(links\w*|rechts\w*|link(?:e|en|er|em|es)\b|recht(?:e|en|er|em|es)\b)", re.I)
NUM_UNIT = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)(?=\s*(?:mm|cm|ml|mg|HU)\b)")

EN_HARMLESS = [("shows", "demonstrates"), ("is seen", "is visible"), ("demonstrates", "shows"),
               ("no evidence of", "no signs of"), ("approximately", "about"), ("consistent with", "in keeping with"),
               ("noted", "seen"), ("bilateral", "on both sides"), ("unremarkable", "normal")]
DE_HARMLESS = [("zeigt", "stellt dar"), ("unauffällig", "regelrecht"), ("keine Hinweise auf", "kein Anhalt für"),
               ("ca.", "etwa"), ("beidseits", "beidseitig"), ("vereinbar mit", "passend zu"),
               ("sichtbar", "erkennbar"), ("Zeichen", "Hinweise")]


def _case_like(src: str, dst: str) -> str:
    return dst.capitalize() if src[:1].isupper() else dst


def negation_drop(text, lang, rng):
    ms = list((EN_NEG if lang == "en" else DE_NEG).finditer(text))
    if not ms:
        return None
    m = rng.choice(ms)
    before = text[max(0, m.start() - 15): m.end() + 20]
    new = text[:m.start()] + text[m.end():]
    head = text[:m.start()].rstrip()
    if m.start() == 0 or head.endswith((".", ":", "\n")) or text[m.start() - 1] == "\n":
        new = new[:m.start()] + new[m.start():m.start() + 1].upper() + new[m.start() + 1:]
    return new, before, new[max(0, m.start() - 15): m.start() + 20]


def laterality_flip(text, lang, rng):
    pat = EN_LAT if lang == "en" else DE_LAT
    ms = list(pat.finditer(text))
    if not ms:
        return None
    m = rng.choice(ms); w = m.group(0); lw = w.lower()
    if lang == "en":
        rep = "right" if lw == "left" else "left"
    elif lw.startswith("links"):
        rep = "rechts" + w[5:]
    elif lw.startswith("rechts"):
        rep = "links" + w[6:]
    elif lw.startswith("link"):
        rep = "recht" + w[4:]
    else:
        rep = "link" + w[5:]
    rep = _case_like(w, rep)
    return text[:m.start()] + rep + text[m.end():], w, rep


def number_change(text, lang, rng):
    ms = list(NUM_UNIT.finditer(text))
    if not ms:
        return None
    m = rng.choice(ms); s = m.group(1)
    idx = max(i for i, c in enumerate(s) if c.isdigit())
    new_digit = str((int(s[idx]) + 3) % 10)
    rep = s[:idx] + new_digit + s[idx + 1:]
    return text[:m.start()] + rep + text[m.end():], s, rep


def harmless(text, lang, rng):
    pairs = [(a, b) for a, b in (EN_HARMLESS if lang == "en" else DE_HARMLESS)
             if re.search(re.escape(a), text, re.I)]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    m = re.search(re.escape(a), text, re.I)
    rep = _case_like(m.group(0), b)
    return text[:m.start()] + rep + text[m.end():], m.group(0), rep


KINDS = {"negation_drop": negation_drop, "laterality_flip": laterality_flip,
         "number_change": number_change, "harmless": harmless}
DANGEROUS = ("negation_drop", "laterality_flip", "number_change")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--de-en", default="results/translations_20261005")
    ap.add_argument("--base-system", default="DeepSeek-V4-Flash")
    args = ap.parse_args()
    out = Path(f"results/perturbation_{args.tag}")
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)

    real = {}
    for line in (Path(args.de_en) / "single_pass" / f"{args.base_system}.jsonl").open(encoding="utf-8"):
        r = json.loads(line); real[r["doc_id"]] = r
    corpus = [json.loads(l) for l in open("data/derived/parrot_de.jsonl", encoding="utf-8")]

    rows, counts = [], {}
    # (direction, lang of the edited text, base name, src, hypothesis-to-edit, reference)
    def bases(r):
        yield ("de->en", "en", "ref", r["src_text"], r["ref_text"], r["ref_text"])
        if r["doc_id"] in real:
            yield ("de->en", "en", args.base_system, r["src_text"], real[r["doc_id"]]["hyp_text"], r["ref_text"])
        # EN->DE: the original German is the "perfect" hypothesis for the English source
        yield ("en->de", "de", "ref", r["ref_text"], r["src_text"], r["src_text"])

    for r in corpus:
        for direction, lang, base, src, hyp, ref in bases(r):
            tagname = f"{base}@{direction}"
            rows.append(dict(set="perturb", system=f"{tagname}__none", base=tagname, kind="none", doc_id=r["doc_id"],
                             step=1, direction=direction, src=src, hyp=hyp, ref=ref, edit=None))
            for kind, fn in KINDS.items():
                rng = random.Random(f"{args.seed}-{r['doc_id']}-{kind}-{tagname}")
                res = fn(hyp, lang, rng)
                if res is None or res[0] == hyp:
                    continue
                new, before, after = res
                rows.append(dict(set="perturb", system=f"{tagname}__{kind}", base=tagname, kind=kind,
                                 doc_id=r["doc_id"], step=1, direction=direction, src=src, hyp=new, ref=ref,
                                 edit={"before": before, "after": after}))
                counts[(tagname, kind)] = counts.get((tagname, kind), 0) + 1
    with (out / "perturbed.jsonl").open("w", encoding="utf-8") as w:
        for x in rows:
            w.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"wrote {out/'perturbed.jsonl'}: {len(rows)} rows")
    for (b, k), n in sorted(counts.items()):
        print(f"  {b:28} {k:16} {n:4d} applicable of {len(corpus)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
