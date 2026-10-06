#!/usr/bin/env python
"""Compare the arms of a mined-glossary run, with the tests the claim needs.

Three things this reports that a per-arm score table cannot.

**glossary vs distractor, paired.** A glossary block lengthens and restructures
the prompt, and that alone moves scores. Only the difference between the relevant
terms and an equally-sized block of irrelevant ones isolates the terms. Both arms
translate the same documents, so the comparison is paired: McNemar's exact test
on the per-document critical-error flag, and a paired bootstrap on BLEU.

**Term adoption.** An injected term the model ignores cannot have caused
anything. Reporting a null result without the adoption rate leaves "the terms did
not help" indistinguishable from "the terms were not used".

**Adopted-but-absent-from-reference.** This is the quantity that explained
Experiment 1: 26% of the RadLex terms the model adopted do not appear in the
human reference, because an ontology label is not report register. A bank mined
from reports should score far lower here *by construction*, and if it does not,
the mining is at fault rather than the idea.

Reference-based figures in this output are contaminated — the terms were mined
from the same English text BLEU scores against. See the module docstring of
scripts/build_glossary_mined.py.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
from pathlib import Path

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def norm(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact binomial test on the discordant pairs."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_bootstrap(a: list[float], b: list[float], *, trials: int = 10000, seed: int = 13) -> float:
    """P(delta <= 0) for the mean of a - b, resampling documents together."""
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    observed = sum(diffs) / n
    worse = 0
    for _ in range(trials):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        if (s <= 0) if observed > 0 else (s >= 0):
            worse += 1
    return worse / trials


def load(path: Path) -> dict[str, dict[str, dict]]:
    by_arm: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for line in path.open():
        row = json.loads(line)
        by_arm[row["arm"]][row["id"]] = row
    return by_arm


def adoption(rows: dict[str, dict]) -> tuple[int, int, int]:
    """(injected, adopted, adopted-but-absent-from-reference) over all documents."""
    injected = adopted = orphan = 0
    for row in rows.values():
        hyp, ref = norm(row["hyp_text"]), norm(row["ref_text"])
        for entry in row.get("glossary_terms") or []:
            # The runner records injected terms as {concept_id, src, tgt}; a
            # bank CSV names the same columns de/en. Accept either.
            if isinstance(entry, dict):
                target = entry.get("tgt") or entry.get("en") or ""
            else:
                target = str(entry)
            if not target:
                continue
            injected += 1
            t = norm(target)
            if t and t in hyp:
                adopted += 1
                if t not in ref:
                    orphan += 1
    return injected, adopted, orphan


def report(path: Path, summary: Path | None) -> None:
    by_arm = load(path)
    print(f"\n{'=' * 78}\n{path}\n{'=' * 78}")

    corpus = {}
    if summary and summary.exists():
        for entry in json.loads(summary.read_text())["arms"]:
            corpus[entry["arm"]] = entry

    print(f"{'arm':11} {'n':>4} {'crit%':>7} {'BLEU*':>7} {'chrF*':>7} {'TER*':>7} "
          f"{'terms':>6} {'adopt':>7} {'orphan':>7}")
    for arm in ("none", "glossary", "distractor"):
        rows = by_arm.get(arm)
        if not rows:
            continue
        c = corpus.get(arm, {})
        crit = sum(r["has_critical_error"] for r in rows.values()) / len(rows)
        inj, ado, orp = adoption(rows)
        terms = c.get("mean_glossary_terms", inj / max(1, len(rows)))
        ar = f"{ado / inj * 100:6.1f}%" if inj else "     --"
        orr = f"{orp / ado * 100:6.1f}%" if ado else "     --"
        print(f"{arm:11} {len(rows):4d} {crit * 100:6.1f}% {c.get('bleu', float('nan')):7.2f} "
              f"{c.get('chrf', float('nan')):7.2f} {c.get('ter', float('nan')):7.2f} "
              f"{terms:6.1f} {ar} {orr}")
    print("* reference-based: contaminated, terms were mined from this corpus' English side")

    for treat, ctrl in (("glossary", "distractor"), ("glossary", "none")):
        if treat not in by_arm or ctrl not in by_arm:
            continue
        shared = sorted(set(by_arm[treat]) & set(by_arm[ctrl]))
        t_only = sum(by_arm[treat][i]["has_critical_error"] and not by_arm[ctrl][i]["has_critical_error"]
                     for i in shared)
        c_only = sum(by_arm[ctrl][i]["has_critical_error"] and not by_arm[treat][i]["has_critical_error"]
                     for i in shared)
        p_mc = mcnemar_exact(t_only, c_only)
        tb = [by_arm[treat][i]["metrics"]["bleu"] for i in shared]
        cb = [by_arm[ctrl][i]["metrics"]["bleu"] for i in shared]
        delta = (sum(tb) - sum(cb)) / len(shared)
        p_bs = paired_bootstrap(tb, cb)
        print(f"\n{treat} vs {ctrl}  (n={len(shared)} paired)")
        print(f"  critical errors: {treat} only {t_only}, {ctrl} only {c_only} "
              f"-> McNemar exact p = {p_mc:.3f}")
        print(f"  sentence BLEU*:  delta {delta:+.2f}, paired bootstrap p = {p_bs:.4f}")

    findings: dict[str, collections.Counter] = {}
    for arm, rows in by_arm.items():
        counter = collections.Counter()
        for row in rows.values():
            for f in row["findings"]:
                if f.get("severity") == "critical":
                    counter[f["detector"]] += 1
        findings[arm] = counter
    detectors = sorted({d for c in findings.values() for d in c})
    if detectors:
        print(f"\n{'critical detector':38} " + " ".join(f"{a:>11}" for a in ("none", "glossary", "distractor")))
        for d in detectors:
            print(f"{d:38} " + " ".join(f"{findings.get(a, {}).get(d, 0):11d}"
                                        for a in ("none", "glossary", "distractor")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+", help="glossary_mined_*.jsonl result files")
    args = ap.parse_args()
    for raw in args.jsonl:
        path = Path(raw)
        # Not with_suffix(): a model id like Qwen3.5-4B makes ".5-4B" look
        # like the suffix, which silently produced an unreadable path and a
        # table of NaNs.
        summary = path.parent / (path.name[: -len(".jsonl")] + ".summary.json")
        report(path, summary if path.name.endswith(".jsonl") else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
