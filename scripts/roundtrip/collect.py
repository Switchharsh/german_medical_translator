#!/usr/bin/env python3
"""Collect round-trip runs into per-step degradation curves.

Usage:
    python3 scripts/roundtrip/collect.py results/roundtrip_<id> [-o OUT.md]

Reads every ``rt_<model>.jsonl`` in the directory and writes a CSV plus a
markdown report. English steps are scored against the human English reference;
German steps against the original German source, so both curves measure drift
from a fixed anchor rather than from the previous step.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path


def load(run_dir: Path) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for path in sorted(run_dir.glob("rt_*.jsonl")):
        rows = [json.loads(line) for line in path.open() if line.strip()]
        if rows:
            runs[path.stem[3:]] = rows
    return runs


def per_step(rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = collections.defaultdict(list)
    for row in rows:
        grouped[int(row["step"])].append(row)

    out: list[dict] = []
    for step in sorted(grouped):
        group = grouped[step]

        def mean(name: str) -> float | None:
            values = [
                v
                for r in group
                for k, v in (r.get("metrics") or {}).items()
                if k == name and isinstance(v, (int, float))
            ]
            return statistics.mean(values) if values else None

        codes: collections.Counter[str] = collections.Counter()
        for row in group:
            for finding in row.get("findings") or []:
                codes[finding["code"]] += 1

        out.append(
            {
                "step": step,
                "cycle": group[0]["cycle"],
                "direction": group[0]["direction"],
                "n": len(group),
                "crit_rate": sum(1 for r in group if r.get("has_critical_error")) / len(group),
                "hop_crit_rate": sum(1 for r in group if r.get("hop_has_critical_error")) / len(group),
                "bleu": mean("bleu"),
                "chrf": mean("chrf"),
                "ter": mean("ter"),
                "chars": statistics.mean(len(r["hyp_text"]) for r in group),
                "neg": codes["negation_dropped"] + codes["negation_introduced"],
                "num": codes["number_or_measurement_mismatch"],
                "lat": codes["laterality_missing_or_flipped"] + codes["laterality_added_or_flipped"],
                "term": codes["terminology_not_preserved"],
            }
        )
    return out


def fmt(value: float | None, spec: str = "{:.1f}") -> str:
    return "—" if value is None else spec.format(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    runs = load(args.run_dir)
    if not runs:
        print(f"no rt_*.jsonl found in {args.run_dir}", file=sys.stderr)
        return 1

    curves = {model: per_step(rows) for model, rows in runs.items()}

    csv_path = args.run_dir / "roundtrip_steps.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "step", "cycle", "direction", "n", "crit_rate", "hop_crit_rate",
             "bleu", "chrf", "ter", "mean_chars", "neg", "num", "lat", "term"]
        )
        for model, curve in sorted(curves.items()):
            for row in curve:
                writer.writerow(
                    [model, row["step"], row["cycle"], row["direction"], row["n"],
                     f"{row['crit_rate']:.4f}", f"{row['hop_crit_rate']:.4f}",
                     fmt(row["bleu"], "{:.2f}"), fmt(row["chrf"], "{:.2f}"),
                     fmt(row["ter"], "{:.2f}"), f"{row['chars']:.0f}",
                     row["neg"], row["num"], row["lat"], row["term"]]
                )

    lines: list[str] = []
    lines.append("# Round-trip degradation — PARROT German radiology reports\n")
    sample = next(iter(curves.values()))
    n_docs = sample[0]["n"]
    n_steps = len(sample)
    lines.append(
        f"{len(curves)} models · {n_docs} documents · {n_steps // 2} cycles "
        f"({n_steps} translation passes).\n"
    )
    lines.append(
        "Odd steps are DE→EN, scored against the **human English reference**. "
        "Even steps are EN→DE, scored against the **original German**. Both "
        "anchors are fixed, so `crit%` is cumulative drift from ground truth, "
        "not agreement with the previous step. Step 1 is the ordinary "
        "single-pass evaluation.\n"
    )
    lines.append(
        "`hop%` is the error rate for that step alone (its own input vs its own "
        "output), which isolates where a loss was introduced.\n"
    )

    lines.append("## Cumulative critical-error rate by step\n")
    header = "| model | " + " | ".join(str(s["step"]) for s in sample) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(sample) + 1))
    for model, curve in sorted(curves.items()):
        cells = " | ".join(f"{r['crit_rate']*100:.0f}" for r in curve)
        lines.append(f"| {model} | {cells} |")
    lines.append("")

    lines.append("## English steps only — BLEU vs the human reference\n")
    en_steps = [s["step"] for s in sample if s["direction"].endswith("->en")]
    lines.append("| model | " + " | ".join(str(s) for s in en_steps) + " |")
    lines.append("|" + "---|" * (len(en_steps) + 1))
    for model, curve in sorted(curves.items()):
        cells = " | ".join(
            fmt(r["bleu"]) for r in curve if r["direction"].endswith("->en")
        )
        lines.append(f"| {model} | {cells} |")
    lines.append("")

    for model, curve in sorted(curves.items()):
        lines.append(f"## {model}\n")
        lines.append(
            "| step | cycle | direction | crit% | hop% | BLEU | chrF | TER | chars | neg | num | lat | term |"
        )
        lines.append("|" + "---|" * 13)
        for row in curve:
            lines.append(
                f"| {row['step']} | {row['cycle']} | {row['direction']} | "
                f"{row['crit_rate']*100:.0f} | {row['hop_crit_rate']*100:.0f} | "
                f"{fmt(row['bleu'])} | {fmt(row['chrf'])} | {fmt(row['ter'])} | "
                f"{row['chars']:.0f} | {row['neg']} | {row['num']} | {row['lat']} | {row['term']} |"
            )
        lines.append("")

    out_path = args.output or (args.run_dir / "ROUNDTRIP_RESULTS.md")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
