#!/usr/bin/env python3
"""Score existing translations with COMET, and check whether it reorders BLEU.

    python3 scripts/score_comet.py results/parrot_de/consolidated -o results/comet_parrot_de.json

Roadmap step 1 ([thesis/07-metric-roadmap.md]): the cheapest possible upgrade,
because every system's output is already on disk and nothing has to be
retranslated.

The question it answers is specific. BLEU ranks these systems one way and the
clinical detectors rank them another; the whole thesis rests on that divergence.
If a *learned semantic* metric reproduces the BLEU order, then the clinical layer
is carrying the argument alone and had better be trustworthy. If COMET reorders
them, the divergence is not an artefact of n-gram matching.

Caveat carried into the output: COMET is trained on WMT news-domain human
judgements and is documented to degrade off-distribution
(arXiv:2402.18747). Radiology reports are a long way from news, so a COMET score
here is evidence, not adjudication.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medmt_eval.metrics.neural import CometScorer  # noqa: E402


def load_system(path: Path) -> tuple[str, list[dict]]:
    seen: dict[str, dict] = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        seen.setdefault(row["id"], row)
    name = path.stem.split("parrot_de_")[-1]
    return name, list(seen.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("results/comet_scores.json"))
    parser.add_argument("--checkpoint", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true",
                        help="Never touch the network (required on compute nodes)")
    args = parser.parse_args()

    files = sorted(args.run_dir.glob("parrot_de_*.jsonl"))
    if not files:
        raise SystemExit(f"no parrot_de_*.jsonl in {args.run_dir}")

    scorer = CometScorer(args.checkpoint, local_files_only=args.local_files_only)
    results: dict[str, dict] = {}
    for path in files:
        name, rows = load_system(path)
        rows = [r for r in rows if r.get("ref_text") and r.get("hyp_text") is not None]
        if not rows:
            print(f"  {name}: no scorable rows, skipped")
            continue
        scores = scorer.score(
            [r["src_text"] for r in rows],
            [r["hyp_text"] for r in rows],
            [r["ref_text"] for r in rows],
            batch_size=args.batch_size,
            gpus=args.gpus,
        )
        results[name] = {
            "comet": scores.system_score,
            "n": len(rows),
            "bleu": statistics.mean(
                r["metrics"]["bleu"] for r in rows if r.get("metrics", {}).get("bleu") is not None
            ),
            "critical_error_rate": sum(1 for r in rows if r.get("has_critical_error")) / len(rows),
        }
        print(f"  {name:22} COMET {scores.system_score:.4f}  "
              f"BLEU {results[name]['bleu']:.2f}  crit {results[name]['critical_error_rate']*100:.1f}%",
              flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"checkpoint": args.checkpoint, "systems": results}, indent=2))
    print(f"\nwrote {args.output}")

    # The actual question: does a learned metric agree with n-gram overlap?
    names = [n for n in results if n != "identity"]
    by_bleu = sorted(names, key=lambda n: -results[n]["bleu"])
    by_comet = sorted(names, key=lambda n: -results[n]["comet"])
    by_crit = sorted(names, key=lambda n: results[n]["critical_error_rate"])
    print("\nrank by BLEU :", " > ".join(by_bleu))
    print("rank by COMET:", " > ".join(by_comet))
    print("rank by crit%:", " > ".join(by_crit))
    print(f"\nBLEU and COMET agree on the ordering: {by_bleu == by_comet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
