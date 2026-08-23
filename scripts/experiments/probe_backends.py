#!/usr/bin/env python3
"""Preflight: can each debate backend take a free-form prompt?

    python3 scripts/experiments/probe_backends.py local:Qwen/Qwen3.5-4B ...

Exits non-zero naming any model whose chat template is translation-only. Loads
tokenizers only, so it costs seconds rather than the twelve minutes it took to
discover the same thing after three model loads (job 4077236).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from medmt_eval.models.chat import supports_freeform_chat  # noqa: E402


def main(specs: list[str]) -> int:
    unusable: list[str] = []
    for spec in specs:
        kind, _, model = spec.partition(":")
        if not model:
            kind, model = "api", kind
        if kind != "local":
            print(f"preflight: {spec} (hosted — template probe not applicable)")
            continue
        ok, reason = supports_freeform_chat(model)
        print(f"preflight: {model} -> {'OK' if ok else 'UNUSABLE'} ({reason})")
        if not ok:
            unusable.append(model)
    if unusable:
        print(
            "ERROR: these models cannot take a free-form debate prompt: "
            + ", ".join(unusable),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
