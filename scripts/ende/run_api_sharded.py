#!/usr/bin/env python
"""Run a hosted-model translation in resumable shards.

    python scripts/ende/run_api_sharded.py --model-id DeepSeek-V4-Flash \\
        --input data/derived/parrot_en_de.jsonl --out results/parrot_en_de_X/parrot_en_de_DeepSeek-V4-Flash.jsonl

`medmt-eval run` writes its output only at the end, so one HTTP 500 after three hours of
translating loses everything (DeepSeek EN->DE, 2026-10-06). Here the corpus is split into
shards; each is a separate run whose output is kept, a failed shard is retried with backoff,
and a re-launch skips finished shards. Shards are merged in the input's original order.

Per-row scores (BLEU / chrF++ / TER, detector findings) are computed per document and are
identical to an unsharded run. The corpus-level summary JSON is NOT produced; no downstream
script here uses it (they read per-document rows).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=12)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=2048,
                    help="per-report output ceiling; the request ceiling is this x batch-size. Reasoning models "
                         "spend most of it on hidden reasoning, so give them generous headroom")
    ap.add_argument("--term-bank", default="data/term_banks/radiology_en_de_starter.csv")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    work = args.out.with_suffix(".shards"); work.mkdir(parents=True, exist_ok=True)
    rows = [l for l in args.input.open(encoding="utf-8") if l.strip()]
    per = -(-len(rows) // args.shards)
    env = {**os.environ, "OPENAI_COMPAT_MAX_WORKERS": os.environ.get("OPENAI_COMPAT_MAX_WORKERS", "3")}
    print(f"preflight: {len(rows)} rows, {args.shards} shards of <= {per}, model {args.model_id}", flush=True)

    outs = []
    for i in range(args.shards):
        part = rows[i * per:(i + 1) * per]
        if not part:
            continue
        sin, sout = work / f"in_{i:02d}.jsonl", work / f"out_{i:02d}.jsonl"
        outs.append(sout)
        if sout.exists():
            print(f"  shard {i:02d}: already done", flush=True); continue
        sin.write_text("".join(part), encoding="utf-8")
        for attempt in range(1, args.retries + 1):
            t0 = time.time()
            cmd = ["medmt-eval", "run", "--input", str(sin), "--model", "openai-compat", "--model-id", args.model_id,
                   "--batch-size", str(args.batch_size), "--num-beams", "1", "--max-new-tokens", str(args.max_new_tokens),
                   "--max-input-tokens", "4096", "--term-bank", args.term_bank,
                   "--output", str(sout), "--summary", str(work / f"sum_{i:02d}.json")]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if r.returncode == 0 and sout.exists():
                print(f"  shard {i:02d}: ok in {time.time() - t0:.0f}s (attempt {attempt})", flush=True); break
            tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["?"]
            print(f"  shard {i:02d}: attempt {attempt} failed after {time.time() - t0:.0f}s: {tail[0][:140]}", flush=True)
            if sout.exists():
                sout.unlink()
            time.sleep(min(600, 30 * attempt * attempt))
        else:
            raise SystemExit(f"shard {i} failed {args.retries} times; re-run to resume from the finished shards")

    merged = [l for o in outs for l in o.open(encoding="utf-8") if l.strip()]
    if len(merged) != len(rows):
        raise SystemExit(f"merge mismatch: {len(merged)} rows vs {len(rows)} input")
    empty = [json.loads(l)["doc_id"] for l in merged if json.loads(l)["hyp_text"].strip() in ("", "None")
             or len(json.loads(l)["hyp_text"]) < 0.3 * len(json.loads(l)["src_text"])]   # blank, the string "None", or cut off
    args.out.write_text("".join(merged), encoding="utf-8")
    print(f"done: {args.out} ({len(merged)} rows); EMPTY hypotheses: {len(empty)} {empty[:10]}", flush=True)
    if empty:
        print("WARNING: empty translations usually mean the reasoning used the whole token ceiling; "
              "re-run those documents with a larger --max-new-tokens before scoring.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
