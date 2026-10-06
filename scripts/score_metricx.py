#!/usr/bin/env python
"""MetricX-24 (hybrid) over a metric-input table, at aligned-segment level.

    .venv-comet/bin/python scripts/score_metricx.py results/metric_inputs_X/de_en_single.jsonl \\
        -o results/metricx_X/de_en_single.jsonl [--model large] [--limit N]

MetricX-24 outputs an ERROR score on 0-25 where LOWER IS BETTER, the opposite polarity to
BLEU and COMET. The hybrid model scores both with a reference (``source/candidate/
reference``) and without one (QE: ``source/candidate``); both are reported. The QE score
is a second reference-free signal, independent of the single gold translation.

Reports are cut into aligned segments first (see medmt_eval.metrics.segmentation).
MetricX takes 1536 tokens, far more than xCOMET, so segments here are large (budget 1100).
Run on a GPU node from the COMET venv (transformers 4.x); compute nodes have no internet,
so fetch with scripts/fetch_metric_models.py first.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "metricx"))

import torch  # noqa: E402
import transformers  # noqa: E402

from medmt_eval.metrics.segmentation import flatten  # noqa: E402
from metricx24 import models  # noqa: E402

MODELS = {"large": ("google/metricx-24-hybrid-large-v2p6", "google/mt5-large"),
          "xl": ("google/metricx-24-hybrid-xl-v2p6", "google/mt5-xl")}


def predict(model, tok, texts, device, batch_size, max_len, attn_budget=2.5e7):
    """Score texts in length-sorted batches.

    Attention memory grows with batch * length^2, so a fixed batch size that is fine for
    short inputs overflows on the longest (job 4474361: a batch of 32 near-1500-token inputs
    asked for a 4.5 GB softmax buffer on a 20 GB MIG slice). The batch is therefore capped
    so that n * L^2 stays under ``attn_budget``.
    """
    enc = [tok(t, max_length=max_len, truncation=True, padding=False)["input_ids"][:-1] for t in texts]  # drop EOS
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    out = [0.0] * len(texts)
    s = 0
    while s < len(order):
        # lengths ascend, so the batch's longest item is the last one we would add
        n = 1
        while (s + n < len(order) and n < batch_size
               and (n + 1) * len(enc[order[s + n]]) ** 2 <= attn_budget):
            n += 1
        idx = order[s:s + n]
        s += n
        ids = [enc[i] for i in idx]
        width = max(len(x) for x in ids)
        input_ids = torch.tensor([x + [tok.pad_token_id] * (width - len(x)) for x in ids], device=device)
        mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids], device=device)
        with torch.no_grad():
            # use_cache=False: MetricX was written for transformers 4.30 and its single-step
            # dummy decoder trips the 4.57 cache/mask bookkeeping (size mismatch 27 vs 26).
            pred = model(input_ids=input_ids, attention_mask=mask, use_cache=False).predictions
        for i, v in zip(idx, pred.float().cpu().tolist()):
            out[i] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--model", choices=sorted(MODELS), default="large")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--budget", type=int, default=1100, help="token budget for a source+hyp+ref segment")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows (smoke test)")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    mname, tname = MODELS[args.model]
    tok = transformers.AutoTokenizer.from_pretrained(tname, local_files_only=args.local_files_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = models.MT5ForRegression.from_pretrained(mname, torch_dtype=dtype,
                                                    local_files_only=args.local_files_only).to(device).eval()
    print(f"preflight: {mname} on {device} ({dtype}), tokenizer {tname}", flush=True)

    rows = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    length = lambda t: len(tok(t, add_special_tokens=False)["input_ids"]) if t else 0
    by_row, triples, approx = flatten(rows, length_fn=length, budget=args.budget)
    print(f"preflight: {len(rows)} rows -> {sum(map(len, by_row))} segments, {len(triples)} unique", flush=True)

    t0 = time.time()
    ref_in = [f"source: {s} candidate: {h} reference: {r}" for s, h, r in triples]
    qe_in = [f"source: {s} candidate: {h}" for s, h, _ in triples]
    ref_sc = predict(model, tok, ref_in, device, args.batch_size, args.max_length)
    qe_sc = predict(model, tok, qe_in, device, args.batch_size, args.max_length)
    print(f"scored {len(triples)} segments x2 in {time.time()-t0:.0f}s", flush=True)

    w = [max(1, length(h)) for _, h, _ in triples]
    with args.output.open("w", encoding="utf-8") as f:
        for r, ids, ap_ in zip(rows, by_row, approx):
            ws = [w[i] for i in ids]
            wavg = lambda sc: sum(sc[i] * w[i] for i in ids) / sum(ws)
            f.write(json.dumps({
                "set": r["set"], "system": r["system"], "doc_id": r["doc_id"], "step": r["step"],
                "direction": r["direction"], "n_segments": len(ids), "approx_share": round(ap_, 4),
                "metricx_ref": round(wavg(ref_sc), 4), "metricx_ref_worst": round(max(ref_sc[i] for i in ids), 4),
                "metricx_qe": round(wavg(qe_sc), 4), "metricx_qe_worst": round(max(qe_sc[i] for i in ids), 4),
            }, ensure_ascii=False) + "\n")
    print(f"done: {args.output} | mean ref {statistics.mean(ref_sc):.2f} qe {statistics.mean(qe_sc):.2f} (0-25, lower better)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
