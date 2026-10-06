#!/usr/bin/env python
"""xCOMET-XL over a metric-input table, at aligned-segment level, with error spans.

    .venv-comet/bin/python scripts/score_xcomet.py results/metric_inputs_X/de_en_single.jsonl \\
        -o results/xcomet_X/de_en_single.jsonl [--limit N]

xCOMET replaces plain COMET as the learned metric: it is also trained on error-span
(MQM) annotations, so it returns WHERE it thinks the translation is wrong and how badly
(minor / major / critical). Those spans are what lets the learned metric be compared with
the rule-based detectors on the same words.

Scores are in [0, 1], HIGHER is better. Two passes: with the reference, and reference-free
(QE, source + hypothesis only).

Input limit: xCOMET concatenates source, hypothesis and reference into ONE sequence of at
most 512 tokens, and 47% of PARROT reports exceed that, so reports are first cut into
aligned segments (medmt_eval.metrics.segmentation, Gale-Church; validated to reproduce the
index pairing exactly in 89% of the reports where that is checkable).

Licence: Unbabel/XCOMET-XL is CC-BY-NC-SA-4.0 and GATED. The Hugging Face account behind
HF_TOKEN must accept it once at huggingface.co/Unbabel/XCOMET-XL. Fetch the model on the
login node first (scripts/fetch_metric_models.py); compute nodes have no internet.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medmt_eval.metrics.segmentation import flatten  # noqa: E402

REPO = "Unbabel/XCOMET-XL"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--budget", type=int, default=440, help="token budget for a source+hyp+ref segment")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from comet import load_from_checkpoint

    try:
        ckpt_dir = snapshot_download(REPO, local_files_only=args.local_files_only)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"cannot obtain {REPO}: {type(e).__name__}. If this is a 403, accept the licence at "
                         f"https://huggingface.co/{REPO} with the account that owns HF_TOKEN. ({str(e)[:160]})")
    ckpt = next(Path(ckpt_dir).rglob("model.ckpt"), None)
    if ckpt is None:
        raise SystemExit(f"no model.ckpt under {ckpt_dir}")
    tok = AutoTokenizer.from_pretrained("facebook/xlm-roberta-xl", local_files_only=args.local_files_only)
    model = load_from_checkpoint(str(ckpt))
    print(f"preflight: {REPO} checkpoint {ckpt}", flush=True)

    rows = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    length = lambda t: len(tok(t, add_special_tokens=False)["input_ids"]) if t else 0
    by_row, triples, approx = flatten(rows, length_fn=length, budget=args.budget)
    print(f"preflight: {len(rows)} rows -> {sum(map(len, by_row))} segments, {len(triples)} unique", flush=True)

    t0 = time.time()
    with_ref = model.predict([{"src": s, "mt": h, "ref": r} for s, h, r in triples],
                             batch_size=args.batch_size, gpus=args.gpus, progress_bar=False)
    qe = model.predict([{"src": s, "mt": h} for s, h, _ in triples],
                       batch_size=args.batch_size, gpus=args.gpus, progress_bar=False)
    print(f"scored {len(triples)} segments x2 in {time.time()-t0:.0f}s", flush=True)

    ref_sc, qe_sc = list(with_ref.scores), list(qe.scores)
    spans = with_ref.metadata.error_spans
    mqm = list(with_ref.metadata.mqm_scores)
    w = [max(1, length(h)) for _, h, _ in triples]
    sev_total = collections.Counter()
    with args.output.open("w", encoding="utf-8") as f:
        for r, ids, ap_ in zip(rows, by_row, approx):
            wavg = lambda sc: sum(sc[i] * w[i] for i in ids) / sum(w[i] for i in ids)
            sev = collections.Counter(); out_spans = []
            for k, i in enumerate(ids):
                for sp in spans[i]:
                    sev[sp["severity"]] += 1
                    out_spans.append({"seg": k, "severity": sp["severity"], "text": sp["text"],
                                      "confidence": round(float(sp.get("confidence", 0.0)), 3)})
            sev_total.update(sev)
            f.write(json.dumps({
                "set": r["set"], "system": r["system"], "doc_id": r["doc_id"], "step": r["step"],
                "direction": r["direction"], "n_segments": len(ids), "approx_share": round(ap_, 4),
                "xcomet": round(wavg(ref_sc), 5), "xcomet_worst": round(min(ref_sc[i] for i in ids), 5),
                "xcomet_qe": round(wavg(qe_sc), 5), "xcomet_mqm": round(wavg(mqm), 5),
                "spans_minor": sev["minor"], "spans_major": sev["major"], "spans_critical": sev["critical"],
                "has_critical_span": sev["critical"] > 0, "spans": out_spans,
            }, ensure_ascii=False) + "\n")
    print(f"done: {args.output} | mean xcomet {statistics.mean(ref_sc):.3f} qe {statistics.mean(qe_sc):.3f} | "
          f"spans {dict(sev_total)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
