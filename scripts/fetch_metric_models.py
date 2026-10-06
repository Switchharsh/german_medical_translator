#!/usr/bin/env python
"""Pre-fetch the learned-metric models on the LOGIN node (compute nodes have no internet).

    .venv-comet/bin/python scripts/fetch_metric_models.py

* xCOMET-XL   Unbabel/XCOMET-XL (13.9 GB, gated=auto: the token's account must have
              accepted the licence once at huggingface.co/Unbabel/XCOMET-XL)
* MetricX-24  google/metricx-24-hybrid-large-v2p6 (4.9 GB, Apache-2.0)
* tokenizers  only the tokenizer/config files of the encoders each metric instantiates,
              NOT their weights (xlm-roberta-xl alone is 28 GB and the metric checkpoints
              carry the fine-tuned weights already).
"""
import os, sys
from huggingface_hub import snapshot_download

tok = os.environ.get("HF_TOKEN")
TOKENIZER_ONLY = ["*.json", "*.model", "*.txt", "sentencepiece*", "spiece*", "tokenizer*"]
jobs = [
    ("Unbabel/XCOMET-XL", None),
    ("google/metricx-24-hybrid-large-v2p6", None),
    ("facebook/xlm-roberta-xl", TOKENIZER_ONLY),
    ("google/mt5-large", TOKENIZER_ONLY),
]
rc = 0
for repo, patterns in jobs:
    try:
        print(f"== {repo} {'(tokenizer/config only)' if patterns else ''}", flush=True)
        path = snapshot_download(repo, token=tok, allow_patterns=patterns)
        print(f"   ok -> {path}", flush=True)
    except Exception as e:
        rc = 1
        print(f"   FAILED {repo}: {type(e).__name__}: {str(e)[:300]}", flush=True)
sys.exit(rc)
