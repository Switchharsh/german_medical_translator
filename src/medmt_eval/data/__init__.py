"""Corpus readers and writers."""

from .io import load_segments, read_rows, write_jsonl, write_parquet

__all__ = [
    "load_segments",
    "read_rows",
    "write_jsonl",
    "write_parquet",
    "himl_sgm",
    "tmx",
]
