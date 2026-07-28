"""Format-neutral IO for the pipeline's normalized records."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from medmt_eval.schema import Segment


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL, CSV, TSV, or Parquet into dictionaries.

    Parquet support is intentionally optional, so a lightweight installation can
    still run the complete rule-based evaluation pipeline.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    suffix = file_path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with file_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number} in {file_path}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"Line {line_number} in {file_path} is not a JSON object.")
                rows.append(row)
        return rows
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with file_path.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover - dependency-specific branch
            raise RuntimeError("Reading Parquet requires `pip install -e '.[storage]'`.") from error
        return pd.read_parquet(file_path).to_dict(orient="records")
    raise ValueError(f"Unsupported input format {suffix!r}; use JSONL, CSV, TSV, or Parquet.")


def load_segments(
    path: str | Path,
    *,
    default_src_lang: str | None = None,
    default_tgt_lang: str | None = None,
    reverse: bool = False,
) -> list[Segment]:
    """Load and validate normalized corpus rows."""
    segments = [
        Segment.from_mapping(
            row,
            default_src_lang=default_src_lang,
            default_tgt_lang=default_tgt_lang,
            position=index,
        )
        for index, row in enumerate(read_rows(path), start=1)
    ]
    if not segments:
        raise ValueError(f"No segments found in {path}.")
    ids = [segment.id for segment in segments]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Input {path} contains duplicate segment IDs.")
    return [segment.reversed() for segment in segments] if reverse else segments


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write stable UTF-8 JSONL and create only the requested parent directory."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str))
            handle.write("\n")
    return file_path


def write_parquet(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write Parquet when the optional storage dependencies are installed."""
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - dependency-specific branch
        raise RuntimeError("Writing Parquet requires `pip install -e '.[storage]'`.") from error
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(file_path, index=False)
    return file_path
