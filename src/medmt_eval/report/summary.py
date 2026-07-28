"""Reporting helpers for the core fluency-versus-safety argument."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from medmt_eval.metrics.surface import score_surface


def _findings(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = row.get("findings", [])
    return raw if isinstance(raw, list) else []


def _metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = row.get("metrics", {})
    return raw if isinstance(raw, dict) else {}


def aggregate_evaluations(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build model × direction × domain rows from per-segment evaluations."""
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("model", "unknown")),
            f"{row.get('src_lang', '?')}→{row.get('tgt_lang', '?')}",
            str(row.get("domain", "unspecified")),
        )
        groups[key].append(row)
    table: list[dict[str, Any]] = []
    for (model, direction, domain), group in sorted(groups.items()):
        total = len(group)
        metric_means: dict[str, float] = {}
        references = [row.get("ref_text") for row in group]
        if all(reference is not None for reference in references):
            # Report true corpus metrics rather than means of sentence metrics.
            metric_means.update(
                score_surface(
                    [str(row["hyp_text"]) for row in group], [str(reference) for reference in references]
                ).corpus
            )
        for metric in ("comet",):
            values = [float(_metrics(row)[metric]) for row in group if _metrics(row).get(metric) is not None]
            if values:
                metric_means[metric] = sum(values) / len(values)
        error_segments: Counter[str] = Counter()
        critical_count = 0
        for row in group:
            codes = {str(finding.get("code")) for finding in _findings(row)}
            error_segments.update(codes)
            critical_count += int(any(finding.get("severity") == "critical" for finding in _findings(row)))
        def rate_for_prefix(prefix: str) -> float:
            return sum(
                any(str(finding.get("code", "")).startswith(prefix) for finding in _findings(row))
                for row in group
            ) / total

        record: dict[str, Any] = {
            "model": model,
            "direction": direction,
            "domain": domain,
            "n_segments": total,
            **metric_means,
            "critical_error_rate": critical_count / total,
            "negation_flip_rate": rate_for_prefix("negation_"),
            "laterality_error_rate": rate_for_prefix("laterality_"),
            "number_error_rate": rate_for_prefix("number_or_measurement_mismatch"),
            "terminology_error_rate": rate_for_prefix("terminology_not_preserved"),
        }
        table.append(record)
    return table


def write_master_table(table: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the compact reporting table as CSV without requiring pandas."""
    records = list(table)
    if not records:
        raise ValueError("No aggregate rows to write.")
    columns = [
        "model", "direction", "domain", "n_segments", "bleu", "chrf", "ter", "comet",
        "critical_error_rate", "negation_flip_rate", "laterality_error_rate", "number_error_rate",
        "terminology_error_rate",
    ]
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return file_path


def plot_divergence(table: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Plot surface BLEU against clinically critical error incidence."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Plotting requires `pip install -e '.[plot]'`.") from error
    records = [record for record in table if record.get("bleu") is not None]
    if not records:
        raise ValueError("The divergence plot needs BLEU scores and at least one aggregate row.")
    figure, axis = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    for record in records:
        label = f"{record['model']} ({record['direction']}, {record['domain']})"
        axis.scatter(record["bleu"], record["critical_error_rate"] * 100, s=62)
        axis.annotate(label, (record["bleu"], record["critical_error_rate"] * 100), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("sacreBLEU (higher is better)")
    axis.set_ylabel("Clinically critical error rate (%)")
    axis.set_title("Surface fluency does not establish clinical preservation")
    axis.grid(alpha=0.25)
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(file_path, dpi=180)
    plt.close(figure)
    return file_path
