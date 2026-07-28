"""Unit tests for multi-model leaderboard aggregation."""

from __future__ import annotations

from medmt_eval.report.summary import aggregate_evaluations


def _eval_row(
    model: str,
    domain: str = "radiology-ct-synthetic",
    src_lang: str = "en",
    tgt_lang: str = "de",
    hyp_text: str = "translation",
    ref_text: str = "reference",
    bleu: float = 50.0,
    chrf: float = 70.0,
    ter: float = 30.0,
    has_critical: bool = False,
    finding_code: str | None = None,
) -> dict:
    """Build a minimal evaluation row for testing aggregation."""
    findings = []
    if finding_code:
        findings.append(
            {"code": finding_code, "severity": "critical" if has_critical else "warning"}
        )
    return {
        "model": model,
        "domain": domain,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "hyp_text": hyp_text,
        "ref_text": ref_text,
        "metrics": {"bleu": bleu, "chrf": chrf, "ter": ter},
        "findings": findings,
        "has_critical_error": has_critical,
    }


def test_aggregate_single_model_single_domain() -> None:
    rows = [
        _eval_row("opus", bleu=60.0),
        _eval_row("opus", bleu=40.0),
    ]
    table = aggregate_evaluations(rows)
    assert len(table) == 1
    row = table[0]
    assert row["model"] == "opus"
    assert row["domain"] == "radiology-ct-synthetic"
    assert row["n_segments"] == 2
    # Corpus BLEU is computed from the actual texts, not the mean of sentence scores.
    assert "bleu" in row


def test_aggregate_multi_model_multi_domain() -> None:
    rows = [
        _eval_row("opus", domain="emea", has_critical=True, finding_code="negation_dropped"),
        _eval_row("opus", domain="himl2015-cochrane"),
        _eval_row("nllb", domain="emea"),
        _eval_row("nllb", domain="himl2015-cochrane", has_critical=True, finding_code="laterality_missing_or_flipped"),
    ]
    table = aggregate_evaluations(rows)
    # 2 models × 2 domains = 4 rows
    assert len(table) == 4
    models = {r["model"] for r in table}
    assert models == {"opus", "nllb"}
    domains = {r["domain"] for r in table}
    assert domains == {"emea", "himl2015-cochrane"}


def test_aggregate_critical_error_rate() -> None:
    rows = [
        _eval_row("m1", has_critical=True, finding_code="negation_dropped"),
        _eval_row("m1", has_critical=False),
        _eval_row("m1", has_critical=False),
    ]
    table = aggregate_evaluations(rows)
    assert len(table) == 1
    assert table[0]["critical_error_rate"] == pytest.approx(1 / 3)


def test_aggregate_finding_type_rates() -> None:
    rows = [
        _eval_row("m1", has_critical=True, finding_code="negation_dropped"),
        _eval_row("m1", has_critical=True, finding_code="negation_dropped"),
        _eval_row("m1", has_critical=False, finding_code="laterality_missing_or_flipped"),
        _eval_row("m1", has_critical=False),
    ]
    table = aggregate_evaluations(rows)
    assert len(table) == 1
    row = table[0]
    assert row["negation_flip_rate"] == pytest.approx(2 / 4)
    assert row["laterality_error_rate"] == pytest.approx(1 / 4)
    assert row["number_error_rate"] == pytest.approx(0.0)
    assert row["terminology_error_rate"] == pytest.approx(0.0)


def test_aggregate_direction_grouping() -> None:
    """Different directions for the same model+domain produce separate rows."""
    rows = [
        _eval_row("opus", src_lang="en", tgt_lang="de"),
        _eval_row("opus", src_lang="de", tgt_lang="en"),
    ]
    table = aggregate_evaluations(rows)
    assert len(table) == 2
    directions = {r["direction"] for r in table}
    assert directions == {"en→de", "de→en"}


# Needed for pytest.approx at module level.
import pytest
