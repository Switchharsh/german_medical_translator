"""Unit tests for iterative round-trip translation."""

from __future__ import annotations

import pytest

from medmt_eval.inference.roundtrip import run_roundtrip, summarise_by_step
from medmt_eval.models.base import GenerationConfig, Translator
from medmt_eval.schema import Segment


# Fake translators must actually change language: the clinical detectors compare
# German cues against English cues, so a double that echoes German text while the
# runner labels it English trips them for the wrong reason.
DE0 = "Kein Erguss. Ein 5 mm Knoten links."
EN0 = "No effusion. A 5 mm nodule on the left."
DE_LOST = "Kein Erguss. Ein Knoten links."
EN_LOST = "No effusion. A nodule on the left."


class _StableTranslator(Translator):
    """A faithful translator: round-trips DE0 <-> EN0 forever with no drift."""

    name = "stable"
    _MAP = {("de", "en"): {DE0: EN0, DE_LOST: EN_LOST},
            ("en", "de"): {EN0: DE0, EN_LOST: DE_LOST}}

    def __init__(self, src: str, tgt: str) -> None:
        self.src, self.tgt = src, tgt

    def translate(self, texts, src_lang, tgt_lang):
        table = self._MAP[(src_lang, tgt_lang)]
        return [table.get(t, t) for t in texts]

    @property
    def generation_config(self):
        return {"adapter": self.name, **GenerationConfig().to_dict()}


class _PinnedTranslator(_StableTranslator):
    """Stands in for Opus: one checkpoint per direction, so it refuses any other."""

    name = "pinned"
    direction_specific = True

    def translate(self, texts, src_lang, tgt_lang):
        assert (src_lang, tgt_lang) == (self.src, self.tgt), "wrong direction for this instance"
        return super().translate(texts, src_lang, tgt_lang)


class _DecayTranslator(_StableTranslator):
    """Loses the measurement on the first de->en hop, then is stable."""

    name = "decay"
    _MAP = {("de", "en"): {DE0: EN_LOST, DE_LOST: EN_LOST},
            ("en", "de"): {EN0: DE0, EN_LOST: DE_LOST}}


def _segments(n=2):
    return [
        Segment(
            id=f"s{i}",
            domain="test",
            src_lang="de",
            tgt_lang="en",
            src_text=DE0,
            ref_text=EN0,
        )
        for i in range(n)
    ]


def test_cycle_count_produces_two_steps_each() -> None:
    """10 cycles must mean 20 translation passes, as requested."""
    rows = run_roundtrip(lambda s, t: _StableTranslator(s, t), _segments(2), cycles=10)
    steps = sorted({row["step"] for row in rows})
    assert steps == list(range(1, 21))
    assert len(rows) == 20 * 2  # steps x segments


def test_directions_alternate_starting_with_source_to_english() -> None:
    rows = run_roundtrip(lambda s, t: _StableTranslator(s, t), _segments(1), cycles=2)
    by_step = {row["step"]: row["direction"] for row in rows}
    assert by_step[1] == "de->en"
    assert by_step[2] == "en->de"
    assert by_step[3] == "de->en"
    assert by_step[4] == "en->de"


def test_cycle_numbering_groups_two_steps() -> None:
    rows = run_roundtrip(lambda s, t: _StableTranslator(s, t), _segments(1), cycles=3)
    pairs = {(row["step"], row["cycle"]) for row in rows}
    assert (1, 1) in pairs and (2, 1) in pairs
    assert (3, 2) in pairs and (4, 2) in pairs
    assert (5, 3) in pairs and (6, 3) in pairs


def test_direction_specific_adapter_gets_one_instance_per_direction() -> None:
    """Opus refuses to switch direction on a live instance, so it needs two."""
    built: list[tuple[str, str]] = []

    def factory(src: str, tgt: str):
        built.append((src, tgt))
        return _PinnedTranslator(src, tgt)

    run_roundtrip(factory, _segments(1), cycles=5)
    assert built == [("de", "en"), ("en", "de")]


def test_shared_adapter_is_loaded_only_once() -> None:
    """Every non-Opus adapter takes the direction as an argument, so one loaded
    instance serves both ways. Building two doubled GPU memory and OOM'd a 14 GB
    model on a 20 GB slice (job 4006929)."""
    built: list[tuple[str, str]] = []

    def factory(src: str, tgt: str):
        built.append((src, tgt))
        return _StableTranslator(src, tgt)

    rows = run_roundtrip(factory, _segments(1), cycles=5)
    assert built == [("de", "en")]
    assert len(rows) == 10  # still ran all 10 steps


def test_output_feeds_into_the_next_step() -> None:
    """Step N+1 must translate step N's output, not the original."""
    rows = run_roundtrip(lambda s, t: _DecayTranslator(s, t), _segments(1), cycles=2)
    by_step = {row["step"]: row for row in rows}
    assert by_step[2]["input_text"] == by_step[1]["hyp_text"]
    assert by_step[3]["input_text"] == by_step[2]["hyp_text"]


def test_origin_anchor_never_changes() -> None:
    """Every step is scored against the untouched original, not the prior step."""
    rows = run_roundtrip(lambda s, t: _DecayTranslator(s, t), _segments(1), cycles=3)
    originals = {row["src_text"] for row in rows}
    assert originals == {DE0}


def test_stable_translator_accumulates_no_drift() -> None:
    rows = run_roundtrip(lambda s, t: _StableTranslator(s, t), _segments(1), cycles=4)
    assert not any(row["has_critical_error"] for row in rows)


def test_decaying_translator_is_flagged_against_origin() -> None:
    """A dropped measurement must surface as a finding versus the original."""
    rows = run_roundtrip(lambda s, t: _DecayTranslator(s, t), _segments(1), cycles=2)
    assert any(row["has_critical_error"] for row in rows)
    codes = {f["code"] for row in rows for f in row["findings"]}
    assert "number_or_measurement_mismatch" in codes


def test_hop_findings_isolate_the_step_that_broke_it() -> None:
    """clinical_vs_input equivalent: step 1 drops the number, later hops are clean
    relative to their own input even though drift from origin persists."""
    rows = run_roundtrip(lambda s, t: _DecayTranslator(s, t), _segments(1), cycles=2)
    by_step = {row["step"]: row for row in rows}
    assert by_step[1]["hop_has_critical_error"] is True
    assert by_step[2]["hop_has_critical_error"] is False
    # ...but drift from the original is still visible at step 2.
    assert by_step[2]["has_critical_error"] is True


def test_summary_has_one_row_per_step() -> None:
    rows = run_roundtrip(lambda s, t: _StableTranslator(s, t), _segments(3), cycles=5)
    summary = summarise_by_step(rows)
    assert len(summary) == 10
    assert [s["step"] for s in summary] == list(range(1, 11))
    assert all(s["n_segments"] == 3 for s in summary)


def test_summary_reports_metrics_and_rates() -> None:
    rows = run_roundtrip(lambda s, t: _DecayTranslator(s, t), _segments(2), cycles=2)
    summary = summarise_by_step(rows)
    first = summary[0]
    assert 0.0 <= first["critical_error_rate"] <= 1.0
    assert first["bleu"] is not None
    assert first["mean_output_chars"] > 0


def test_rejects_zero_cycles() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_roundtrip(lambda s, t: _StableTranslator(s, t), _segments(1), cycles=0)


def test_rejects_mixed_directions() -> None:
    mixed = [
        Segment(id="a", domain="t", src_lang="de", tgt_lang="en", src_text="x", ref_text="y"),
        Segment(id="b", domain="t", src_lang="en", tgt_lang="de", src_text="x", ref_text="y"),
    ]
    with pytest.raises(ValueError, match="single translation direction"):
        run_roundtrip(lambda s, t: _StableTranslator(s, t), mixed, cycles=1)


def test_empty_input_returns_nothing() -> None:
    assert run_roundtrip(lambda s, t: _StableTranslator(s, t), [], cycles=3) == []
