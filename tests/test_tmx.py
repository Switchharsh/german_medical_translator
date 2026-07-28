"""Unit tests for the EMEA TMX parser with alignment-quality filtering."""

from __future__ import annotations

import gzip
import textwrap
from pathlib import Path

import pytest

from medmt_eval.data.tmx import (
    _is_degenerate,
    _is_length_ratio_outlier,
    load_emea_from_tmx,
)

_SAMPLE_TMX = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8" ?>
    <tmx version="1.4">
    <header srclang="de" adminlang="de" segtype="sentence"
            creationtool="Uplug" datatype="PlainText" />
      <body>
        <tu>
          <tuv xml:lang="de"><seg>Was ist Abilify?</seg></tuv>
          <tuv xml:lang="en"><seg>What is Abilify?</seg></tuv>
        </tu>
        <tu>
          <tuv xml:lang="de"><seg>Abilify ist ein Arzneimittel.</seg></tuv>
          <tuv xml:lang="en"><seg>Abilify is a medicine.</seg></tuv>
        </tu>
        <tu>
          <tuv xml:lang="de"><seg>EMEA/H/C/471</seg></tuv>
          <tuv xml:lang="en"><seg>EMEA/ H/ C/ 471</seg></tuv>
        </tu>
        <tu>
          <tuv xml:lang="de"><seg>ABILIFY</seg></tuv>
          <tuv xml:lang="en"><seg>ABILIFY</seg></tuv>
        </tu>
        <tu>
          <tuv xml:lang="de"><seg>Dies ist ein sehr langer deutscher Satz der viele Worte enthaelt um das Laengenverhaeltnis zu testen und sollte gefiltert werden.</seg></tuv>
          <tuv xml:lang="en"><seg>Short.</seg></tuv>
        </tu>
        <tu>
          <tuv xml:lang="de"><seg>Was ist Abilify?</seg></tuv>
          <tuv xml:lang="en"><seg>What is Abilify?</seg></tuv>
        </tu>
      </body>
    </tmx>
""")


def _write_tmx_gz(content: str, path: Path) -> Path:
    file_path = path / "test.tmx.gz"
    with gzip.open(file_path, "wt", encoding="utf-8") as fh:
        fh.write(content)
    return file_path


# ---------------------------------------------------------------------------
# Individual filter functions
# ---------------------------------------------------------------------------

def test_degenerate_empty() -> None:
    assert _is_degenerate("", "hello", min_length=3) is True
    assert _is_degenerate("hello", "", min_length=3) is True


def test_degenerate_short() -> None:
    assert _is_degenerate("AB", "AB", min_length=3) is True
    assert _is_degenerate("ABC", "DEF", min_length=3) is False


def test_degenerate_identical_across_languages() -> None:
    assert _is_degenerate("ABILIFY", "ABILIFY", min_length=3) is True
    assert _is_degenerate("EMEA/H/C/471", "EMEA/H/C/471", min_length=3) is True


def test_length_ratio_normal() -> None:
    assert _is_length_ratio_outlier("Hello world", "Hallo Welt", max_ratio=3.0) is False


def test_length_ratio_extreme() -> None:
    long_text = "Dies ist ein sehr langer deutscher Satz der viele Worte enthaelt."
    short_text = "Short."
    assert _is_length_ratio_outlier(long_text, short_text, max_ratio=3.0) is True


def test_length_ratio_empty_side() -> None:
    assert _is_length_ratio_outlier("", "something", max_ratio=3.0) is True


# ---------------------------------------------------------------------------
# Full round-trip: TMX → filtered Segment list
# ---------------------------------------------------------------------------

def test_load_emea_filters_degenerate_and_duplicates(tmp_path: Path) -> None:
    tmx_path = _write_tmx_gz(_SAMPLE_TMX, tmp_path)
    segments = load_emea_from_tmx(
        tmx_path,
        src_lang="de",
        tgt_lang="en",
        min_length=3,
        max_length_ratio=3.0,
    )
    # Expected:
    #   "Was ist Abilify?" / "What is Abilify?" — valid (appears twice, deduped)
    #   "Abilify ist ein Arzneimittel." / "Abilify is a medicine." — valid
    #   "EMEA/H/C/471" / "EMEA/ H/ C/ 471" — valid (not identical, length OK)
    #   "ABILIFY" / "ABILIFY" — degenerate (identical across languages)
    #   Long/Short extreme ratio — filtered by length ratio
    #   Duplicate of first pair — deduped
    assert len(segments) == 3
    assert segments[0].src_text == "Was ist Abilify?"
    assert segments[1].src_text == "Abilify ist ein Arzneimittel."
    assert segments[2].src_text == "EMEA/H/C/471"


def test_load_emea_sample_size(tmp_path: Path) -> None:
    tmx_path = _write_tmx_gz(_SAMPLE_TMX, tmp_path)
    segments = load_emea_from_tmx(
        tmx_path,
        sample_size=1,
        seed=42,
    )
    assert len(segments) == 1


def test_load_emea_segment_schema(tmp_path: Path) -> None:
    tmx_path = _write_tmx_gz(_SAMPLE_TMX, tmp_path)
    segments = load_emea_from_tmx(tmx_path)
    seg = segments[0]
    assert seg.id.startswith("emea-")
    assert seg.domain == "emea"
    assert seg.src_lang == "de"
    assert seg.tgt_lang == "en"
    assert seg.ref_text is not None
