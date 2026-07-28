"""Unit tests for the HimL SGML → Segment converter."""

from __future__ import annotations

import gzip
import io
import tarfile
import textwrap
from pathlib import Path

import pytest

from medmt_eval.data.himl_sgm import _parse_segments, load_himl_from_tar


# ---------------------------------------------------------------------------
# _parse_segments — pure text parsing
# ---------------------------------------------------------------------------

_SAMPLE_EN = textwrap.dedent("""\
    <srcset setid="himl" srclang="en">
    <doc sysid="ref" docid="CD000011" genre="health" origlang="en">
    <p>
    <seg id="1">Interventions for enhancing medication adherence</seg>
    <seg id="2">Ways to help people follow prescribed medicines</seg>
    <seg id="3">Background</seg>
    </p>
    </doc>
    </srcset>
""")

_SAMPLE_DE = textwrap.dedent("""\
    <tstset trglang="de" setid="himl" srclang="en">
    <doc sysid="Edinburgh" docid="CD000011" genre="health" origlang="en">
    <p>
    <seg id="1">Maßnahmen zur Verbesserung der Medikationsadhärenz</seg>
    <seg id="2">Wie man den Patienten helfen kann, die verschriebenen Medikamente zu befolgen</seg>
    <seg id="3">Hintergrund</seg>
    </p>
    </doc>
    </tstset>
""")


def test_parse_segments_extracts_ids_and_text() -> None:
    segs = _parse_segments(_SAMPLE_EN)
    assert segs == {
        "1": "Interventions for enhancing medication adherence",
        "2": "Ways to help people follow prescribed medicines",
        "3": "Background",
    }


def test_parse_segments_unescapes_entities() -> None:
    raw = '<seg id="1">A &amp; B &lt; C &gt; D &quot;quoted&quot;</seg>'
    segs = _parse_segments(raw)
    assert segs["1"] == 'A & B < C > D "quoted"'


def test_parse_segments_collapses_whitespace() -> None:
    raw = '<seg id="1">Line one\n   line two\n   line three</seg>'
    segs = _parse_segments(raw)
    assert segs["1"] == "Line one line two line three"


# ---------------------------------------------------------------------------
# Alignment — mismatched IDs raise
# ---------------------------------------------------------------------------

_MISMATCH_EN = textwrap.dedent("""\
    <seg id="1">Hello</seg>
    <seg id="2">World</seg>
""")
_MISMATCH_DE = textwrap.dedent("""\
    <seg id="1">Hallo</seg>
    <seg id="3">Welt</seg>
""")


def _make_tarball(members: dict[str, str], tmp_path: Path) -> Path:
    """Create a .tgz file containing the given {name: content} members."""
    tar_path = tmp_path / "test.tgz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for name, content in members.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return tar_path


def test_mismatched_seg_ids_raises(tmp_path: Path) -> None:
    tar_path = _make_tarball(
        {
            "himl-test-2015/cochrane.all.en.sgm": _MISMATCH_EN,
            "himl-test-2015/cochrane.all.de.sgm": _MISMATCH_DE,
            "himl-test-2015/nhs24.all.en.sgm": _SAMPLE_EN,
            "himl-test-2015/nhs24.all.de.sgm": _SAMPLE_DE,
            "himl-test-2015/himl.testing.en.sgm": _SAMPLE_EN,
            "himl-test-2015/himl.testing.de.sgm": _SAMPLE_DE,
        },
        tmp_path,
    )
    with pytest.raises(ValueError, match="Segment-ID mismatch"):
        load_himl_from_tar(tar_path, year=2015)


# ---------------------------------------------------------------------------
# Full round-trip: tar → aligned Segment list
# ---------------------------------------------------------------------------

def test_full_round_trip_2015(tmp_path: Path) -> None:
    tar_path = _make_tarball(
        {
            "himl-test-2015/cochrane.all.en.sgm": _SAMPLE_EN,
            "himl-test-2015/cochrane.all.de.sgm": _SAMPLE_DE,
            "himl-test-2015/nhs24.all.en.sgm": _SAMPLE_EN,
            "himl-test-2015/nhs24.all.de.sgm": _SAMPLE_DE,
            "himl-test-2015/himl.testing.en.sgm": _SAMPLE_EN,
            "himl-test-2015/himl.testing.de.sgm": _SAMPLE_DE,
        },
        tmp_path,
    )
    segments = load_himl_from_tar(tar_path, year=2015)
    # 3 subsets × 3 segments each = 9
    assert len(segments) == 9
    # Check first segment fields.
    first = segments[0]
    assert first.id.startswith("himl2015-cochrane-")
    assert first.domain == "himl2015-cochrane"
    assert first.src_lang == "en"
    assert first.tgt_lang == "de"
    assert first.src_text == "Interventions for enhancing medication adherence"
    assert first.ref_text == "Maßnahmen zur Verbesserung der Medikationsadhärenz"


def test_full_round_trip_2017(tmp_path: Path) -> None:
    tar_path = _make_tarball(
        {
            "himl-test-2017/cochrane_output.en.sgm": _SAMPLE_EN,
            "himl-test-2017/cochrane_output.de.sgm": _SAMPLE_DE,
            "himl-test-2017/nhs_output.en.sgm": _SAMPLE_EN,
            "himl-test-2017/nhs_output.de.sgm": _SAMPLE_DE,
        },
        tmp_path,
    )
    segments = load_himl_from_tar(tar_path, year=2017)
    assert len(segments) == 6
    assert segments[0].domain == "himl2017-cochrane"
