"""Unit tests for the PARROT radiology-report converter."""

from __future__ import annotations

import json

import pytest

from medmt_eval.data.parrot import load_parrot_records, normalise_area, parrot_segments


def _write(tmp_path, rows):
    path = tmp_path / "parrot.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _record(no, language="German", report="Kein Pleuraerguss.", translation="No pleural effusion.", **extra):
    row = {
        "no": no,
        "language": language,
        "modality": "CT",
        "area": "chest",
        "report": report,
        "translation": translation,
        "icd": "J90",
        "contributor_code": "someone",
        "country": "Germany",
        "subspecialty": "CH",
    }
    row.update(extra)
    return row


def test_normalise_area_folds_whitespace_and_separator_variants() -> None:
    # PARROT's `area` is free text: "head" and "head " are distinct raw strings.
    assert normalise_area("head") == normalise_area("head ") == "head"
    assert normalise_area("abdomen,pelvis") == normalise_area("abdomen, pelvis") == "abdomen, pelvis"
    assert normalise_area("  Head   and  Neck ") == "head and neck"
    assert normalise_area("") == "unspecified"
    assert normalise_area(None) == "unspecified"


def test_filters_to_german_by_language_not_country(tmp_path) -> None:
    # The `country` field is dirty in the real corpus (the value "German" appears
    # as a country), so filtering must key on `language`.
    rows = [
        _record(1, language="German", country="Switzerland"),
        _record(2, language="French", country="German"),
        _record(3, language="Polish", country="Poland"),
    ]
    records = load_parrot_records(_write(tmp_path, rows))
    assert [r["no"] for r in records] == [1]


def test_skips_records_missing_either_side(tmp_path) -> None:
    rows = [
        _record(1),
        _record(2, translation="   "),
        _record(3, report=""),
    ]
    records = load_parrot_records(_write(tmp_path, rows))
    assert [r["no"] for r in records] == [1]


def test_segments_use_german_as_source_by_default(tmp_path) -> None:
    path = _write(tmp_path, [_record(7)])
    segment = parrot_segments(path)[0]
    assert segment.id == "parrot-7"
    assert segment.src_lang == "de" and segment.tgt_lang == "en"
    assert segment.src_text == "Kein Pleuraerguss."
    assert segment.ref_text == "No pleural effusion."
    assert segment.domain == "parrot-ct"
    assert segment.metadata["area"] == "chest"
    assert segment.metadata["icd"] == "J90"


def test_segments_can_be_inverted_to_en_de(tmp_path) -> None:
    path = _write(tmp_path, [_record(7)])
    segment = parrot_segments(path, src_lang="en", tgt_lang="de")[0]
    assert segment.src_lang == "en" and segment.tgt_lang == "de"
    assert segment.src_text == "No pleural effusion."
    assert segment.ref_text == "Kein Pleuraerguss."


def test_rejects_unsupported_language_pair(tmp_path) -> None:
    path = _write(tmp_path, [_record(1)])
    with pytest.raises(ValueError, match="de/en"):
        parrot_segments(path, src_lang="de", tgt_lang="fr")


def test_raises_when_no_usable_records(tmp_path) -> None:
    path = _write(tmp_path, [_record(1, language="Polish")])
    with pytest.raises(ValueError, match="No usable records"):
        parrot_segments(path)


def test_duplicate_ids_are_rejected(tmp_path) -> None:
    path = _write(tmp_path, [_record(5), _record(5, report="Anderer Befund.")])
    with pytest.raises(ValueError, match="duplicate segment IDs"):
        parrot_segments(path)
