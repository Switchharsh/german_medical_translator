"""Tests for report section extraction."""

from __future__ import annotations

from medmt_eval.data.sections import (
    MEDICAL_ROLES, classify_header, coverage, extract, extract_parallel, split_sections,
)

DE = (
    "Klinik, Fragestellung, Rechtfertigende Indikation: Pankreatitis.\n"
    "Technik: CT des Abdomens, 3 mm Schichtdicke, 100 kV.\n"
    "Befund: Kein Pleuraerguss. Ein 5 mm Knoten links.\n"
    "Beurteilung: Unauffälliger Befund."
)
EN = (
    "Clinical Information: Pancreatitis.\n"
    "Technique: CT of the abdomen, 3 mm slice thickness, 100 kV.\n"
    "Findings: No pleural effusion. A 5 mm nodule on the left.\n"
    "Impression: Unremarkable."
)


def test_headers_are_classified_by_role() -> None:
    assert classify_header("Befund", "de") == "findings"
    assert classify_header("Beurteilung", "de") == "impression"
    assert classify_header("Technik", "de") == "technique"
    assert classify_header("Fragestellung", "de") == "indication"
    assert classify_header("Findings", "en") == "findings"
    assert classify_header("Impression", "en") == "impression"


def test_combined_header_counts_as_findings() -> None:
    """'Befund und Beurteilung' also matches the impression pattern on its second
    half; findings must win so the section is not classified twice."""
    assert classify_header("Befund und Beurteilung", "de") == "findings"
    assert classify_header("Findings and Impression", "en") == "findings"


def test_unknown_headers_fall_back_to_other() -> None:
    assert classify_header("Nierentransplantat", "de") == "other"


def test_split_returns_every_section_in_order() -> None:
    roles = [s.role for s in split_sections(DE, "de")]
    assert roles == ["indication", "technique", "findings", "impression"]


def test_split_of_unstructured_text_is_empty() -> None:
    assert split_sections("Kein Pleuraerguss. Ein 5 mm Knoten links.", "de") == []


def test_untitled_preamble_is_kept_as_other() -> None:
    """A report whose body precedes its only header must not lose that body."""
    text = "Freitext ohne Kopfzeile.\nBeurteilung: Unauffällig."
    roles = [s.role for s in split_sections(text, "de")]
    assert roles == ["other", "impression"]


def test_extract_keeps_only_the_medical_sections() -> None:
    got = extract(DE, "de", MEDICAL_ROLES)
    assert "Pleuraerguss" in got and "Unauffälliger" in got
    assert "Pankreatitis" not in got      # indication dropped
    assert "Schichtdicke" not in got      # technique dropped


def test_technique_numbers_are_excluded() -> None:
    """The acquisition preamble supplies ~30% of critical findings as protocol
    parameters — slice thickness, kV — that are not patient facts."""
    got = extract(DE, "de", MEDICAL_ROLES)
    assert "100 kV" not in got
    assert "3 mm" not in got
    assert "5 mm" in got                  # the clinical measurement survives


def test_extract_returns_none_when_no_requested_section_exists() -> None:
    assert extract("Freitext ohne Kopfzeilen.", "de", MEDICAL_ROLES) is None


def test_parallel_extraction_requires_both_sides() -> None:
    """Scoring an extracted German findings section against a whole English
    report would measure the extraction, not the translation."""
    assert extract_parallel(DE, "No headers here at all.", "de", "en") is None
    assert extract_parallel("Kein Kopf.", EN, "de", "en") is None


def test_parallel_extraction_returns_both_when_alignable() -> None:
    pair = extract_parallel(DE, EN, "de", "en")
    assert pair is not None
    src, ref = pair
    assert "Pleuraerguss" in src and "pleural effusion" in ref
    assert "kV" not in src and "kV" not in ref


def test_coverage_counts_each_case() -> None:
    pairs = [(DE, EN), (DE, "no headers"), ("no headers", EN), ("a", "b")]
    got = coverage(pairs, "de", "en")
    assert got == {"n": 4, "both": 1, "source_only": 1, "reference_only": 1, "neither": 1}
