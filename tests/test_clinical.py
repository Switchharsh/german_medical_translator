from medmt_eval.taxonomy.clinical import ClinicalSafetyEvaluator, TerminologyBank, TermPair, extract_numbers


def test_detects_critical_negation_laterality_and_number_loss() -> None:
    findings = ClinicalSafetyEvaluator().evaluate(
        "No 5 mm nodule is present in the left upper lobe.",
        "Ein 6 mm großer Knoten ist im rechten Oberlappen sichtbar.",
        "en",
        "de",
    )
    codes = {finding.code for finding in findings}
    assert "negation_dropped" in codes
    assert "laterality_missing_or_flipped" in codes
    assert "laterality_added_or_flipped" in codes
    assert "number_or_measurement_mismatch" in codes
    assert all(finding.severity == "critical" for finding in findings)


def test_preserves_equivalent_measurement_with_decimal_comma() -> None:
    findings = ClinicalSafetyEvaluator().evaluate(
        "A 1.5 cm lesion is present.",
        "Eine 15 mm große Läsion ist vorhanden.",
        "en",
        "de",
    )
    assert not [finding for finding in findings if finding.code == "number_or_measurement_mismatch"]


def test_detects_de_to_en_negation_drop() -> None:
    findings = ClinicalSafetyEvaluator().evaluate(
        "Keine akute Fraktur der rechten sechsten Rippe.",
        "An acute fracture of the right sixth rib is present.",
        "de",
        "en",
    )
    assert any(finding.code == "negation_dropped" for finding in findings)


def test_extracts_percentage_as_measurement() -> None:
    mention = extract_numbers("Less than 10 % of the lungs are involved.")[0]
    assert mention.dimension == "percent"
    assert mention.canonical_value == 10


def test_optional_term_bank_marks_missing_expected_target_term() -> None:
    bank = TerminologyBank([TermPair("RID1", "pleural effusion", "Pleuraerguss")])
    findings = ClinicalSafetyEvaluator(bank).evaluate(
        "No pleural effusion.", "Kein Erguss im Brustraum.", "en", "de"
    )
    assert [finding.code for finding in findings] == ["terminology_not_preserved"]


# ---------------------------------------------------------------------------
# Term-bank surface matching (inflection tolerance)
# ---------------------------------------------------------------------------

def test_term_pattern_matches_english_plural() -> None:
    """Regression: an exact-boundary match rejects "lymph nodes" for the bank
    entry "lymph node", flagging correct translations as terminology failures.
    On PARROT this single effect produced 572 of 735 terminology findings."""
    import re
    from medmt_eval.taxonomy.clinical import term_surface_pattern

    pattern = term_surface_pattern("lymph node")
    assert re.search(pattern, "multiple enlarged lymph nodes", re.IGNORECASE)
    assert re.search(pattern, "a single lymph node", re.IGNORECASE)


def test_term_pattern_matches_german_inflection() -> None:
    import re
    from medmt_eval.taxonomy.clinical import term_surface_pattern

    assert re.search(term_surface_pattern("Lymphknoten"), "vergrößerte Lymphknotens", re.IGNORECASE)
    assert re.search(term_surface_pattern("Erguss"), "kein Ergusses", re.IGNORECASE)


def test_term_pattern_matches_multiword_plural() -> None:
    import re
    from medmt_eval.taxonomy.clinical import term_surface_pattern

    pattern = term_surface_pattern("pleural effusion")
    assert re.search(pattern, "bilateral pleural effusions", re.IGNORECASE)
    assert re.search(pattern, "no pleural effusion", re.IGNORECASE)


def test_term_pattern_does_not_match_unrelated_longer_word() -> None:
    """The suffix allowance must not let the match drift onto a different
    concept: "lymph nodules" is not "lymph nodes"."""
    import re
    from medmt_eval.taxonomy.clinical import term_surface_pattern

    pattern = term_surface_pattern("lymph node")
    assert not re.search(pattern, "lymph nodules were absent", re.IGNORECASE)
    assert not re.search(pattern, "no lymphatic tissue seen", re.IGNORECASE)


def test_inflected_target_term_is_no_longer_flagged() -> None:
    """End-to-end: a correct plural translation must not raise a finding."""
    bank = TerminologyBank([TermPair("RID1", "lymph node", "Lymphknoten")])
    findings = ClinicalSafetyEvaluator(bank).evaluate(
        "Vergrößerte Lymphknoten paraaortal.",
        "Enlarged lymph nodes in the para-aortic region.",
        "de",
        "en",
    )
    assert [f for f in findings if f.code == "terminology_not_preserved"] == []


def test_genuinely_missing_term_still_flagged() -> None:
    bank = TerminologyBank([TermPair("RID1", "lymph node", "Lymphknoten")])
    findings = ClinicalSafetyEvaluator(bank).evaluate(
        "Vergrößerte Lymphknoten paraaortal.",
        "Enlarged structures in the para-aortic region.",
        "de",
        "en",
    )
    assert any(f.code == "terminology_not_preserved" for f in findings)


# ---------------------------------------------------------------------------
# Languages without detector lexicons (e.g. Turkish)
# ---------------------------------------------------------------------------

def test_detector_coverage_reports_reduced_support() -> None:
    ev = ClinicalSafetyEvaluator()
    assert ev.detector_coverage("de", "en") == {
        "negation": True, "laterality": True,
        "number_or_measurement": True, "terminology": True,
    }
    # TR has no cue lexicon and no term-bank column; only the language-agnostic
    # number check survives.
    assert ev.detector_coverage("tr", "en") == {
        "negation": False, "laterality": False,
        "number_or_measurement": True, "terminology": False,
    }


def test_uncovered_source_language_does_not_raise() -> None:
    """A bare dict lookup on the cue tables used to KeyError for any language
    outside en/de, taking the whole run down."""
    findings = ClinicalSafetyEvaluator().evaluate(
        "Sağ akciğerde nodül.", "A nodule in the right lung.", "tr", "en"
    )
    assert isinstance(findings, list)


def test_uncovered_source_does_not_fabricate_laterality_findings() -> None:
    """The laterality detector diffs source against target. With no Turkish
    lexicon the source side is always empty, so an unguarded diff would report
    every English laterality mention as 'added' — inventing findings."""
    findings = ClinicalSafetyEvaluator().evaluate(
        "Sağ akciğerde 5 mm nodül.", "A 5 mm nodule in the right lung.", "tr", "en"
    )
    assert not [f for f in findings if f.code.startswith("laterality")]


def test_uncovered_source_does_not_fabricate_negation_findings() -> None:
    findings = ClinicalSafetyEvaluator().evaluate(
        "Plevral efüzyon yok.", "No pleural effusion.", "tr", "en"
    )
    assert not [f for f in findings if f.code.startswith("negation")]


def test_number_check_still_works_for_uncovered_languages() -> None:
    """Numbers are language-agnostic, so this detector must remain fully active."""
    findings = ClinicalSafetyEvaluator().evaluate(
        "Sağ akciğerde 8 mm nodül.", "A 5 mm nodule in the lung.", "tr", "en"
    )
    assert any(f.code == "number_or_measurement_mismatch" for f in findings)


def test_term_bank_skips_languages_it_has_no_column_for() -> None:
    """The bank has en/de columns only. Without a guard the lookup silently fell
    back to the German term, searching Turkish text for German words."""
    bank = TerminologyBank([TermPair("RID1", "pleural effusion", "Pleuraerguss")])
    assert bank.expected_terms("Plevral efüzyon mevcut.", "tr", "en") == []
    assert len(bank.expected_terms("Pleuraerguss vorhanden.", "de", "en")) == 1
