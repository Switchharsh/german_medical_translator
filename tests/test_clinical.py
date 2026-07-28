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
