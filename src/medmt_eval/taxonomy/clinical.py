"""Transparent, deterministic first-pass clinical safety detectors.

These rules are intentionally inspectable. They identify review candidates and
should be validated against bilingual clinician annotations before being treated
as a clinical-safety metric.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

from medmt_eval.schema import ClinicalFinding, normalise_language


_NEGATION_PATTERNS = {
    "en": re.compile(
        r"\b(?:no|not|without|neither|nor|denies|denied|absence\s+of|negative\s+for)\b",
        re.IGNORECASE,
    ),
    "de": re.compile(
        r"\b(?:kein(?:e[mnr]?|en|er|es)?|ohne|nicht|weder|negativ\s+für|ausschluss)\b",
        re.IGNORECASE,
    ),
}

_LATERALITY_PATTERNS = {
    "en": {
        "left": re.compile(r"\bleft(?:-sided)?\b", re.IGNORECASE),
        "right": re.compile(r"\bright(?:-sided)?\b", re.IGNORECASE),
        "bilateral": re.compile(r"\b(?:bilateral(?:ly)?|both\s+sides?)\b", re.IGNORECASE),
    },
    "de": {
        "left": re.compile(r"\b(?:links(?:seitig(?:e[nrms]?)?)?|linke[nrms]?)\b", re.IGNORECASE),
        "right": re.compile(r"\b(?:rechts(?:seitig(?:e[nrms]?)?)?|rechte[nrms]?)\b", re.IGNORECASE),
        "bilateral": re.compile(r"\b(?:beidseits|beidseitig(?:e[nrms]?)?)\b", re.IGNORECASE),
    },
}

_UNIT_PATTERN = (
    r"mm|cm|m|µm|um|mg|g|kg|µg|ug|ml|mL|l|L|cc|cm3|cm³|%|mmhg|mmol/l|mg/dl|"
    r"HU|bpm|/min"
)
_MEASUREMENT_PATTERN = re.compile(
    rf"(?<![\w.])(?P<value>[+-]?\d+(?:[.,]\d+)?)\s*(?P<unit>{_UNIT_PATTERN})(?!\w)",
    re.IGNORECASE,
)
_BARE_NUMBER_PATTERN = re.compile(r"(?<![\w.,])[+-]?\d+(?:[.,]\d+)?(?![\w.,])")

# Values are (dimension, multiplier to an agreed base unit). Unrecognised units
# remain comparable only to themselves, which avoids silently accepting a unit swap.
_UNIT_NORMALISATION: dict[str, tuple[str, Decimal]] = {
    "µm": ("length_mm", Decimal("0.001")),
    "um": ("length_mm", Decimal("0.001")),
    "mm": ("length_mm", Decimal("1")),
    "cm": ("length_mm", Decimal("10")),
    "m": ("length_mm", Decimal("1000")),
    "µg": ("mass_mg", Decimal("0.001")),
    "ug": ("mass_mg", Decimal("0.001")),
    "mg": ("mass_mg", Decimal("1")),
    "g": ("mass_mg", Decimal("1000")),
    "kg": ("mass_mg", Decimal("1000000")),
    "ml": ("volume_ml", Decimal("1")),
    "l": ("volume_ml", Decimal("1000")),
    "cc": ("volume_ml", Decimal("1")),
    "cm3": ("volume_ml", Decimal("1")),
    "cm³": ("volume_ml", Decimal("1")),
    "%": ("percent", Decimal("1")),
    "mmhg": ("pressure_mmhg", Decimal("1")),
    "mmol/l": ("mmol_l", Decimal("1")),
    "mg/dl": ("mg_dl", Decimal("1")),
    "hu": ("hu", Decimal("1")),
    "bpm": ("bpm", Decimal("1")),
    "/min": ("per_min", Decimal("1")),
}


@dataclass(frozen=True)
class NumberMention:
    raw: str
    value: Decimal
    unit: str | None
    dimension: str
    canonical_value: Decimal

    @property
    def key(self) -> tuple[str, Decimal]:
        return self.dimension, self.canonical_value


@dataclass(frozen=True)
class TermPair:
    concept_id: str
    en: str
    de: str


class TerminologyBank:
    """Small pluggable EN-DE term bank (RadLex/UMLS exports can be converted to it)."""

    def __init__(self, terms: Iterable[TermPair] = ()) -> None:
        self.terms = tuple(terms)

    @classmethod
    def from_csv(cls, path: str | Path) -> "TerminologyBank":
        with Path(path).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"concept_id", "en", "de"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"Term bank {path} must contain concept_id,en,de columns.")
            return cls(
                TermPair(str(row["concept_id"]), str(row["en"]), str(row["de"]))
                for row in reader
                if row.get("en") and row.get("de")
            )

    def expected_terms(self, source_text: str, src_lang: str, tgt_lang: str) -> list[TermPair]:
        source, target = normalise_language(src_lang), normalise_language(tgt_lang)
        if source == target:
            return []
        matched: list[TermPair] = []
        for pair in self.terms:
            source_term = pair.en if source == "en" else pair.de
            if re.search(rf"(?<!\w){re.escape(source_term)}(?!\w)", source_text, re.IGNORECASE):
                matched.append(pair)
        return matched


def _normalise_decimal(raw_value: str) -> Decimal:
    # Medical reports in this evaluation use decimal commas in German. Thousands
    # separators should be normalised upstream if they are meaningful data.
    try:
        return Decimal(raw_value.replace(",", "."))
    except InvalidOperation as error:  # pragma: no cover - regex guarantees a number
        raise ValueError(f"Could not parse numeric value {raw_value!r}") from error


def extract_numbers(text: str) -> list[NumberMention]:
    """Extract measurements and bare numbers, normalising compatible units."""
    mentions: list[NumberMention] = []
    occupied: list[tuple[int, int]] = []
    for match in _MEASUREMENT_PATTERN.finditer(text):
        raw_unit = match.group("unit")
        unit = raw_unit.lower().replace("³", "3")
        value = _normalise_decimal(match.group("value"))
        dimension, multiplier = _UNIT_NORMALISATION.get(unit, (f"unit:{unit}", Decimal("1")))
        mentions.append(
            NumberMention(
                raw=match.group(0),
                value=value,
                unit=unit,
                dimension=dimension,
                canonical_value=value * multiplier,
            )
        )
        occupied.append(match.span())
    for match in _BARE_NUMBER_PATTERN.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in occupied):
            continue
        value = _normalise_decimal(match.group(0))
        mentions.append(
            NumberMention(
                raw=match.group(0),
                value=value,
                unit=None,
                dimension="unitless",
                canonical_value=value,
            )
        )
    return mentions


def _laterality(text: str, lang: str) -> set[str]:
    found = {name for name, pattern in _LATERALITY_PATTERNS[lang].items() if pattern.search(text)}
    if "bilateral" in found:
        found.update({"left", "right"})
        found.remove("bilateral")
    return found


class ClinicalSafetyEvaluator:
    """Apply Stage-1 clinical information-loss rules to a source/hypothesis pair."""

    def __init__(self, term_bank: TerminologyBank | None = None) -> None:
        self.term_bank = term_bank or TerminologyBank()

    def evaluate(self, source_text: str, hypothesis: str, src_lang: str, tgt_lang: str) -> list[ClinicalFinding]:
        source, target = normalise_language(src_lang), normalise_language(tgt_lang)
        findings: list[ClinicalFinding] = []
        findings.extend(self._negation(source_text, hypothesis, source, target))
        findings.extend(self._laterality(source_text, hypothesis, source, target))
        findings.extend(self._numbers(source_text, hypothesis))
        findings.extend(self._terminology(source_text, hypothesis, source, target))
        return findings

    @staticmethod
    def _negation(source_text: str, hypothesis: str, src_lang: str, tgt_lang: str) -> list[ClinicalFinding]:
        source_cues = [match.group(0) for match in _NEGATION_PATTERNS[src_lang].finditer(source_text)]
        target_cues = [match.group(0) for match in _NEGATION_PATTERNS[tgt_lang].finditer(hypothesis)]
        if bool(source_cues) == bool(target_cues):
            return []
        direction = "dropped" if source_cues else "introduced"
        return [
            ClinicalFinding(
                code=f"negation_{direction}",
                severity="critical",
                detector="segment_negation_cues",
                message=f"Negation appears {direction} between source and translation.",
                source_evidence=", ".join(source_cues) or None,
                target_evidence=", ".join(target_cues) or None,
                details={"source_cue_count": len(source_cues), "target_cue_count": len(target_cues)},
            )
        ]

    @staticmethod
    def _laterality(source_text: str, hypothesis: str, src_lang: str, tgt_lang: str) -> list[ClinicalFinding]:
        expected, observed = _laterality(source_text, src_lang), _laterality(hypothesis, tgt_lang)
        if not expected and not observed:
            return []
        findings: list[ClinicalFinding] = []
        missing = expected - observed
        added = observed - expected
        if missing:
            findings.append(
                ClinicalFinding(
                    code="laterality_missing_or_flipped",
                    severity="critical",
                    detector="laterality_lexicon",
                    message=f"Expected laterality {sorted(missing)} is absent from translation.",
                    source_evidence=", ".join(sorted(expected)),
                    target_evidence=", ".join(sorted(observed)) or None,
                    details={"expected": sorted(expected), "observed": sorted(observed)},
                )
            )
        if added:
            findings.append(
                ClinicalFinding(
                    code="laterality_added_or_flipped",
                    severity="critical",
                    detector="laterality_lexicon",
                    message=f"Unexpected laterality {sorted(added)} appears in translation.",
                    source_evidence=", ".join(sorted(expected)) or None,
                    target_evidence=", ".join(sorted(observed)),
                    details={"expected": sorted(expected), "observed": sorted(observed)},
                )
            )
        return findings

    @staticmethod
    def _numbers(source_text: str, hypothesis: str) -> list[ClinicalFinding]:
        expected, observed = extract_numbers(source_text), extract_numbers(hypothesis)
        expected_counts = Counter(item.key for item in expected)
        observed_counts = Counter(item.key for item in observed)
        if expected_counts == observed_counts:
            return []
        missing = list((expected_counts - observed_counts).elements())
        added = list((observed_counts - expected_counts).elements())
        return [
            ClinicalFinding(
                code="number_or_measurement_mismatch",
                severity="critical",
                detector="number_unit_parser",
                message="Numbers or normalised measurements do not match between source and translation.",
                source_evidence="; ".join(item.raw for item in expected) or None,
                target_evidence="; ".join(item.raw for item in observed) or None,
                details={
                    "missing_normalised": [(unit, str(value)) for unit, value in missing],
                    "added_normalised": [(unit, str(value)) for unit, value in added],
                },
            )
        ]

    def _terminology(self, source_text: str, hypothesis: str, src_lang: str, tgt_lang: str) -> list[ClinicalFinding]:
        findings: list[ClinicalFinding] = []
        for pair in self.term_bank.expected_terms(source_text, src_lang, tgt_lang):
            expected = pair.de if tgt_lang == "de" else pair.en
            if not re.search(rf"(?<!\w){re.escape(expected)}(?!\w)", hypothesis, re.IGNORECASE):
                source_term = pair.en if src_lang == "en" else pair.de
                findings.append(
                    ClinicalFinding(
                        code="terminology_not_preserved",
                        severity="major",
                        detector="term_bank_exact_match",
                        message=f"Term-bank concept {pair.concept_id} was not found as its expected target term.",
                        source_evidence=source_term,
                        target_evidence=expected,
                        details={"concept_id": pair.concept_id, "expected_target_term": expected},
                    )
                )
        return findings
