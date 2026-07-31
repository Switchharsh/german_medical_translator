"""Convert the PARROT multilingual radiology-report dataset into the pipeline schema.

PARROT (https://github.com/PARROT-reports/PARROT_v1.0) is an open collection of
fictional-but-radiologist-authored radiology reports in 14 languages, each with a
contributor-supplied English translation.  This module extracts a single-language
subset (German by default) and emits normalized ``Segment`` rows.

Two properties of the source data drive the implementation:

* ``language`` is the reliable language field.  ``country`` is dirty (the value
  "German" appears as a country for 50 records), so filtering must use
  ``language`` and never ``country``.
* ``area`` is free text rather than a controlled vocabulary: 55 distinct strings
  for the German subset alone, including whitespace-only variants ("head" vs
  "head ") and separator variants ("abdomen, pelvis" vs "abdomen,pelvis").
  ``normalise_area`` folds these together so results can be grouped by body region.

Translation-provenance caveat: the PARROT paper states only that "contributors
provided an English translation"; no professional translation, review, or QA step
is documented.  The English side is therefore an *unverified-provenance* human
reference rather than a certified gold standard, and results scored against it
should say so.

PARROT's README states the dataset "should not be used for training" — this
converter is for building evaluation sets only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from medmt_eval.schema import Segment

# Values of the `language` field that denote German.
_GERMAN_ALIASES = {"german", "deutsch", "de"}

_WHITESPACE = re.compile(r"\s+")


def normalise_area(value: Any) -> str:
    """Fold PARROT's free-text `area` field into a stable grouping key.

    Collapses internal/trailing whitespace, lowercases, and normalises comma
    spacing so "abdomen,pelvis", "abdomen, pelvis" and "Abdomen , Pelvis " all
    map to the same key.  Returns "unspecified" for empty values.
    """
    text = _WHITESPACE.sub(" ", str(value or "").strip()).lower()
    if not text:
        return "unspecified"
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return ", ".join(parts) if parts else "unspecified"


def _language_matches(value: Any, aliases: set[str]) -> bool:
    return str(value or "").strip().lower() in aliases


def load_parrot_records(
    path: str | Path,
    *,
    language_aliases: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Read PARROT JSONL and return records for one language with usable text.

    Records missing either the original report or its English translation are
    skipped, since both sides are required for reference-based scoring.
    """
    aliases = language_aliases or _GERMAN_ALIASES
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    records: list[dict[str, Any]] = []
    with file_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} in {file_path}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} in {file_path} is not a JSON object.")
            if not _language_matches(row.get("language"), aliases):
                continue
            report = str(row.get("report") or "").strip()
            translation = str(row.get("translation") or "").strip()
            if not report or not translation:
                continue
            records.append(row)
    if not records:
        raise ValueError(f"No usable records for the requested language in {path}.")
    return records


def parrot_segments(
    path: str | Path,
    *,
    src_lang: str = "de",
    tgt_lang: str = "en",
    language_aliases: set[str] | None = None,
) -> list[Segment]:
    """Build Segments from PARROT.

    With ``src_lang="de"`` the original German report is the source and the
    contributor's English translation is the reference.  With ``src_lang="en"``
    the pair is inverted (English translation as source, original German report
    as reference) — defensible, but note the human translator worked in the
    opposite direction, so translationese runs backwards relative to a corpus
    authored natively in English.
    """
    if {src_lang, tgt_lang} != {"de", "en"}:
        raise ValueError("PARROT conversion supports the de/en pair only.")

    records = load_parrot_records(path, language_aliases=language_aliases)
    segments: list[Segment] = []
    for row in records:
        report = str(row.get("report") or "").strip()
        translation = str(row.get("translation") or "").strip()
        german_first = src_lang == "de"
        source_text = report if german_first else translation
        reference_text = translation if german_first else report

        modality = str(row.get("modality") or "unspecified").strip() or "unspecified"
        area = normalise_area(row.get("area"))
        identifier = f"parrot-{row.get('no')}"
        segments.append(
            Segment(
                id=identifier,
                # Group by modality so results break out by exam type; the raw
                # and normalised area both live in metadata for finer slicing.
                domain=f"parrot-{modality.lower()}",
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_text=source_text,
                ref_text=reference_text,
                doc_id=identifier,
                metadata={
                    "modality": modality,
                    "area": area,
                    "area_raw": str(row.get("area") or ""),
                    "icd": str(row.get("icd") or ""),
                    "subspecialty": str(row.get("subspecialty") or ""),
                    "contributor_code": str(row.get("contributor_code") or ""),
                    "country": str(row.get("country") or ""),
                },
            )
        )

    ids = [segment.id for segment in segments]
    if len(ids) != len(set(ids)):
        raise ValueError(f"PARROT input {path} produced duplicate segment IDs.")
    return segments


def parrot_rows(
    path: str | Path,
    *,
    src_lang: str = "de",
    tgt_lang: str = "en",
    language_aliases: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return PARROT segments as plain dicts ready for JSONL serialisation."""
    return [
        segment.to_dict()
        for segment in parrot_segments(
            path,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            language_aliases=language_aliases,
        )
    ]
