"""Stable, serialisable records shared by loaders, scorers, and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


_LANGUAGE_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "de": "de",
    "deu": "de",
    "ger": "de",
    "german": "de",
    "deutsch": "de",
    "tr": "tr",
    "tur": "tr",
    "turkish": "tr",
    "türkçe": "tr",
    "turkce": "tr",
}

# Languages with hand-written clinical detectors (negation cues, laterality
# lexicon). Text in any other supported language still flows through the
# pipeline — surface metrics and the language-agnostic number checks apply —
# but the cue-based detectors cannot inspect it. See
# taxonomy.clinical for how that degradation is handled.
DETECTOR_LANGUAGES = frozenset({"en", "de"})


def normalise_language(value: str) -> str:
    """Return the pipeline's two-letter language code or raise a useful error."""
    code = _LANGUAGE_ALIASES.get(value.strip().lower())
    if code is None:
        supported = ", ".join(sorted(set(_LANGUAGE_ALIASES.values())).__iter__())
        raise ValueError(f"Unsupported language {value!r}; supported codes are {supported}.")
    return code


def has_detector_support(lang: str) -> bool:
    """Is `lang` covered by the cue-based clinical detectors?"""
    return normalise_language(lang) in DETECTOR_LANGUAGES


@dataclass(frozen=True)
class Segment:
    """One aligned source/reference translation unit."""

    id: str
    domain: str
    src_lang: str
    tgt_lang: str
    src_text: str
    ref_text: str | None = None
    doc_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, Any],
        *,
        default_src_lang: str | None = None,
        default_tgt_lang: str | None = None,
        position: int | None = None,
    ) -> "Segment":
        """Build a segment while accepting common corpus field aliases."""
        def first(*keys: str) -> Any:
            for key in keys:
                value = row.get(key)
                if value is not None and value != "":
                    return value
            return None

        segment_id = first("id", "segment_id", "uid")
        if segment_id is None:
            segment_id = f"segment-{position if position is not None else 0:06d}"
        src_text = first("src_text", "source_text", "source", "src")
        if src_text is None:
            raise ValueError(f"Segment {segment_id!r} has no source text.")
        src_lang = first("src_lang", "source_lang", "language", "lang") or default_src_lang
        tgt_lang = first("tgt_lang", "target_lang") or default_tgt_lang
        if src_lang is None or tgt_lang is None:
            raise ValueError(
                f"Segment {segment_id!r} needs src_lang and tgt_lang (or CLI defaults)."
            )
        src_lang, tgt_lang = normalise_language(str(src_lang)), normalise_language(str(tgt_lang))
        if src_lang == tgt_lang:
            raise ValueError(f"Segment {segment_id!r} uses the same source and target language.")
        ref_text = first("ref_text", "reference_text", "reference", "target_text", "target", "ref")
        reserved = {
            "id", "segment_id", "uid", "domain", "src_lang", "source_lang", "language", "lang",
            "tgt_lang", "target_lang", "src_text", "source_text", "source", "src", "ref_text",
            "reference_text", "reference", "target_text", "target", "ref", "doc_id", "document_id",
            "hyp_text", "hypothesis", "translation", "mt", "prediction", "model", "generation",
            "metrics", "findings", "has_critical_error",
        }
        metadata = {key: value for key, value in row.items() if key not in reserved}
        return cls(
            id=str(segment_id),
            domain=str(first("domain") or "unspecified"),
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_text=str(src_text),
            ref_text=None if ref_text is None else str(ref_text),
            doc_id=None if first("doc_id", "document_id") is None else str(first("doc_id", "document_id")),
            metadata=metadata,
        )

    def reversed(self) -> "Segment":
        """Reverse an aligned segment for the opposite-direction evaluation."""
        if self.ref_text is None:
            raise ValueError(f"Segment {self.id!r} cannot be reversed without ref_text.")
        return Segment(
            id=self.id,
            domain=self.domain,
            src_lang=self.tgt_lang,
            tgt_lang=self.src_lang,
            src_text=self.ref_text,
            ref_text=self.src_text,
            doc_id=self.doc_id,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClinicalFinding:
    """Structured, reviewable result emitted by a clinical safety detector."""

    code: str
    severity: str  # MQM-aligned: minor, major, or critical
    detector: str
    message: str
    source_evidence: str | None = None
    target_evidence: str | None = None
    confidence: str = "heuristic"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentEvaluation:
    """A hypothesis, its metrics, and all safety findings for one segment."""

    segment: Segment
    hyp_text: str
    model: str
    metrics: dict[str, float | None] = field(default_factory=dict)
    findings: list[ClinicalFinding] = field(default_factory=list)
    generation: dict[str, Any] = field(default_factory=dict)

    @property
    def has_critical_error(self) -> bool:
        return any(finding.severity == "critical" for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        data = self.segment.to_dict()
        data.update(
            {
                "hyp_text": self.hyp_text,
                "model": self.model,
                "metrics": self.metrics,
                "findings": [finding.to_dict() for finding in self.findings],
                "has_critical_error": self.has_critical_error,
                "generation": self.generation,
            }
        )
        return data
