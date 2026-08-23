"""Split a radiology report into its sections, and keep only the medical ones.

A PARROT report is one free-text string. It typically opens with material that is
*about* the examination rather than about the patient's anatomy — the clinical
question, the justifying indication, the acquisition protocol, consent — and only
then reaches the findings and the impression.

That preamble is not neutral for this project's metrics:

* It inflates BLEU slightly and uniformly (≈1.1–1.7 points across systems), so it
  does not change rankings.
* It contributes a **disproportionate share of critical clinical findings**: on a
  sample of 109 German reports, 30% of located critical findings sat in the
  preamble, and every one of them was a ``number_or_measurement_mismatch``
  arising from acquisition parameters — slice thickness, T1/T2 weighting, kV,
  contrast volume. Those are protocol metadata, not patient facts, and counting
  them as clinical errors overstates the error rate.

So this module exists to answer "can the *medical* text be translated" rather
than "can the whole document be reproduced".

Two hard limits, both measured on the German subset (296 reports):

* Only **35%** carry a findings/impression header on *both* the German and the
  English side. Extraction is impossible for the rest, so callers must decide
  whether to fall back to the whole report or drop the document — hence
  :func:`extract_parallel` returning ``None`` rather than guessing.
* Headers do not map one-to-one across languages. One report's German has
  clinical-question, findings and impression sections while its English
  translation has only findings and impression: the translator dropped a whole
  section. Extraction therefore selects by *role* on each side independently
  rather than trying to align section counts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Section roles, coarsest useful taxonomy for this corpus.
ROLES = ("indication", "technique", "findings", "impression", "other")

#: Roles that constitute "the medical text" by default. The clinical indication
#: is medical content too, but it is a *question* rather than an observation and
#: its English rendering is the least standardised part of the corpus, so it is
#: opt-in rather than default.
MEDICAL_ROLES = ("findings", "impression")

_PATTERNS: dict[str, dict[str, re.Pattern[str]]] = {
    "de": {
        # "Befund und Beurteilung" must be tried before bare "Befund", and is
        # tagged findings; its impression half is not separately addressable.
        "findings": re.compile(
            r"^[ \t]*(Befund(?:\s*(?:und|/|\+)\s*Beurteilung)?|Befunde|Bildbefund)\s*:",
            re.M | re.I),
        "impression": re.compile(
            r"^[ \t]*((?:Zusammenfassende\s+)?Beurteilung|Zusammenfassung|Fazit|Schlussfolgerung)\s*:",
            re.M | re.I),
        "technique": re.compile(
            r"^[ \t]*(Technik|Untersuchungstechnik|Methode|Aufkl[äa]rung\s+und\s+Einwilligung|Kontrastmittel)\s*:",
            re.M | re.I),
        "indication": re.compile(
            r"^[ \t]*(Klinik[^:\n]*|Fragestellung|Indikation|Rechtfertigende\s+Indikation|Anamnese|Vorbefund)\s*:",
            re.M | re.I),
    },
    "en": {
        "findings": re.compile(
            r"^[ \t]*(Findings?(?:\s*(?:and|/|\+)\s*(?:Impression|Assessment|Conclusion))?|Report)\s*:",
            re.M | re.I),
        "impression": re.compile(
            r"^[ \t]*(Impression|Assessment|Conclusion|Summary|Interpretation)\s*:",
            re.M | re.I),
        "technique": re.compile(
            r"^[ \t]*(Technique|Procedure|Method|Protocol|Consent|Contrast(?:\s+agent)?)\s*:",
            re.M | re.I),
        "indication": re.compile(
            r"^[ \t]*(Clinical[^:\n]*|Question|Indication|History|Justification[^:\n]*|Comparison)\s*:",
            re.M | re.I),
    },
}

# Any line-initial "Word...:" is a candidate boundary, so that an unrecognised
# header still terminates the preceding section instead of being swallowed by it.
_ANY_HEADER = re.compile(r"^[ \t]*([^\W\d_][^:\n]{1,70})\s*:", re.M | re.U)


@dataclass(frozen=True)
class Section:
    role: str
    header: str
    body: str

    @property
    def text(self) -> str:
        """Header and body, as they appear in the report."""
        return f"{self.header}: {self.body}".strip()


def classify_header(header: str, lang: str) -> str:
    """Map a header string to a role, or "other" if unrecognised."""
    table = _PATTERNS.get(lang.lower())
    if not table:
        return "other"
    probe = f"{header}:"
    # findings before impression: "Befund und Beurteilung" is a findings header
    # that also matches the impression pattern on its second half.
    for role in ("findings", "impression", "technique", "indication"):
        if table[role].match(probe):
            return role
    return "other"


def split_sections(text: str, lang: str) -> list[Section]:
    """Split a report at line-initial headers. Empty when the report has none.

    Text before the first header is not a section — it is an untitled preamble —
    and is returned with role "other" only if it is non-empty, so that a report
    written as free text with a trailing "Beurteilung:" does not silently lose
    its body.
    """
    matches = list(_ANY_HEADER.finditer(text))
    if not matches:
        return []
    sections: list[Section] = []
    lead = text[: matches[0].start()].strip()
    if lead:
        sections.append(Section("other", "", lead))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        header = match.group(1).strip()
        body = text[match.end():end].strip()
        sections.append(Section(classify_header(header, lang), header, body))
    return sections


def extract(text: str, lang: str, roles: tuple[str, ...] = MEDICAL_ROLES) -> str | None:
    """Concatenate the sections whose role is in ``roles``.

    Returns ``None`` when the report has no section of any requested role, which
    is the caller's signal that this document cannot be reduced to medical text —
    not that it has no medical content.
    """
    sections = split_sections(text, lang)
    if not sections:
        return None
    kept = [s for s in sections if s.role in roles]
    if not kept:
        return None
    return "\n".join(s.text for s in kept).strip()


def extract_parallel(
    source: str,
    reference: str,
    src_lang: str,
    tgt_lang: str,
    roles: tuple[str, ...] = MEDICAL_ROLES,
) -> tuple[str, str] | None:
    """Extract the same roles from both sides, or ``None`` if either lacks them.

    Both sides are required. Scoring an extracted German findings section against
    a whole English report would measure the extraction, not the translation.
    """
    src = extract(source, src_lang, roles)
    ref = extract(reference, tgt_lang, roles)
    if src is None or ref is None:
        return None
    return src, ref


def coverage(pairs: list[tuple[str, str]], src_lang: str, tgt_lang: str,
             roles: tuple[str, ...] = MEDICAL_ROLES) -> dict[str, int]:
    """How many pairs can be reduced to medical text — report before relying on it."""
    both = source_only = reference_only = neither = 0
    for source, reference in pairs:
        has_src = extract(source, src_lang, roles) is not None
        has_ref = extract(reference, tgt_lang, roles) is not None
        if has_src and has_ref:
            both += 1
        elif has_src:
            source_only += 1
        elif has_ref:
            reference_only += 1
        else:
            neither += 1
    return {
        "n": len(pairs), "both": both, "source_only": source_only,
        "reference_only": reference_only, "neither": neither,
    }
