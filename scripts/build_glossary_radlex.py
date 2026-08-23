#!/usr/bin/env python3
"""Extract an EN-DE radiology glossary from RadLex.

    python3 scripts/build_glossary_radlex.py datasets/RadLex.owl \
        -o data/term_banks/radlex_de_en.csv

Two input formats, and they are not equivalent.

**RadLex.owl (use this).** Carries the DRG German translation as explicit
language tags: 12,100 of them, on ``rdfs:label`` and ``RID:Synonym``. Nothing has
to be guessed.

**RADLEX.csv (BioPortal export).** Does *not* carry the translation. Its
``Preferred_name_German`` column is populated for one concept in 46,900, and the
only German present is untagged inside the pipe-separated ``Synonyms`` field. The
CSV path below recovers ~389 pairs from that by rule, and is kept only as a
fallback.

RadLex is the right terminology for this project — it is radiology-native, so it
cannot contain the cross-domain polysemy that makes a general encyclopedic
source dangerous here (see thesis/08-terminology.md).

The catch in this export: the German translation is **not** in the
``Preferred_name_German`` column. That column is populated for exactly one of
46,900 concepts. The German terms are instead mixed, untagged, into the
pipe-separated ``Synonyms`` field alongside English variants and Latin anatomy.
So German has to be *recognised* rather than read off.

For the CSV fallback, language identification is done with rules rather than a
statistical detector: langdetect is built for sentences and on single medical
terms called "Epiduralhämatom" Estonian and "posteriore Schallverstärkung"
Swedish.

The same scoring is reused on the OWL path for a different job. RadLex's German
label is frequently the Latin anatomical name — RID10012's ``rdfs:label@de`` is
"Aponeurosis palatina" while its ``RID:Synonym@de`` is "Gaumenaponeurose". Latin
is correct German anatomical nomenclature and is not what a dictated report says,
so the most German-looking of the tagged German strings becomes the preferred
term and the rest become aliases.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medmt_eval.glossary import Glossary, GlossaryEntry  # noqa: E402

GERMAN_COLUMN = "http://www.radlex.org/RID/Preferred_name_German"

_UMLAUT = re.compile(r"[äöüÄÖÜß]")
# Function words that essentially never appear in an English or Latin term.
_GERMAN_WORDS = re.compile(
    r"(?<!\w)(der|die|das|des|dem|den|und|mit|ohne|zum|zur|von|vom|beim|im|am"
    r"|eines|einer|einem|nach|über|unter|zwischen|innerhalb)(?!\w)",
    re.IGNORECASE,
)
# Derivational endings that are German-specific in this vocabulary.
_GERMAN_SUFFIX = re.compile(
    r"(?<!\w)\w+(ung|ungen|heit|keit|schaft|chen|lein|erung|zeichen|knochen"
    r"|drüse|höhle|gefäß|muskel|nerv|band|haut)(?!\w)",
    re.IGNORECASE,
)
# Latin declension endings; two or more words carrying them means Latin, not
# German, and Latin anatomy is the bulk of what else lives in this field.
_LATIN_WORD = re.compile(
    r"(?<!\w)\w+(us|um|is|ae|ii|orum|arum|ibus|atis|oideus|oidei|alis|ales)(?!\w)"
)
# A capitalised word that is not the first token: German capitalises every noun.
_INNER_CAPITAL = re.compile(r"(?<=\s)[A-ZÄÖÜ][a-zäöüß]{2,}")


def german_score(text: str) -> int:
    """Positive means German. Rules, in rough order of how much they are trusted."""
    if not text or len(text) < 3:
        return -10
    score = 0
    if _UMLAUT.search(text):
        score += 3
    if _GERMAN_WORDS.search(text):
        score += 3
    if _GERMAN_SUFFIX.search(text):
        score += 2
    score += min(2, len(_INNER_CAPITAL.findall(text)))
    latin_words = len(_LATIN_WORD.findall(text))
    if latin_words >= 2:
        score -= 4
    elif latin_words == 1 and score < 3:
        score -= 2
    return score


def is_german(text: str, threshold: int = 3) -> bool:
    return german_score(text) >= threshold


# ── OWL ────────────────────────────────────────────────────────────────────
_OWL_NS = {
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
}
_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
_ABOUT = f"{{{_OWL_NS['rdf']}}}about"


def _tagged(element, tag_suffix: str, lang: str) -> list[str]:
    """Direct children whose tag ends with `tag_suffix` and carry `xml:lang`."""
    out: list[str] = []
    for child in element:
        if child.get(_LANG) != lang:
            continue
        if child.tag.split("}")[-1] != tag_suffix:
            continue
        text = (child.text or "").strip()
        if text:
            out.append(text)
    return out


def parse_owl(path: Path, keep_obsolete: bool = False) -> list[GlossaryEntry]:
    """Read EN/DE pairs straight from the language tags.

    RadLex serialises the same concept more than once: an ``owl:Class`` block
    carries most labels, and a later ``rdf:Description`` block carries more —
    together 47,414 German tags, of which only 21,810 are in the class section.
    Parsing classes alone silently loses more than half the translation, so
    every element bearing an ``rdf:about`` under ``/RID/`` is read and merged on
    the RID.
    """
    import xml.etree.ElementTree as ET

    text = path.read_text(encoding="utf-8", errors="replace")
    if "</rdf:RDF>" not in text[-4096:]:
        # A file still being downloaded, or a bad transfer, cuts off mid-element
        # and ElementTree refuses all of it. Recover what is complete rather than
        # discard a 65 MB ontology, but say so — the tail is silently missing.
        end = text.rfind("</owl:Class>")
        if end == -1:
            raise SystemExit(f"{path} is incomplete and holds no complete class.")
        text = text[: end + len("</owl:Class>")] + "\n</rdf:RDF>\n"
        print(
            f"WARNING: {path} has no closing </rdf:RDF> — it is truncated or "
            f"still downloading. Parsing the first {end / len(text):.0%} only; "
            f"re-run once the download completes.",
            file=sys.stderr,
        )
    root = ET.fromstring(text)

    english: dict[str, list[str]] = {}
    german: dict[str, list[str]] = {}
    english_alias: dict[str, list[str]] = {}
    obsolete: set[str] = set()

    for element in root.iter():
        about = element.get(_ABOUT)
        if not about or "/RID/" not in about:
            continue
        rid = about.rsplit("/", 1)[-1]
        if not rid.startswith("RID"):
            continue
        for child in element:
            tag = child.tag.split("}")[-1]
            value = (child.text or "").strip()
            if tag == "deprecated" and value.lower() in ("true", "1"):
                obsolete.add(rid)
                continue
            if not value:
                continue
            lang = child.get(_LANG)
            if tag == "label" and lang == "en":
                english.setdefault(rid, []).append(value)
            elif tag == "label" and lang == "de":
                german.setdefault(rid, []).append(value)
            elif tag == "Synonym" and lang == "de":
                german.setdefault(rid, []).append(value)
            elif tag == "Synonym" and lang == "en":
                english_alias.setdefault(rid, []).append(value)

    entries: list[GlossaryEntry] = []
    for rid, de_terms in german.items():
        if rid not in english:
            continue
        if not keep_obsolete and rid in obsolete:
            continue
        de_terms = list(dict.fromkeys(de_terms))
        en_terms = list(dict.fromkeys(english[rid]))
        # RadLex's German label is often the Latin anatomical name; a report uses
        # the vernacular. Prefer whichever tagged German string scores most
        # German and keep the rest — Latin included — as aliases.
        preferred_de = max(de_terms, key=german_score)
        entries.append(GlossaryEntry(
            concept_id=rid,
            source="radlex",
            en=en_terms[0],
            de=preferred_de,
            aliases_en=tuple(dict.fromkeys(en_terms[1:] + english_alias.get(rid, []))),
            aliases_de=tuple(g for g in de_terms if g != preferred_de),
            tree=("A00.radlex",),
        ))
    return entries


def _open(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path,
                        help="RadLex.owl (preferred) or RADLEX.csv[.gz] (fallback)")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("data/term_banks/radlex_de_en.csv"))
    parser.add_argument("--threshold", type=int, default=3,
                        help="German score at or above which a synonym is accepted")
    parser.add_argument("--keep-obsolete", action="store_true")
    args = parser.parse_args()

    if str(args.source).lower().endswith(".owl"):
        entries = parse_owl(args.source, keep_obsolete=args.keep_obsolete)
        glossary = Glossary(entries)
        glossary.to_csv(args.output)
        with_alias = sum(1 for e in glossary if e.aliases_de)
        print(f"parsed {args.source} (explicit xml:lang tags — nothing inferred)")
        print(f"\nwrote {args.output}")
        print(f"  {len(glossary)} EN-DE pairs")
        print(f"  {with_alias} with additional German surface forms")
        return 0

    with _open(args.source) as handle:
        rows = list(csv.DictReader(handle))
    print(f"read {len(rows)} RadLex concepts from CSV (fallback path)")

    entries: list[GlossaryEntry] = []
    from_column = 0
    for row in rows:
        if not args.keep_obsolete and str(row.get("Obsolete", "")).strip().lower() in ("true", "1"):
            continue
        english = (row.get("Preferred Label") or "").strip()
        if not english:
            continue
        rid = (row.get("Class ID") or "").rsplit("/", 1)[-1]

        german = (row.get(GERMAN_COLUMN) or "").strip()
        if german:
            from_column += 1
        german_aliases: list[str] = []
        english_aliases: list[str] = []
        for synonym in (row.get("Synonyms") or "").split("|"):
            synonym = synonym.strip()
            if not synonym or synonym.lower() == english.lower():
                continue
            (german_aliases if is_german(synonym, args.threshold) else english_aliases).append(synonym)

        if not german and german_aliases:
            # Longest German synonym as the preferred term: the short ones are
            # usually abbreviations, and short forms are exactly what caused the
            # alias collisions documented in thesis/08-terminology.md.
            german = max(german_aliases, key=len)
            german_aliases = [g for g in german_aliases if g != german]
        if not german:
            continue

        entries.append(GlossaryEntry(
            concept_id=rid, source="radlex", en=english, de=german,
            aliases_en=tuple(dict.fromkeys(english_aliases)),
            aliases_de=tuple(dict.fromkeys(german_aliases)),
            # RadLex is entirely radiology, so the MeSH-branch filter that a
            # general source needs is unnecessary; tag every concept as in-domain
            # so in_branches("ACE") keeps it when banks are merged.
            tree=("A00.radlex",),
        ))

    glossary = Glossary(entries)
    glossary.to_csv(args.output)
    print(f"\nwrote {args.output}")
    print(f"  {len(glossary)} EN-DE pairs")
    print(f"  from the German column: {from_column}")
    print(f"  recovered from untagged synonyms: {len(glossary) - from_column}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
