"""A provenance-tracked bilingual glossary that can be shown to a translator.

CSV schema (``concept_id,source,en,de,aliases_en,aliases_de``; ``tr`` optional):

    concept_id   stable id in the originating terminology (RID…, MeSH D…, Q…)
    source       which terminology it came from — see PROVENANCE below
    en, de, tr   the preferred term per language
    aliases_*    ``|``-separated synonyms; these carry the colloquial variants
                 that formal ontologies omit

PROVENANCE IS NOT DECORATION. Some terminologies are themselves
machine-translated — German MeSH is a DeepL first pass with curation on top.
Injecting an MT-derived term and then measuring whether the model reproduced it
measures agreement with an MT system, not correctness. ``Glossary.exclude_mt``
drops those entries, and ``mt_derived_sources`` names them.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from medmt_eval.schema import normalise_language
from medmt_eval.taxonomy.clinical import term_surface_pattern

#: Terminologies whose target-language side was produced (wholly or partly) by
#: machine translation. Kept explicit so the choice is auditable.
MT_DERIVED_SOURCES = frozenset({"mesh-zbmed"})

_SUPPORTED_LANGUAGES = ("en", "de", "tr")

#: Aliases shorter than this are discarded — see GlossaryEntry.surface_forms for
#: why two-letter aliases are actively harmful on clinical German.
MIN_ALIAS_LENGTH = 4

#: Preferred labels below this length are dropped too. A two-character term
#: matched case-insensitively hits function words and units regardless of how
#: authoritative its source is ("Es" is the MeSH preferred label for the
#: psychoanalytic *id*).
MIN_TERM_LENGTH = 3

#: Surface forms that are correct for their concept but mean something else in a
#: radiology report. These are false friends, not errors in the source
#: terminology, so they are suppressed here rather than fixed upstream.
#:
#:   Darstellung   alias of "chemical synthesis"; in a report it means depiction
#:   Technik/Tech  alias of "technology"; in a report it means imaging technique
#:   Höhe          MeSH label for "altitude"; in a report it means level
DEFAULT_STOPLIST = frozenset({
    "darstellung", "technik", "tech", "höhe", "hoehe", "synthese",
})

#: Closed-class function words. RadLex ships these as first-class concepts —
#: RID28454 is literally ``kein -> none`` and RID28475 is ``nicht -> no`` — and a
#: glossary entry for them is worse than useless. No model needs to be told what
#: "mit" means, and "kein = none" is an actively wrong instruction: "Kein
#: Erguss" is "No effusion", never "None effusion". Negation is one of the three
#: critical error classes this project measures, so injecting a bad negation
#: gloss could manufacture exactly the failure under study.
FUNCTION_WORD_STOPLIST = frozenset({
    # German
    "kein", "keine", "keiner", "keinen", "keinem", "nicht", "nichts",
    "und", "oder", "aber", "mit", "ohne", "nach", "vor", "bei", "von", "vom",
    "für", "im", "am", "zum", "zur", "der", "die", "das", "des", "dem", "den",
    "ein", "eine", "einer", "eines", "einem", "einen", "als", "wie", "auch",
    # English
    "no", "not", "none", "and", "or", "but", "with", "without", "after",
    "before", "the", "a", "an", "of", "in", "at", "for", "as", "also",
})

#: Word tokeniser for the selection index; keeps German umlauts and digits.
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class GlossaryEntry:
    concept_id: str
    source: str
    en: str = ""
    de: str = ""
    tr: str = ""
    aliases_en: tuple[str, ...] = field(default_factory=tuple)
    aliases_de: tuple[str, ...] = field(default_factory=tuple)
    #: MeSH tree codes, ``|``-separated (e.g. "A03.556|C06.405"). The leading
    #: letter is the branch and is what makes domain filtering possible.
    tree: tuple[str, ...] = field(default_factory=tuple)

    @property
    def branches(self) -> frozenset[str]:
        return frozenset(code[0].upper() for code in self.tree if code)

    def term(self, lang: str) -> str:
        return getattr(self, normalise_language(lang), "")

    def aliases(self, lang: str) -> tuple[str, ...]:
        return getattr(self, f"aliases_{normalise_language(lang)}", ())

    def surface_forms(
        self,
        lang: str,
        *,
        min_alias_length: int = MIN_ALIAS_LENGTH,
        stoplist: frozenset[str] = DEFAULT_STOPLIST,
        drop_function_words: bool = True,
    ) -> tuple[str, ...]:
        """Preferred term plus usable aliases.

        Short aliases are dropped, and the threshold is not cosmetic. Wikidata
        stores chemical symbols and initialisms as aliases, and matched
        case-insensitively against German clinical text they are catastrophic:

            Cm (curium)      matches "cm"  — the centimetre unit, everywhere
            CT (circuit training) matches "CT" — the modality, everywhere
            In (indium)      matches "in"
            Es (einsteinium) matches "es"
            Am (americium)   matches "am"

        On PARROT that produced 125 spurious "Indium" hits and made "CT" resolve
        to circuit training. A 4-character floor removes every one of these while
        keeping real synonyms ("Kreistraining", "Milchglastrübung").
        """
        preferred = self.term(lang)
        forms = [preferred] if preferred and len(preferred) >= MIN_TERM_LENGTH else []
        forms += [a for a in self.aliases(lang) if a and len(a) >= min_alias_length]
        banned = stoplist | (FUNCTION_WORD_STOPLIST if drop_function_words else frozenset())
        forms = [f for f in forms if f.lower() not in banned]
        return tuple(dict.fromkeys(forms))  # de-duplicate, keep order

    def covers(self, src_lang: str, tgt_lang: str) -> bool:
        return bool(self.term(src_lang)) and bool(self.term(tgt_lang))


def _split_aliases(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split("|") if part.strip())


class Glossary:
    """An ordered, de-duplicated collection of :class:`GlossaryEntry`."""

    def __init__(self, entries: Iterable[GlossaryEntry] = ()) -> None:
        self.entries = tuple(entries)
        # Built lazily per language, see _index().
        self._index_cache: dict[str, dict[str, list[tuple[GlossaryEntry, str]]]] = {}
        self._pattern_cache: dict[str, re.Pattern[str]] = {}

    def _pattern(self, form: str) -> re.Pattern[str]:
        cached = self._pattern_cache.get(form)
        if cached is None:
            cached = re.compile(term_surface_pattern(form), re.IGNORECASE)
            self._pattern_cache[form] = cached
        return cached

    def _index(self, lang: str) -> dict[str, list[tuple[GlossaryEntry, str]]]:
        """First-word -> candidate (entry, surface form), for one language.

        Scanning all 23k entries with a regex against every document is O(n·m)
        and took minutes on a 120-report sample. Every pattern this module
        builds begins with the term's first word matched literally (only the
        *final* word may take an inflectional suffix), so that word must occur
        verbatim in the text. Indexing on it turns the scan into a lookup over
        the words actually present, and the regex then runs on a handful of
        candidates rather than the whole glossary.
        """
        lang = normalise_language(lang)
        cached = self._index_cache.get(lang)
        if cached is not None:
            return cached
        index: dict[str, list[tuple[GlossaryEntry, str]]] = {}
        for entry in self.entries:
            for form in entry.surface_forms(lang):
                words = form.split()
                if not words:
                    continue
                index.setdefault(words[0].lower(), []).append((entry, form))
        self._index_cache[lang] = index
        return index

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    # ── io ────────────────────────────────────────────────────────────────
    @classmethod
    def from_csv(cls, path: str | Path) -> "Glossary":
        with Path(path).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "concept_id" not in reader.fieldnames:
                raise ValueError(f"Glossary {path} needs at least a concept_id column.")
            return cls(
                GlossaryEntry(
                    concept_id=str(row["concept_id"]).strip(),
                    source=str(row.get("source", "") or "unknown").strip(),
                    en=str(row.get("en", "") or "").strip(),
                    de=str(row.get("de", "") or "").strip(),
                    tr=str(row.get("tr", "") or "").strip(),
                    aliases_en=_split_aliases(row.get("aliases_en")),
                    aliases_de=_split_aliases(row.get("aliases_de")),
                    tree=_split_aliases(row.get("tree")),
                )
                for row in reader
                if row.get("concept_id")
            )

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["concept_id", "source", "en", "de", "tr", "aliases_en", "aliases_de", "tree"]
            )
            for e in self.entries:
                writer.writerow([
                    e.concept_id, e.source, e.en, e.de, e.tr,
                    "|".join(e.aliases_en), "|".join(e.aliases_de), "|".join(e.tree),
                ])

    # ── filtering ─────────────────────────────────────────────────────────
    def exclude_mt(self, extra: Iterable[str] = ()) -> "Glossary":
        """Drop entries whose target side came out of a machine translator."""
        banned = MT_DERIVED_SOURCES | {s.strip().lower() for s in extra}
        return Glossary(e for e in self.entries if e.source.lower() not in banned)

    def in_branches(self, branches: Iterable[str], *, keep_untreed: bool = False) -> "Glossary":
        """Keep only concepts in the given MeSH tree branches.

        A general encyclopedic source linked to MeSH carries the whole tree,
        including Z (Geographicals) and L (Information Science). Those are the
        source of the worst polysemy on clinical German: "Technik" resolving to
        L-branch *technology*, "Darstellung" to a D-branch *chemical synthesis*.
        For radiology the useful branches are A (Anatomy), C (Diseases) and
        E (Diagnostic and Therapeutic Techniques and Equipment).

        Entries with no tree code are dropped unless ``keep_untreed``, because an
        unclassifiable concept cannot be shown to be in-domain.
        """
        wanted = frozenset(b.strip().upper() for b in branches if b.strip())
        return Glossary(
            e for e in self.entries
            if (e.branches & wanted) or (keep_untreed and not e.tree)
        )

    def covering(self, src_lang: str, tgt_lang: str) -> "Glossary":
        return Glossary(e for e in self.entries if e.covers(src_lang, tgt_lang))

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.entries:
            counts[e.source] = counts.get(e.source, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    # ── the anti-circularity split ────────────────────────────────────────
    def split(self, holdout: float = 0.5, seed: int = 13) -> tuple["Glossary", "Glossary"]:
        """Partition into (inject, evaluate) halves by concept.

        Showing a model a term and then scoring whether it used that term is not
        an experiment — the terminology detector would improve by construction.
        Splitting by concept means the injected half and the scored half never
        overlap, so a gain on the scored half is generalisation rather than
        copying.
        """
        if not 0.0 < holdout < 1.0:
            raise ValueError("holdout must be strictly between 0 and 1.")
        ordered = sorted(self.entries, key=lambda e: e.concept_id)
        rng = random.Random(seed)
        rng.shuffle(ordered)
        cut = int(len(ordered) * (1.0 - holdout))
        return Glossary(ordered[:cut]), Glossary(ordered[cut:])

    # ── selection ─────────────────────────────────────────────────────────
    def select(
        self,
        source_text: str,
        src_lang: str,
        tgt_lang: str,
        *,
        limit: int = 40,
    ) -> list[GlossaryEntry]:
        """Entries whose source-language surface form occurs in ``source_text``.

        Matching reuses the evaluator's inflection-tolerant pattern so that
        "Lymphknotens" finds the entry "Lymphknoten". Longer terms are preferred
        when the list has to be truncated: a multi-word term is more specific and
        more likely to be the one a model gets wrong.
        """
        src, tgt = normalise_language(src_lang), normalise_language(tgt_lang)
        if src == tgt:
            return []
        index = self._index(src)
        # A single-word term carries the inflectional suffix itself
        # ("Pleuraerguss" -> "Pleuraergusses"), so the indexed key does not
        # appear verbatim in the text. _TERM_SUFFIX is at most two characters,
        # so trying the word and its two shorter prefixes restores every match
        # the linear scan used to find. Multi-word terms are unaffected: only
        # their final word may inflect, and they are indexed on their first.
        keys: set[str] = set()
        for raw in _WORD.findall(source_text):
            word = raw.lower()
            keys.add(word)
            keys.add(word[:-1])
            keys.add(word[:-2])
        keys.discard("")
        candidates: dict[str, tuple[int, GlossaryEntry]] = {}
        for word in keys:
            for entry, form in index.get(word, ()):
                if not entry.covers(src, tgt):
                    continue
                # Longer surface forms win: a multi-word term is more specific
                # and more likely to be the one a model gets wrong.
                previous = candidates.get(entry.concept_id)
                if previous is not None and previous[0] >= len(form):
                    continue
                if self._pattern(form).search(source_text):
                    candidates[entry.concept_id] = (len(form), entry)
        ranked = sorted(candidates.values(), key=lambda pair: -pair[0])
        return [entry for _, entry in ranked[:limit]]

    def distractors(
        self,
        source_text: str,
        src_lang: str,
        tgt_lang: str,
        *,
        count: int,
        seed: int = 13,
    ) -> list[GlossaryEntry]:
        """``count`` entries that do NOT occur in ``source_text`` — the control arm.

        A glossary block makes the prompt longer, more structured and more
        domain-flavoured. Any of those could move the score on their own. The
        control supplies an equally long block of irrelevant terms, so a gain
        over the control is attributable to the terms being *right* rather than
        to the block being present.
        """
        src, tgt = normalise_language(src_lang), normalise_language(tgt_lang)
        relevant = {e.concept_id for e in self.select(source_text, src, tgt, limit=10**6)}
        pool = [
            e for e in self.entries
            if e.covers(src, tgt) and e.concept_id not in relevant
        ]
        rng = random.Random(f"{seed}:{source_text[:64]}")
        rng.shuffle(pool)
        return pool[:count]

    # ── rendering ─────────────────────────────────────────────────────────
    def render(
        self,
        entries: Sequence[GlossaryEntry],
        src_lang: str,
        tgt_lang: str,
        *,
        header: str | None = None,
    ) -> str:
        """Format entries as a prompt block. Empty string when there are none.

        Returning "" rather than an empty header matters: a header with no terms
        under it tells the model a glossary exists and is empty, which is a
        different instruction from not mentioning one.
        """
        src, tgt = normalise_language(src_lang), normalise_language(tgt_lang)
        lines = [
            f"- {e.term(src)} = {e.term(tgt)}"
            for e in entries
            if e.term(src) and e.term(tgt)
            # Guard the target side too: an entry can be selected on a legitimate
            # source term and still gloss it to a function word.
            and e.term(tgt).lower() not in FUNCTION_WORD_STOPLIST
        ]
        if not lines:
            return ""
        if header is None:
            header = (
                "Approved terminology for this text. Where the source uses the "
                "term on the left, the translation must use the term on the right."
            )
        return header + "\n" + "\n".join(lines)


def load_glossary(path: str | Path | None, *, exclude_mt: bool = True) -> Glossary:
    """Load a glossary, optionally dropping machine-translated terminologies."""
    if not path:
        return Glossary()
    glossary = Glossary.from_csv(path)
    return glossary.exclude_mt() if exclude_mt else glossary
