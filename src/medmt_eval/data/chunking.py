"""Split long documents for translation, then reassemble the pieces.

Some models cannot process a whole radiology report. The limit is on the
*encoder*, so it truncates the input no matter how large ``--max-new-tokens``
is. Verified from the model configs:

    Helsinki-NLP/opus-mt-*   max_position_embeddings = 512
    facebook/nllb-200-*      max_position_embeddings = 1024

The longest German PARROT report is 4,029 characters (~1,343 tokens) and the
median Turkish one is ~2,080 (~693 tokens), so both models overflow. NLLB's
output stayed at 26 % of source length on Turkish even after the output ceiling
was raised, which is what first exposed this.

The approach here is translate-chunked / score-whole: a document is split on
sentence boundaries, each chunk is translated independently, and the results are
joined back into one hypothesis. Evaluation then runs on the full report, so the
critical-error rate stays comparable to unchunked runs and to the German
results. Scoring per chunk would inflate apparent quality — each chunk holds
fewer numbers, so there are fewer chances to mismatch.

The cost is context: a model translating one sentence cannot see the rest of the
report, so an unresolved pronoun or a back-reference to an earlier measurement
may be rendered differently than it would be in full context. A chunked run is
therefore not strictly comparable to an unchunked run of the same model, and
should be labelled as chunked wherever it is reported.
"""

from __future__ import annotations

import re

# A sentence ends at . ! ? followed by whitespace — but several common patterns
# in medical reports contain a period that does NOT end a sentence. Getting this
# wrong matters more than usual here: numeric fidelity is the only detector
# active for uncovered languages, so splitting "1.5 cm" into "1." + "5 cm" would
# manufacture exactly the errors being measured.
#
# Two hazards were found in the PARROT Turkish corpus:
#   * 83 decimal points   — "1.5 cm"
#   * 11 lone capitals    — all of them "A." for Latin *arteria*
#                           ("A. iliaca interna", "A. hepatica dextra"),
#                           none a personal initial.
#
# The guard matches the boundary itself rather than using a lookbehind: with
# re.split the lookbehind is evaluated at the whitespace, so it inspects the
# period instead of the character before it, and silently never fires.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[^\sA-ZÇĞİÖŞÜ0-9])([.!?])\s+")

# Rough characters-per-token. Deliberately conservative: over-estimating tokens
# yields smaller chunks, which is the safe direction when the alternative is
# silent truncation.
_CHARS_PER_TOKEN = 3.0

CHUNK_JOINER = " "


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, protecting decimals and abbreviations.

    The terminator stays attached to the sentence it ends.
    """
    parts: list[str] = []
    last = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        # end(1) is just after the . ! ? so the terminator is kept.
        parts.append(text[last : match.end(1)].strip())
        last = match.end()
    tail = text[last:].strip()
    if tail:
        parts.append(tail)
    return [part for part in parts if part]


def chunk_text(text: str, max_tokens: int = 400) -> list[str]:
    """Greedily pack sentences into chunks that fit ``max_tokens``.

    A single sentence longer than the budget is emitted on its own rather than
    split mid-sentence: an over-long chunk that the model may truncate is less
    damaging than a fragment that is not a coherent unit of meaning, and such
    sentences are rare (the longest in PARROT Turkish is ~103 characters).
    """
    if not text.strip():
        return []
    budget = max(1, int(max_tokens * _CHARS_PER_TOKEN))
    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(text):
        if not current:
            current = sentence
        elif len(current) + len(CHUNK_JOINER) + len(sentence) <= budget:
            current = f"{current}{CHUNK_JOINER}{sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(
    texts: list[str], max_tokens: int = 400
) -> tuple[list[str], list[int]]:
    """Flatten documents into chunks, keeping the per-document chunk counts.

    Returns ``(chunks, counts)`` where ``counts[i]`` is how many chunks document
    ``i`` produced, so ``reassemble`` can put the translations back together.
    An empty document yields zero chunks and is reassembled as an empty string.
    """
    chunks: list[str] = []
    counts: list[int] = []
    for text in texts:
        pieces = chunk_text(text, max_tokens=max_tokens)
        counts.append(len(pieces))
        chunks.extend(pieces)
    return chunks, counts


def reassemble(translated_chunks: list[str], counts: list[int]) -> list[str]:
    """Join translated chunks back into one hypothesis per document."""
    expected = sum(counts)
    if len(translated_chunks) != expected:
        raise ValueError(
            f"Expected {expected} translated chunks for {len(counts)} documents, "
            f"got {len(translated_chunks)}."
        )
    documents: list[str] = []
    position = 0
    for count in counts:
        piece = translated_chunks[position : position + count]
        position += count
        documents.append(CHUNK_JOINER.join(part.strip() for part in piece if part.strip()))
    return documents
