"""Unit tests for sentence-boundary chunking and reassembly."""

from __future__ import annotations

import pytest

from medmt_eval.data.chunking import (
    chunk_documents,
    chunk_text,
    reassemble,
    split_sentences,
)


def test_splits_on_sentence_boundaries() -> None:
    text = "Kein Pleuraerguss. Kein Pneumothorax. Herz normal gross."
    assert split_sentences(text) == [
        "Kein Pleuraerguss.",
        "Kein Pneumothorax.",
        "Herz normal gross.",
    ]


def test_does_not_split_decimals() -> None:
    """Critical: numeric fidelity is the only detector active for uncovered
    languages, so splitting "1.5 cm" into "1." + "5 cm" would manufacture
    exactly the errors being measured. PARROT Turkish has 83 decimal points."""
    text = "Ein Knoten von 1.5 cm. Kein Erguss."
    parts = split_sentences(text)
    assert parts == ["Ein Knoten von 1.5 cm.", "Kein Erguss."]
    assert any("1.5" in p for p in parts)


def test_does_not_split_on_latin_anatomical_abbreviations() -> None:
    """All 11 lone-capital periods in the PARROT Turkish corpus are "A." for
    Latin *arteria* ("A. iliaca interna"), not personal initials. Splitting
    there would cut a vessel name in half mid-phrase."""
    text = "5F-RIM kateter ile sol A. iliaca interna sondalandı. Bulgu yok."
    assert split_sentences(text) == [
        "5F-RIM kateter ile sol A. iliaca interna sondalandı.",
        "Bulgu yok.",
    ]


def test_does_not_split_on_personal_initials() -> None:
    text = "Befund von M. Meddeb. Kein Erguss."
    assert split_sentences(text) == ["Befund von M. Meddeb.", "Kein Erguss."]


def test_handles_question_and_exclamation() -> None:
    assert split_sentences("Koiling mumkun mu? Evet!") == ["Koiling mumkun mu?", "Evet!"]


def test_chunk_packs_sentences_up_to_budget() -> None:
    # ~3 chars per token, so 10 tokens ≈ 30 chars.
    text = "Aaa bbb ccc. Ddd eee fff. Ggg hhh iii."
    chunks = chunk_text(text, max_tokens=10)
    assert len(chunks) > 1
    # No chunk may exceed the budget unless it is a single long sentence.
    for chunk in chunks:
        assert len(chunk) <= 30 or " " not in chunk.rstrip(".")


def test_short_document_is_one_chunk() -> None:
    assert chunk_text("Kein Pleuraerguss.", max_tokens=400) == ["Kein Pleuraerguss."]


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_text("", max_tokens=400) == []
    assert chunk_text("   ", max_tokens=400) == []


def test_oversized_sentence_is_emitted_whole() -> None:
    """Splitting mid-sentence is worse than an over-long chunk: the fragment
    stops being a coherent unit of meaning for the translator."""
    long_sentence = "Wort " * 100 + "ende."
    chunks = chunk_text(long_sentence, max_tokens=10)
    assert len(chunks) == 1


def test_chunk_documents_tracks_counts() -> None:
    docs = ["A. B. C.", "D.", ""]
    chunks, counts = chunk_documents(docs, max_tokens=2)  # ~6 chars
    assert sum(counts) == len(chunks)
    assert counts[2] == 0  # empty document contributes nothing


def test_reassemble_restores_one_hypothesis_per_document() -> None:
    chunks = ["No effusion.", "No pneumothorax.", "Heart normal."]
    assert reassemble(chunks, [2, 1]) == [
        "No effusion. No pneumothorax.",
        "Heart normal.",
    ]


def test_reassemble_handles_empty_document() -> None:
    assert reassemble(["Only one."], [0, 1]) == ["", "Only one."]


def test_reassemble_rejects_wrong_chunk_count() -> None:
    """A mismatch means the translator dropped or duplicated a chunk; failing
    loudly is far better than silently misaligning every later document."""
    with pytest.raises(ValueError, match="Expected 3 translated chunks"):
        reassemble(["a", "b"], [2, 1])


def test_round_trip_preserves_document_count() -> None:
    docs = ["Aaa. Bbb. Ccc.", "Ddd eee.", "Fff. Ggg."]
    chunks, counts = chunk_documents(docs, max_tokens=3)
    # Identity "translation" of each chunk.
    assert len(reassemble(list(chunks), counts)) == len(docs)


def test_numbers_survive_a_chunk_round_trip() -> None:
    """End-to-end guard on the property that actually matters here."""
    doc = "Ein Knoten von 1.5 cm im linken Oberlappen. Ein zweiter von 8 mm. Kein Erguss."
    chunks, counts = chunk_documents([doc], max_tokens=8)
    rebuilt = reassemble(chunks, counts)[0]
    for number in ("1.5", "8"):
        assert number in rebuilt
