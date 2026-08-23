"""Tests for the injectable glossary and the dictionary-augmented experiment."""

from __future__ import annotations

import pytest

from medmt_eval.glossary import Glossary, GlossaryEntry
from medmt_eval.inference.glossary_mt import (
    ARMS, build_prompt, entries_for_arm, run_glossary_experiment, summarise_by_arm,
)
from medmt_eval.models.chat import ChatBackend
from medmt_eval.schema import Segment

DE = "Kein Pleuraerguss. Ein 5 mm Lymphknoten links."
EN = "No pleural effusion. A 5 mm lymph node on the left."


def _glossary() -> Glossary:
    return Glossary([
        GlossaryEntry("RID1", "radlex", en="pleural effusion", de="Pleuraerguss",
                      tree=("C08.528",)),
        GlossaryEntry("RID2", "radlex", en="lymph node", de="Lymphknoten",
                      tree=("A15.382",)),
        GlossaryEntry("RID3", "radlex", en="pneumothorax", de="Pneumothorax",
                      tree=("C08.528",)),
        GlossaryEntry("RID4", "radlex", en="aorta", de="Aorta", tree=("A07.231",)),
    ])


class _Echo(ChatBackend):
    """Records prompts and returns a fixed translation."""

    name = "echo"

    def __init__(self, reply: str = EN) -> None:
        self.prompts: list[str] = []
        self._reply = reply

    def complete(self, user: str, system: str | None = None) -> str:
        self.prompts.append(user)
        return self._reply

    @property
    def config(self):
        return {"backend": "echo"}


def _segments(n: int = 2):
    return [
        Segment(id=f"s{i}", domain="t", src_lang="de", tgt_lang="en",
                src_text=DE, ref_text=EN)
        for i in range(n)
    ]


# ── selection ──────────────────────────────────────────────────────────────
def test_select_finds_inflected_terms() -> None:
    got = _glossary().select("Kein Pleuraergusses nachweisbar.", "de", "en")
    assert [e.concept_id for e in got] == ["RID1"]


def test_select_ignores_terms_absent_from_the_text() -> None:
    ids = {e.concept_id for e in _glossary().select(DE, "de", "en")}
    assert ids == {"RID1", "RID2"}


def test_short_aliases_are_dropped() -> None:
    """A two-letter alias matched case-insensitively hits 'cm' and 'in'.

    On PARROT this made curium match the centimetre unit and CT match circuit
    training — 125 spurious hits for indium alone.
    """
    entry = GlossaryEntry("Q1", "wikidata", en="curium", de="Curium",
                          aliases_de=("Cm", "Kurium"))
    assert entry.surface_forms("de") == ("Curium", "Kurium")


def test_two_character_preferred_labels_are_dropped() -> None:
    entry = GlossaryEntry("Q2", "wikidata", en="id", de="Es")
    assert entry.surface_forms("de") == ()


def test_stoplisted_false_friends_are_dropped() -> None:
    """'Darstellung' is a valid alias of chemical synthesis and means depiction
    in a radiology report."""
    entry = GlossaryEntry("Q3", "wikidata", en="chemical synthesis", de="Synthese",
                          aliases_de=("Darstellung", "Synthesechemie"))
    assert entry.surface_forms("de") == ("Synthesechemie",)


def test_branch_filter_keeps_only_requested_mesh_branches() -> None:
    g = Glossary([
        GlossaryEntry("a", "wikidata", en="aorta", de="Aorta", tree=("A07.231",)),
        GlossaryEntry("z", "wikidata", en="Germany", de="Deutschland", tree=("Z01.542",)),
    ])
    assert [e.concept_id for e in g.in_branches("ACE")] == ["a"]


def test_branch_filter_drops_untreed_entries_by_default() -> None:
    g = Glossary([GlossaryEntry("x", "wikidata", en="thing", de="Ding")])
    assert len(g.in_branches("ACE")) == 0
    assert len(g.in_branches("ACE", keep_untreed=True)) == 1


def test_mt_derived_sources_are_excluded() -> None:
    """German MeSH is a DeepL first pass; scoring against it measures agreement
    with an MT system, not correctness."""
    g = Glossary([
        GlossaryEntry("m", "mesh-zbmed", en="fever", de="Fieber"),
        GlossaryEntry("r", "radlex", en="aorta", de="Aorta"),
    ])
    assert [e.concept_id for e in g.exclude_mt()] == ["r"]


def test_split_is_disjoint_and_deterministic() -> None:
    inject, evaluate = _glossary().split(holdout=0.5, seed=7)
    assert not ({e.concept_id for e in inject} & {e.concept_id for e in evaluate})
    assert len(inject) + len(evaluate) == 4
    again, _ = _glossary().split(holdout=0.5, seed=7)
    assert [e.concept_id for e in again] == [e.concept_id for e in inject]


def test_render_returns_empty_string_when_no_terms() -> None:
    """An empty header would tell the model a glossary exists and is empty."""
    assert _glossary().render([], "de", "en") == ""


# ── arms ───────────────────────────────────────────────────────────────────
def test_none_arm_prompt_never_mentions_a_glossary() -> None:
    prompt = build_prompt(DE, "de", "en", "")
    assert "=" not in prompt.split(DE)[0]
    assert "TERM" not in prompt.upper()


def test_distractor_arm_matches_the_glossary_arm_in_size() -> None:
    g = _glossary()
    relevant = entries_for_arm("glossary", g, DE, "de", "en")
    distractors = entries_for_arm("distractor", g, DE, "de", "en")
    assert len(distractors) == len(relevant) > 0


def test_distractor_terms_do_not_occur_in_the_source() -> None:
    g = _glossary()
    for entry in entries_for_arm("distractor", g, DE, "de", "en"):
        assert entry.de not in DE


def test_unknown_arm_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown arm"):
        entries_for_arm("nope", _glossary(), DE, "de", "en")


# ── experiment ─────────────────────────────────────────────────────────────
def test_experiment_produces_one_row_per_segment_and_arm() -> None:
    backend = _Echo()
    rows = run_glossary_experiment(backend, _segments(3), _glossary())
    assert len(rows) == 3 * len(ARMS)
    assert {r["arm"] for r in rows} == set(ARMS)


def test_glossary_arm_actually_puts_terms_in_the_prompt() -> None:
    backend = _Echo()
    run_glossary_experiment(backend, _segments(1), _glossary(), arms=["glossary"])
    assert "Pleuraerguss = pleural effusion" in backend.prompts[0]


def test_none_arm_sends_no_terms() -> None:
    backend = _Echo()
    run_glossary_experiment(backend, _segments(1), _glossary(), arms=["none"])
    assert "pleural effusion" not in backend.prompts[0]


def test_summary_has_one_row_per_arm_in_fixed_order() -> None:
    rows = run_glossary_experiment(_Echo(), _segments(2), _glossary())
    summary = summarise_by_arm(rows)
    assert [s["arm"] for s in summary] == list(ARMS)
    assert all(0.0 <= s["critical_error_rate"] <= 1.0 for s in summary)


def test_empty_input_returns_nothing() -> None:
    assert run_glossary_experiment(_Echo(), [], _glossary()) == []


def test_function_words_are_never_glossary_entries() -> None:
    """RadLex ships RID28454 as `kein -> none`. Injecting that is worse than
    useless: "Kein Erguss" is "No effusion", never "None effusion", and negation
    is one of the three critical error classes."""
    entry = GlossaryEntry("RID28454", "radlex", en="none", de="kein")
    assert entry.surface_forms("de") == ()
    assert entry.surface_forms("en") == ()


def test_function_word_targets_are_not_rendered() -> None:
    """An entry can be selected on a legitimate source term and still gloss to a
    function word; the rendered block must not carry it."""
    g = Glossary([
        GlossaryEntry("RID28475", "radlex", en="no", de="nicht"),
        GlossaryEntry("RID1", "radlex", en="pleural effusion", de="Pleuraerguss"),
    ])
    block = g.render(list(g), "de", "en")
    assert "Pleuraerguss = pleural effusion" in block
    assert "= no" not in block


def test_content_words_survive_the_function_word_filter() -> None:
    entry = GlossaryEntry("RID1", "radlex", en="lesion", de="Läsion")
    assert entry.surface_forms("de") == ("Läsion",)
