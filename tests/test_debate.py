"""Tests for the multi-agent translation debate."""

from __future__ import annotations

import pytest

from medmt_eval.glossary import Glossary, GlossaryEntry
from medmt_eval.inference.debate import (
    PERSONAS, Agent, debate_one, parse_revision, run_debate, summarise_debate,
)
from medmt_eval.models.chat import ChatBackend
from medmt_eval.schema import Segment

DE = "Kein Pleuraerguss. Ein 5 mm Lymphknoten links."
EN_GOOD = "No pleural effusion. A 5 mm lymph node on the left."
EN_BAD = "Pleural effusion. A lymph node on the right."


class _Scripted(ChatBackend):
    """Returns queued replies in order; records every prompt it was sent."""

    def __init__(self, name: str, replies: list[str]) -> None:
        self.name = name
        self._replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, user: str, system: str | None = None) -> str:
        self.prompts.append(user)
        return self._replies.pop(0) if self._replies else EN_GOOD

    @property
    def config(self):
        return {"backend": "scripted", "model_id": self.name}


def _agents(replies: dict[str, list[str]] | None = None) -> list[Agent]:
    replies = replies or {}
    glossary = Glossary([
        GlossaryEntry("RID1", "radlex", en="pleural effusion", de="Pleuraerguss"),
        GlossaryEntry("RID2", "radlex", en="lymph node", de="Lymphknoten"),
    ])
    return [
        Agent(name, _Scripted(f"m-{name}", replies.get(name, [])), PERSONAS[name], glossary)
        for name in ("anatomist", "safety", "linguist")
    ]


def _segment() -> Segment:
    return Segment(id="s1", domain="t", src_lang="de", tgt_lang="en",
                   src_text=DE, ref_text=EN_GOOD)


# ── parsing ────────────────────────────────────────────────────────────────
def test_parse_revision_splits_critique_from_translation() -> None:
    critique, text = parse_revision(
        "CRITIQUE: linguist dropped the negation.\nTRANSLATION: No pleural effusion."
    )
    assert critique == "linguist dropped the negation."
    assert text == "No pleural effusion."


def test_parse_revision_keeps_multiline_translations_whole() -> None:
    _, text = parse_revision("CRITIQUE: fine.\nTRANSLATION: Line one.\nLine two.")
    assert text == "Line one.\nLine two."


def test_parse_revision_tolerates_an_unformatted_reply() -> None:
    """A model that ignores the format must not lose its translation."""
    critique, text = parse_revision("No pleural effusion.")
    assert critique == "" and text == "No pleural effusion."


def test_critique_prose_is_never_used_as_the_translation() -> None:
    _, text = parse_revision("CRITIQUE: bad.\nTRANSLATION: Good text.")
    assert "bad" not in text


# ── protocol ───────────────────────────────────────────────────────────────
def test_round_zero_asks_every_agent_independently() -> None:
    agents = _agents()
    result = debate_one(agents, _segment(), rounds=0)
    opening = [t for t in result["transcript"] if t["round"] == 0]
    assert {t["agent"] for t in opening} == {"anatomist", "safety", "linguist"}


def test_each_agent_sees_its_own_persona() -> None:
    agents = _agents()
    debate_one(agents, _segment(), rounds=0)
    for agent in agents:
        assert agent.persona[:40] in agent.backend.prompts[0]


def test_glossary_terms_reach_the_prompt() -> None:
    agents = _agents()
    debate_one(agents, _segment(), rounds=0)
    assert "Pleuraerguss = pleural effusion" in agents[0].backend.prompts[0]


def test_revision_round_shows_every_proposal_to_every_agent() -> None:
    agents = _agents({
        "anatomist": [EN_GOOD, "CRITIQUE: ok\nTRANSLATION: " + EN_GOOD],
        "safety": [EN_BAD, "CRITIQUE: ok\nTRANSLATION: " + EN_GOOD],
        "linguist": [EN_GOOD, "CRITIQUE: ok\nTRANSLATION: " + EN_GOOD],
    })
    debate_one(agents, _segment(), rounds=1)
    revise_prompt = agents[0].backend.prompts[1]
    assert "[anatomist]" in revise_prompt
    assert "[safety]" in revise_prompt
    assert EN_BAD in revise_prompt


def test_identical_openings_converge_at_round_zero_without_revising() -> None:
    agents = _agents()  # every scripted backend defaults to EN_GOOD
    result = debate_one(agents, _segment(), rounds=3)
    assert result["converged_at_round"] == 0
    assert result["n_rounds_run"] == 0


def test_disagreement_runs_the_requested_rounds() -> None:
    agents = _agents({
        "anatomist": [EN_GOOD] + ["CRITIQUE: a\nTRANSLATION: A"] * 3,
        "safety": [EN_BAD] + ["CRITIQUE: b\nTRANSLATION: B"] * 3,
        "linguist": ["Third variant."] + ["CRITIQUE: c\nTRANSLATION: C"] * 3,
    })
    result = debate_one(agents, _segment(), rounds=2)
    assert result["n_rounds_run"] == 2
    assert result["converged_at_round"] is None


def test_convergence_ignores_whitespace_and_case() -> None:
    agents = _agents({
        "anatomist": ["No pleural effusion."],
        "safety": ["  no   PLEURAL effusion.  "],
        "linguist": ["No pleural effusion."],
    })
    assert debate_one(agents, _segment(), rounds=2)["converged_at_round"] == 0


def test_agent_keeping_its_own_text_is_allowed() -> None:
    """An empty TRANSLATION must fall back to the agent's previous proposal, not
    replace a real translation with an empty string."""
    agents = _agents({
        "anatomist": [EN_GOOD, "CRITIQUE: no change needed.\nTRANSLATION: "],
        "safety": [EN_BAD, "CRITIQUE: x\nTRANSLATION: " + EN_BAD],
        "linguist": ["Other.", "CRITIQUE: y\nTRANSLATION: Other."],
    })
    result = debate_one(agents, _segment(), rounds=1)
    assert result["final_proposals"]["anatomist"] == EN_GOOD


def test_rejects_a_single_agent() -> None:
    with pytest.raises(ValueError, match="at least two"):
        debate_one(_agents()[:1], _segment(), rounds=1)


# ── scoring ────────────────────────────────────────────────────────────────
def test_run_debate_scores_the_debate_and_every_solo_agent() -> None:
    rows = run_debate(_agents(), [_segment()], rounds=1)
    variants = rows[0]["variants"]
    assert "debate" in variants
    assert {"solo:anatomist", "solo:safety", "solo:linguist"} <= set(variants)


def test_solo_variant_is_the_agents_opening_proposal() -> None:
    agents = _agents({
        "anatomist": [EN_BAD],
        "safety": [EN_GOOD],
        "linguist": [EN_GOOD],
    })
    rows = run_debate(agents, [_segment()], rounds=0)
    assert rows[0]["variants"]["solo:anatomist"]["hyp_text"] == EN_BAD


def test_clinical_findings_are_recorded_per_variant() -> None:
    agents = _agents({
        "anatomist": [EN_BAD], "safety": [EN_BAD], "linguist": [EN_BAD],
    })
    rows = run_debate(agents, [_segment()], rounds=0)
    assert rows[0]["variants"]["solo:anatomist"]["has_critical_error"] is True


def test_summary_covers_every_variant_plus_meta() -> None:
    rows = run_debate(_agents(), [_segment()], rounds=1)
    summary = summarise_debate(rows)
    labels = {s["variant"] for s in summary}
    assert "debate" in labels and "_meta" in labels
    meta = next(s for s in summary if s["variant"] == "_meta")
    assert 0.0 <= meta["converged_fraction"] <= 1.0


def test_empty_input_returns_nothing() -> None:
    assert run_debate(_agents(), [], rounds=1) == []


class _FakeTransformers:
    """Stands in for the transformers module in the capability probe."""

    def __init__(self, raiser=None):
        outer = self

        class _Tok:
            def apply_chat_template(self, *_a, **_k):
                if outer._raiser:
                    raise outer._raiser
                return "<|user|>Reply with OK."

        class _Auto:
            @staticmethod
            def from_pretrained(*_a, **_k):
                return _Tok()

        self._raiser = raiser
        self.AutoTokenizer = _Auto


def test_translation_only_models_are_rejected_with_a_clear_reason(monkeypatch) -> None:
    """TranslateGemma's template is translation-only, so a debate prompt raises a
    Jinja error deep inside generation. Job 4077236 spent 12 minutes and three
    model loads discovering that; the probe must catch it up front."""
    import sys as _sys

    import jinja2

    from medmt_eval.models import chat as chat_mod

    monkeypatch.setitem(
        _sys.modules, "transformers",
        _FakeTransformers(jinja2.exceptions.TemplateError(
            "User role must provide `content` as an iterable"
        )),
    )
    ok, reason = chat_mod.supports_freeform_chat("google/translategemma-4b-it")
    assert ok is False
    assert "translation-only" in reason


def test_chat_capable_models_pass_the_probe(monkeypatch) -> None:
    import sys as _sys

    from medmt_eval.models import chat as chat_mod

    monkeypatch.setitem(_sys.modules, "transformers", _FakeTransformers())
    ok, reason = chat_mod.supports_freeform_chat("Qwen/Qwen3.5-4B")
    assert ok is True and reason == "ok"


def test_local_backend_refuses_to_load_an_unusable_model(monkeypatch) -> None:
    """The failure must surface at load, not after three models are in memory."""
    from medmt_eval.models import chat as chat_mod

    monkeypatch.setattr(
        chat_mod, "supports_freeform_chat",
        lambda *_a, **_k: (False, "chat template rejects a plain instruction"),
    )
    backend = chat_mod.LocalChat("google/translategemma-4b-it")
    with pytest.raises(chat_mod.UnsupportedChatModel, match="translategemma"):
        backend.complete("hello")
