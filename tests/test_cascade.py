"""Tests for the sequential translate -> terminologise -> arbitrate pipeline."""

from __future__ import annotations

import pytest

from medmt_eval.glossary import Glossary, GlossaryEntry
from medmt_eval.inference.cascade import (
    PERSONAS, Stage, cascade_one, parse_stage_reply, run_cascade, summarise_cascade,
)
from medmt_eval.models.chat import ChatBackend
from medmt_eval.schema import Segment

DE = "Kein Pleuraerguss. Ein 5 mm Lymphknoten links."
DRAFT = "Pleural effusion. A 5 mm lymph node on the left."
FIXED = "No pleural effusion. A 5 mm lymph node on the left."
FINAL = "No pleural effusion. A 5 mm left-sided lymph node."


class _Scripted(ChatBackend):
    def __init__(self, name: str, replies: list[str]) -> None:
        self.name = name
        self._replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, user: str, system: str | None = None) -> str:
        self.prompts.append(user)
        return self._replies.pop(0) if self._replies else ""

    @property
    def config(self):
        return {"backend": "scripted"}


def _glossary() -> Glossary:
    return Glossary([
        GlossaryEntry("RID1", "radlex", en="pleural effusion", de="Pleuraerguss"),
        GlossaryEntry("RID2", "radlex", en="lymph node", de="Lymphknoten"),
    ])


def _stages(replies=None):
    replies = replies or {}
    return [
        Stage("1-translate", "translate",
              _Scripted("mt", replies.get("translate", [DRAFT]))),
        Stage("2-terminologise", "terminologise",
              _Scripted("term", replies.get("terminologise",
                                            [f"TRANSLATION: {FIXED}\nISSUES: negation dropped."])),
              glossary=_glossary()),
        Stage("3-arbitrate", "arbitrate",
              _Scripted("expert", replies.get("arbitrate",
                                              [f"TRANSLATION: {FINAL}\nRULING: accepted."]))),
    ]


def _segment():
    return Segment(id="s1", domain="t", src_lang="de", tgt_lang="en",
                   src_text=DE, ref_text=FIXED)


def test_parse_takes_translation_before_the_note() -> None:
    note, text, ok = parse_stage_reply("TRANSLATION: X\nISSUES: a")
    assert (note, text, ok) == ("a", "X", True)


def test_parse_survives_a_truncated_note() -> None:
    """TRANSLATION comes first precisely so that running out of tokens costs the
    note rather than the translation."""
    note, text, ok = parse_stage_reply("TRANSLATION: X")
    assert (text, ok) == ("X", True)


def test_parse_keeps_multiline_translations_whole() -> None:
    _, text, ok = parse_stage_reply("TRANSLATION: Line one.\nLine two.\n\nRULING: fine.")
    assert (text, ok) == ("Line one.\nLine two.", True)


def test_parse_handles_the_legacy_note_first_order() -> None:
    _, text, ok = parse_stage_reply("ISSUES: fine\n\nNo pleural effusion.")
    assert (text, ok) == ("No pleural effusion.", True)


def test_commentary_without_a_translation_is_a_parse_failure() -> None:
    """Job 4077545: a 4B model wrote a long ISSUES list and never reached
    TRANSLATION. Returning the whole reply scored the model's prose as its
    translation on 17 of 40 documents — BLEU fell 17.9 points and the critical
    error rate rose 22.5, and all of it was an artefact."""
    note, text, ok = parse_stage_reply("ISSUES:\n- changed a to b\n- changed c to d")
    assert ok is False
    assert text == ""
    assert "changed a to b" in note


def test_unformatted_reply_is_treated_as_the_translation() -> None:
    assert parse_stage_reply("just text") == ("", "just text", True)


def test_stages_run_in_order_and_each_produces_text() -> None:
    result = cascade_one(_stages(), _segment())
    assert [s["stage"] for s in result["stages"]] == [
        "1-translate", "2-terminologise", "3-arbitrate"]
    assert result["final_text"] == FINAL


def test_terminologist_sees_the_source_and_the_draft() -> None:
    stages = _stages()
    cascade_one(stages, _segment())
    prompt = stages[1].backend.prompts[0]
    assert DE in prompt and DRAFT in prompt


def test_arbiter_sees_source_draft_correction_and_the_reported_issues() -> None:
    stages = _stages()
    cascade_one(stages, _segment())
    prompt = stages[2].backend.prompts[0]
    assert DE in prompt and DRAFT in prompt and FIXED in prompt
    assert "negation dropped" in prompt


def test_parse_failures_are_reported_not_hidden() -> None:
    """A stage whose replies do not parse contributes carried-forward text, so
    its scores say nothing about the stage. That must be visible."""
    stages = _stages({"terminologise": ["ISSUES:\n- lots of commentary and no translation"]})
    rows = run_cascade(stages, [_segment()])
    summary = summarise_cascade(rows)
    term = next(s for s in summary if s["stage"] == "2-terminologise")
    assert term["parse_failures"] == 1
    # and the draft was carried forward rather than replaced by commentary
    assert rows[0]["stages"]["2-terminologise"]["hyp_text"] == DRAFT


def test_only_the_glossary_stage_is_shown_terms() -> None:
    """The arbiter must not see the dictionary: it exists to catch corrections
    the dictionary got wrong."""
    stages = _stages()
    cascade_one(stages, _segment())
    assert "Pleuraerguss = pleural effusion" in stages[1].backend.prompts[0]
    assert "Pleuraerguss = pleural effusion" not in stages[2].backend.prompts[0]
    assert "Pleuraerguss = pleural effusion" not in stages[0].backend.prompts[0]


def test_empty_translation_keeps_the_previous_text() -> None:
    """A stage replying 'no change' must not blank the report."""
    stages = _stages({"arbitrate": ["RULING: nothing to change."]})
    result = cascade_one(stages, _segment())
    assert result["final_text"] == FIXED


def test_first_stage_must_be_a_translate_role() -> None:
    bad = [Stage("x", "arbitrate", _Scripted("a", []))]
    with pytest.raises(ValueError, match="must have role 'translate'"):
        cascade_one(bad, _segment())


def test_unknown_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown role"):
        Stage("x", "nope", _Scripted("a", []))


def test_every_stage_is_scored_separately() -> None:
    rows = run_cascade(_stages(), [_segment()])
    stages = rows[0]["stages"]
    assert set(stages) == {"1-translate", "2-terminologise", "3-arbitrate"}
    assert stages["1-translate"]["has_critical_error"] is True   # negation dropped
    assert stages["2-terminologise"]["has_critical_error"] is False


def test_changed_previous_records_whether_a_stage_edited_anything() -> None:
    rows = run_cascade(_stages(), [_segment()])
    assert rows[0]["changed_previous"]["2-terminologise"] is True
    assert rows[0]["changed_previous"]["3-arbitrate"] is True


def test_a_passive_stage_is_recorded_as_no_change() -> None:
    stages = _stages({"arbitrate": [f"TRANSLATION: {FIXED}\nRULING: fine."]})
    rows = run_cascade(stages, [_segment()])
    assert rows[0]["changed_previous"]["3-arbitrate"] is False


def test_summary_reports_per_stage_deltas_in_pipeline_order() -> None:
    rows = run_cascade(_stages(), [_segment()])
    summary = summarise_cascade(rows)
    assert [s["stage"] for s in summary] == [
        "1-translate", "2-terminologise", "3-arbitrate"]
    assert summary[0]["bleu_delta"] is None       # nothing precedes stage 1
    assert summary[1]["bleu_delta"] is not None
    assert summary[1]["critical_delta"] == -1.0   # fixed the only document


def test_empty_input_returns_nothing() -> None:
    assert run_cascade(_stages(), []) == []


def test_local_backend_drops_tensors_the_model_does_not_accept(monkeypatch) -> None:
    """Hy-MT2-7B's tokenizer returns token_type_ids and its generate() rejects
    them outright; Qwen's and Gemma's do not. Job 4077495 died on this."""
    from medmt_eval.models import chat as chat_mod

    seen: dict = {}

    class _Tensor(list):
        @property
        def shape(self):
            return (1, 2)

    class _Enc(dict):
        def to(self, _device):
            return self

    class _Tok:
        pad_token = "<pad>"
        pad_token_id = 0
        padding_side = "left"

        def apply_chat_template(self, *_a, **_k):
            return "prompt"

        def __call__(self, *_a, **_k):
            return _Enc(
                input_ids=_Tensor([[1, 2]]),
                attention_mask=_Tensor([[1, 1]]),
                token_type_ids=_Tensor([[0, 0]]),
            )

        def decode(self, *_a, **_k):
            return "out"

    class _Out:
        def __getitem__(self, _idx):
            return [3, 4]

    class _Model:
        device = "cpu"

        def forward(self, input_ids=None, attention_mask=None):  # no token_type_ids
            ...

        def generate(self, **kwargs):
            seen.update(kwargs)
            return _Out()

    class _Torch:
        @staticmethod
        def inference_mode():
            import contextlib

            return contextlib.nullcontext()

    backend = chat_mod.LocalChat("tencent/Hy-MT2-7B")
    backend._tokenizer = _Tok()
    backend._model = _Model()
    backend._torch = _Torch()
    monkeypatch.setattr(chat_mod, "supports_freeform_chat", lambda *_a, **_k: (True, "ok"))

    assert backend.complete("hi") == "out"
    assert "token_type_ids" not in seen
    assert "input_ids" in seen and "attention_mask" in seen
