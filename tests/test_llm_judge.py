"""Tests for the clinical LLM-as-judge metric."""

from __future__ import annotations

import pytest

from medmt_eval.metrics.llm_judge import (
    CATEGORIES, agreement_with_detectors, parse_judgement, run_llm_judge, summarise_judge,
)
from medmt_eval.models.chat import ChatBackend

_ERR = ('{"errors":[{"category":"negation","severity":"critical",'
        '"span":"Pleural effusion","explanation":"kein was dropped"}]}')
_CLEAN = '{"errors":[]}'


class _Fixed(ChatBackend):
    def __init__(self, reply: str, name: str = "judge-model") -> None:
        self.name = name
        self._reply = reply

    def complete(self, user: str, system: str | None = None) -> str:
        return self._reply

    @property
    def config(self):
        return {"backend": "fixed"}


def _rows(n=2):
    return [{"id": f"s{i}", "src_text": "Kein Erguss.", "hyp_text": "Effusion.",
             "has_critical_error": True} for i in range(n)]


def test_parses_json_wrapped_in_prose() -> None:
    got = parse_judgement("Sure, here:\n" + _ERR + "\nHope that helps.")
    assert got[0]["category"] == "negation" and got[0]["severity"] == "critical"


def test_unknown_category_falls_back_to_other() -> None:
    got = parse_judgement('{"errors":[{"category":"vibes","severity":"critical","span":"x"}]}')
    assert got[0]["category"] == "other"


def test_unknown_severity_falls_back_to_minor() -> None:
    """An unrecognised severity must not be promoted to critical."""
    got = parse_judgement('{"errors":[{"category":"negation","severity":"catastrophic","span":"x"}]}')
    assert got[0]["severity"] == "minor"


def test_unparseable_reply_yields_no_errors_but_is_flagged() -> None:
    """Treating a broken reply as 'clean' would bias every system toward looking
    good, so the run must record that it could not be parsed."""
    rows = run_llm_judge(_Fixed("I could not comply."), _rows(1))
    assert rows[0]["judge_errors"] == []
    assert rows[0]["judge_parse_ok"] is False
    assert summarise_judge(rows)["unparseable_replies"] == 1


def test_clean_translation_reports_no_errors() -> None:
    rows = run_llm_judge(_Fixed(_CLEAN), _rows(1))
    assert rows[0]["judge_has_critical"] is False
    assert rows[0]["judge_parse_ok"] is True


def test_critical_findings_are_counted() -> None:
    rows = run_llm_judge(_Fixed(_ERR), _rows(3))
    assert all(r["judge_has_critical"] for r in rows)
    assert summarise_judge(rows)["critical_error_rate"] == 1.0


def test_judge_refuses_to_grade_its_own_output() -> None:
    with pytest.raises(ValueError, match="system under test"):
        run_llm_judge(_Fixed(_CLEAN, name="glm-5.2"), _rows(1),
                      system_under_test="glm-5.2")


def test_self_evaluation_guard_ignores_namespacing() -> None:
    with pytest.raises(ValueError, match="system under test"):
        run_llm_judge(_Fixed(_CLEAN, name="z-ai/glm-5.2"), _rows(1),
                      system_under_test="glm-5.2")


def test_a_different_judge_is_allowed() -> None:
    rows = run_llm_judge(_Fixed(_CLEAN, name="other-model"), _rows(1),
                         system_under_test="glm-5.2")
    assert len(rows) == 1


def test_agreement_separates_the_recall_gap() -> None:
    """Documents the judge flags and the rules miss are the whole point."""
    rows = [
        {"has_critical_error": True,  "judge_has_critical": True},
        {"has_critical_error": False, "judge_has_critical": True},   # recall gap
        {"has_critical_error": True,  "judge_has_critical": False},
        {"has_critical_error": False, "judge_has_critical": False},
    ]
    got = agreement_with_detectors(rows)
    assert got["judge_only"] == 1 and got["rules_only"] == 1
    assert got["both_flag"] == 1 and got["neither"] == 1
    assert got["agreement"] == 0.5


def test_summary_of_nothing_is_empty() -> None:
    assert summarise_judge([]) == {}
