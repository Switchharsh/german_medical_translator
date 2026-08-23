"""LLM-as-judge with a *clinical* rubric, emitting MQM-style error spans.

The rule-based detectors in :mod:`medmt_eval.taxonomy.clinical` are high
precision and closed-class: they find dropped negations, flipped laterality and
changed numbers because someone wrote a rule for each. Their recall is unknown
and demonstrably imperfect — three models rendered ``BWK 12`` (twelfth *thoracic*
vertebra) as ``L12`` (twelfth *lumbar*) and no detector fired, because nobody had
written an anatomical-abbreviation rule.

This module is the open-class counterpart. It follows GEMBA-MQM
(arXiv:2310.13988), which prompts a strong model to annotate MQM error spans, with
two deliberate departures:

* **The rubric is clinical, not generic.** GEMBA's categories are
  accuracy/fluency/terminology. Those cannot distinguish a mistranslated
  adjective from a mistranslated vertebra level, which is the only distinction
  this project cares about. The categories here are the clinical failure modes,
  and severity is defined by effect on patient management.
* **It is scored against the source, never a reference.** A reference-based judge
  would penalise legitimate paraphrase, which is the failure mode that motivated
  the whole two-layer design.

Known limits, stated because they bound what the output can be used for:

* **Prompt sensitivity is severe.** Moving GEMBA from a bare prompt to a
  rubric-style one moved correlation with human judgement from 0.09 to 0.35
  (RUBRIC-MQM, ACL 2025 Industry). This rubric is unvalidated against clinicians;
  until it is, treat findings as *candidates* for human review, not as ground
  truth.
* **Self-preference.** A model must not judge its own output. ``run_llm_judge``
  refuses when the judge name matches the system under test.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Sequence

from medmt_eval.models.chat import ChatBackend
from medmt_eval.schema import language_name

#: Clinical error categories. Deliberately parallel to the rule-based detectors
#: so the two can be diffed, plus the open-class categories the rules cannot
#: express.
CATEGORIES = (
    "negation",          # presence/absence inverted or dropped
    "laterality",        # left/right/bilateral wrong or missing
    "measurement",       # number, unit or dimension altered
    "anatomy",           # wrong structure, level or abbreviation expansion
    "finding",           # a diagnosis or observation added, dropped or changed
    "certainty",         # hedging strengthened or weakened
    "terminology",       # non-standard term that a clinician would not write
    "omission",          # source content missing from the translation
)

#: Severity is defined by clinical consequence, not by linguistic magnitude.
SEVERITIES = ("critical", "major", "minor")

_SYSTEM = (
    "You are a bilingual consultant radiologist auditing a machine translation of "
    "a radiology report. You are precise, you do not invent problems, and you "
    "judge only against the source text."
)

_PROMPT = """Audit this translation for errors that would change patient care.

SOURCE ({source_lang}):
{source}

TRANSLATION ({target_lang}):
{hypothesis}

Report every error you find, using these categories:
- negation: presence/absence inverted or dropped
- laterality: left/right/bilateral wrong or missing
- measurement: a number, unit or dimension changed
- anatomy: wrong structure, wrong level, or an abbreviation expanded wrongly
- finding: a diagnosis or observation added, dropped or changed
- certainty: hedging strengthened or weakened
- terminology: a term no clinician in the target language would write
- omission: source content missing from the translation

Severity is about clinical consequence, not wording:
- critical: would change diagnosis, management or urgency
- major: clinically relevant but unlikely to change management on its own
- minor: stylistic or cosmetic

Rules:
- Judge ONLY against the source. A different wording that preserves the meaning
  is NOT an error.
- Quote the exact offending span from the translation.
- If the translation is clinically faithful, return an empty list.

Reply with JSON only, no other text:
{{"errors": [{{"category": "...", "severity": "...", "span": "...", "explanation": "..."}}]}}"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_judgement(reply: str) -> list[dict[str, str]]:
    """Extract the error list from a judge reply.

    Models wrap JSON in prose or fences often enough that a bare ``json.loads``
    fails on otherwise-good output, so the outermost brace-delimited block is
    taken. A reply that cannot be parsed yields an empty list rather than a
    crash — but see :func:`run_llm_judge`, which counts those separately, because
    silently treating unparseable output as "no errors found" would bias every
    system toward looking clean.
    """
    match = _JSON_BLOCK.search(reply)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    errors = payload.get("errors")
    if not isinstance(errors, list):
        return []
    cleaned: list[dict[str, str]] = []
    for item in errors:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "")).strip().lower()
        severity = str(item.get("severity", "")).strip().lower()
        cleaned.append({
            "category": category if category in CATEGORIES else "other",
            "severity": severity if severity in SEVERITIES else "minor",
            "span": str(item.get("span", "")).strip(),
            "explanation": str(item.get("explanation", "")).strip(),
        })
    return cleaned


def judge_one(
    backend: ChatBackend,
    source: str,
    hypothesis: str,
    src_lang: str,
    tgt_lang: str,
) -> tuple[list[dict[str, str]], bool]:
    """Audit one translation. Returns (errors, parsed_ok)."""
    reply = backend.complete(
        _PROMPT.format(
            source_lang=language_name(src_lang),
            target_lang=language_name(tgt_lang),
            source=source,
            hypothesis=hypothesis,
        ),
        system=_SYSTEM,
    )
    parsed_ok = bool(_JSON_BLOCK.search(reply))
    return parse_judgement(reply), parsed_ok


def run_llm_judge(
    backend: ChatBackend,
    rows: Sequence[dict[str, Any]],
    *,
    src_key: str = "src_text",
    hyp_key: str = "hyp_text",
    src_lang: str = "de",
    tgt_lang: str = "en",
    system_under_test: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Audit every row, returning the input rows augmented with judge findings.

    ``system_under_test`` guards against a model grading its own homework, which
    is a live risk here: the hosted models in this benchmark are exactly the
    models one would reach for as a judge.
    """
    if system_under_test and backend.name.rsplit("/", 1)[-1].lower() == (
        system_under_test.rsplit("/", 1)[-1].lower()
    ):
        raise ValueError(
            f"Judge {backend.name!r} is the system under test. Pick a different "
            f"judge; self-evaluation inflates the score."
        )

    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if progress and index % 10 == 0:
            progress(f"judging {index}/{len(rows)}")
        errors, parsed_ok = judge_one(
            backend, row[src_key], row[hyp_key], src_lang, tgt_lang
        )
        critical = [e for e in errors if e["severity"] == "critical"]
        out.append({
            **row,
            "judge_model": backend.name,
            "judge_errors": errors,
            "judge_parse_ok": parsed_ok,
            "judge_has_critical": bool(critical),
            "judge_n_critical": len(critical),
        })
    return out


def summarise_judge(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate judge findings, keeping unparseable replies visible."""
    import collections

    if not rows:
        return {}
    n = len(rows)
    categories: collections.Counter[str] = collections.Counter()
    severities: collections.Counter[str] = collections.Counter()
    for row in rows:
        for error in row.get("judge_errors") or []:
            categories[error["category"]] += 1
            severities[error["severity"]] += 1
    unparsed = sum(1 for r in rows if not r.get("judge_parse_ok", True))
    return {
        "n": n,
        "judge_model": rows[0].get("judge_model"),
        "critical_error_rate": sum(1 for r in rows if r.get("judge_has_critical")) / n,
        "mean_errors_per_doc": sum(len(r.get("judge_errors") or []) for r in rows) / n,
        "by_category": dict(categories.most_common()),
        "by_severity": dict(severities.most_common()),
        # A run with many unparseable replies is not a clean run: those documents
        # contribute zero findings and silently drag the error rate down.
        "unparseable_replies": unparsed,
    }


def agreement_with_detectors(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Confusion between the rule detectors and the judge, per document.

    This is the number the metric roadmap actually wants: documents the judge
    flags and the rules miss are the recall gap, and they are where the
    ``BWK 12`` → ``L12`` class of error lives.
    """
    both = rules_only = judge_only = neither = 0
    for row in rows:
        rules = bool(row.get("has_critical_error"))
        judge = bool(row.get("judge_has_critical"))
        if rules and judge:
            both += 1
        elif rules:
            rules_only += 1
        elif judge:
            judge_only += 1
        else:
            neither += 1
    total = max(1, len(rows))
    return {
        "n": len(rows),
        "both_flag": both,
        "rules_only": rules_only,
        "judge_only": judge_only,
        "neither": neither,
        "agreement": (both + neither) / total,
    }
