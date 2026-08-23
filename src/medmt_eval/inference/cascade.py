"""A sequential review pipeline: translate, then correct, then arbitrate.

This is a different protocol from :mod:`medmt_eval.inference.debate`. There, three
peers propose in parallel and revise symmetrically. Here each stage sees the work
of the stage before it and is asked for something different:

    1. translate     an MT model produces the first translation from the source
    2. terminologise a model *with the glossary* sees source + draft and corrects
                     what it believes is wrong
    3. arbitrate     a domain expert *without* the glossary sees source, draft and
                     correction, weighs the disagreement, and issues the final text

The asymmetry is the point. A round-robin debate measures consensus; a cascade
measures **marginal contribution**, because every stage's output is scored
separately and stage *n* is only credited with what it changed relative to
stage *n-1*. That distinction matters here: the symmetric debate moved BLEU and
left the clinical error rate untouched, and it could not say which participant
was responsible.

Two hazards the design has to answer for:

* **Later stages can make things worse.** An arbiter free to rewrite can undo a
  correct fix. Scoring every stage is what makes that visible instead of hidden
  behind a final number, and ``stage_deltas`` reports it directly.
* **A reviewer that sees a draft anchors on it.** Stage 2 and 3 are shown the
  source first and the draft second, and are asked to justify changes against the
  *source*, not to prefer the draft.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from medmt_eval.glossary import Glossary
from medmt_eval.metrics.surface import score_surface
from medmt_eval.models.chat import ChatBackend
from medmt_eval.schema import Segment, language_name
from medmt_eval.taxonomy.clinical import ClinicalSafetyEvaluator

ROLES = ("translate", "terminologise", "arbitrate")

PERSONAS: dict[str, str] = {
    "translate": (
        "You are a machine translation system for medical text. Translate "
        "faithfully and completely."
    ),
    "terminologise": (
        "You are a radiology terminologist reviewing a draft translation against "
        "an approved bilingual glossary. Your job is to find where the draft "
        "departs from the source or from approved terminology — a wrong or "
        "missing term, a changed measurement, a dropped negation, a flipped "
        "laterality — and to correct it. Do not restyle text that is already "
        "correct."
    ),
    "arbitrate": (
        "You are a senior German radiologist with native command of both German "
        "and English. You are shown a source report, a draft translation, and a "
        "terminologist's corrected version. The terminologist works from a "
        "glossary and can be wrong: a glossary term is not always right in "
        "context, and a 'correction' can introduce an error. Decide what the "
        "report should say, preferring the reading that preserves clinical fact, "
        "and issue the final translation."
    ),
}

_TRANSLATE = (
    "{persona}\n\n"
    "Translate the following medical text from {source_lang} to {target_lang}.\n"
    "Return only the translation. No preamble, no commentary.\n\n{text}"
)

_TERMINOLOGISE = (
    "{persona}\n\n"
    "SOURCE ({source_lang}):\n{text}\n\n"
    "{glossary_block}"
    "DRAFT TRANSLATION ({target_lang}):\n{draft}\n\n"
    "Compare the draft against the source. Correct anything the draft gets "
    "wrong; leave correct text alone.\n\n"
    "Reply in exactly this format, TRANSLATION first:\n"
    "TRANSLATION: the full corrected translation.\n"
    "ISSUES: a short list of what you changed and why, or 'none'."
)

_ARBITRATE = (
    "{persona}\n\n"
    "SOURCE ({source_lang}):\n{text}\n\n"
    "DRAFT ({target_lang}), from a translation model:\n{draft}\n\n"
    "CORRECTED ({target_lang}), from a terminologist working with a glossary:\n"
    "{corrected}\n\n"
    "The terminologist reported: {issues}\n\n"
    "Judge the disagreement against the source. Accept the corrections that are "
    "right, reject any that are not, and fix anything both of them missed.\n\n"
    "Reply in exactly this format, TRANSLATION first:\n"
    "TRANSLATION: the final translation.\n"
    "RULING: one or two sentences on what you accepted or rejected."
)

_TEMPLATES = {
    "translate": _TRANSLATE,
    "terminologise": _TERMINOLOGISE,
    "arbitrate": _ARBITRATE,
}

_MARKER = re.compile(r"^\s*TRANSLATION\s*:\s*", re.IGNORECASE | re.MULTILINE)
# The note ends at a TRANSLATION marker, a blank line, or the end of the reply.
# Without the blank-line stop it swallowed everything after it, so a reply of the
# shape "ISSUES: ...\n\n<translation>" yielded an empty translation.
_NOTE = re.compile(
    r"^\s*(?:ISSUES|RULING)\s*:[ \t]*(.*?)(?=^\s*TRANSLATION\s*:|\n[ \t]*\n|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def parse_stage_reply(reply: str) -> tuple[str, str, bool]:
    """Split a reply into (note, translation, parse_ok).

    Getting this wrong is expensive and silent. In job 4077545 a 4B model wrote a
    long ``ISSUES:`` list and never reached ``TRANSLATION:`` — only 1 reply in 40
    contained the marker at all — and the fallback returned the *whole reply*,
    commentary included, as the translation. Seventeen of forty documents were
    scored on the model's prose. BLEU fell 17.9 points and the critical-error
    rate rose 22.5, and every one of those numbers was an artefact.

    So there are three cases, not two:

    * a ``TRANSLATION:`` marker is present -> take what follows (parse_ok)
    * a note marker is present but no ``TRANSLATION:`` -> the model answered in
      the wrong shape. Strip the note block and use the remainder if there is
      one; otherwise report failure so the caller can carry the previous text
      forward rather than score commentary.
    * no marker at all -> the reply is a bare translation (parse_ok)
    """
    marker = _MARKER.search(reply)
    note_match = _NOTE.search(reply)
    note = note_match.group(1).strip() if note_match else ""

    if marker:
        body = reply[marker.end():]
        # The note now follows the translation, so cut it off if present.
        trailing = _NOTE.search(body)
        text = (body[: trailing.start()] if trailing else body).strip()
        if trailing:
            note = trailing.group(1).strip()
        return note, text, bool(text)
    if note_match:
        remainder = reply[note_match.end():].strip()
        # A note with nothing after it is commentary, not a translation.
        return note, remainder, bool(remainder)
    return "", reply.strip(), True


@dataclass
class Stage:
    name: str
    role: str
    backend: ChatBackend
    persona: str = ""
    glossary: Glossary = field(default_factory=Glossary)
    max_terms: int = 40

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"Unknown role {self.role!r}; choose from {ROLES}.")
        if not self.persona:
            self.persona = PERSONAS[self.role]

    def glossary_block(self, text: str, src: str, tgt: str) -> str:
        entries = self.glossary.select(text, src, tgt, limit=self.max_terms)
        block = self.glossary.render(entries, src, tgt)
        return f"APPROVED TERMINOLOGY\n{block}\n\n" if block else ""


def cascade_one(stages: Sequence[Stage], segment: Segment) -> dict[str, Any]:
    """Run the pipeline over one document, returning every stage's output."""
    if not stages:
        raise ValueError("A cascade needs at least one stage.")
    if stages[0].role != "translate":
        raise ValueError("The first stage must have role 'translate'.")

    src, tgt = segment.src_lang, segment.tgt_lang
    source_name, target_name = language_name(src), language_name(tgt)
    outputs: list[dict[str, Any]] = []
    draft = corrected = ""
    issues = "nothing reported"

    for stage in stages:
        block = stage.glossary_block(segment.src_text, src, tgt)
        prompt = _TEMPLATES[stage.role].format(
            persona=stage.persona,
            source_lang=source_name,
            target_lang=target_name,
            text=segment.src_text,
            glossary_block=block,
            draft=draft,
            corrected=corrected,
            issues=issues or "nothing reported",
        )
        reply = stage.backend.complete(prompt)
        if stage.role == "translate":
            note, text, parse_ok = "", reply.strip(), True
            draft = text
        else:
            note, text, parse_ok = parse_stage_reply(reply)
            # A failed parse or an empty translation means "no change", never
            # "replace the report with commentary".
            if not parse_ok or not text:
                text = corrected or draft
            if stage.role == "terminologise":
                corrected, issues = text, note
        outputs.append({
            "stage": stage.name,
            "role": stage.role,
            "backend": stage.backend.name,
            "note": note,
            "text": text,
            "parse_ok": parse_ok,
            "n_glossary_terms": block.count("\n- "),
        })

    return {"id": segment.id, "stages": outputs, "final_text": outputs[-1]["text"]}


def run_cascade(
    stages: Sequence[Stage],
    segments: Sequence[Segment],
    *,
    safety_evaluator: ClinicalSafetyEvaluator | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run the pipeline over a corpus, scoring **every stage** independently."""
    if not segments:
        return []
    evaluator = safety_evaluator or ClinicalSafetyEvaluator()

    results: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if progress:
            progress(f"cascade {index + 1}/{len(segments)}  {segment.id}")
        results.append(cascade_one(stages, segment))

    references = [s.ref_text for s in segments]
    scorable = all(r is not None for r in references)
    names = [s.name for s in stages]

    surface: dict[str, list[dict[str, float]]] = {}
    for position, name in enumerate(names):
        hypotheses = [r["stages"][position]["text"] for r in results]
        surface[name] = (
            list(score_surface(hypotheses, [str(x) for x in references]).sentence)
            if scorable else [{} for _ in hypotheses]
        )

    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        result = results[index]
        scored: dict[str, Any] = {}
        for position, name in enumerate(names):
            text = result["stages"][position]["text"]
            findings = evaluator.evaluate(
                segment.src_text, text, segment.src_lang, segment.tgt_lang
            )
            scored[name] = {
                "hyp_text": text,
                "role": result["stages"][position]["role"],
                "backend": result["stages"][position]["backend"],
                "note": result["stages"][position]["note"],
                "parse_ok": result["stages"][position].get("parse_ok", True),
                "n_glossary_terms": result["stages"][position]["n_glossary_terms"],
                "metrics": surface[name][index],
                "findings": [f.to_dict() for f in findings],
                "has_critical_error": any(f.severity == "critical" for f in findings),
            }
        # Did each stage actually change anything? A stage that rewrites nothing
        # and a stage that rewrites everything are very different behaviours and
        # the scores alone do not distinguish them.
        changed = {
            names[i]: result["stages"][i]["text"].strip()
            != result["stages"][i - 1]["text"].strip()
            for i in range(1, len(names))
        }
        rows.append({
            "id": segment.id,
            "domain": segment.domain,
            "src_text": segment.src_text,
            "ref_text": segment.ref_text,
            "stages": scored,
            "stage_order": names,
            "changed_previous": changed,
        })
    return rows


def summarise_cascade(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per stage, in pipeline order, with the delta it contributed."""
    import statistics

    if not rows:
        return []
    names = rows[0]["stage_order"]
    summary: list[dict[str, Any]] = []
    previous_bleu: float | None = None
    previous_crit: float | None = None

    for name in names:
        entries = [row["stages"][name] for row in rows]

        def mean(metric: str) -> float | None:
            values = [
                value for entry in entries
                for key, value in (entry.get("metrics") or {}).items()
                if key == metric and isinstance(value, (int, float))
            ]
            return statistics.mean(values) if values else None

        bleu = mean("bleu")
        crit = sum(1 for e in entries if e["has_critical_error"]) / len(entries)
        changed = [
            row["changed_previous"].get(name) for row in rows
            if name in row["changed_previous"]
        ]
        summary.append({
            "stage": name,
            "role": entries[0]["role"],
            "backend": entries[0]["backend"],
            "n": len(entries),
            "bleu": bleu,
            "chrf": mean("chrf"),
            "ter": mean("ter"),
            "critical_error_rate": crit,
            "bleu_delta": None if previous_bleu is None or bleu is None else bleu - previous_bleu,
            "critical_delta": None if previous_crit is None else crit - previous_crit,
            "changed_fraction": (sum(1 for c in changed if c) / len(changed)) if changed else None,
            # A stage whose replies do not parse contributes carried-forward text,
            # so its scores say nothing about the stage. Never hide this.
            "parse_failures": sum(1 for e in entries if not e.get("parse_ok", True)),
            "mean_glossary_terms": statistics.mean(e["n_glossary_terms"] for e in entries),
        })
        previous_bleu, previous_crit = bleu, crit
    return summary
