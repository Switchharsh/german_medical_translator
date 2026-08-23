"""Experiment 1 — does handing the model a glossary improve the translation?

Three arms, one model, one prompt template. Only the block of terminology
changes between them:

``none``
    No glossary block. The baseline.
``glossary``
    The terms that actually occur in this document, source = target.
``distractor``
    The *same number* of terms, drawn from the same glossary, that do **not**
    occur in this document.

The distractor arm is what makes this an experiment rather than a demo. A
glossary block makes the prompt longer, more structured and more domain-flavoured,
and any of those could move the score on its own. Comparing ``glossary`` against
``none`` measures the block; comparing it against ``distractor`` measures the
terms. Only the second comparison supports the claim that a dictionary helps.

Scoring uses the same two layers as everything else. One subtlety governs the
terminology detector: if the injected terms and the scored terms are the same
set, the terminology metric improves by construction and means nothing. Pass a
``holdout`` split (see :meth:`Glossary.split`) so the two are disjoint.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from medmt_eval.glossary import Glossary, GlossaryEntry
from medmt_eval.metrics.surface import score_surface
from medmt_eval.models.chat import ChatBackend
from medmt_eval.schema import Segment, language_name
from medmt_eval.taxonomy.clinical import ClinicalSafetyEvaluator

ARMS = ("none", "glossary", "distractor")

_TEMPLATE = (
    "You are translating a medical document from {source_lang} to {target_lang}.\n"
    "{glossary_block}"
    "Return only the translation. No preamble, no commentary, no quotation marks.\n\n"
    "{text}"
)


def build_prompt(
    text: str,
    src_lang: str,
    tgt_lang: str,
    glossary_block: str = "",
) -> str:
    """Render the translation prompt (pure function, so the arms are testable).

    The block is padded with blank lines only when non-empty, so the ``none`` arm
    produces a prompt with no trace of a glossary having been considered.
    """
    block = f"\n{glossary_block}\n\n" if glossary_block else ""
    return _TEMPLATE.format(
        source_lang=language_name(src_lang),
        target_lang=language_name(tgt_lang),
        glossary_block=block,
        text=text,
    )


def entries_for_arm(
    arm: str,
    glossary: Glossary,
    text: str,
    src_lang: str,
    tgt_lang: str,
    *,
    limit: int = 40,
    seed: int = 13,
) -> list[GlossaryEntry]:
    """Which terms this arm shows the model for this document."""
    if arm == "none":
        return []
    relevant = glossary.select(text, src_lang, tgt_lang, limit=limit)
    if arm == "glossary":
        return relevant
    if arm == "distractor":
        # Matched in count so the two blocks are the same size. A document with
        # no relevant terms gets no distractors either, keeping the arms paired.
        return glossary.distractors(
            text, src_lang, tgt_lang, count=len(relevant), seed=seed
        )
    raise ValueError(f"Unknown arm {arm!r}; choose from {ARMS}.")


def run_glossary_experiment(
    backend: ChatBackend,
    segments: Sequence[Segment],
    glossary: Glossary,
    *,
    arms: Sequence[str] = ARMS,
    limit: int = 40,
    seed: int = 13,
    safety_evaluator: ClinicalSafetyEvaluator | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Translate every segment once per arm. Returns one row per (segment, arm)."""
    if not segments:
        return []
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"Unknown arm(s) {sorted(unknown)}; choose from {ARMS}.")

    evaluator = safety_evaluator or ClinicalSafetyEvaluator()
    rows: list[dict[str, Any]] = []

    for arm in arms:
        hypotheses: list[str] = []
        used: list[list[GlossaryEntry]] = []
        for index, segment in enumerate(segments):
            if progress and index % 10 == 0:
                progress(f"{arm}: {index}/{len(segments)}")
            entries = entries_for_arm(
                arm, glossary, segment.src_text, segment.src_lang, segment.tgt_lang,
                limit=limit, seed=seed,
            )
            block = glossary.render(entries, segment.src_lang, segment.tgt_lang)
            prompt = build_prompt(segment.src_text, segment.src_lang, segment.tgt_lang, block)
            hypotheses.append(backend.complete(prompt))
            used.append(entries)

        references = [s.ref_text for s in segments]
        surface = (
            list(score_surface(hypotheses, [str(r) for r in references]).sentence)
            if all(r is not None for r in references)
            else [{} for _ in hypotheses]
        )

        for index, segment in enumerate(segments):
            findings = evaluator.evaluate(
                segment.src_text, hypotheses[index], segment.src_lang, segment.tgt_lang
            )
            rows.append({
                "id": segment.id,
                "domain": segment.domain,
                "arm": arm,
                "model": backend.name,
                "src_text": segment.src_text,
                "ref_text": segment.ref_text,
                "hyp_text": hypotheses[index],
                "metrics": surface[index],
                "findings": [f.to_dict() for f in findings],
                "has_critical_error": any(f.severity == "critical" for f in findings),
                "n_glossary_terms": len(used[index]),
                "glossary_terms": [
                    {"concept_id": e.concept_id,
                     "src": e.term(segment.src_lang),
                     "tgt": e.term(segment.tgt_lang)}
                    for e in used[index]
                ],
                "generation": backend.config,
            })
    return rows


def summarise_by_arm(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate to one record per arm — the table the experiment reports."""
    import collections
    import statistics

    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)

    summary: list[dict[str, Any]] = []
    for arm in ARMS:
        group = grouped.get(arm)
        if not group:
            continue

        def mean(name: str) -> float | None:
            values = [
                value for row in group
                for key, value in (row.get("metrics") or {}).items()
                if key == name and isinstance(value, (int, float))
            ]
            return statistics.mean(values) if values else None

        codes: collections.Counter[str] = collections.Counter()
        for row in group:
            for finding in row.get("findings") or []:
                codes[finding["code"]] += 1

        summary.append({
            "arm": arm,
            "n": len(group),
            "critical_error_rate": sum(1 for r in group if r["has_critical_error"]) / len(group),
            "bleu": mean("bleu"),
            "chrf": mean("chrf"),
            "ter": mean("ter"),
            "mean_glossary_terms": statistics.mean(r["n_glossary_terms"] for r in group),
            "finding_counts": dict(sorted(codes.items())),
        })
    return summary
