"""Experiment 2 — three specialists argue about a translation and converge.

Each agent gets a different **persona** and a different **glossary slice**, so
they disagree for substantive reasons rather than by sampling noise. The personas
are deliberately aligned with this project's error taxonomy, one per critical
detector, so that a disagreement between agents is about the thing the evaluation
actually measures:

    anatomist   — laterality, anatomical structures, modality conventions
    safety      — negation, numbers, measurements, hedging and certainty
    linguist    — register, idiom, standard target-language terminology

Protocol per document:

    round 0   each agent independently proposes a translation
    round 1+  each agent sees every current proposal and revises its own,
              having been asked to say what it is changing and why
    final     a synthesis step merges the surviving proposals into one answer

Convergence is checked after each round; identical proposals stop the debate
early rather than burning turns on agreement. The full transcript is kept, so a
result can always be traced to the exchange that produced it.

A caution that belongs in the design, not the write-up: this is a *consensus*
mechanism, and consensus is not correctness. Three models sharing a blind spot
will agree confidently and wrongly, and the debate will look like it worked. The
transcript and the per-round scores exist so that can be checked rather than
assumed.
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

PERSONAS: dict[str, str] = {
    "anatomist": (
        "You are a board-certified radiologist reviewing a translated report. "
        "Your responsibility is anatomical and spatial fidelity: laterality "
        "(left/right/bilateral), anatomical structures and their standard names, "
        "vertebral and rib levels, and modality-specific conventions. You care "
        "that BWK 12 is a thoracic vertebra and not a lumbar one."
    ),
    "safety": (
        "You are a clinical safety reviewer. Your responsibility is the facts that "
        "change management: negation (present vs absent), every number, every "
        "measurement and its unit, dosages, and the strength of hedging — "
        "'suspicious for' is not 'consistent with'. You object whenever a "
        "translation adds certainty the source did not have, or drops any of it."
    ),
    "linguist": (
        "You are a professional medical translator. Your responsibility is that "
        "the output reads as a native report in the target language, in the "
        "correct register, using the terminology a clinician in that language "
        "would actually write. You object to calques, to literal renderings that "
        "no radiologist would write, and to inconsistent terminology."
    ),
}

_PROPOSE = (
    "{persona}\n\n"
    "Translate the following medical text from {source_lang} to {target_lang}.\n"
    "{glossary_block}"
    "Return ONLY the translation. No preamble, no commentary.\n\n"
    "{text}"
)

_REVISE = (
    "{persona}\n\n"
    "Source text ({source_lang}):\n{text}\n\n"
    "{glossary_block}"
    "Three translators produced these candidate translations into {target_lang}:\n\n"
    "{proposals}\n\n"
    "Your own is {own_label}. Considering the others from your specialism, produce "
    "your best translation now. You may keep yours unchanged if you believe it is "
    "correct.\n\n"
    "Reply in exactly this format:\n"
    "CRITIQUE: one or two sentences on what the others got wrong or right, from "
    "your specialism only.\n"
    "TRANSLATION: the full translation, on one line or several, and nothing after it."
)

_SYNTHESIS = (
    "You are the senior radiologist signing off this report. Three specialists "
    "translated it from {source_lang} to {target_lang} and reviewed each other:\n"
    "- an anatomist (laterality, structures, levels)\n"
    "- a clinical safety reviewer (negation, numbers, measurements, hedging)\n"
    "- a medical translator (register and terminology)\n\n"
    "Source text:\n{text}\n\n"
    "Their final candidates:\n\n{proposals}\n\n"
    "Produce the single translation you would sign. Where they disagree, prefer "
    "the reading that preserves clinical fact over the one that reads better.\n"
    "Return ONLY the translation. No preamble, no commentary."
)

_TRANSLATION_MARKER = re.compile(r"^\s*TRANSLATION\s*:\s*", re.IGNORECASE | re.MULTILINE)
_CRITIQUE_MARKER = re.compile(r"^\s*CRITIQUE\s*:\s*(.*?)(?=^\s*TRANSLATION\s*:|\Z)",
                              re.IGNORECASE | re.MULTILINE | re.DOTALL)


def parse_revision(reply: str) -> tuple[str, str]:
    """Split a revision reply into (critique, translation).

    A model that ignores the format and returns a bare translation must not have
    its whole reply — critique prose included — treated as the translation, so an
    unmarked reply yields an empty critique and the reply as the translation only
    when no marker is present at all.
    """
    match = _TRANSLATION_MARKER.search(reply)
    critique_match = _CRITIQUE_MARKER.search(reply)
    critique = critique_match.group(1).strip() if critique_match else ""
    if match:
        return critique, reply[match.end():].strip()
    return critique, reply.strip()


@dataclass
class Agent:
    name: str
    backend: ChatBackend
    persona: str
    glossary: Glossary = field(default_factory=Glossary)

    def glossary_block(self, text: str, src_lang: str, tgt_lang: str, limit: int) -> str:
        entries = self.glossary.select(text, src_lang, tgt_lang, limit=limit)
        return self.glossary.render(entries, src_lang, tgt_lang)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def debate_one(
    agents: Sequence[Agent],
    segment: Segment,
    *,
    rounds: int = 2,
    limit: int = 40,
    synthesiser: ChatBackend | None = None,
) -> dict[str, Any]:
    """Run the protocol for one document. Returns the transcript and final text."""
    if len(agents) < 2:
        raise ValueError("A debate needs at least two agents.")
    src, tgt = segment.src_lang, segment.tgt_lang
    source_name, target_name = language_name(src), language_name(tgt)

    transcript: list[dict[str, Any]] = []
    proposals: dict[str, str] = {}

    # round 0 — independent proposals
    for agent in agents:
        block = agent.glossary_block(segment.src_text, src, tgt, limit)
        prompt = _PROPOSE.format(
            persona=agent.persona,
            source_lang=source_name, target_lang=target_name,
            glossary_block=f"\n{block}\n\n" if block else "",
            text=segment.src_text,
        )
        proposals[agent.name] = agent.backend.complete(prompt).strip()
        transcript.append({
            "round": 0, "agent": agent.name, "critique": "",
            "translation": proposals[agent.name], "n_glossary_terms": block.count("\n- "),
        })

    converged_at: int | None = None
    if len({_normalise(v) for v in proposals.values()}) == 1:
        converged_at = 0

    # rounds 1..N — see everything, revise
    for round_index in range(1, rounds + 1):
        if converged_at is not None:
            break
        rendered = "\n\n".join(
            f"[{name}]\n{text}" for name, text in proposals.items()
        )
        revised: dict[str, str] = {}
        for agent in agents:
            block = agent.glossary_block(segment.src_text, src, tgt, limit)
            prompt = _REVISE.format(
                persona=agent.persona,
                source_lang=source_name, target_lang=target_name,
                glossary_block=f"{block}\n\n" if block else "",
                text=segment.src_text,
                proposals=rendered,
                own_label=f"[{agent.name}]",
            )
            critique, translation = parse_revision(agent.backend.complete(prompt))
            revised[agent.name] = translation or proposals[agent.name]
            transcript.append({
                "round": round_index, "agent": agent.name,
                "critique": critique, "translation": revised[agent.name],
                "n_glossary_terms": block.count("\n- "),
            })
        proposals = revised
        if len({_normalise(v) for v in proposals.values()}) == 1:
            converged_at = round_index

    # final synthesis
    rendered = "\n\n".join(f"[{name}]\n{text}" for name, text in proposals.items())
    judge = synthesiser or agents[0].backend
    final = judge.complete(_SYNTHESIS.format(
        source_lang=source_name, target_lang=target_name,
        text=segment.src_text, proposals=rendered,
    )).strip()

    return {
        "id": segment.id,
        "transcript": transcript,
        "final_proposals": dict(proposals),
        "final_text": final,
        "converged_at_round": converged_at,
        "n_rounds_run": max((t["round"] for t in transcript), default=0),
        "synthesiser": judge.name,
    }


def run_debate(
    agents: Sequence[Agent],
    segments: Sequence[Segment],
    *,
    rounds: int = 2,
    limit: int = 40,
    synthesiser: ChatBackend | None = None,
    safety_evaluator: ClinicalSafetyEvaluator | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Debate every segment and score the outcome. One row per segment.

    Each agent's round-0 proposal is scored too, so the debate can be compared
    against the individual models that fed it. A debate that lands below its own
    best participant is a real and reportable outcome.
    """
    if not segments:
        return []
    evaluator = safety_evaluator or ClinicalSafetyEvaluator()

    results: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if progress:
            progress(f"debate {index + 1}/{len(segments)}  {segment.id}")
        results.append(debate_one(
            agents, segment, rounds=rounds, limit=limit, synthesiser=synthesiser
        ))

    references = [s.ref_text for s in segments]
    scorable = all(r is not None for r in references)

    # Score the final answer and every agent's opening proposal on the same scale.
    variants: dict[str, list[str]] = {"debate": [r["final_text"] for r in results]}
    for agent in agents:
        variants[f"solo:{agent.name}"] = [
            next(t["translation"] for t in r["transcript"]
                 if t["round"] == 0 and t["agent"] == agent.name)
            for r in results
        ]

    surface: dict[str, list[dict[str, float]]] = {}
    for label, hypotheses in variants.items():
        surface[label] = (
            list(score_surface(hypotheses, [str(r) for r in references]).sentence)
            if scorable else [{} for _ in hypotheses]
        )

    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        result = results[index]
        scored: dict[str, Any] = {}
        for label, hypotheses in variants.items():
            findings = evaluator.evaluate(
                segment.src_text, hypotheses[index], segment.src_lang, segment.tgt_lang
            )
            scored[label] = {
                "hyp_text": hypotheses[index],
                "metrics": surface[label][index],
                "findings": [f.to_dict() for f in findings],
                "has_critical_error": any(f.severity == "critical" for f in findings),
            }
        rows.append({
            "id": segment.id,
            "domain": segment.domain,
            "src_text": segment.src_text,
            "ref_text": segment.ref_text,
            "variants": scored,
            "converged_at_round": result["converged_at_round"],
            "n_rounds_run": result["n_rounds_run"],
            "transcript": result["transcript"],
            "synthesiser": result["synthesiser"],
            "agents": [
                {"name": a.name, "backend": a.backend.name, "glossary_size": len(a.glossary)}
                for a in agents
            ],
        })
    return rows


def summarise_debate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per variant — the debate against each of its participants."""
    import statistics

    if not rows:
        return []
    labels = list(rows[0]["variants"])
    summary: list[dict[str, Any]] = []
    for label in labels:
        entries = [row["variants"][label] for row in rows]

        def mean(name: str) -> float | None:
            values = [
                value for entry in entries
                for key, value in (entry.get("metrics") or {}).items()
                if key == name and isinstance(value, (int, float))
            ]
            return statistics.mean(values) if values else None

        summary.append({
            "variant": label,
            "n": len(entries),
            "critical_error_rate": sum(1 for e in entries if e["has_critical_error"]) / len(entries),
            "bleu": mean("bleu"),
            "chrf": mean("chrf"),
            "ter": mean("ter"),
        })
    converged = [r["converged_at_round"] for r in rows if r["converged_at_round"] is not None]
    summary.append({
        "variant": "_meta",
        "n": len(rows),
        "converged_fraction": len(converged) / len(rows),
        "mean_convergence_round": statistics.mean(converged) if converged else None,
        "mean_rounds_run": statistics.mean(r["n_rounds_run"] for r in rows),
    })
    return summary
