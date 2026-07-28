"""Model-agnostic translation and two-layer scoring orchestration."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from medmt_eval.metrics.neural import CometScorer
from medmt_eval.metrics.surface import SurfaceScores, score_surface
from medmt_eval.models.base import Translator
from medmt_eval.schema import Segment, SegmentEvaluation
from medmt_eval.taxonomy.clinical import ClinicalSafetyEvaluator


def _single_direction(segments: Sequence[Segment]) -> tuple[str, str]:
    directions = {(segment.src_lang, segment.tgt_lang) for segment in segments}
    if len(directions) != 1:
        raise ValueError(
            "A translation invocation may contain one direction only; split a mixed-direction corpus first."
        )
    return next(iter(directions))


def translate_segments(translator: Translator, segments: Sequence[Segment]) -> list[dict[str, Any]]:
    """Translate validated segments and retain all source/reference provenance."""
    if not segments:
        return []
    src_lang, tgt_lang = _single_direction(segments)
    hypotheses = translator.translate([segment.src_text for segment in segments], src_lang, tgt_lang)
    if len(hypotheses) != len(segments):
        raise RuntimeError(
            f"Translator {translator.name!r} returned {len(hypotheses)} outputs for {len(segments)} inputs."
        )
    generation = translator.generation_config
    output: list[dict[str, Any]] = []
    for segment, hypothesis in zip(segments, hypotheses):
        row = segment.to_dict()
        row.update({"hyp_text": hypothesis, "model": translator.name, "generation": generation})
        output.append(row)
    return output


def evaluate_hypotheses(
    segments: Sequence[Segment],
    hypotheses: Sequence[str],
    *,
    model: str = "unknown",
    generation: dict[str, Any] | None = None,
    safety_evaluator: ClinicalSafetyEvaluator | None = None,
    comet: CometScorer | None = None,
    comet_batch_size: int = 8,
    comet_gpus: int = 0,
) -> tuple[list[SegmentEvaluation], dict[str, Any]]:
    """Apply surface/neural and clinical-loss scoring to aligned hypotheses."""
    if len(segments) != len(hypotheses):
        raise ValueError("Segments and hypotheses must have the same length.")
    if not segments:
        raise ValueError("Cannot evaluate an empty corpus.")
    safety_evaluator = safety_evaluator or ClinicalSafetyEvaluator()
    all_have_references = all(segment.ref_text is not None for segment in segments)
    surface: SurfaceScores | None = None
    if all_have_references:
        surface = score_surface(list(hypotheses), [str(segment.ref_text) for segment in segments])
    neural_scores = None
    if comet is not None:
        neural_scores = comet.score(
            [segment.src_text for segment in segments],
            hypotheses,
            [segment.ref_text for segment in segments] if all_have_references else None,
            batch_size=comet_batch_size,
            gpus=comet_gpus,
        )

    evaluations: list[SegmentEvaluation] = []
    for index, (segment, hypothesis) in enumerate(zip(segments, hypotheses)):
        metrics: dict[str, float | None] = {}
        if surface is not None:
            metrics.update(surface.sentence[index])
        if neural_scores is not None:
            metrics["comet"] = neural_scores.segment_scores[index]
        findings = safety_evaluator.evaluate(segment.src_text, hypothesis, segment.src_lang, segment.tgt_lang)
        evaluations.append(
            SegmentEvaluation(
                segment=segment,
                hyp_text=hypothesis,
                model=model,
                metrics=metrics,
                findings=findings,
                generation=generation or {},
            )
        )

    summary: dict[str, Any] = _summary(evaluations)
    summary["model"] = model
    summary["n_segments"] = len(evaluations)
    summary["has_references"] = all_have_references
    if surface is not None:
        summary["surface"] = {**surface.corpus, "signatures": surface.signatures}
    if neural_scores is not None:
        summary["neural"] = {
            "comet": neural_scores.system_score,
            "checkpoint": neural_scores.checkpoint,
            "reference_free": neural_scores.reference_free,
        }
    return evaluations, summary


def _summary(evaluations: Sequence[SegmentEvaluation]) -> dict[str, Any]:
    count = len(evaluations)
    error_segments = Counter()
    finding_counts = Counter()
    critical = 0
    severity_counts = Counter()
    for evaluation in evaluations:
        codes = {finding.code for finding in evaluation.findings}
        error_segments.update(codes)
        finding_counts.update(finding.code for finding in evaluation.findings)
        severity_counts.update(finding.severity for finding in evaluation.findings)
        critical += int(evaluation.has_critical_error)
    return {
        "clinical": {
            "critical_error_segments": critical,
            "critical_error_rate": critical / count if count else 0.0,
            "error_segment_rates": {code: value / count for code, value in sorted(error_segments.items())},
            "finding_counts": dict(sorted(finding_counts.items())),
            "severity_counts": dict(sorted(severity_counts.items())),
        }
    }
