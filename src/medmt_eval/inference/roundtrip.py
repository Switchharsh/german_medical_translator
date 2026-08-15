"""Iterative round-trip translation: DE → EN → DE → EN … , scored every step.

This is *not* the discredited use of back-translation as a stand-in for a
reference. Every step here is scored against a fixed anchor that never changes:

    English outputs  ->  the human English reference
    German outputs   ->  the original German source

So what accumulates across cycles is measured drift from ground truth, not a
model agreeing with itself. Round 1's English output is exactly the ordinary
single-pass evaluation, which makes the first point of every curve directly
comparable to the main benchmark.

Two evaluations are recorded per step:

``clinical_vs_origin``
    Detector findings comparing the **original German** against the current
    output. For English outputs this is the usual de→en check; for German
    outputs it is de→de, which the negation, laterality and number detectors
    all support. This is the cumulative-loss signal.

``clinical_vs_input``
    Findings for that hop alone — the text that went in versus the text that
    came out. Isolates which step introduced a given error.

Direction-specific models (Opus ships one checkpoint per language pair, and its
adapter refuses to switch direction on a live instance) are handled by taking a
factory rather than a translator: one instance is built per direction and reused
across all cycles.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from medmt_eval.metrics.surface import score_surface
from medmt_eval.models.base import Translator
from medmt_eval.schema import Segment
from medmt_eval.taxonomy.clinical import ClinicalSafetyEvaluator

TranslatorFactory = Callable[[str, str], Translator]


def _surface_for(hypotheses: Sequence[str], references: Sequence[str | None]) -> list[dict[str, float]]:
    """Per-segment surface metrics, or empty dicts when references are absent."""
    if any(reference is None for reference in references):
        return [{} for _ in hypotheses]
    scores = score_surface(list(hypotheses), [str(reference) for reference in references])
    return list(scores.sentence)


def run_roundtrip(
    factory: TranslatorFactory,
    segments: Sequence[Segment],
    *,
    cycles: int = 10,
    safety_evaluator: ClinicalSafetyEvaluator | None = None,
    chunk_max_tokens: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Translate back and forth for ``cycles`` cycles, scoring after every step.

    One cycle is two steps: source-language → English, then English → back.
    ``cycles=10`` therefore performs 20 translation passes over the corpus.

    Returns one row per (segment, step). Row 1 of each segment corresponds to the
    standard single-pass evaluation.
    """
    if not segments:
        return []
    if cycles < 1:
        raise ValueError("cycles must be at least 1.")

    directions = {(segment.src_lang, segment.tgt_lang) for segment in segments}
    if len(directions) != 1:
        raise ValueError("Round-trip requires a single translation direction.")
    source_lang, target_lang = next(iter(directions))

    evaluator = safety_evaluator or ClinicalSafetyEvaluator()

    # Built once and reused: constructing a translator loads model weights.
    # Only build a second instance when the adapter is pinned to one direction
    # (Opus). Loading two copies of every other model doubled GPU memory and
    # OOM'd a 14 GB model on a 20 GB slice.
    forward = factory(source_lang, target_lang)
    backward = (
        factory(target_lang, source_lang) if forward.direction_specific else forward
    )

    origin_texts = [segment.src_text for segment in segments]
    english_reference = [segment.ref_text for segment in segments]

    current = list(origin_texts)
    current_lang = source_lang
    rows: list[dict[str, Any]] = []

    for step in range(1, cycles * 2 + 1):
        into_english = current_lang == source_lang
        translator = forward if into_english else backward
        from_lang, to_lang = (
            (source_lang, target_lang) if into_english else (target_lang, source_lang)
        )
        if progress:
            progress(f"step {step}/{cycles * 2}  {from_lang}->{to_lang}")

        previous = list(current)
        current = _translate(translator, previous, from_lang, to_lang, chunk_max_tokens)
        current_lang = to_lang

        # Anchor: English steps score against the human reference; German steps
        # score against the untouched original.
        anchor = english_reference if into_english else origin_texts
        surface = _surface_for(current, anchor)

        for index, segment in enumerate(segments):
            origin_findings = evaluator.evaluate(
                origin_texts[index], current[index], source_lang, current_lang
            )
            hop_findings = evaluator.evaluate(
                previous[index], current[index], from_lang, to_lang
            )
            rows.append(
                {
                    "id": segment.id,
                    "domain": segment.domain,
                    "cycle": (step + 1) // 2,
                    "step": step,
                    "direction": f"{from_lang}->{to_lang}",
                    "src_text": origin_texts[index],
                    "ref_text": english_reference[index],
                    "input_text": previous[index],
                    "hyp_text": current[index],
                    "metrics": surface[index],
                    "findings": [finding.to_dict() for finding in origin_findings],
                    "has_critical_error": any(
                        finding.severity == "critical" for finding in origin_findings
                    ),
                    "hop_findings": [finding.to_dict() for finding in hop_findings],
                    "hop_has_critical_error": any(
                        finding.severity == "critical" for finding in hop_findings
                    ),
                    "model": translator.name,
                    "generation": translator.generation_config,
                }
            )
    return rows


def _translate(
    translator: Translator,
    texts: Sequence[str],
    src_lang: str,
    tgt_lang: str,
    chunk_max_tokens: int | None,
) -> list[str]:
    """One translation pass, chunking when the model cannot take whole documents."""
    if chunk_max_tokens:
        from medmt_eval.data.chunking import chunk_documents, reassemble

        chunks, counts = chunk_documents(list(texts), max_tokens=chunk_max_tokens)
        translated = translator.translate(chunks, src_lang, tgt_lang)
        if len(translated) != len(chunks):
            raise RuntimeError(
                f"Translator {translator.name!r} returned {len(translated)} outputs "
                f"for {len(chunks)} chunks."
            )
        return reassemble(translated, counts)
    result = translator.translate(list(texts), src_lang, tgt_lang)
    if len(result) != len(texts):
        raise RuntimeError(
            f"Translator {translator.name!r} returned {len(result)} outputs for {len(texts)} inputs."
        )
    return list(result)


def summarise_by_step(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-step rows into one record per step: the degradation curve."""
    import collections
    import statistics

    grouped: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[int(row["step"])].append(row)

    summary: list[dict[str, Any]] = []
    for step in sorted(grouped):
        group = grouped[step]
        codes: collections.Counter[str] = collections.Counter()
        for row in group:
            for finding in row.get("findings") or []:
                codes[finding["code"]] += 1

        def mean_metric(name: str) -> float | None:
            values = [
                value
                for row in group
                for key, value in (row.get("metrics") or {}).items()
                if key == name and isinstance(value, (int, float))
            ]
            return statistics.mean(values) if values else None

        summary.append(
            {
                "step": step,
                "cycle": group[0]["cycle"],
                "direction": group[0]["direction"],
                "n_segments": len(group),
                "critical_error_rate": sum(
                    1 for row in group if row.get("has_critical_error")
                )
                / len(group),
                "bleu": mean_metric("bleu"),
                "chrf": mean_metric("chrf"),
                "ter": mean_metric("ter"),
                "finding_counts": dict(sorted(codes.items())),
                "mean_output_chars": statistics.mean(len(row["hyp_text"]) for row in group),
            }
        )
    return summary
