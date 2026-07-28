"""sacreBLEU metrics with explicit, persisted signatures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SurfaceScores:
    corpus: dict[str, float]
    signatures: dict[str, str]
    sentence: list[dict[str, float]]


def _metric_objects() -> tuple[Any, Any, Any]:
    try:
        from sacrebleu.metrics import BLEU, CHRF, TER
    except ImportError as error:  # pragma: no cover - core dependency missing
        raise RuntimeError("Surface scoring requires sacrebleu; run `pip install -e .`.") from error
    # chrF++ is chrF with word n-grams enabled (word_order=2).
    return (
        BLEU(tokenize="13a", effective_order=True),
        CHRF(word_order=2),
        TER(normalized=False, no_punct=False, asian_support=False),
    )


def _validate(hypotheses: Sequence[str], references: Sequence[str]) -> None:
    if not hypotheses:
        raise ValueError("Cannot score an empty corpus.")
    if len(hypotheses) != len(references):
        raise ValueError("Hypotheses and references must have the same length.")
    if any(reference is None for reference in references):
        raise ValueError("Reference-based surface metrics require ref_text for every segment.")


def score_surface(hypotheses: Sequence[str], references: Sequence[str]) -> SurfaceScores:
    """Compute corpus and sentence sacreBLEU, chrF++, and TER scores."""
    _validate(hypotheses, references)
    bleu, chrf, ter = _metric_objects()
    reference_streams = [list(references)]
    corpus = {
        "bleu": float(bleu.corpus_score(list(hypotheses), reference_streams).score),
        "chrf": float(chrf.corpus_score(list(hypotheses), reference_streams).score),
        "ter": float(ter.corpus_score(list(hypotheses), reference_streams).score),
    }
    sentence = [
        {
            "bleu": float(bleu.sentence_score(hypothesis, [reference]).score),
            "chrf": float(chrf.sentence_score(hypothesis, [reference]).score),
            "ter": float(ter.sentence_score(hypothesis, [reference]).score),
        }
        for hypothesis, reference in zip(hypotheses, references)
    ]
    return SurfaceScores(
        corpus=corpus,
        signatures={
            "bleu": f"BLEU|{bleu.get_signature().format()}",
            "chrf": f"chrF++|{chrf.get_signature().format()}",
            "ter": f"TER|{ter.get_signature().format()}",
        },
        sentence=sentence,
    )


def corpus_metric(name: str, hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Compute one reference-based corpus metric for bootstrap resampling."""
    scores = score_surface(hypotheses, references).corpus
    try:
        return scores[name.lower()]
    except KeyError as error:
        raise ValueError("Metric must be one of bleu, chrf, or ter.") from error
