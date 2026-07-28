"""Batch translation and end-to-end evaluation orchestration."""

from .runner import evaluate_hypotheses, translate_segments

__all__ = ["evaluate_hypotheses", "translate_segments"]
