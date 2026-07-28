"""Dependency-light paired bootstrap utilities for reproducible comparisons."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile of an empty sequence.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass(frozen=True)
class BootstrapResult:
    observed_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    resamples: int
    seed: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def paired_bootstrap_metric(
    baseline: Sequence[str],
    candidate: Sequence[str],
    references: Sequence[str],
    metric: Callable[[Sequence[str], Sequence[str]], float],
    *,
    resamples: int = 2000,
    seed: int = 13,
) -> BootstrapResult:
    """Resample aligned segments and return candidate-minus-baseline CI.

    ``metric`` receives an entire resampled corpus, so corpus BLEU and TER keep
    their correct non-additive definitions. The p-value is a two-sided bootstrap
    sign estimate; report it as an exploratory comparison, not a randomisation test.
    """
    size = len(baseline)
    if size == 0 or len(candidate) != size or len(references) != size:
        raise ValueError("Baseline, candidate, and references must be non-empty and aligned.")
    if resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples.")
    observed = metric(candidate, references) - metric(baseline, references)
    generator = random.Random(seed)
    differences: list[float] = []
    for _ in range(resamples):
        indices = [generator.randrange(size) for _ in range(size)]
        candidate_sample = [candidate[index] for index in indices]
        baseline_sample = [baseline[index] for index in indices]
        reference_sample = [references[index] for index in indices]
        differences.append(metric(candidate_sample, reference_sample) - metric(baseline_sample, reference_sample))
    non_positive = sum(value <= 0 for value in differences) / resamples
    non_negative = sum(value >= 0 for value in differences) / resamples
    return BootstrapResult(
        observed_difference=observed,
        ci_low=_quantile(differences, 0.025),
        ci_high=_quantile(differences, 0.975),
        p_value=min(1.0, 2 * min(non_positive, non_negative)),
        resamples=resamples,
        seed=seed,
    )


def bootstrap_proportion_ci(
    values: Sequence[bool | int], *, resamples: int = 2000, seed: int = 13
) -> tuple[float, float, float]:
    """Return observed proportion and its percentile 95% bootstrap CI."""
    if not values:
        raise ValueError("Cannot bootstrap an empty list.")
    generator = random.Random(seed)
    numeric = [int(bool(value)) for value in values]
    size = len(numeric)
    samples = [
        sum(numeric[generator.randrange(size)] for _ in range(size)) / size for _ in range(resamples)
    ]
    return sum(numeric) / size, _quantile(samples, 0.025), _quantile(samples, 0.975)


def mcnemar_exact(baseline_errors: Sequence[bool], candidate_errors: Sequence[bool]) -> dict[str, int | float]:
    """Exact two-sided McNemar test for paired critical-error incidence."""
    if len(baseline_errors) != len(candidate_errors) or not baseline_errors:
        raise ValueError("Paired non-empty error indicators are required.")
    baseline_only = sum(bool(base) and not bool(candidate) for base, candidate in zip(baseline_errors, candidate_errors))
    candidate_only = sum(not bool(base) and bool(candidate) for base, candidate in zip(baseline_errors, candidate_errors))
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower_tail = sum(math.comb(discordant, index) for index in range(0, min(baseline_only, candidate_only) + 1))
        p_value = min(1.0, 2 * lower_tail / (2**discordant))
    return {
        "baseline_only_errors": baseline_only,
        "candidate_only_errors": candidate_only,
        "discordant_pairs": discordant,
        "p_value": p_value,
    }
