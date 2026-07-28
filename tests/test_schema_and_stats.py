from medmt_eval.schema import Segment
from medmt_eval.stats.bootstrap import bootstrap_proportion_ci, mcnemar_exact, paired_bootstrap_metric


def test_segment_aliases_and_reverse() -> None:
    segment = Segment.from_mapping(
        {"id": "x", "source": "No effusion.", "reference": "Kein Erguss.", "src_lang": "EN", "tgt_lang": "DE"}
    )
    assert segment.src_lang == "en"
    assert segment.reversed().src_text == "Kein Erguss."


def test_bootstrap_is_reproducible() -> None:
    score = lambda hypotheses, references: sum(hypothesis == reference for hypothesis, reference in zip(hypotheses, references))
    first = paired_bootstrap_metric(["a", "x"], ["a", "b"], ["a", "b"], score, resamples=100, seed=4)
    second = paired_bootstrap_metric(["a", "x"], ["a", "b"], ["a", "b"], score, resamples=100, seed=4)
    assert first == second
    observed, low, high = bootstrap_proportion_ci([True, False, True], resamples=100, seed=2)
    assert observed == 2 / 3
    assert 0 <= low <= high <= 1


def test_mcnemar_handles_no_discordant_pairs() -> None:
    result = mcnemar_exact([True, False], [True, False])
    assert result["p_value"] == 1.0
