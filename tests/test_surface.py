from medmt_eval.metrics.surface import score_surface


def test_perfect_surface_scores_and_reproducibility_signature() -> None:
    scores = score_surface(["Kein Pleuraerguss."], ["Kein Pleuraerguss."])
    assert scores.corpus["bleu"] > 99.99
    assert scores.corpus["chrf"] == 100.0
    assert scores.corpus["ter"] == 0.0
    assert scores.signatures["bleu"].startswith("BLEU|")
