"""Optional COMET-family integration; checkpoint use is always explicit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class NeuralScores:
    system_score: float
    segment_scores: list[float]
    checkpoint: str
    reference_free: bool


class CometScorer:
    """Score translations with a supplied COMET/XCOMET checkpoint identifier.

    Examples: ``Unbabel/wmt22-comet-da`` (reference based),
    ``Unbabel/wmt22-cometkiwi-da`` (reference free, gated), and XCOMET checkpoints.
    """

    def __init__(self, checkpoint: str, *, reference_free: bool = False) -> None:
        self.checkpoint = checkpoint
        self.reference_free = reference_free
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from comet import download_model, load_from_checkpoint
        except ImportError as error:  # pragma: no cover - optional integration
            raise RuntimeError("COMET scoring requires `pip install -e '.[neural]'`.") from error
        self._model = load_from_checkpoint(download_model(self.checkpoint))
        return self._model

    def score(
        self,
        sources: Sequence[str],
        hypotheses: Sequence[str],
        references: Sequence[str | None] | None = None,
        *,
        batch_size: int = 8,
        gpus: int = 0,
    ) -> NeuralScores:
        if len(sources) != len(hypotheses):
            raise ValueError("Sources and hypotheses must have the same length.")
        if not self.reference_free:
            if references is None or len(references) != len(sources) or any(ref is None for ref in references):
                raise ValueError("Reference-based COMET needs a reference for every segment.")
        records = [
            {
                "src": source,
                "mt": hypothesis,
                **({} if self.reference_free else {"ref": str(references[index])}),
            }
            for index, (source, hypothesis) in enumerate(zip(sources, hypotheses))
        ]
        prediction = self._load().predict(records, batch_size=batch_size, gpus=gpus)
        return NeuralScores(
            system_score=float(prediction.system_score),
            segment_scores=[float(score) for score in prediction.scores],
            checkpoint=self.checkpoint,
            reference_free=self.reference_free,
        )
