"""Small common interface for all MT back ends."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove a leaked <think>...</think> reasoning block, if present.

    Some instruction-tuned models (e.g. Qwen3.5) default to a "thinking"
    mode that wraps a reasoning trace in <think> tags before the actual
    answer. Adapters pass enable_thinking=False to the chat template where
    supported, but this is a second line of defense for templates/models
    that emit the tag anyway (unclosed tags are left as-is rather than
    guessed-at, since truncating on an unclosed tag risks losing an
    otherwise-cut-off legitimate translation).
    """
    return _THINK_BLOCK.sub("", text).strip()


@dataclass(frozen=True)
class GenerationConfig:
    batch_size: int = 8
    num_beams: int = 4
    max_input_tokens: int = 512
    max_new_tokens: int = 512
    device: str | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


class Translator(ABC):
    """A direction-aware machine translation implementation."""

    name: str

    @abstractmethod
    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Translate a non-empty list, preserving its order."""

    @property
    @abstractmethod
    def generation_config(self) -> dict[str, object]:
        """Return reproducibility metadata persisted with each result."""


class IdentityTranslator(Translator):
    """Explicit no-download adapter for smoke tests and pipeline demos only."""

    name = "identity"

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self._config = config or GenerationConfig()

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        if src_lang == tgt_lang:
            raise ValueError("IdentityTranslator is not a same-language translation shortcut.")
        return list(texts)

    @property
    def generation_config(self) -> dict[str, object]:
        return {"adapter": "identity", **self._config.to_dict()}
