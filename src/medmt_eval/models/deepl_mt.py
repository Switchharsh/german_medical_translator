"""DeepL API adapter for the medmt-eval Translator interface.

Requires the ``deepl`` extra (``pip install -e '.[deepl]'``), which brings in
the ``requests`` library.  The adapter hits the DeepL ``/v2/translate``
endpoint (free or paid tier) and is registered in the factory as ``deepl``.

A DeepL API key is required — set it via the ``DEEPL_AUTH_KEY`` environment
variable or pass ``api_key`` to the constructor.
"""

from __future__ import annotations

import os
from typing import Any

import requests as _requests

from medmt_eval.models.base import GenerationConfig, Translator
from medmt_eval.schema import normalise_language

_DEEPL_LANG_MAP = {"en": "EN", "de": "DE"}

# Free-tier vs. paid-tier endpoint.
_FREE_API = "https://api-free.deepl.com/v2/translate"
_PAID_API = "https://api.deepl.com/v2/translate"


def _get_api_key() -> str:
    key = os.environ.get("DEEPL_AUTH_KEY", "")
    if not key:
        raise RuntimeError(
            "DeepL adapter requires an API key. "
            "Set the DEEPL_AUTH_KEY environment variable or pass api_key= to the constructor."
        )
    return key


class DeepLTranslator(Translator):
    """HTTP adapter for the DeepL translation API."""

    name = "deepl"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        free_tier: bool = True,
        config: GenerationConfig | None = None,
        **_kwargs: Any,
    ) -> None:
        self._api_key = api_key or _get_api_key
        self._free_tier = free_tier
        self._config = config or GenerationConfig()
        # Lazily resolved on first call.
        self._resolved_key: str | None = api_key

    def _key(self) -> str:
        if self._resolved_key is None:
            self._resolved_key = self._api_key() if callable(self._api_key) else str(self._api_key)
        return self._resolved_key

    @property
    def _endpoint(self) -> str:
        return _FREE_API if self._free_tier else _PAID_API

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Translate via the DeepL API, batching to stay under payload limits."""
        source = _DEEPL_LANG_MAP[normalise_language(src_lang)]
        target = _DEEPL_LANG_MAP[normalise_language(tgt_lang)]
        # DeepL free tier has a 128 KB body limit; batch conservatively.
        batch_size = max(1, self._config.batch_size)
        translations: list[str] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = _requests.post(
                self._endpoint,
                headers={
                    "Authorization": f"DeepL-Auth-Key {self._key()}",
                    "Content-Type": "application/json",
                },
                json={
                    "text": batch,
                    "source_lang": source,
                    "target_lang": target,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            translations.extend(item["text"] for item in data["translations"])
        return translations

    @property
    def generation_config(self) -> dict[str, object]:
        return {
            "adapter": self.name,
            "free_tier": self._free_tier,
            **self._config.to_dict(),
        }
