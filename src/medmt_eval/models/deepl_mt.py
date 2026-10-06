"""DeepL API adapter for the medmt-eval Translator interface.

Requires the ``deepl`` extra (``pip install -e '.[deepl]'``), which brings in
the ``requests`` library.  The adapter hits the DeepL ``/v2/translate``
endpoint (free or paid tier) and is registered in the factory as ``deepl``.

A DeepL API key is required — set ``DEEPL_API_KEY`` (or ``DEEPL_AUTH_KEY``) or
pass ``api_key`` to the constructor. The endpoint is chosen from the key: DeepL
free-plan keys end in ``:fx`` and are only valid on ``api-free.deepl.com``.

DeepL is a hosted service with no GPU-node network access here, so it runs from
the login node.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

import requests as _requests

from medmt_eval.models.base import GenerationConfig, Translator
from medmt_eval.schema import normalise_language

# Source codes are bare; target English must name a variant. The bare "EN" target
# is deprecated. It currently resolves to US spelling (checked: identical to
# EN-US, and PARROT's references are 78:0 US:UK), but that is DeepL's to change.
_DEEPL_SOURCE_MAP = {"en": "EN", "de": "DE"}
_DEEPL_TARGET_MAP = {"en": "EN-US", "de": "DE"}

# 429 = slow down. 456 = monthly character quota exhausted: retrying cannot help
# and burns the caller's time, so it is fatal.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504, 529})
_QUOTA_EXCEEDED = 456
_MAX_ATTEMPTS = 6

# Free-tier vs. paid-tier endpoint.
_FREE_API = "https://api-free.deepl.com/v2/translate"
_PAID_API = "https://api.deepl.com/v2/translate"


_KEY_ENV_VARS = ("DEEPL_API_KEY", "DEEPL_AUTH_KEY")


def _get_api_key() -> str:
    for name in _KEY_ENV_VARS:
        key = os.environ.get(name, "").strip()
        if key:
            return key
    raise RuntimeError(
        "DeepL adapter requires an API key. Set one of "
        f"{' / '.join(_KEY_ENV_VARS)} or pass api_key= to the constructor."
    )


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
        # The key decides, not the flag: a ":fx" key on the paid host (or the
        # reverse) is a 403 that reads like a bad key.
        return _FREE_API if self._key().endswith(":fx") else _PAID_API

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = _requests.post(
                    self._endpoint,
                    headers={
                        "Authorization": f"DeepL-Auth-Key {self._key()}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60,
                )
            except (_requests.ConnectionError, _requests.Timeout):
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
                time.sleep(min(60.0, 2.0**attempt) + random.random())
                continue
            if response.status_code == _QUOTA_EXCEEDED:
                raise RuntimeError(
                    "DeepL quota exhausted (HTTP 456). The free plan allows "
                    "1,000,000 characters per month; check /v2/usage."
                )
            if response.status_code in _RETRY_STATUS and attempt < _MAX_ATTEMPTS - 1:
                wait = response.headers.get("Retry-After", "")
                delay = float(wait) if wait.replace(".", "", 1).isdigit() else 2.0**attempt
                time.sleep(min(60.0, delay) + random.random())
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("unreachable")

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Translate via the DeepL API, batching to stay under payload limits."""
        source = _DEEPL_SOURCE_MAP[normalise_language(src_lang)]
        target = _DEEPL_TARGET_MAP[normalise_language(tgt_lang)]
        # DeepL free tier has a 128 KB body limit; batch conservatively.
        batch_size = max(1, self._config.batch_size)
        translations: list[str] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            data = self._post({"text": batch, "source_lang": source, "target_lang": target})
            translations.extend(item["text"] for item in data["translations"])
        return translations

    @property
    def generation_config(self) -> dict[str, object]:
        return {
            "adapter": self.name,
            "free_tier": self._key().endswith(":fx"),
            "target_variant": _DEEPL_TARGET_MAP["en"],
            **self._config.to_dict(),
        }
