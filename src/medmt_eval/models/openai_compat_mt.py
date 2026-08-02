"""Adapter for any OpenAI-compatible ``/v1/chat/completions`` endpoint.

Vendor-neutral by design: the model is whatever ``model_id`` names, and the
adapter records what the endpoint *says* actually answered.

That last point is not theoretical. The gateway this was built against
(``api.hcnsec.cn``) was observed serving ``nvidia/nemotron-3-ultra-550b-a55b``
in response to two separate ``DeepSeek-V4-Pro`` requests, with no error — while
``glm-5.2`` correctly returned ``z-ai/glm-5.2``. So routing honesty varies per
alias on the same gateway, and the ``/v1/models`` catalogue is not reliable
either (every entry there claims ``"owned_by": "openai"``).

The only trustworthy signal is the ``model`` field of an actual response, which
``_check_served_model`` verifies on every call.

Design notes:

* **Small numbered batches, not one mega-request.** The context window is large
  enough to hold the whole corpus, but packing hundreds of reports into a single
  call risks long-context recall degradation on later items, and one malformed
  or dropped item destroys alignment for everything after it. Batches of ~8 keep
  a failure cheap and the per-item quality closer to the single-segment
  translations every other model in this benchmark produced.
* **Concurrency is where the speedup comes from.** Requests are issued from a
  thread pool, so wall-clock is roughly ``n_batches / max_workers`` request
  round-trips rather than ``n_batches``. Observed generation rate on this
  gateway was ~27 tokens/s (9.5 s for a 258-token reply), so a batch of eight
  radiology reports should land in the tens of seconds.

Batch responses are parsed back by index. Any batch whose response cannot be
parsed cleanly is retried one item at a time, so a formatting failure costs
extra time rather than silently misaligning the output.
"""

from __future__ import annotations

import os
import re
import threading as _threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests as _requests

from medmt_eval.models.base import GenerationConfig, Translator, strip_thinking
from medmt_eval.schema import normalise_language

_DEFAULT_BASE_URL = "https://api.hcnsec.cn/v1/chat/completions"
_DEFAULT_MODEL = "glm-5.2"

_LANGUAGE_NAMES = {"en": "English", "de": "German"}

# Generous: this endpoint has been observed taking ~3 minutes per request.
_DEFAULT_TIMEOUT = 900

_SINGLE_TEMPLATE = (
    "Translate the following medical text from {source_lang} to {target_lang}. "
    "Return only the translation, with no explanation, no preamble, and no quotation marks.\n\n"
    "{text}"
)

_BATCH_HEADER = (
    "Translate each of the following {n} medical texts from {source_lang} to "
    "{target_lang}.\n\n"
    "Rules:\n"
    "- Output exactly {n} translations.\n"
    "- Prefix each translation with its number in the form '[[k]]' on its own line, "
    "using the same numbering as the input.\n"
    "- Do not merge, split, reorder, or omit any item.\n"
    "- Output only the translations. No commentary, no preamble, no repetition of the source.\n\n"
)

# Matches the '[[k]]' item markers used to split a batch response.
_ITEM_MARKER = re.compile(r"\[\[\s*(\d+)\s*\]\]")


def _get_api_key() -> str:
    key = os.environ.get("OPENAI_COMPAT_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise RuntimeError(
            "This adapter requires an API key. Set OPENAI_COMPAT_API_KEY (or "
            "DEEPSEEK_API_KEY) in the environment, or pass api_key= to the constructor."
        )
    return key


def build_batch_prompt(texts: list[str], src_lang: str, tgt_lang: str) -> str:
    """Render a numbered multi-item translation prompt (pure function, testable)."""
    source_name = _LANGUAGE_NAMES[normalise_language(src_lang)]
    target_name = _LANGUAGE_NAMES[normalise_language(tgt_lang)]
    header = _BATCH_HEADER.format(n=len(texts), source_lang=source_name, target_lang=target_name)
    body = "\n\n".join(f"[[{index + 1}]]\n{text}" for index, text in enumerate(texts))
    return header + body


def parse_batch_response(content: str, expected: int) -> list[str] | None:
    """Split a numbered batch response into ``expected`` items.

    Returns None when the response cannot be parsed into exactly the expected
    number of correctly-numbered items, so the caller can fall back to
    one-at-a-time translation rather than emitting misaligned output.
    """
    matches = list(_ITEM_MARKER.finditer(content))
    if len(matches) != expected:
        return None
    if [int(match.group(1)) for match in matches] != list(range(1, expected + 1)):
        return None
    items: list[str] = []
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(content)
        items.append(content[match.end():end].strip())
    if any(not item for item in items):
        return None
    return items


class OpenAICompatTranslator(Translator):
    """Chat-completions translation adapter with numbered batching and concurrency."""

    name = "openai-compat"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        config: GenerationConfig | None = None,
        max_workers: int | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        temperature: float = 0.0,
        strict_model: bool | None = None,
        **_kwargs: Any,
    ) -> None:
        self.model_id = model_id or _DEFAULT_MODEL
        self.base_url = base_url or os.environ.get("OPENAI_COMPAT_BASE_URL") or _DEFAULT_BASE_URL
        self._config = config or GenerationConfig(batch_size=8)
        self._resolved_key: str | None = api_key
        self._timeout = timeout
        # Greedy by default so runs are reproducible, matching the local models.
        self._temperature = temperature
        # Whatever the endpoint reports as the serving model, collected across
        # all requests and written into generation_config so results are never
        # attributed to a model that did not produce them.
        self._served_models: set[str] = set()
        self._served_lock = _threading.Lock()
        if strict_model is None:
            strict_model = os.environ.get("ALLOW_MODEL_SUBSTITUTION", "") not in {"1", "true", "yes"}
        self._strict_model = strict_model
        # Concurrency is the real lever on a ~3 min/request endpoint. Kept well
        # under the documented 830 RPM ceiling.
        self._max_workers = max_workers or int(os.environ.get("OPENAI_COMPAT_MAX_WORKERS", "8"))

    def _key(self) -> str:
        if self._resolved_key is None:
            self._resolved_key = _get_api_key()
        return self._resolved_key

    def _complete(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
        }
        if self._config.max_new_tokens:
            # Batches return several full reports, so scale the ceiling with the
            # batch size rather than using the per-segment value directly.
            payload["max_tokens"] = self._config.max_new_tokens * max(1, self._config.batch_size)
        response = _requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        self._check_served_model(data.get("model"))
        # Reasoning models on this gateway return their chain of thought in a
        # separate `reasoning_content` field; only `content` is the answer.
        return str(data["choices"][0]["message"]["content"])

    @staticmethod
    def _same_model(requested: str, served: str) -> bool:
        """Is `served` the same model as `requested`, allowing for namespacing?

        Endpoints commonly answer an unqualified alias with its fully-qualified
        name — asking for ``glm-5.2`` yields ``z-ai/glm-5.2``, and
        ``DeepSeek-V4-Flash`` yields ``deepseek-ai/deepseek-v4-flash``. Those are
        the same model and must not be flagged.

        A genuine substitution looks different: ``DeepSeek-V4-Pro`` answered by
        ``nvidia/nemotron-3-ultra-550b-a55b``, where the final path component
        does not match at all. Comparing only the component after the last "/"
        separates the two cases.
        """
        def leaf(name: str) -> str:
            return name.strip().lower().rsplit("/", 1)[-1]

        return leaf(requested) == leaf(served)

    def _check_served_model(self, served: Any) -> None:
        """Record which model actually answered, and refuse a silent substitution.

        This gateway has been observed returning a completely different model
        from the one requested (asking for "DeepSeek-V4-Pro" was served
        "nvidia/nemotron-3-ultra-550b-a55b"; "Kimi-K2.6" was served
        "thinkingmachines/inkling") with no error. Left unchecked, the benchmark
        would label results with the requested name while measuring something
        else entirely, which is far worse than a failed run.
        """
        if not served:
            return
        served = str(served)
        with self._served_lock:
            self._served_models.add(served)
        if self._strict_model and not self._same_model(self.model_id, served):
            raise RuntimeError(
                f"Endpoint served model {served!r} but {self.model_id!r} was requested. "
                f"Results would be mislabelled. Set strict_model=False (or "
                f"ALLOW_MODEL_SUBSTITUTION=1) to record the served model and "
                f"continue anyway."
            )

    def _translate_one(self, text: str, src_lang: str, tgt_lang: str) -> str:
        source_name = _LANGUAGE_NAMES[normalise_language(src_lang)]
        target_name = _LANGUAGE_NAMES[normalise_language(tgt_lang)]
        prompt = _SINGLE_TEMPLATE.format(
            source_lang=source_name, target_lang=target_name, text=text
        )
        return strip_thinking(self._complete(prompt))

    def _translate_batch(self, batch: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        if len(batch) == 1:
            return [self._translate_one(batch[0], src_lang, tgt_lang)]
        try:
            content = self._complete(build_batch_prompt(batch, src_lang, tgt_lang))
            items = parse_batch_response(strip_thinking(content), len(batch))
            if items is not None:
                return [strip_thinking(item) for item in items]
        except Exception:  # noqa: BLE001 - fall back rather than lose the batch
            pass
        # Unparseable or failed batch: redo it one item at a time so a
        # formatting glitch costs time instead of corrupting alignment.
        return [self._translate_one(text, src_lang, tgt_lang) for text in batch]

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        batch_size = max(1, self._config.batch_size)
        batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]
        if not batches:
            return []
        workers = max(1, min(self._max_workers, len(batches)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # map preserves input order, so results stay aligned with `texts`.
            results = list(
                pool.map(lambda batch: self._translate_batch(batch, src_lang, tgt_lang), batches)
            )
        flattened = [item for batch_result in results for item in batch_result]
        if len(flattened) != len(texts):
            raise RuntimeError(
                f"Adapter returned {len(flattened)} translations for {len(texts)} inputs."
            )
        return flattened

    @property
    def generation_config(self) -> dict[str, object]:
        with self._served_lock:
            served = sorted(self._served_models)
        return {
            "adapter": self.name,
            # What was asked for...
            "model_id": self.model_id,
            # ...and what the endpoint said actually answered. These can differ:
            # this gateway substitutes models silently, so the served value is
            # the one to trust when attributing results.
            "served_models": served,
            "model_substitution": bool(served) and any(
                not self._same_model(self.model_id, s) for s in served
            ),
            # Recorded because the gateway, not the model name, determines what
            # actually served the request.
            "base_url": self.base_url,
            "strict_model": self._strict_model,
            "temperature": self._temperature,
            "max_workers": self._max_workers,
            "timeout_s": self._timeout,
            **self._config.to_dict(),
        }
