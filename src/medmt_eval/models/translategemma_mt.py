"""Google TranslateGemma adapter using the processor's apply_chat_template.

TranslateGemma uses a specialized chat template where each user message
contains ``source_lang_code`` and ``target_lang_code`` fields alongside
the text.  It requires ``AutoProcessor`` (not just ``AutoTokenizer``)
and ``AutoModelForImageTextToText``.

Usage example from the model card::

    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "source_lang_code": "en",
            "target_lang_code": "de",
            "text": "No pleural effusion.",
        }]
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
"""

from __future__ import annotations

from typing import Any

from medmt_eval.models.base import GenerationConfig, Translator, strip_thinking
from medmt_eval.schema import normalise_language


def _imports() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as error:
        raise RuntimeError(
            "TranslateGemma adapter requires `pip install -e '.[mt]'`."
        ) from error
    return torch, AutoProcessor, AutoModelForImageTextToText


class TranslateGemmaTranslator(Translator):
    """Google TranslateGemma adapter with the official chat template."""

    name = "translategemma"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        config: GenerationConfig | None = None,
        **_kwargs: Any,
    ) -> None:
        self.model_id = model_id or "google/translategemma-4b-it"
        self._config = config or GenerationConfig(batch_size=1)
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        torch, AutoProcessor, AutoModelForImageTextToText = _imports()
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto" if self._config.device is None else None,
        )
        if self._config.device:
            self._model.to(self._config.device)
        self._device = next(self._model.parameters()).device
        self._model.eval()

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        self._load()
        assert self._processor is not None and self._model is not None and self._torch is not None
        source_code = normalise_language(src_lang)
        target_code = normalise_language(tgt_lang)
        results: list[str] = []
        for text in texts:
            messages = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "source_lang_code": source_code,
                    "target_lang_code": target_code,
                    "text": text,
                }],
            }]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._device, dtype=self._torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]
            with self._torch.inference_mode():
                generation = self._model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self._config.max_new_tokens,
                )
            decoded = self._processor.decode(
                generation[0][input_len:], skip_special_tokens=True
            )
            results.append(strip_thinking(decoded))
        return results

    @property
    def generation_config(self) -> dict[str, object]:
        return {
            "adapter": self.name,
            "model_id": self.model_id,
            **self._config.to_dict(),
        }
