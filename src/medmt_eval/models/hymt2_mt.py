"""Tencent Hy-MT2 translation model adapter.

Hy-MT2 models (1.8B, 7B, 30B-A3B) use a chat-template-based prompt format.
The recommended English prompt is:

    Translate the following text into {target_lang}. Note that you should
    only output the translated result without any additional explanation:
    {source_text}

The model uses ``tokenizer.apply_chat_template`` with a standard
``[{"role": "user", "content": prompt}]`` message structure.

Default inference parameters (from the model card):
    temperature=0.7, top_p=0.6, top_k=20, repetition_penalty=1.05

We override temperature/top_p/top_k with greedy decoding (do_sample=False,
num_beams=1) for reproducibility, matching the other adapters in this pipeline.
"""

from __future__ import annotations

from typing import Any

from medmt_eval.models.base import GenerationConfig, Translator, strip_thinking
from medmt_eval.schema import normalise_language

_LANGUAGE_NAMES = {"en": "English", "de": "German"}

_DEFAULT_MODEL_ID = "tencent/Hy-MT2-1.8B"

_PROMPT_TEMPLATE = (
    "Translate the following text into {target_lang}. "
    "Note that you should only output the translated result "
    "without any additional explanation:\n\n{text}"
)


def _imports() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Hy-MT2 adapter requires `pip install -e '.[mt]'`."
        ) from error
    return torch, AutoTokenizer, AutoModelForCausalLM


def build_hymt2_prompt(text: str, tgt_lang: str) -> str:
    """Build the raw user-message content for Hy-MT2."""
    target_name = _LANGUAGE_NAMES[normalise_language(tgt_lang)]
    return _PROMPT_TEMPLATE.format(target_lang=target_name, text=text)


class HyMT2Translator(Translator):
    """Tencent Hy-MT2 causal-LM translation adapter."""

    name = "hymt2"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        config: GenerationConfig | None = None,
        **_kwargs: Any,
    ) -> None:
        self.model_id = model_id or _DEFAULT_MODEL_ID
        self._config = config or GenerationConfig(batch_size=1)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        torch, AutoTokenizer, AutoModelForCausalLM = _imports()
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto" if self._config.device is None else None,
            trust_remote_code=True,
        )
        if self._config.device:
            self._model.to(self._config.device)
        self._device = next(self._model.parameters()).device
        self._model.eval()

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        self._load()
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        results: list[str] = []
        for start in range(0, len(texts), max(1, self._config.batch_size)):
            batch = texts[start : start + self._config.batch_size]
            batch_outputs: list[str] = []
            for text in batch:
                prompt = build_hymt2_prompt(text, tgt_lang)
                messages = [{"role": "user", "content": prompt}]
                # return_dict=True is required so this always returns a
                # BatchEncoding with an explicit input_ids tensor — some
                # transformers versions return a plain tensor from
                # return_tensors="pt" alone and some return a dict, and
                # calling model.generate(result, ...) positionally on a dict
                # crashes deep inside generate() with an opaque
                # AttributeError on `.shape`. Being explicit here avoids
                # depending on that version-specific default.
                try:
                    encoded = self._tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    # Hy-MT2's own template doesn't document a "thinking" mode
                    # (unlike Qwen3.5, which does), but trust_remote_code=True
                    # means its exact template is out of our control — try the
                    # flag defensively, fall back if unsupported.
                    encoded = self._tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                    )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                input_length = encoded["input_ids"].shape[-1]
                with self._torch.inference_mode():
                    output = self._model.generate(
                        **encoded,
                        max_new_tokens=self._config.max_new_tokens,
                        do_sample=False,
                        num_beams=self._config.num_beams,
                        repetition_penalty=1.05,
                    )
                decoded = self._tokenizer.decode(
                    output[0][input_length:],
                    skip_special_tokens=True,
                )
                batch_outputs.append(strip_thinking(decoded))
            results.extend(batch_outputs)
        return results

    @property
    def generation_config(self) -> dict[str, object]:
        return {
            "adapter": self.name,
            "model_id": self.model_id,
            **self._config.to_dict(),
        }
