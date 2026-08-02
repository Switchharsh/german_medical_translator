"""Transformers-backed adapters loaded only when a translation command runs."""

from __future__ import annotations

from typing import Any

from medmt_eval.models.base import GenerationConfig, Translator
from medmt_eval.schema import normalise_language


def _imports() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - requires optional dependency
        raise RuntimeError(
            "Model inference requires `pip install -e '.[mt]'` before running this command."
        ) from error
    return torch, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


class _TransformersTranslator(Translator):
    """Shared batching and device handling for sequence-to-sequence adapters."""

    default_model_id: str

    def __init__(self, model_id: str | None = None, config: GenerationConfig | None = None) -> None:
        self.model_id = model_id or self.default_model_id
        self._config = config or GenerationConfig()
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None

    def _load_seq2seq(self) -> None:
        if self._model is not None:
            return
        torch, AutoTokenizer, AutoModelForSeq2SeqLM, _ = _imports()
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        self._device = self._config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    def _generate(self, inputs: list[str], **generate_kwargs: Any) -> list[str]:
        self._load_seq2seq()
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        outputs: list[str] = []
        for start in range(0, len(inputs), self._config.batch_size):
            batch = inputs[start : start + self._config.batch_size]
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._config.max_input_tokens,
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                token_ids = self._model.generate(
                    **encoded,
                    num_beams=self._config.num_beams,
                    max_new_tokens=self._config.max_new_tokens,
                    max_length=None,
                    **generate_kwargs,
                )
            outputs.extend(self._tokenizer.batch_decode(token_ids, skip_special_tokens=True))
        return outputs

    @property
    def generation_config(self) -> dict[str, object]:
        return {
            "adapter": self.name,
            "model_id": self.model_id,
            **self._config.to_dict(),
        }


class OpusMTTranslator(_TransformersTranslator):
    """Direction-specific Helsinki-NLP Marian checkpoints."""

    name = "opus"
    default_model_id = "Helsinki-NLP/opus-mt-en-de"
    # Helsinki-NLP ships one checkpoint per direction, so each pair needs an
    # explicit entry; an unmapped direction raises rather than silently using
    # the wrong model.
    _MODELS = {
        ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
        ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
        ("en", "tr"): "Helsinki-NLP/opus-mt-en-tr",
        ("tr", "en"): "Helsinki-NLP/opus-mt-tr-en",
    }

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        direction = (normalise_language(src_lang), normalise_language(tgt_lang))
        if direction not in self._MODELS:
            raise ValueError(f"Opus adapter has no checkpoint for {direction}.")
        expected = self._MODELS[direction]
        if self._model is not None and self.model_id != expected:
            raise ValueError("One Opus translator instance can only be used for one direction.")
        # Auto-select the checkpoint for this direction, unless the caller gave
        # an explicit --model-id. Comparing against default_model_id (rather
        # than a hardcoded direction) keeps this correct as _MODELS grows.
        if self.model_id == self.default_model_id:
            self.model_id = expected
        return self._generate(texts)


class NLLBTranslator(_TransformersTranslator):
    """NLLB-200 adapter with explicit source language and forced target BOS token."""

    name = "nllb"
    default_model_id = "facebook/nllb-200-distilled-1.3B"
    _NLLB_CODES = {"en": "eng_Latn", "de": "deu_Latn", "tr": "tur_Latn"}

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        source, target = normalise_language(src_lang), normalise_language(tgt_lang)
        self._load_seq2seq()
        assert self._tokenizer is not None
        self._tokenizer.src_lang = self._NLLB_CODES[source]
        target_code = self._NLLB_CODES[target]
        # convert_tokens_to_ids works across all transformers versions;
        # the older lang_code_to_id dict attribute was removed in newer releases.
        target_id = self._tokenizer.convert_tokens_to_ids(target_code)
        return self._generate(texts, forced_bos_token_id=target_id)


class MADLADTranslator(_TransformersTranslator):
    """MADLAD-400 MT adapter, which uses a target-language prefix."""

    name = "madlad"
    default_model_id = "google/madlad400-3b-mt"

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        target = normalise_language(tgt_lang)
        return self._generate([f"<2{target}> {text}" for text in texts])


class TowerTranslator(Translator):
    """Instruction-tuned causal-LM translation adapter for TowerInstruct-like models."""

    name = "tower"
    default_model_id = "Unbabel/TowerInstruct-7B-v0.2"

    def __init__(self, model_id: str | None = None, config: GenerationConfig | None = None) -> None:
        self.model_id = model_id or self.default_model_id
        self._config = config or GenerationConfig(batch_size=1)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        torch, AutoTokenizer, _, AutoModelForCausalLM = _imports()
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self._device = self._config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        source, target = normalise_language(src_lang), normalise_language(tgt_lang)
        language_name = {"en": "English", "de": "German"}
        prompts = [
            f"Translate the following medical text from {language_name[source]} to "
            f"{language_name[target]}. Return only the translation.\n\n{text}"
            for text in texts
        ]
        self._load()
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        results: list[str] = []
        for start in range(0, len(prompts), self._config.batch_size):
            batch = prompts[start : start + self._config.batch_size]
            encoded = self._tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                output = self._model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=self._config.num_beams,
                    max_new_tokens=self._config.max_new_tokens,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            prompt_length = encoded["input_ids"].shape[1]
            results.extend(self._tokenizer.batch_decode(output[:, prompt_length:], skip_special_tokens=True))
        return [result.strip() for result in results]

    @property
    def generation_config(self) -> dict[str, object]:
        return {"adapter": self.name, "model_id": self.model_id, **self._config.to_dict()}
