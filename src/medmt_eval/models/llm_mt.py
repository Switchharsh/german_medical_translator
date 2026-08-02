"""Generalized prompted-LLM adapter for any causal-LM via HuggingFace transformers.

Unlike the TowerTranslator which is hardcoded to one prompt template, this
adapter is parameterized by ``model_id`` and a prompt template string, making
it easy to benchmark different instruction-tuned LLMs (e.g. Llama, Qwen,
Mistral) on medical translation.

The prompt template accepts three placeholders:
    ``{source_lang}`` — full source language name (e.g. "English")
    ``{target_lang}`` — full target language name (e.g. "German")
    ``{text}`` — the source text to translate

For a large (e.g. 20B+) LLM baseline, this in-process transformers.generate()
loop is not the efficient path on this cluster — see colipri_setup.sh for the
established pattern of standing up a vLLM OpenAI-compatible server
(``vllm serve ... --tensor-parallel-size N``) and querying it over HTTP
instead. That would be a separate SLURM job (its own multi-GPU allocation)
plus a small HTTP-client adapter here; not implemented, since small/medium
models run adequately through this adapter as-is.
"""

from __future__ import annotations

from typing import Any

from medmt_eval.models.base import GenerationConfig, Translator, strip_thinking
from medmt_eval.schema import language_name, normalise_language

_DEFAULT_TEMPLATE = (
    "Translate the following medical text from {source_lang} to {target_lang}. "
    "Return only the translation, with no explanation.\n\n{text}"
)


def _imports() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "PromptedLLMTranslator requires `pip install -e '.[mt]'`."
        ) from error
    return torch, AutoTokenizer, AutoModelForCausalLM


def build_prompt(
    text: str,
    src_lang: str,
    tgt_lang: str,
    template: str = _DEFAULT_TEMPLATE,
) -> str:
    """Construct a translation prompt without loading a model (pure function, easily testable)."""
    source_name = language_name(src_lang)
    target_name = language_name(tgt_lang)
    return template.format(source_lang=source_name, target_lang=target_name, text=text)


class PromptedLLMTranslator(Translator):
    """Causal-LM translation adapter with a configurable prompt template."""

    name = "prompted-llm"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        config: GenerationConfig | None = None,
        prompt_template: str = _DEFAULT_TEMPLATE,
        **_kwargs: Any,
    ) -> None:
        self.model_id = model_id or "meta-llama/Llama-3.1-8B-Instruct"
        self._config = config or GenerationConfig(batch_size=1)
        self._template = prompt_template
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        torch, AutoTokenizer, AutoModelForCausalLM = _imports()
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Decoder-only models MUST be left-padded for batched generation. With
        # the default right-padding, pad tokens sit between the prompt and the
        # continuation for every sequence shorter than the longest in the
        # batch, so the model generates from a position preceded by padding
        # and the output is corrupted. transformers warns about this at
        # generate() time ("right-padding was detected"); it is silent in the
        # sense that generation still "succeeds" and produces degraded text
        # rather than raising.
        self._tokenizer.padding_side = "left"
        # device_map="auto" shards across all visible GPUs when more than one
        # is available (needed for large models, e.g. Qwen3.5-27B at ~54GB
        # bf16, which does not fit a single 40GB A100). See HyMT2Translator
        # for the same fix and why a plain --device cuda hint must not force
        # single-GPU placement when multiple GPUs are actually visible.
        multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
        use_auto_map = multi_gpu or self._config.device is None
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto" if use_auto_map else None,
            trust_remote_code=True,
        )
        if use_auto_map:
            self._device = next(self._model.parameters()).device
        else:
            self._device = self._config.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
        self._model.eval()

    def _build_chat_prompt(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Render one message through the tokenizer's chat template.

        Raw text prompting (no chat template) feeds an instruction-tuned model a
        bare continuation prompt rather than a proper chat turn, which some
        models (e.g. Qwen3.5, which defaults to a "thinking" mode that wraps
        output in <think>...</think>) handle poorly or not as documented.
        apply_chat_template is the model-agnostic way to get correct turn
        formatting; enable_thinking=False is passed where the template accepts
        it (Qwen-family templates support this kwarg) to keep output as a bare
        translation instead of a reasoning trace.
        """
        assert self._tokenizer is not None
        content = build_prompt(text, src_lang, tgt_lang, self._template)
        messages = [{"role": "user", "content": content}]
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Template doesn't accept enable_thinking (non-Qwen models).
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        self._load()
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        prompts = [self._build_chat_prompt(text, src_lang, tgt_lang) for text in texts]
        results: list[str] = []
        for start in range(0, len(prompts), self._config.batch_size):
            batch = prompts[start : start + self._config.batch_size]
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._config.max_input_tokens,
                add_special_tokens=False,  # chat template already added them
            )
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
            results.extend(
                self._tokenizer.batch_decode(output[:, prompt_length:], skip_special_tokens=True)
            )
        return [strip_thinking(result) for result in results]

    @property
    def generation_config(self) -> dict[str, object]:
        return {
            "adapter": self.name,
            "model_id": self.model_id,
            "prompt_template": self._template,
            **self._config.to_dict(),
        }
