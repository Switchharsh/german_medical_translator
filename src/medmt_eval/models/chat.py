"""A minimal chat interface, for experiments that need arbitrary prompts.

The :class:`~medmt_eval.models.base.Translator` interface takes text and returns
a translation. The glossary and debate experiments need something weaker and more
general — send a prompt, get a string — because what varies between arms *is* the
prompt. Building them on this interface rather than on Translator keeps the
comparison exact: every arm goes through one code path and differs only in the
text it sends.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any


class ChatBackend(ABC):
    """One model, addressable with a system + user prompt."""

    name: str

    @abstractmethod
    def complete(self, user: str, system: str | None = None) -> str:
        """Return the assistant's reply as plain text."""

    @property
    @abstractmethod
    def config(self) -> dict[str, object]:
        """Reproducibility metadata persisted with every result."""


class OpenAICompatChat(ChatBackend):
    """Hosted model behind an OpenAI-compatible endpoint.

    Delegates to :class:`OpenAICompatTranslator` rather than re-implementing the
    HTTP call, so experiments inherit its retry/backoff *and* its served-model
    guard. That guard is the reason not to shortcut this: the gateway has been
    observed answering with a different model than requested, and an experiment
    that lost the check would silently attribute one model's output to another.
    """

    def __init__(self, model_id: str, *, api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 2048,
                 **kwargs: Any) -> None:
        from medmt_eval.models.base import GenerationConfig
        from medmt_eval.models.openai_compat_mt import OpenAICompatTranslator

        self.name = model_id
        self._impl = OpenAICompatTranslator(
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
            config=GenerationConfig(batch_size=1, max_new_tokens=max_tokens),
            **kwargs,
        )

    def complete(self, user: str, system: str | None = None) -> str:
        return self._impl.chat(user, system=system).strip()

    @property
    def config(self) -> dict[str, object]:
        return self._impl.generation_config


class UnsupportedChatModel(RuntimeError):
    """Raised when a model's chat template cannot express a free-form prompt."""


_PROBE_MESSAGE = [{"role": "user", "content": "Reply with OK."}]


def supports_freeform_chat(model_id: str, *, local_files_only: bool = False) -> tuple[bool, str]:
    """Can this model be given an arbitrary instruction? (ok, reason).

    Translation-only models are not chat models. TranslateGemma's template
    requires each message's content to be a structured mapping carrying
    ``source_lang_code``/``target_lang_code``, so it can express "translate this"
    and nothing else — no persona, no critique, no glossary discussion. Feeding
    it a debate prompt raises a Jinja TemplateError deep inside generation, which
    is a slow and confusing way to discover a design constraint (job 4077236 died
    12 minutes in, after loading three models).

    Only the tokenizer is loaded, so this is cheap enough to run in a preflight.
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, local_files_only=local_files_only
        )
    except Exception as error:  # noqa: BLE001 - report, do not crash the probe
        return False, f"tokenizer unavailable: {type(error).__name__}: {error}"
    try:
        tokenizer.apply_chat_template(
            _PROBE_MESSAGE, tokenize=False, add_generation_prompt=True
        )
    except Exception as error:  # noqa: BLE001
        return False, (
            f"chat template rejects a plain instruction "
            f"({type(error).__name__}). This looks like a translation-only "
            f"model; it cannot take a persona or a critique."
        )
    return True, "ok"


class LocalChat(ChatBackend):
    """Local instruction-tuned causal LM via transformers.

    Shares the conventions the translation adapters had to learn the hard way:
    the chat template is applied (not raw text), thinking is disabled where the
    template supports it, and any leaked ``<think>`` block is stripped.
    """

    def __init__(self, model_id: str, *, device: str | None = None,
                 max_tokens: int = 2048, temperature: float = 0.0) -> None:
        self.name = model_id
        self.model_id = model_id
        self._device = device
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ok, reason = supports_freeform_chat(self.model_id)
        if not ok:
            raise UnsupportedChatModel(f"{self.model_id}: {reason}")

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        multi_gpu = torch.cuda.device_count() > 1
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            **({"device_map": "auto"} if multi_gpu else {}),
        )
        if not multi_gpu:
            self._model.to(self._device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model.eval()

    def complete(self, user: str, system: str | None = None) -> str:
        from medmt_eval.models.base import strip_thinking

        self._load()
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": user})
        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            # Older templates do not accept enable_thinking; strip_thinking below
            # is the fallback for any reasoning trace that leaks through.
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        encoded = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        # Some tokenizers emit tensors the model's forward does not accept —
        # Hy-MT2-7B's returns token_type_ids, which generate() rejects outright
        # ("model_kwargs are not used by the model"). Qwen and Gemma do not, so
        # this is per-model and cannot be assumed away. Keep only what the model
        # actually takes rather than hardcoding a deny-list, so the next tokenizer
        # with an extra field does not break the same way.
        accepted = set(
            inspect.signature(self._model.forward).parameters
        ) | {"inputs", "input_ids"}
        encoded = {k: v for k, v in encoded.items() if k in accepted}
        with self._torch.inference_mode():
            output = self._model.generate(
                **encoded,
                do_sample=self._temperature > 0,
                **({"temperature": self._temperature} if self._temperature > 0 else {}),
                max_new_tokens=self._max_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        reply = self._tokenizer.decode(
            output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return strip_thinking(reply).strip()

    @property
    def config(self) -> dict[str, object]:
        return {
            "backend": "local", "model_id": self.model_id,
            "max_new_tokens": self._max_tokens, "temperature": self._temperature,
        }


def create_chat_backend(spec: str, **kwargs: Any) -> ChatBackend:
    """Build a backend from ``api:<model>`` or ``local:<hf-id>``.

    A bare string is treated as ``api:`` — the hosted models are the ones these
    experiments are normally run against, because a debate needs many short
    round-trips and paying model-load time per turn locally is prohibitive.
    """
    kind, _, model = spec.partition(":")
    if not model:
        kind, model = "api", kind
    kind = kind.lower()
    if kind in {"api", "openai-compat", "hosted"}:
        return OpenAICompatChat(model, **kwargs)
    if kind == "local":
        return LocalChat(model, **{k: v for k, v in kwargs.items() if k != "api_key"})
    raise ValueError(f"Unknown chat backend {kind!r}; use 'api:<model>' or 'local:<hf-id>'.")
