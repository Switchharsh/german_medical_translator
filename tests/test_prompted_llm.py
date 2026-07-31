"""Unit tests for PromptedLLMTranslator — prompt construction logic only.

These tests verify that the correct prompt is built for each language pair
without loading a real model (no GPU or download needed).
"""

from __future__ import annotations

from medmt_eval.models.llm_mt import PromptedLLMTranslator, build_prompt, _DEFAULT_TEMPLATE
from medmt_eval.models.base import GenerationConfig, strip_thinking


def test_build_prompt_en_to_de() -> None:
    prompt = build_prompt("No pleural effusion.", "en", "de")
    assert "English" in prompt
    assert "German" in prompt
    assert "No pleural effusion." in prompt
    assert "only the translation" in prompt.lower() or "no explanation" in prompt.lower()


def test_build_prompt_de_to_en() -> None:
    prompt = build_prompt("Kein Pleuraerguss.", "de", "en")
    assert "German" in prompt
    assert "English" in prompt
    assert "Kein Pleuraerguss." in prompt


def test_build_prompt_custom_template() -> None:
    template = "Translate from {source_lang} to {target_lang}: {text}"
    prompt = build_prompt("Hello", "en", "de", template=template)
    assert prompt == "Translate from English to German: Hello"


def test_prompted_llm_translator_properties() -> None:
    translator = PromptedLLMTranslator(model_id="test/model", prompt_template="T: {text}")
    assert translator.name == "prompted-llm"
    assert translator.model_id == "test/model"
    config = translator.generation_config
    assert config["adapter"] == "prompted-llm"
    assert config["model_id"] == "test/model"
    assert config["prompt_template"] == "T: {text}"


def test_prompted_llm_default_model() -> None:
    translator = PromptedLLMTranslator()
    assert "llama" in translator.model_id.lower() or "Llama" in translator.model_id


def test_prompted_llm_config_passes_through() -> None:
    config = GenerationConfig(batch_size=2, num_beams=1, max_input_tokens=256)
    translator = PromptedLLMTranslator(config=config)
    gen = translator.generation_config
    assert gen["batch_size"] == 2
    assert gen["num_beams"] == 1
    assert gen["max_input_tokens"] == 256


class _FakeTokenizerWithThinking:
    """Mimics a Qwen-style tokenizer whose chat template accepts enable_thinking."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking=None):
        assert tokenize is False and add_generation_prompt is True
        return f"<thinking={enable_thinking}>{messages[0]['content']}"


class _FakeTokenizerNoThinking:
    """Mimics a plain chat template with no enable_thinking kwarg support."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return f"<chat>{messages[0]['content']}"


def test_build_chat_prompt_disables_thinking_when_supported() -> None:
    translator = PromptedLLMTranslator(model_id="test/model")
    translator._tokenizer = _FakeTokenizerWithThinking()
    rendered = translator._build_chat_prompt("No pleural effusion.", "en", "de")
    assert rendered.startswith("<thinking=False>")
    assert "No pleural effusion." in rendered


def test_build_chat_prompt_falls_back_without_thinking_kwarg() -> None:
    translator = PromptedLLMTranslator(model_id="test/model")
    translator._tokenizer = _FakeTokenizerNoThinking()
    rendered = translator._build_chat_prompt("No pleural effusion.", "en", "de")
    assert rendered.startswith("<chat>")
    assert "No pleural effusion." in rendered


def test_strip_thinking_removes_leaked_reasoning_block() -> None:
    raw = "<think>Let me consider the medical terminology...</think>Kein Pleuraerguss."
    assert strip_thinking(raw) == "Kein Pleuraerguss."


def test_strip_thinking_is_noop_without_think_tags() -> None:
    assert strip_thinking("Kein Pleuraerguss.") == "Kein Pleuraerguss."


def test_strip_thinking_handles_multiline_block() -> None:
    raw = "<think>\nline one\nline two\n</think>\nDer Patient verneint Brustschmerzen."
    assert strip_thinking(raw) == "Der Patient verneint Brustschmerzen."


class _RecordingTokenizer:
    """Stands in for a real tokenizer so _load()'s side effects are visible."""

    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "right"  # transformers' default — must be overridden


class _StubModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        raise AssertionError("parameters() should not be reached in this test")


def test_load_sets_left_padding_for_decoder_only_batching(monkeypatch) -> None:
    """Regression test: decoder-only models must be left-padded for batched
    generation. With right-padding, pad tokens land between the prompt and the
    continuation for every sequence shorter than the longest in its batch,
    silently corrupting output (transformers only warns, it does not raise).
    Observed live as a repeated "right-padding was detected" warning during
    the qwen35-27b benchmark run.

    This drives the real _load() via a patched _imports() so the padding_side
    assignment in the production code path is what's asserted."""
    import medmt_eval.models.llm_mt as llm_mt

    tok = _RecordingTokenizer()

    class _FakeTokenizerCls:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return tok

    class _FakeModelCls:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _StubModel()

    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

    class _FakeTorchMod:
        bfloat16 = "bf16"
        cuda = _FakeCuda

    monkeypatch.setattr(
        llm_mt, "_imports", lambda: (_FakeTorchMod, _FakeTokenizerCls, _FakeModelCls)
    )

    translator = PromptedLLMTranslator(model_id="test/model", config=GenerationConfig(device="cpu"))
    translator._load()

    assert tok.padding_side == "left", "decoder-only batching requires left padding"
