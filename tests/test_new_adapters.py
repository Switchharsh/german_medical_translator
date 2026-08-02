"""Unit tests for Hy-MT2 and TranslateGemma adapters — prompt construction only."""

from __future__ import annotations

from medmt_eval.models.hymt2_mt import HyMT2Translator, build_hymt2_prompt
from medmt_eval.models.translategemma_mt import TranslateGemmaTranslator
from medmt_eval.models.base import GenerationConfig


# ---------------------------------------------------------------------------
# Hy-MT2
# ---------------------------------------------------------------------------

def test_hymt2_prompt_en_to_de() -> None:
    prompt = build_hymt2_prompt("No pleural effusion.", "de")
    assert "German" in prompt
    assert "No pleural effusion." in prompt
    assert "only output the translated result" in prompt


def test_hymt2_prompt_de_to_en() -> None:
    prompt = build_hymt2_prompt("Kein Pleuraerguss.", "en")
    assert "English" in prompt
    assert "Kein Pleuraerguss." in prompt


def test_hymt2_properties() -> None:
    t = HyMT2Translator(model_id="tencent/Hy-MT2-1.8B")
    assert t.name == "hymt2"
    gc = t.generation_config
    assert gc["adapter"] == "hymt2"
    assert gc["model_id"] == "tencent/Hy-MT2-1.8B"


def test_hymt2_default_model() -> None:
    t = HyMT2Translator()
    assert "Hy-MT2" in t.model_id


def test_hymt2_config_passthrough() -> None:
    config = GenerationConfig(batch_size=2, num_beams=1)
    t = HyMT2Translator(config=config)
    gc = t.generation_config
    assert gc["batch_size"] == 2
    assert gc["num_beams"] == 1


class _FakeTorch:
    """Just enough of the torch API surface for translate() to run."""

    class inference_mode:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False


class _FakeBatchEncoding(dict):
    """Mimics a real BatchEncoding: dict-like, plus a broken .shape attr access
    (this is exactly what caused the live AttributeError: calling .shape on
    the dict itself, instead of on encoded["input_ids"], raises through
    BatchEncoding.__getattr__)."""

    def to(self, device):
        return self


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def to(self, device):
        return self

    def __getitem__(self, item):
        return self


class _FakeHyMT2Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, return_tensors,
                             return_dict, enable_thinking=None):
        assert return_dict is True, "must request return_dict=True to get a real BatchEncoding"
        return _FakeBatchEncoding(input_ids=_FakeTensor((1, 7)))

    def decode(self, ids, skip_special_tokens):
        return "Kein Pleuraerguss."


class _FakeHyMT2Model:
    def generate(self, **kwargs):
        assert "input_ids" in kwargs, "must unpack the encoding, not pass it positionally"
        return [_FakeTensor((1, 10))]


def test_hymt2_translate_unpacks_batch_encoding_correctly() -> None:
    """Regression test: apply_chat_template(..., return_tensors='pt') can
    return a BatchEncoding (dict-like) rather than a bare tensor depending on
    the transformers/tokenizer version. Calling model.generate(encoding, ...)
    positionally on that dict crashes inside generate() with an opaque
    AttributeError on `.shape` — this was caught on a live GPU smoke test
    against the real Hy-MT2-1.8B checkpoint, not by any prior unit test."""
    t = HyMT2Translator(model_id="tencent/Hy-MT2-1.8B", config=GenerationConfig(batch_size=1, num_beams=1))
    t._tokenizer = _FakeHyMT2Tokenizer()
    t._model = _FakeHyMT2Model()
    t._torch = _FakeTorch()
    t._device = "cpu"
    result = t.translate(["No pleural effusion."], "en", "de")
    assert result == ["Kein Pleuraerguss."]


# ---------------------------------------------------------------------------
# TranslateGemma
# ---------------------------------------------------------------------------

def test_translategemma_properties() -> None:
    t = TranslateGemmaTranslator(model_id="google/translategemma-4b-it")
    assert t.name == "translategemma"
    gc = t.generation_config
    assert gc["adapter"] == "translategemma"
    assert gc["model_id"] == "google/translategemma-4b-it"


def test_translategemma_default_model() -> None:
    t = TranslateGemmaTranslator()
    assert "translategemma" in t.model_id


def test_translategemma_config_passthrough() -> None:
    config = GenerationConfig(batch_size=1, max_new_tokens=1024)
    t = TranslateGemmaTranslator(config=config)
    gc = t.generation_config
    assert gc["max_new_tokens"] == 1024


# ---------------------------------------------------------------------------
# Turkish language support in model adapters
# ---------------------------------------------------------------------------

def test_language_name_covers_turkish() -> None:
    """Three separate copies of an en/de-only name map used to exist, one per
    adapter. Adding Turkish to some but not others produced KeyError crashes
    mid-run (SLURM jobs 3941775/3941776)."""
    from medmt_eval.schema import language_name

    assert language_name("tr") == "Turkish"
    assert language_name("turkish") == "Turkish"
    assert language_name("de") == "German"
    assert language_name("en") == "English"


def test_opus_maps_turkish_directions() -> None:
    from medmt_eval.models.transformers_mt import OpusMTTranslator

    assert OpusMTTranslator._MODELS[("tr", "en")] == "Helsinki-NLP/opus-mt-tr-en"
    assert OpusMTTranslator._MODELS[("en", "tr")] == "Helsinki-NLP/opus-mt-en-tr"


def test_opus_selects_checkpoint_for_non_default_direction() -> None:
    """The selection used to be gated on `direction != ("en","de")`, a hidden
    coupling to the default checkpoint. It must key off default_model_id so it
    stays correct as more directions are added."""
    from medmt_eval.models.transformers_mt import OpusMTTranslator

    t = OpusMTTranslator()
    assert t.model_id == t.default_model_id
    try:
        t.translate([], "tr", "en")   # no model load: empty input short-circuits
    except Exception:
        pass
    assert t.model_id == "Helsinki-NLP/opus-mt-tr-en"


def test_opus_rejects_unmapped_direction() -> None:
    from medmt_eval.models.transformers_mt import OpusMTTranslator

    import pytest as _pytest
    with _pytest.raises(ValueError, match="no checkpoint"):
        OpusMTTranslator().translate(["x"], "de", "tr")


def test_nllb_has_turkish_code() -> None:
    from medmt_eval.models.transformers_mt import NLLBTranslator

    assert NLLBTranslator._NLLB_CODES["tr"] == "tur_Latn"


def test_prompt_builders_accept_turkish() -> None:
    """Each adapter builds its prompt from the shared name map; all must work."""
    from medmt_eval.models.llm_mt import build_prompt
    from medmt_eval.models.hymt2_mt import build_hymt2_prompt
    from medmt_eval.models.openai_compat_mt import build_batch_prompt

    assert "Turkish" in build_prompt("Efüzyon yok.", "tr", "en")
    assert "English" in build_prompt("Efüzyon yok.", "tr", "en")
    assert "English" in build_hymt2_prompt("Efüzyon yok.", "en")
    assert "Turkish" in build_batch_prompt(["a", "b"], "tr", "en")
