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
