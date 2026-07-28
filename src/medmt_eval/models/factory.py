"""Factory that keeps model selection explicit and reproducible."""

from __future__ import annotations

from medmt_eval.models.base import GenerationConfig, IdentityTranslator, Translator
from medmt_eval.models.transformers_mt import MADLADTranslator, NLLBTranslator, OpusMTTranslator, TowerTranslator


# Lazy-imported to avoid pulling in optional dependencies at module load time.
def _deepl_cls():
    from medmt_eval.models.deepl_mt import DeepLTranslator
    return DeepLTranslator


def _llm_cls():
    from medmt_eval.models.llm_mt import PromptedLLMTranslator
    return PromptedLLMTranslator


def _hymt2_cls():
    from medmt_eval.models.hymt2_mt import HyMT2Translator
    return HyMT2Translator


def _translategemma_cls():
    from medmt_eval.models.translategemma_mt import TranslateGemmaTranslator
    return TranslateGemmaTranslator


def create_translator(
    name: str,
    *,
    model_id: str | None = None,
    batch_size: int = 8,
    num_beams: int = 4,
    max_input_tokens: int = 512,
    max_new_tokens: int = 512,
    device: str | None = None,
    prompt_template: str | None = None,
    api_key: str | None = None,
    free_tier: bool = True,
) -> Translator:
    """Create a lazy model adapter without triggering a model download."""
    config = GenerationConfig(
        batch_size=batch_size,
        num_beams=num_beams,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    adapters = {
        "identity": lambda: IdentityTranslator(config),
        "opus": lambda: OpusMTTranslator(model_id=model_id, config=config),
        "nllb": lambda: NLLBTranslator(model_id=model_id, config=config),
        "madlad": lambda: MADLADTranslator(model_id=model_id, config=config),
        "tower": lambda: TowerTranslator(model_id=model_id, config=config),
        "deepl": lambda: _deepl_cls()(api_key=api_key, free_tier=free_tier, config=config),
        "prompted-llm": lambda: _llm_cls()(
            model_id=model_id, config=config,
            **({"prompt_template": prompt_template} if prompt_template else {}),
        ),
        "hymt2": lambda: _hymt2_cls()(model_id=model_id, config=config),
        "translategemma": lambda: _translategemma_cls()(model_id=model_id, config=config),
    }
    key = name.lower()
    if key not in adapters:
        choices = ", ".join(sorted(adapters))
        raise ValueError(f"Unknown model adapter {name!r}; choose one of {choices}.")
    return adapters[key]()
