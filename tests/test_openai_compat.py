"""Unit tests for the OpenAI-compatible chat-completions adapter.

Covers prompt construction and batch-response parsing without any network call.
Parsing is the risky part: a batch response that parses *wrongly* rather than
failing would misalign every downstream translation, so the parser is required
to return None on anything it cannot verify item-for-item.
"""

from __future__ import annotations

from medmt_eval.models.base import GenerationConfig
from medmt_eval.models.openai_compat_mt import (
    OpenAICompatTranslator,
    build_batch_prompt,
    parse_batch_response,
)


def test_batch_prompt_numbers_every_item() -> None:
    prompt = build_batch_prompt(["Kein Pleuraerguss.", "Kein Pneumothorax."], "de", "en")
    assert "[[1]]" in prompt and "[[2]]" in prompt
    assert "German" in prompt and "English" in prompt
    assert "Kein Pleuraerguss." in prompt and "Kein Pneumothorax." in prompt
    # The count must be stated so the model knows how many items to emit.
    assert "2" in prompt


def test_parse_batch_response_splits_numbered_items() -> None:
    content = "[[1]]\nNo pleural effusion.\n\n[[2]]\nNo pneumothorax."
    assert parse_batch_response(content, 2) == ["No pleural effusion.", "No pneumothorax."]


def test_parse_batch_response_tolerates_marker_spacing() -> None:
    content = "[[ 1 ]] First.\n[[ 2 ]] Second."
    assert parse_batch_response(content, 2) == ["First.", "Second."]


def test_parse_batch_response_rejects_wrong_item_count() -> None:
    # Model dropped an item — must fail rather than silently misalign.
    assert parse_batch_response("[[1]]\nOnly one.", 2) is None


def test_parse_batch_response_rejects_misnumbered_items() -> None:
    # Numbering out of order means we cannot trust the mapping back to inputs.
    assert parse_batch_response("[[1]]\nA.\n[[3]]\nB.", 2) is None


def test_parse_batch_response_rejects_empty_item() -> None:
    assert parse_batch_response("[[1]]\nA.\n[[2]]\n   ", 2) is None


def test_parse_batch_response_rejects_unnumbered_output() -> None:
    # A model that ignores the format entirely must trigger the fallback.
    assert parse_batch_response("No pleural effusion. No pneumothorax.", 2) is None


def test_translate_batches_and_preserves_order(monkeypatch) -> None:
    """Order must survive batching + the thread pool.

    Note the 5-item / batch_size=2 split leaves a trailing batch of one, which
    takes the single-item prompt path rather than the numbered-batch path — the
    fake below has to handle both shapes.
    """
    translator = OpenAICompatTranslator(api_key="k", config=GenerationConfig(batch_size=2))

    def fake_complete(prompt: str) -> str:
        import re

        items = re.findall(r"\[\[(\d+)\]\]\n(.+)", prompt)
        if items:  # numbered batch prompt
            return "\n".join(f"[[{n}]]\nEN:{text}" for n, text in items)
        # Single-item prompt: the source text is the last line of the template.
        return "EN:" + prompt.strip().splitlines()[-1]

    monkeypatch.setattr(translator, "_complete", fake_complete)
    out = translator.translate(["a", "b", "c", "d", "e"], "de", "en")
    assert out == ["EN:a", "EN:b", "EN:c", "EN:d", "EN:e"]


def test_unparseable_batch_falls_back_to_single_items(monkeypatch) -> None:
    """A malformed batch response must degrade to per-item calls, not corrupt output."""
    translator = OpenAICompatTranslator(api_key="k", config=GenerationConfig(batch_size=3))
    calls: list[str] = []

    def fake_complete(prompt: str) -> str:
        calls.append(prompt)
        if "[[2]]" in prompt:          # this is the batch prompt
            return "totally unformatted response"
        return "SINGLE"                 # single-item prompts

    monkeypatch.setattr(translator, "_complete", fake_complete)
    out = translator.translate(["a", "b", "c"], "de", "en")
    assert out == ["SINGLE", "SINGLE", "SINGLE"]
    # One failed batch call, then one call per item.
    assert len(calls) == 4


def test_generation_config_records_endpoint() -> None:
    """The gateway URL determines what actually served the request, so it is
    recorded with the results rather than only the model name."""
    translator = OpenAICompatTranslator(api_key="k", base_url="https://example.invalid/v1/chat/completions")
    config = translator.generation_config
    assert config["adapter"] == "openai-compat"
    assert config["base_url"] == "https://example.invalid/v1/chat/completions"
    assert config["temperature"] == 0.0


def test_strict_mode_rejects_silent_model_substitution() -> None:
    """The gateway was observed serving nvidia/nemotron-3-ultra-550b-a55b for a
    DeepSeek-V4-Pro request. Labelling those results 'deepseek' would be worse
    than failing, so strict mode raises."""
    translator = OpenAICompatTranslator(api_key="k", model_id="DeepSeek-V4-Pro")
    try:
        translator._check_served_model("nvidia/nemotron-3-ultra-550b-a55b")
    except RuntimeError as error:
        assert "nemotron" in str(error)
        assert "DeepSeek-V4-Pro" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError on model substitution")


def test_matching_model_passes_strict_check() -> None:
    translator = OpenAICompatTranslator(api_key="k", model_id="DeepSeek-V4-Pro")
    translator._check_served_model("deepseek-v4-pro")  # case-insensitive
    assert translator.generation_config["model_substitution"] is False


def test_namespaced_name_is_not_a_substitution() -> None:
    """Endpoints answer an unqualified alias with its fully-qualified name.
    Requesting 'glm-5.2' and being served 'z-ai/glm-5.2' is the same model —
    flagging it would block every legitimate run (observed live on job 3935932).
    """
    translator = OpenAICompatTranslator(api_key="k", model_id="glm-5.2")
    translator._check_served_model("z-ai/glm-5.2")  # must not raise
    assert translator.generation_config["model_substitution"] is False


def test_namespaced_deepseek_flash_is_not_a_substitution() -> None:
    translator = OpenAICompatTranslator(api_key="k", model_id="DeepSeek-V4-Flash")
    translator._check_served_model("deepseek-ai/deepseek-v4-flash")
    assert translator.generation_config["model_substitution"] is False


def test_real_substitution_still_detected_despite_namespacing() -> None:
    """The namespace allowance must not let a genuine swap through: the leaf
    component differs entirely for the observed Kimi -> inkling substitution."""
    translator = OpenAICompatTranslator(api_key="k", model_id="Kimi-K2.6", strict_model=False)
    translator._check_served_model("thinkingmachines/inkling")
    assert translator.generation_config["model_substitution"] is True


def test_non_strict_mode_records_substitution_instead_of_raising() -> None:
    translator = OpenAICompatTranslator(
        api_key="k", model_id="DeepSeek-V4-Pro", strict_model=False
    )
    translator._check_served_model("nvidia/nemotron-3-ultra-550b-a55b")
    config = translator.generation_config
    assert config["served_models"] == ["nvidia/nemotron-3-ultra-550b-a55b"]
    assert config["model_substitution"] is True


def test_env_var_can_disable_strict_model(monkeypatch) -> None:
    monkeypatch.setenv("ALLOW_MODEL_SUBSTITUTION", "1")
    translator = OpenAICompatTranslator(api_key="k", model_id="DeepSeek-V4-Pro")
    assert translator._strict_model is False
    translator._check_served_model("something-else")  # must not raise


def test_missing_api_key_raises_clearly(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    translator = OpenAICompatTranslator()
    try:
        translator._key()
    except RuntimeError as error:
        assert "OPENAI_COMPAT_API_KEY" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError for missing API key")


def test_dated_snapshot_is_not_a_substitution() -> None:
    """The gateway pins DeepSeek-V4-Flash to a dated build; same model, so the
    strict check must accept it (this killed job 4008964)."""
    translator = OpenAICompatTranslator(api_key="k", model_id="DeepSeek-V4-Flash")
    translator._check_served_model("deepseek-ai/deepseek-v4-flash-0731")
    assert translator.generation_config["model_substitution"] is False


def test_snapshot_tag_does_not_excuse_a_different_name() -> None:
    """Stripping a version tag must not let a genuinely different model pass."""
    translator = OpenAICompatTranslator(
        api_key="k", model_id="DeepSeek-V4-Pro", strict_model=False
    )
    translator._check_served_model("nvidia/nemotron-3-ultra-550b-a55b-0731")
    assert translator.generation_config["model_substitution"] is True


def test_version_difference_alone_is_tolerated_but_family_is_not() -> None:
    assert OpenAICompatTranslator._same_model("glm-5.2", "z-ai/glm-5.2-v2")
    assert not OpenAICompatTranslator._same_model("glm-5.2", "z-ai/glm-4.6")


class _Resp:
    def __init__(self, status, headers=None, payload=None):
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload or {
            "model": "m", "choices": [{"message": {"content": "ok"}}]
        }

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)


def test_rate_limit_is_retried_then_succeeds(monkeypatch) -> None:
    """A 429 must be retried with backoff rather than killing the run."""
    calls = []
    slept = []
    monkeypatch.setattr("medmt_eval.models.openai_compat_mt._time.sleep", slept.append)

    def fake_post(*_a, **_k):
        calls.append(1)
        return _Resp(429) if len(calls) < 3 else _Resp(200)

    monkeypatch.setattr("medmt_eval.models.openai_compat_mt._requests.post", fake_post)
    translator = OpenAICompatTranslator(api_key="k", model_id="m")
    assert translator._complete("hi") == "ok"
    assert len(calls) == 3
    assert len(slept) == 2 and all(s > 0 for s in slept)


def test_retry_honours_retry_after_header(monkeypatch) -> None:
    slept = []
    monkeypatch.setattr("medmt_eval.models.openai_compat_mt._time.sleep", slept.append)
    calls = []

    def fake_post(*_a, **_k):
        calls.append(1)
        return _Resp(429, {"Retry-After": "7"}) if len(calls) < 2 else _Resp(200)

    monkeypatch.setattr("medmt_eval.models.openai_compat_mt._requests.post", fake_post)
    OpenAICompatTranslator(api_key="k", model_id="m")._complete("hi")
    assert slept == [7.0]


def test_persistent_rate_limit_finally_raises(monkeypatch) -> None:
    import requests
    monkeypatch.setattr("medmt_eval.models.openai_compat_mt._time.sleep", lambda _s: None)
    monkeypatch.setattr(
        "medmt_eval.models.openai_compat_mt._requests.post", lambda *a, **k: _Resp(429)
    )
    translator = OpenAICompatTranslator(api_key="k", model_id="m")
    try:
        translator._complete("hi")
    except requests.HTTPError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected HTTPError after retries are exhausted")


def test_rate_limited_batch_does_not_fan_out_into_single_requests(monkeypatch) -> None:
    """The per-item fallback must not multiply a 429 by the batch size."""
    import requests
    posts = []
    monkeypatch.setattr("medmt_eval.models.openai_compat_mt._time.sleep", lambda _s: None)

    def fake_post(*_a, **_k):
        posts.append(1)
        return _Resp(429)

    monkeypatch.setattr("medmt_eval.models.openai_compat_mt._requests.post", fake_post)
    translator = OpenAICompatTranslator(api_key="k", model_id="m")
    try:
        translator._translate_batch(["a", "b", "c", "d"], "de", "en")
    except requests.HTTPError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected HTTPError to propagate")
    # 1 batch request x (1 + _MAX_RETRIES) attempts, and no per-item retries.
    from medmt_eval.models.openai_compat_mt import _MAX_RETRIES
    assert len(posts) == _MAX_RETRIES + 1


def test_origin_timeout_is_retried(monkeypatch) -> None:
    """Cloudflare 524 killed job 4009014; it must be retried like any timeout."""
    calls = []
    monkeypatch.setattr("medmt_eval.models.openai_compat_mt._time.sleep", lambda _s: None)

    def fake_post(*_a, **_k):
        calls.append(1)
        return _Resp(524) if len(calls) < 3 else _Resp(200)

    monkeypatch.setattr("medmt_eval.models.openai_compat_mt._requests.post", fake_post)
    assert OpenAICompatTranslator(api_key="k", model_id="m")._complete("hi") == "ok"
    assert len(calls) == 3


def test_timed_out_batch_does_fan_out_into_single_requests(monkeypatch) -> None:
    """Unlike a 429, a timeout is cured by sending SMALLER requests, so the
    per-item fallback must still run."""
    monkeypatch.setattr("medmt_eval.models.openai_compat_mt._time.sleep", lambda _s: None)
    seen = []

    def fake_post(*_a, **kw):
        prompt = kw["json"]["messages"][0]["content"]
        seen.append(prompt)
        # The batch prompt carries [[1]] .. [[n]]; single prompts do not.
        if "[[2]]" in prompt:
            return _Resp(524)
        return _Resp(200, payload={"model": "m",
                                   "choices": [{"message": {"content": "out"}}]})

    monkeypatch.setattr("medmt_eval.models.openai_compat_mt._requests.post", fake_post)
    translator = OpenAICompatTranslator(api_key="k", model_id="m")
    assert translator._translate_batch(["a", "b", "c"], "de", "en") == ["out"] * 3
    assert sum(1 for p in seen if "[[2]]" not in p) == 3  # one call per item
