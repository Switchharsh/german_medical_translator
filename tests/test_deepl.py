"""Unit tests for the DeepL adapter with mocked HTTP calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from medmt_eval.models.deepl_mt import DeepLTranslator


def _mock_response(translations: list[str]) -> MagicMock:
    """Create a mock requests.Response returning DeepL-format JSON."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"translations": [{"text": t} for t in translations]}
    return resp


@patch("medmt_eval.models.deepl_mt._requests.post")
def test_deepl_translate_single_batch(mock_post: MagicMock) -> None:
    mock_post.return_value = _mock_response(["Kein Pleuraerguss."])
    translator = DeepLTranslator(api_key="test-key-123")
    result = translator.translate(["No pleural effusion."], "en", "de")
    assert result == ["Kein Pleuraerguss."]
    # Verify request was made correctly.
    call_kwargs = mock_post.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert body["source_lang"] == "EN"
    assert body["target_lang"] == "DE"
    assert body["text"] == ["No pleural effusion."]
    assert "test-key-123" in call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {})).get(
        "Authorization", call_kwargs[1].get("headers", {}).get("Authorization", "")
    )


@patch("medmt_eval.models.deepl_mt._requests.post")
def test_deepl_translate_multiple_texts(mock_post: MagicMock) -> None:
    mock_post.return_value = _mock_response(["Satz eins.", "Satz zwei."])
    translator = DeepLTranslator(api_key="k", free_tier=True)
    result = translator.translate(["Sentence one.", "Sentence two."], "en", "de")
    assert result == ["Satz eins.", "Satz zwei."]
    # Verify free-tier endpoint was used.
    url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url", "")
    assert "api-free.deepl.com" in url


@patch("medmt_eval.models.deepl_mt._requests.post")
def test_deepl_translate_paid_tier(mock_post: MagicMock) -> None:
    mock_post.return_value = _mock_response(["Test."])
    translator = DeepLTranslator(api_key="k", free_tier=False)
    translator.translate(["Test."], "de", "en")
    url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url", "")
    assert "api.deepl.com" in url


@patch("medmt_eval.models.deepl_mt._requests.post")
def test_deepl_translate_batches(mock_post: MagicMock) -> None:
    mock_post.side_effect = [_mock_response(["A", "B"]), _mock_response(["C"])]
    translator = DeepLTranslator(api_key="k")
    translator._config = translator._config.__class__(batch_size=2)
    result = translator.translate(["t1", "t2", "t3"], "en", "de")
    assert result == ["A", "B", "C"]
    assert mock_post.call_count == 2


def test_deepl_generation_config() -> None:
    translator = DeepLTranslator(api_key="k", free_tier=True)
    config = translator.generation_config
    assert config["adapter"] == "deepl"
    assert config["free_tier"] is True


def test_deepl_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPL_AUTH_KEY", raising=False)
    translator = DeepLTranslator(free_tier=True)
    with pytest.raises(RuntimeError, match="API key"):
        translator.translate(["test"], "en", "de")
