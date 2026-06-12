"""Tests for Ollama chat-model auto-selection (daily_format.model_select)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from obsidian_rag.daily_format.model_select import PREFERRED_MODELS, select_model


# ---------------------------------------------------------------------------
# Mock Ollama response helpers (mirrors tests/test_server_startup.py)
# ---------------------------------------------------------------------------


@dataclass
class MockModel:
    model: str


@dataclass
class MockListResponse:
    models: list


def _client_with(*model_names: str) -> MagicMock:
    """Return a mock ollama.Client whose list() reports the given models."""
    client = MagicMock()
    client.list.return_value = MockListResponse(
        models=[MockModel(model=name) for name in model_names]
    )
    return client


# ---------------------------------------------------------------------------
# Configured model
# ---------------------------------------------------------------------------


def test_configured_present_exact_match():
    """A configured model that is pulled with the exact name is returned."""
    client = _client_with("llama3.2:latest", "nomic-embed-text:latest")

    assert select_model(client, "llama3.2:latest") == "llama3.2:latest"


def test_configured_present_without_tag():
    """A configured model without a tag matches a pulled tagged model."""
    client = _client_with("llama3.2:latest")

    assert select_model(client, "llama3.2") == "llama3.2"


def test_configured_missing_exits_with_pull_hint():
    """A configured model that is not pulled exits with an 'ollama pull' hint."""
    client = _client_with("llama3.2:latest")

    with pytest.raises(SystemExit) as exc_info:
        select_model(client, "mistral-small")

    assert "ollama pull mistral-small" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Auto-selection preference order
# ---------------------------------------------------------------------------


def test_preference_order_picks_earliest_preferred():
    """The first PREFERRED_MODELS entry that is pulled wins."""
    client = _client_with(
        "llama3.2:latest", "qwen3.5:9b", "gemma4:12b-mlx", "nomic-embed-text:latest"
    )

    assert select_model(client, None) == "gemma4:12b-mlx"


def test_preference_order_skips_unpulled_entries():
    """Preferred entries that are not pulled are skipped in order."""
    client = _client_with("llama3.2:latest", "qwen3.5:9b")

    assert select_model(client, None) == "qwen3.5:9b"


def test_tag_suffix_matching():
    """A preferred name with a tag matches the identically tagged pulled model."""
    client = _client_with("gemma4:26b-mlx")

    assert select_model(client, None) == "gemma4:26b-mlx"


def test_preferred_without_tag_matches_tagged_pulled_model():
    """'llama3.2' in PREFERRED_MODELS matches a pulled 'llama3.2:latest'."""
    client = _client_with("llama3.2:latest")

    assert select_model(client, None) == "llama3.2"


# ---------------------------------------------------------------------------
# Fallback to first non-embed model
# ---------------------------------------------------------------------------


def test_fallback_to_first_non_embed_model():
    """With no preferred model pulled, the first non-embed model is chosen."""
    client = _client_with("nomic-embed-text:latest", "phi4:latest", "mistral:7b")

    assert select_model(client, None) == "phi4:latest"


def test_embed_models_never_auto_selected():
    """Embedding-only model lists exit instead of selecting an embed model."""
    client = _client_with("nomic-embed-text:latest", "mxbai-embed-large:latest")

    with pytest.raises(SystemExit) as exc_info:
        select_model(client, None)

    assert "chat model" in str(exc_info.value).lower()


def test_no_models_pulled_exits():
    """An empty model list exits with an explanation."""
    client = _client_with()

    with pytest.raises(SystemExit) as exc_info:
        select_model(client, None)

    assert "chat model" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Ollama unreachable
# ---------------------------------------------------------------------------


def test_list_failure_raises_connection_error():
    """A failing client.list() surfaces as ConnectionError mentioning 'ollama serve'."""
    client = MagicMock()
    client.list.side_effect = ConnectionError("Connection refused")

    with pytest.raises(ConnectionError) as exc_info:
        select_model(client, None)

    assert "ollama serve" in str(exc_info.value)


def test_list_failure_with_configured_model_raises_connection_error():
    """Connection failures take precedence over configured-model validation."""
    client = MagicMock()
    client.list.side_effect = OSError("boom")

    with pytest.raises(ConnectionError) as exc_info:
        select_model(client, "llama3.2")

    assert "ollama serve" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Constant sanity
# ---------------------------------------------------------------------------


def test_preferred_models_order():
    """PREFERRED_MODELS lists the documented candidates in priority order."""
    assert list(PREFERRED_MODELS) == [
        "gemma4:26b-mlx",
        "gemma4:12b-mlx",
        "qwen3.5:9b",
        "ministral-3:8b",
        "llama3.2",
    ]
