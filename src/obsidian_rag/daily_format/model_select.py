"""Ollama chat-model auto-selection for the daily-note formatter.

Picks the chat model used to format raw daily notes: a configured model is
validated against the pulled models, otherwise the best available model is
auto-selected from :data:`PREFERRED_MODELS`, falling back to the first pulled
non-embedding model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ollama

logger = logging.getLogger(__name__)

# Chat models tried in priority order when no model is configured explicitly.
PREFERRED_MODELS: tuple[str, ...] = (
    "gemma4:26b-mlx",
    "gemma4:12b-mlx",
    "qwen3.5:9b",
    "ministral-3:8b",
    "llama3.2",
)


def _pulled_model_names(client: ollama.Client) -> tuple[str, ...]:
    """Return the names of all pulled Ollama models (e.g. ``llama3.2:latest``).

    Raises ConnectionError mentioning ``ollama serve`` if Ollama is
    unreachable, so callers can leave work queued and retry later.
    """
    try:
        response = client.list()
    except Exception as exc:
        raise ConnectionError(
            "Ollama is not reachable.\n"
            "Fix: ensure Ollama is running (ollama serve)"
        ) from exc
    return tuple(model.model for model in response.models)


def _is_pulled(name: str, pulled: tuple[str, ...]) -> bool:
    """Return True if *name* matches a pulled model exactly or without its tag.

    Mirrors the matching in ``server._check_ollama_health``: ``llama3.2``
    matches a pulled ``llama3.2:latest``, while a fully tagged name such as
    ``gemma4:26b-mlx`` must match exactly (so it never matches ``gemma4:12b-mlx``).
    """
    return any(name == model or name == model.split(":")[0] for model in pulled)


def select_model(client: ollama.Client, configured: str | None) -> str:
    """Return the chat model to use for formatting daily notes.

    If *configured* is given it is validated against the pulled models and
    returned; a missing model raises SystemExit with an ``ollama pull`` hint.
    Otherwise the first pulled :data:`PREFERRED_MODELS` entry is returned,
    falling back to the first pulled non-embedding model. Raises SystemExit
    when no chat model is available, and ConnectionError when Ollama is down.
    """
    pulled = _pulled_model_names(client)

    if configured is not None:
        if _is_pulled(configured, pulled):
            logger.debug("Using configured chat model: %s", configured)
            return configured
        raise SystemExit(
            f"Chat model '{configured}' not found in Ollama.\n"
            f"Fix: run: ollama pull {configured}"
        )

    for candidate in PREFERRED_MODELS:
        if _is_pulled(candidate, pulled):
            logger.info("Auto-selected chat model: %s", candidate)
            return candidate

    for name in pulled:
        if "embed" not in name.lower():
            logger.info("Auto-selected fallback chat model: %s", name)
            return name

    raise SystemExit(
        "No chat model is available in Ollama for daily-note formatting "
        "(embedding models cannot be used).\n"
        "Fix: run: ollama pull llama3.2"
    )
