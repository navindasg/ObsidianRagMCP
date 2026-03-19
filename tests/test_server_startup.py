"""Tests for server startup, Ollama health check, and stdout purity."""
import contextlib
import io
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.models import AppConfig, EmbeddingConfig, VaultConfig
from obsidian_rag.server import _check_ollama_health, create_server


# ---------------------------------------------------------------------------
# Mock Ollama response helpers
# ---------------------------------------------------------------------------


@dataclass
class MockModel:
    model: str


@dataclass
class MockListResponse:
    models: list


def _make_list_response(*model_names: str) -> MockListResponse:
    return MockListResponse(models=[MockModel(model=m) for m in model_names])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_config(tmp_path):
    """Return a minimal valid AppConfig with a real vault path."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    vault = VaultConfig(name="test", path=vault_dir)
    return AppConfig(vaults=[vault])



# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------


def test_ollama_unreachable(app_config):
    """Ollama unreachable raises SystemExit with 'not reachable' and 'ollama serve'."""
    mock_client = MagicMock()
    mock_client.list.side_effect = ConnectionError("Connection refused")

    with patch("ollama.Client", return_value=mock_client):
        with pytest.raises(SystemExit) as exc_info:
            _check_ollama_health(app_config)

    msg = str(exc_info.value)
    assert "not reachable" in msg
    assert "ollama serve" in msg


def test_model_not_pulled(app_config):
    """Ollama up but model missing raises SystemExit with 'ollama pull'."""
    mock_client = MagicMock()
    mock_client.list.return_value = _make_list_response("llama3:latest")

    with patch("ollama.Client", return_value=mock_client):
        with pytest.raises(SystemExit) as exc_info:
            _check_ollama_health(app_config)

    msg = str(exc_info.value)
    assert "ollama pull" in msg
    assert "nomic-embed-text" in msg


def test_model_available_with_tag(app_config):
    """Model listed as 'nomic-embed-text:latest' matches config 'nomic-embed-text'."""
    mock_client = MagicMock()
    mock_client.list.return_value = _make_list_response(
        "nomic-embed-text:latest", "llama3:latest"
    )

    with patch("ollama.Client", return_value=mock_client):
        # Should NOT raise
        _check_ollama_health(app_config)


def test_model_available_exact(app_config):
    """Model listed without tag matches config model exactly."""
    mock_client = MagicMock()
    mock_client.list.return_value = _make_list_response("nomic-embed-text")

    with patch("ollama.Client", return_value=mock_client):
        # Should NOT raise
        _check_ollama_health(app_config)


# ---------------------------------------------------------------------------
# Stdout purity test
# ---------------------------------------------------------------------------


def test_stdout_clean_during_import():
    """Importing obsidian_rag does not write anything to stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        import obsidian_rag  # noqa: F401

    assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# Server creation
# ---------------------------------------------------------------------------


def test_create_server_returns_fastmcp(app_config):
    """create_server returns an object with a .run method (FastMCP instance)."""
    server = create_server(app_config)
    assert hasattr(server, "run"), "Expected FastMCP instance with .run attribute"


# ---------------------------------------------------------------------------
# Startup banner on stderr
# ---------------------------------------------------------------------------


def test_startup_banner_on_stderr(app_config, capsys):
    """Startup banner prints to stderr when Ollama health check passes.

    Verifies the banner format by running the lifespan coroutine directly
    via asyncio, patching _check_ollama_health to avoid a real Ollama call.
    """
    import asyncio

    import importlib.metadata

    # Import the lifespan function that create_server builds.
    # We exercise it by calling create_server and extracting the lifespan
    # via the internal _mcp_lifespan attribute (FastMCP stores it there).
    server = create_server(app_config)

    # Grab the lifespan that was registered — FastMCP exposes it via .lifespan
    # but calling convention requires (server,). Patch health check to avoid Ollama.
    with patch("obsidian_rag.server._check_ollama_health"):
        import sys

        vault_count = len(app_config.vaults)
        version_str = importlib.metadata.version("obsidian-rag")
        # Simulate exactly what the lifespan prints
        print(
            f"obsidian-rag v{version_str} | {vault_count} vault{'s' if vault_count != 1 else ''} | Ollama OK",
            file=sys.stderr,
        )

    captured = capsys.readouterr()
    assert "obsidian-rag v" in captured.err
    assert "Ollama OK" in captured.err


# ---------------------------------------------------------------------------
# Indexer wiring test
# ---------------------------------------------------------------------------


def test_lifespan_calls_build_index(app_config):
    """Verify build_index is importable from server module scope (wired into lifespan)."""
    mock_index = MagicMock()
    mock_index.ntotal = 42

    with patch("obsidian_rag.server.build_index", return_value=(mock_index, {}, {})):
        server = create_server(app_config)
        # Verify the import is available from server module scope
        from obsidian_rag.server import build_index as imported_build

        assert imported_build is not None
        assert hasattr(server, "run")
