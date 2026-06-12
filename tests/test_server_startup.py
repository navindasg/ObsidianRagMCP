"""Tests for server startup, Ollama health check, and stdout purity."""
import contextlib
import io
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.models import AppConfig, EmbeddingConfig, RerankConfig, VaultConfig
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
# Rerank model validation tests
# ---------------------------------------------------------------------------


def test_rerank_model_missing_raises_exit(app_config):
    """When rerank.enabled=True and model is not pulled, _check_ollama_health raises SystemExit."""
    app_config.rerank = RerankConfig(enabled=True, model="nonexistent-model", top_n=20)

    mock_client = MagicMock()
    # Only embedding model is pulled, NOT the rerank model
    mock_client.list.return_value = _make_list_response("nomic-embed-text")

    with patch("ollama.Client", return_value=mock_client):
        with pytest.raises(SystemExit) as exc_info:
            _check_ollama_health(app_config)

    msg = str(exc_info.value)
    assert "Rerank model" in msg
    assert "ollama pull" in msg


def test_rerank_disabled_skips_model_check(app_config):
    """When rerank.enabled=False, rerank model is NOT checked even if absent."""
    app_config.rerank = RerankConfig(enabled=False, model="nonexistent-model", top_n=20)

    mock_client = MagicMock()
    # Only embedding model is pulled — no rerank model
    mock_client.list.return_value = _make_list_response("nomic-embed-text")

    with patch("ollama.Client", return_value=mock_client):
        # Should NOT raise — rerank disabled
        _check_ollama_health(app_config)


def test_rerank_model_present_passes(app_config):
    """When rerank.enabled=True and model IS pulled, health check passes."""
    app_config.rerank = RerankConfig(enabled=True, model="llama3.2", top_n=20)

    mock_client = MagicMock()
    # Both embedding and rerank models are available
    mock_client.list.return_value = _make_list_response("nomic-embed-text", "llama3.2")

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
    """The REAL lifespan prints the startup banner to stderr — and only stderr.

    stdout carries the MCP stdio protocol; anything printed there corrupts it.
    """
    import asyncio

    mock_index = MagicMock()
    mock_index.ntotal = 0

    with (
        patch("obsidian_rag.server.build_index", return_value=(mock_index, {}, {})),
        patch("obsidian_rag.server._check_ollama_health"),
        patch("obsidian_rag.server.VaultWatcher"),
        patch("obsidian_rag.server.register_tools"),
    ):
        server = create_server(app_config)

        async def run():
            async with server._lifespan_manager():
                pass

        asyncio.run(run())

    captured = capsys.readouterr()
    assert "obsidian-rag v" in captured.err
    assert "Ollama OK" in captured.err
    assert "Indexing complete" in captured.err
    assert captured.out == "", "stdout must stay clean for the MCP stdio protocol"


# ---------------------------------------------------------------------------
# Indexer wiring test
# ---------------------------------------------------------------------------


def test_lifespan_calls_build_index(app_config):
    """The lifespan builds an index per vault and wires it into the context."""
    import asyncio

    mock_index = MagicMock()
    mock_index.ntotal = 42
    mock_metadata = {"0": {"file": "a.md"}}
    mock_hashes = {"a.md": "hash"}
    captured: list[dict] = []

    with (
        patch(
            "obsidian_rag.server.build_index",
            return_value=(mock_index, mock_metadata, mock_hashes),
        ) as mock_build,
        patch("obsidian_rag.server._check_ollama_health"),
        patch("obsidian_rag.server.VaultWatcher"),
        patch("obsidian_rag.server.register_tools"),
    ):
        server = create_server(app_config)

        async def run():
            async with server._lifespan_manager():
                captured.append(dict(server._lifespan_result))

        asyncio.run(run())

        mock_build.assert_called_once_with(app_config, app_config.vaults[0])

    vault_indexes = captured[0]["vault_indexes"]
    entry = vault_indexes[app_config.vaults[0].name]
    assert entry["index"] is mock_index
    assert entry["metadata"] == mock_metadata
    assert entry["file_hashes"] == mock_hashes
    assert entry["vault_config"] is app_config.vaults[0]


# ---------------------------------------------------------------------------
# Integration tests for server wiring (tools + watcher)
# ---------------------------------------------------------------------------


def test_create_server_registers_tools(app_config):
    """create_server calls register_tools once with the mcp instance and config."""
    with patch("obsidian_rag.server.register_tools") as mock_register:
        server = create_server(app_config)
        mock_register.assert_called_once()
        call_args = mock_register.call_args
        # First positional arg should be the FastMCP instance, second is config
        assert call_args[0][0] is server
        assert call_args[0][1] is app_config


def test_lifespan_starts_watcher(app_config):
    """Lifespan creates VaultWatcher and calls start() after indexing completes."""
    import asyncio
    import threading

    mock_index = MagicMock()
    mock_index.ntotal = 0

    with (
        patch("obsidian_rag.server.build_index", return_value=(mock_index, {}, {})),
        patch("obsidian_rag.server._check_ollama_health"),
        patch("obsidian_rag.server.VaultWatcher") as MockWatcher,
        patch("obsidian_rag.server.register_tools"),
    ):
        mock_watcher_instance = MagicMock()
        MockWatcher.return_value = mock_watcher_instance

        server = create_server(app_config)

        async def run():
            # FastMCP stores custom lifespan result in _lifespan_result;
            # use _lifespan_manager() to drive the custom lifespan coroutine.
            async with server._lifespan_manager():
                pass

        asyncio.run(run())

        MockWatcher.assert_called_once()
        call_args = MockWatcher.call_args
        # First arg: vault_indexes dict, second: config, keyword: index_lock
        assert isinstance(call_args[0][0], dict)
        assert call_args[0][1] is app_config
        # threading.Lock() returns a _thread.lock; verify via hasattr duck-typing
        lock_arg = call_args[1]["index_lock"]
        assert hasattr(lock_arg, "acquire") and hasattr(lock_arg, "release")

        mock_watcher_instance.start.assert_called_once()


def test_lifespan_stops_watcher_on_shutdown(app_config):
    """Lifespan calls watcher.stop() when context exits."""
    import asyncio

    mock_index = MagicMock()
    mock_index.ntotal = 0

    with (
        patch("obsidian_rag.server.build_index", return_value=(mock_index, {}, {})),
        patch("obsidian_rag.server._check_ollama_health"),
        patch("obsidian_rag.server.VaultWatcher") as MockWatcher,
        patch("obsidian_rag.server.register_tools"),
    ):
        mock_watcher_instance = MagicMock()
        MockWatcher.return_value = mock_watcher_instance

        server = create_server(app_config)

        async def run():
            async with server._lifespan_manager():
                pass

        asyncio.run(run())

        mock_watcher_instance.stop.assert_called_once()


def test_lifespan_yields_index_lock(app_config):
    """Lifespan yields context dict containing 'index_lock' as a threading.Lock."""
    import asyncio
    import threading

    mock_index = MagicMock()
    mock_index.ntotal = 0
    captured: list[dict] = []

    with (
        patch("obsidian_rag.server.build_index", return_value=(mock_index, {}, {})),
        patch("obsidian_rag.server._check_ollama_health"),
        patch("obsidian_rag.server.VaultWatcher") as MockWatcher,
        patch("obsidian_rag.server.register_tools"),
    ):
        MockWatcher.return_value = MagicMock()

        server = create_server(app_config)

        async def run():
            async with server._lifespan_manager():
                # FastMCP stores the yielded lifespan context in _lifespan_result
                captured.append(dict(server._lifespan_result))

        asyncio.run(run())

        assert captured, "Lifespan never yielded a context"
        ctx = captured[0]
        assert "index_lock" in ctx
        # threading.Lock() returns a _thread.lock; verify via duck-typing
        lock = ctx["index_lock"]
        assert hasattr(lock, "acquire") and hasattr(lock, "release")
