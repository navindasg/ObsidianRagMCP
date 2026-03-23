"""Unit tests for obsidian_rag.tools — all 6 MCP tool handlers.

RED phase: These tests are written before the implementation exists.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.models import AppConfig, VaultConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vault_path(tmp_path):
    """Create a real vault directory with some markdown files."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # notes/hello.md
    notes_dir = vault / "notes"
    notes_dir.mkdir()
    (notes_dir / "hello.md").write_text(
        "---\ntags:\n  - greeting\n---\n# Hello\n\nHello world content.",
        encoding="utf-8",
    )

    # projects/readme.md
    projects_dir = vault / "projects"
    projects_dir.mkdir()
    (projects_dir / "readme.md").write_text(
        "---\ntags:\n  - project\n---\n# Projects\n\n## Setup\n\nSetup guide content.",
        encoding="utf-8",
    )

    # python.md at root
    (vault / "python.md").write_text(
        "# Python Notes\n\nPython programming content.",
        encoding="utf-8",
    )

    return vault


@pytest.fixture
def app_config(vault_path):
    """Return an AppConfig with one vault pointing to tmp vault."""
    return AppConfig(vaults=[VaultConfig(name="test", path=vault_path)])


@pytest.fixture
def mock_faiss_index():
    """Return a mock FAISS index."""
    idx = MagicMock()
    idx.ntotal = 2
    return idx


@pytest.fixture
def mock_vault_indexes(vault_path, mock_faiss_index):
    """Return a mock vault_indexes dict matching server lifespan context shape."""
    return {
        "test": {
            "index": mock_faiss_index,
            "metadata": {
                "0": {
                    "chunk_id": 0,
                    "file": "notes/hello.md",
                    "heading_path": "# Hello",
                    "text": "Hello world content.",
                    "tags": ["greeting"],
                    "folder": "notes",
                    "vault": "test",
                    "modified_ts": 1700000000.0,
                    "char_count": 20,
                },
                "1": {
                    "chunk_id": 1,
                    "file": "projects/readme.md",
                    "heading_path": "# Projects > ## Setup",
                    "text": "Setup guide content.",
                    "tags": ["project"],
                    "folder": "projects",
                    "vault": "test",
                    "modified_ts": 1700000100.0,
                    "char_count": 20,
                },
            },
            "file_hashes": {
                "notes/hello.md": "abc123",
                "projects/readme.md": "def456",
            },
            "vault_config": VaultConfig(name="test", path=vault_path),
        }
    }


@pytest.fixture
def mock_ctx(mock_vault_indexes, app_config):
    """Return a mock FastMCP context with lifespan_context set."""
    ctx = MagicMock()
    ctx.lifespan_context = {
        "vault_indexes": mock_vault_indexes,
        "config": app_config,
        "index_lock": threading.Lock(),
    }
    return ctx


# ---------------------------------------------------------------------------
# Helper: build a FastMCP mock and call register_tools
# ---------------------------------------------------------------------------


def make_mcp_and_register(config):
    """Return (mcp_mock, registered_tool_names) after calling register_tools."""
    from obsidian_rag.tools import register_tools

    registered = {}

    mcp = MagicMock()

    def fake_tool(fn=None, **kwargs):
        """Capture registered tool functions."""
        if fn is not None:
            registered[fn.__name__] = fn
            return fn

        def decorator(f):
            registered[f.__name__] = f
            return f

        return decorator

    mcp.tool = fake_tool
    register_tools(mcp, config)
    return mcp, registered


# ---------------------------------------------------------------------------
# register_tools — conditional registration
# ---------------------------------------------------------------------------


def test_register_tools_respects_enabled(app_config):
    """Only tools listed in config.tools.enabled are registered."""
    from obsidian_rag.models import ToolsConfig

    app_config.tools = ToolsConfig(enabled=["search"])
    _, registered = make_mcp_and_register(app_config)
    assert "search" in registered, "search should be registered"
    assert "read_note" not in registered, "read_note should NOT be registered"
    assert "list_notes" not in registered


def test_register_tools_all_enabled(app_config):
    """All 6 tools are registered when all are enabled."""
    _, registered = make_mcp_and_register(app_config)
    for tool_name in ("search", "read_note", "list_notes", "find_notes", "vault_stats", "reindex"):
        assert tool_name in registered, f"{tool_name} should be registered"


# ---------------------------------------------------------------------------
# search tool
# ---------------------------------------------------------------------------


def test_search_tool(mock_ctx):
    """search returns dict with 'results' key."""
    from obsidian_rag.tools import register_tools

    _, registered = make_mcp_and_register(mock_ctx.lifespan_context["config"])
    search_fn = registered["search"]

    fake_embedding = [0.1] * 768
    fake_results = {
        "results": [
            {
                "source_path": "notes/hello.md",
                "heading_path": "# Hello",
                "relevance_score": 0.95,
                "snippet": "Hello world content.",
                "vault_name": "test",
            }
        ]
    }

    with (
        patch("obsidian_rag.tools.ollama.Client") as mock_client_cls,
        patch("obsidian_rag.tools.retriever_search", return_value=fake_results),
    ):
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=[fake_embedding])
        mock_client_cls.return_value = mock_client

        result = search_fn(query="test query", ctx=mock_ctx)

    assert "results" in result
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 1
    assert result["results"][0]["source_path"] == "notes/hello.md"


def test_search_vault_scoping(mock_ctx):
    """search with vault_name only searches the specified vault's index."""
    from obsidian_rag.tools import register_tools

    _, registered = make_mcp_and_register(mock_ctx.lifespan_context["config"])
    search_fn = registered["search"]

    fake_embedding = [0.1] * 768

    captured_vault_name = {}

    def fake_search(index, metadata, query_embedding, vault_name=None, **kwargs):
        captured_vault_name["val"] = vault_name
        return {"results": []}

    with (
        patch("obsidian_rag.tools.ollama.Client") as mock_client_cls,
        patch("obsidian_rag.tools.retriever_search", side_effect=fake_search),
    ):
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=[fake_embedding])
        mock_client_cls.return_value = mock_client

        search_fn(query="test", vault_name="test", ctx=mock_ctx)

    assert captured_vault_name.get("val") == "test"


def test_search_empty_index(mock_ctx):
    """search on empty index returns {'results': [], 'message': ...}."""
    from obsidian_rag.tools import register_tools

    _, registered = make_mcp_and_register(mock_ctx.lifespan_context["config"])
    search_fn = registered["search"]

    fake_embedding = [0.1] * 768
    empty_result = {"results": [], "message": "No matching results found."}

    with (
        patch("obsidian_rag.tools.ollama.Client") as mock_client_cls,
        patch("obsidian_rag.tools.retriever_search", return_value=empty_result),
    ):
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=[fake_embedding])
        mock_client_cls.return_value = mock_client

        result = search_fn(query="nothing", ctx=mock_ctx)

    assert result["results"] == []
    assert "message" in result


# ---------------------------------------------------------------------------
# read_note tool
# ---------------------------------------------------------------------------


def test_read_note_valid(mock_ctx):
    """read_note returns path, content, frontmatter for an existing file."""
    from obsidian_rag.tools import register_tools

    _, registered = make_mcp_and_register(mock_ctx.lifespan_context["config"])
    read_note_fn = registered["read_note"]

    result = read_note_fn(path="notes/hello.md", ctx=mock_ctx)

    assert "error" not in result
    assert result["path"] == "notes/hello.md"
    assert "content" in result
    assert "Hello" in result["content"]
    assert "frontmatter" in result
    assert isinstance(result["frontmatter"], dict)


def test_read_note_not_found(mock_ctx):
    """read_note returns error dict for missing file."""
    from obsidian_rag.tools import register_tools

    _, registered = make_mcp_and_register(mock_ctx.lifespan_context["config"])
    read_note_fn = registered["read_note"]

    result = read_note_fn(path="missing.md", ctx=mock_ctx)

    assert result["error"] == "File not found"
    assert result["path"] == "missing.md"
    assert "suggestion" in result


def test_read_note_path_traversal(mock_ctx):
    """read_note rejects path traversal attempts."""
    from obsidian_rag.tools import register_tools

    _, registered = make_mcp_and_register(mock_ctx.lifespan_context["config"])
    read_note_fn = registered["read_note"]

    result = read_note_fn(path="../../etc/passwd", ctx=mock_ctx)

    assert result["error"] == "Path outside vault"
    assert "suggestion" in result


# ---------------------------------------------------------------------------
# list_notes tool
# ---------------------------------------------------------------------------


def test_list_notes(mock_ctx):
    """list_notes returns list of dicts with path, size, modified, tag_count."""
    from obsidian_rag.tools import register_tools

    _, registered = make_mcp_and_register(mock_ctx.lifespan_context["config"])
    list_notes_fn = registered["list_notes"]

    result = list_notes_fn(ctx=mock_ctx)

    assert "notes" in result
    assert isinstance(result["notes"], list)
    assert len(result["notes"]) >= 1

    note = result["notes"][0]
    assert "path" in note
    assert "size" in note
    assert "modified" in note
    assert "tag_count" in note


def test_list_notes_prefix_filter(mock_ctx):
    """list_notes with path_prefix returns only files under that prefix."""
    from obsidian_rag.tools import register_tools

    _, registered = make_mcp_and_register(mock_ctx.lifespan_context["config"])
    list_notes_fn = registered["list_notes"]

    result = list_notes_fn(path_prefix="projects/", ctx=mock_ctx)

    assert "notes" in result
    for note in result["notes"]:
        assert note["path"].startswith("projects/"), f"Unexpected path: {note['path']}"


# ---------------------------------------------------------------------------
# find_notes tool
# ---------------------------------------------------------------------------


def test_find_notes_filename(mock_ctx):
    """find_notes matches by filename substring."""
    from obsidian_rag.tools import register_tools

    _, registered = make_mcp_and_register(mock_ctx.lifespan_context["config"])
    find_notes_fn = registered["find_notes"]

    result = find_notes_fn(query="hello", ctx=mock_ctx)

    assert "results" in result
    files = [r["file"] for r in result["results"]]
    assert any("hello" in f for f in files), f"Expected 'hello' in files: {files}"


def test_find_notes_heading(mock_ctx):
    """find_notes matches by heading_path substring."""
    from obsidian_rag.tools import register_tools

    _, registered = make_mcp_and_register(mock_ctx.lifespan_context["config"])
    find_notes_fn = registered["find_notes"]

    result = find_notes_fn(query="setup", ctx=mock_ctx)

    assert "results" in result
    headings = [r["heading_path"] for r in result["results"]]
    assert any("Setup" in h for h in headings), f"Expected 'Setup' in headings: {headings}"


def test_find_notes_case_insensitive(mock_ctx):
    """find_notes is case-insensitive."""
    from obsidian_rag.tools import register_tools

    _, registered = make_mcp_and_register(mock_ctx.lifespan_context["config"])
    find_notes_fn = registered["find_notes"]

    result = find_notes_fn(query="HELLO", ctx=mock_ctx)

    assert "results" in result
    files = [r["file"] for r in result["results"]]
    assert any("hello" in f for f in files), f"Expected 'hello' in files: {files}"


# ---------------------------------------------------------------------------
# vault_stats tool
# ---------------------------------------------------------------------------


def test_vault_stats(mock_ctx):
    """vault_stats returns per-vault stats and aggregate totals."""
    from obsidian_rag.tools import register_tools

    _, registered = make_mcp_and_register(mock_ctx.lifespan_context["config"])
    vault_stats_fn = registered["vault_stats"]

    result = vault_stats_fn(ctx=mock_ctx)

    assert "vaults" in result
    assert isinstance(result["vaults"], list)
    assert len(result["vaults"]) == 1

    vault_entry = result["vaults"][0]
    assert "vault" in vault_entry
    assert "note_count" in vault_entry
    assert "chunk_count" in vault_entry
    assert "index_age" in vault_entry
    assert "embedding_model" in vault_entry

    assert "total_notes" in result
    assert "total_chunks" in result
    assert isinstance(result["total_notes"], int)
    assert isinstance(result["total_chunks"], int)


# ---------------------------------------------------------------------------
# reindex tool
# ---------------------------------------------------------------------------


def test_reindex_returns_immediately(mock_ctx):
    """reindex returns {status: started} immediately without blocking."""
    from obsidian_rag.tools import register_tools, _reindex_locks

    # Clear any existing locks
    _reindex_locks.clear()

    _, registered = make_mcp_and_register(mock_ctx.lifespan_context["config"])
    reindex_fn = registered["reindex"]

    with patch("obsidian_rag.tools.build_index") as mock_build:
        mock_build.return_value = (MagicMock(ntotal=0), {}, {})

        result = reindex_fn(vault_name="test", ctx=mock_ctx)

    assert result["status"] == "started"
    assert result["vault"] == "test"
    assert "message" in result

    # Cleanup
    _reindex_locks.clear()


def test_reindex_duplicate_rejected(mock_ctx):
    """Second reindex call while first is running returns already_running."""
    from obsidian_rag.tools import register_tools, _reindex_locks

    # Simulate a running reindex
    _reindex_locks.clear()
    _reindex_locks["test"] = True

    _, registered = make_mcp_and_register(mock_ctx.lifespan_context["config"])
    reindex_fn = registered["reindex"]

    result = reindex_fn(vault_name="test", ctx=mock_ctx)

    assert result["status"] == "already_running"

    _reindex_locks.clear()


def test_reindex_worker_acquires_lock(mock_ctx):
    """_reindex_worker acquires index_lock before mutating vault_indexes."""
    from obsidian_rag import tools

    vault_indexes = mock_ctx.lifespan_context["vault_indexes"]
    config = mock_ctx.lifespan_context["config"]
    vault_config = vault_indexes["test"]["vault_config"]

    mock_lock = MagicMock()
    mock_lock.__enter__ = MagicMock(return_value=None)
    mock_lock.__exit__ = MagicMock(return_value=False)

    new_index = MagicMock(ntotal=3)
    new_metadata = {"2": {"chunk_id": 2, "file": "new.md", "heading_path": "# New"}}
    new_hashes = {"new.md": "ghijk"}

    with patch("obsidian_rag.tools.build_index", return_value=(new_index, new_metadata, new_hashes)):
        tools._reindex_worker(
            vault_indexes=vault_indexes,
            vault_name="test",
            config=config,
            vault_config=vault_config,
            index_lock=mock_lock,
        )

    mock_lock.__enter__.assert_called_once()
    mock_lock.__exit__.assert_called_once()

    # Verify vault_indexes was updated
    assert vault_indexes["test"]["index"] is new_index
    assert vault_indexes["test"]["metadata"] == new_metadata
    assert vault_indexes["test"]["file_hashes"] == new_hashes
