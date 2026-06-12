"""Integration tests for the note_context MCP tool handler.

TDD RED phase: tests written before implementation exists.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_rag.models import AppConfig, ToolsConfig, VaultConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_vault_root():
    """Return the path to the sample vault in the test fixtures directory."""
    return Path(__file__).parent / "fixtures" / "sample_vault"


@pytest.fixture
def vault_path(tmp_path, fixture_vault_root):
    """Create a temp vault directory containing fixture notes."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # Copy wikilinks-callouts.md (the "source" note with forward links)
    shutil.copy(fixture_vault_root / "wikilinks-callouts.md", vault / "wikilinks-callouts.md")

    # Copy wsn-pipeline.md (the "target" note referenced by wikilinks-callouts)
    shutil.copy(fixture_vault_root / "wsn-pipeline.md", vault / "wsn-pipeline.md")

    return vault


@pytest.fixture
def app_config(vault_path):
    """Return a minimal AppConfig with one vault."""
    return AppConfig(vaults=[VaultConfig(name="test", path=vault_path)])


@pytest.fixture
def mock_metadata(vault_path):
    """Return metadata dict containing a chunk from wsn-pipeline.md that links back."""
    return {
        "0": {
            "chunk_id": 0,
            "file": "wsn-pipeline.md",
            "heading_path": "# WSN Pipeline",
            "text": "See [[wikilinks-callouts]] for Obsidian feature docs.",
            "tags": ["engineering", "pipeline"],
            "folder": "",
            "vault": "test",
            "modified_ts": 1700000000.0,
            "char_count": 52,
        },
        "1": {
            "chunk_id": 1,
            "file": "wikilinks-callouts.md",
            "heading_path": "# Obsidian Features > ## Wikilinks",
            "text": "See [[wsn-pipeline]] for details. Also check [[2024-01-15|yesterday's note]].",
            "tags": ["reference"],
            "folder": "",
            "vault": "test",
            "modified_ts": 1700000100.0,
            "char_count": 75,
        },
    }


@pytest.fixture
def mock_faiss_index():
    idx = MagicMock()
    idx.ntotal = 2
    return idx


@pytest.fixture
def mock_vault_indexes(vault_path, mock_faiss_index, mock_metadata):
    return {
        "test": {
            "index": mock_faiss_index,
            "metadata": mock_metadata,
            "file_hashes": {},
            "vault_config": VaultConfig(name="test", path=vault_path),
        }
    }


@pytest.fixture
def mock_ctx(mock_vault_indexes, app_config):
    ctx = MagicMock()
    ctx.lifespan_context = {
        "vault_indexes": mock_vault_indexes,
        "config": app_config,
        "index_lock": threading.Lock(),
    }
    return ctx


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_note_context_fn(config):
    """Register tools and return the note_context handler."""
    from obsidian_rag.tools import register_tools

    registered = {}

    mcp = MagicMock()

    def fake_tool(fn=None, **kwargs):
        if fn is not None:
            registered[fn.__name__] = fn
            return fn

        def decorator(f):
            registered[f.__name__] = f
            return f

        return decorator

    mcp.tool = fake_tool
    register_tools(mcp, config)
    return registered.get("note_context")


# ---------------------------------------------------------------------------
# note_context tests
# ---------------------------------------------------------------------------


def test_note_context_returns_content_and_links(mock_ctx):
    """note_context returns a dict with 'note', 'forward_links', 'backlinks' keys."""
    note_context_fn = make_note_context_fn(mock_ctx.lifespan_context["config"])
    assert note_context_fn is not None, "note_context tool should be registered"

    result = note_context_fn(path="wikilinks-callouts.md", ctx=mock_ctx)

    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert "note" in result
    assert "forward_links" in result
    assert "backlinks" in result
    assert isinstance(result["note"], dict)
    assert isinstance(result["forward_links"], list)
    assert isinstance(result["backlinks"], list)


def test_note_context_note_has_path_and_content(mock_ctx):
    """note_context 'note' dict contains path and content fields."""
    note_context_fn = make_note_context_fn(mock_ctx.lifespan_context["config"])

    result = note_context_fn(path="wikilinks-callouts.md", ctx=mock_ctx)

    assert result["note"]["path"] == "wikilinks-callouts.md"
    assert "content" in result["note"]
    assert "Obsidian Features" in result["note"]["content"]


def test_forward_links_parsed(mock_ctx):
    """note_context extracts forward wikilinks including wsn-pipeline and 2024-01-15."""
    note_context_fn = make_note_context_fn(mock_ctx.lifespan_context["config"])

    result = note_context_fn(path="wikilinks-callouts.md", ctx=mock_ctx)

    forward_paths = [fl["path"] for fl in result["forward_links"]]

    # wsn-pipeline.md exists in the vault
    assert any("wsn-pipeline" in p for p in forward_paths), (
        f"Expected wsn-pipeline in forward_links, got: {forward_paths}"
    )

    # 2024-01-15 does not exist in the vault (exists=False)
    non_existent = [fl for fl in result["forward_links"] if "2024-01-15" in fl["path"]]
    assert len(non_existent) == 1
    assert non_existent[0]["exists"] is False


def test_forward_links_have_exists_boolean(mock_ctx):
    """Each forward_link entry has an 'exists' boolean field."""
    note_context_fn = make_note_context_fn(mock_ctx.lifespan_context["config"])

    result = note_context_fn(path="wikilinks-callouts.md", ctx=mock_ctx)

    for link in result["forward_links"]:
        assert "exists" in link
        assert isinstance(link["exists"], bool)


def test_embed_excluded_from_forward_links(mock_ctx):
    """Embed syntax (![[wsn-pipeline#Architecture]]) is NOT included in forward_links."""
    note_context_fn = make_note_context_fn(mock_ctx.lifespan_context["config"])

    result = note_context_fn(path="wikilinks-callouts.md", ctx=mock_ctx)

    # wikilinks-callouts.md has ![[wsn-pipeline#Architecture]] — the Architecture
    # section link should not produce a separate duplicate entry.
    # The embed must not appear; forward_links should have wsn-pipeline at most once.
    wsn_entries = [fl for fl in result["forward_links"] if "wsn-pipeline" in fl["path"]]
    assert len(wsn_entries) == 1, (
        f"wsn-pipeline should appear exactly once (not duplicated via embed): {wsn_entries}"
    )


def test_backlinks_found(mock_ctx):
    """note_context returns backlinks from metadata referencing wikilinks-callouts."""
    note_context_fn = make_note_context_fn(mock_ctx.lifespan_context["config"])

    result = note_context_fn(path="wikilinks-callouts.md", ctx=mock_ctx)

    # wsn-pipeline.md has [[wikilinks-callouts]] in its chunk text
    backlink_sources = [bl["source_path"] for bl in result["backlinks"]]
    assert any("wsn-pipeline" in s for s in backlink_sources), (
        f"Expected wsn-pipeline.md in backlinks, got: {backlink_sources}"
    )


def test_missing_note_returns_error(mock_ctx):
    """note_context returns structured error dict for non-existent path."""
    note_context_fn = make_note_context_fn(mock_ctx.lifespan_context["config"])

    result = note_context_fn(path="does-not-exist.md", ctx=mock_ctx)

    assert "error" in result
    assert "path" in result
    assert "suggestion" in result
    assert result["path"] == "does-not-exist.md"
    assert "find_notes" in result["suggestion"]


def test_missing_note_does_not_have_note_key(mock_ctx):
    """Error response for missing note must NOT contain 'note' key."""
    note_context_fn = make_note_context_fn(mock_ctx.lifespan_context["config"])

    result = note_context_fn(path="missing.md", ctx=mock_ctx)

    assert "note" not in result


def test_note_context_path_traversal_rejected(mock_ctx):
    """note_context rejects path traversal attempts."""
    note_context_fn = make_note_context_fn(mock_ctx.lifespan_context["config"])

    result = note_context_fn(path="../../etc/passwd", ctx=mock_ctx)

    assert "error" in result
    assert result["error"] == "Path outside vault"


def test_note_context_vault_name_resolves_correct_vault(tmp_path):
    """With 2 vaults, vault_name parameter selects the correct one."""
    vault_a = tmp_path / "vault_a"
    vault_a.mkdir()
    (vault_a / "note-in-a.md").write_text("# Note A\n\nContent in vault A.", encoding="utf-8")

    vault_b = tmp_path / "vault_b"
    vault_b.mkdir()
    (vault_b / "note-in-b.md").write_text("# Note B\n\nContent in vault B.", encoding="utf-8")

    config = AppConfig(
        vaults=[
            VaultConfig(name="vault_a", path=vault_a),
            VaultConfig(name="vault_b", path=vault_b),
        ]
    )

    vault_indexes = {
        "vault_a": {
            "index": MagicMock(),
            "metadata": {},
            "file_hashes": {},
            "vault_config": VaultConfig(name="vault_a", path=vault_a),
        },
        "vault_b": {
            "index": MagicMock(),
            "metadata": {},
            "file_hashes": {},
            "vault_config": VaultConfig(name="vault_b", path=vault_b),
        },
    }

    ctx = MagicMock()
    ctx.lifespan_context = {
        "vault_indexes": vault_indexes,
        "config": config,
        "index_lock": threading.Lock(),
    }

    note_context_fn = make_note_context_fn(config)
    assert note_context_fn is not None

    result = note_context_fn(path="note-in-b.md", vault_name="vault_b", ctx=ctx)

    assert "error" not in result, f"Unexpected error: {result}"
    assert "Note B" in result["note"]["content"]


def test_tool_not_registered_when_disabled(app_config):
    """note_context is NOT registered when removed from config.tools.enabled."""
    from obsidian_rag.models import ToolsConfig

    app_config.tools = ToolsConfig(enabled=["search", "read_note"])

    note_context_fn = make_note_context_fn(app_config)

    assert note_context_fn is None, "note_context should not be registered when disabled"


def test_note_context_deduplicates_forward_links(mock_ctx, vault_path):
    """A target linked twice in one note appears once in forward_links."""
    note = vault_path / "dupes.md"
    note.write_text(
        "# Dupes\n\nFirst [[wsn-pipeline]] and again [[wsn-pipeline]] here.",
        encoding="utf-8",
    )

    note_context_fn = make_note_context_fn(mock_ctx.lifespan_context["config"])
    result = note_context_fn(path="dupes.md", ctx=mock_ctx)

    assert "error" not in result
    targets = [link["path"] for link in result["forward_links"]]
    assert len(targets) == len(set(targets)), f"Duplicate forward links: {targets}"
