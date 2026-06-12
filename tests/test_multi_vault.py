"""Multi-vault isolation, cross-vault search, and vault_name scoping tests.

RED phase: Tests written before verifying all expected behaviors.
Validates VAULT-01 (independent indexes), VAULT-02 (no cross-contamination),
VAULT-03 (scoped search), D-18 (cross-vault merge), D-20 (invalid vault error).
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.models import AppConfig, ToolsConfig, VaultConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_VAULT_A = FIXTURES_DIR / "sample_vault"
SAMPLE_VAULT_B = FIXTURES_DIR / "sample_vault_b"


@pytest.fixture
def two_vault_dirs(tmp_path):
    """Copy sample vaults to tmp_path and return (vault_a_path, vault_b_path)."""
    vault_a = tmp_path / "vault_a"
    vault_b = tmp_path / "vault_b"
    shutil.copytree(SAMPLE_VAULT_A, vault_a)
    shutil.copytree(SAMPLE_VAULT_B, vault_b)
    return vault_a, vault_b


@pytest.fixture
def two_vault_config(two_vault_dirs):
    """Return AppConfig with two vaults: vault-a and vault-b."""
    vault_a_path, vault_b_path = two_vault_dirs
    return AppConfig(
        vaults=[
            VaultConfig(name="vault-a", path=vault_a_path),
            VaultConfig(name="vault-b", path=vault_b_path),
        ]
    )


def _make_chunk_meta(chunk_id: int, file: str, vault: str, text: str, tags: list[str] | None = None) -> dict:
    """Build a minimal chunk metadata dict."""
    return {
        "chunk_id": chunk_id,
        "file": file,
        "heading_path": f"# {file}",
        "text": text,
        "tags": tags or [],
        "folder": file.split("/")[0] if "/" in file else "",
        "vault": vault,
        "modified_ts": 1700000000.0,
        "char_count": len(text),
    }


@pytest.fixture
def two_vault_indexes(two_vault_dirs):
    """Return a mock vault_indexes dict with two independent vaults."""
    vault_a_path, vault_b_path = two_vault_dirs

    idx_a = MagicMock()
    idx_a.ntotal = 2
    idx_b = MagicMock()
    idx_b.ntotal = 1

    meta_a = {
        "0": _make_chunk_meta(0, "notes/hello.md", "vault-a", "Hello world from vault A", ["greeting"]),
        "1": _make_chunk_meta(1, "projects/readme.md", "vault-a", "Projects readme from vault A", ["project"]),
    }
    meta_b = {
        "10": _make_chunk_meta(10, "vault-b-note.md", "vault-b", "Distributed systems from vault B", ["vault-b"]),
    }

    return {
        "vault-a": {
            "index": idx_a,
            "metadata": meta_a,
            "file_hashes": {},
            "vault_config": VaultConfig(name="vault-a", path=vault_a_path),
        },
        "vault-b": {
            "index": idx_b,
            "metadata": meta_b,
            "file_hashes": {},
            "vault_config": VaultConfig(name="vault-b", path=vault_b_path),
        },
    }


@pytest.fixture
def mock_ctx_two_vaults(two_vault_indexes, two_vault_config):
    """Return a mock FastMCP context with two-vault lifespan context."""
    ctx = MagicMock()
    ctx.lifespan_context = {
        "vault_indexes": two_vault_indexes,
        "config": two_vault_config,
        "index_lock": threading.Lock(),
    }
    return ctx


def make_mcp_and_register(config):
    """Register tools with given config and return registered tool functions."""
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
    return mcp, registered


# ---------------------------------------------------------------------------
# Test 1: Two vaults produce independent indexes with no cross-contamination
# ---------------------------------------------------------------------------


def test_two_vaults_independent_indexes(two_vault_indexes):
    """vault_indexes has 2 entries with distinct metadata (VAULT-01, VAULT-02)."""
    assert len(two_vault_indexes) == 2
    assert "vault-a" in two_vault_indexes
    assert "vault-b" in two_vault_indexes

    meta_a = two_vault_indexes["vault-a"]["metadata"]
    meta_b = two_vault_indexes["vault-b"]["metadata"]

    # No cross-contamination: vault-a metadata only has vault-a entries
    for chunk in meta_a.values():
        assert chunk["vault"] == "vault-a", f"Expected vault-a, got {chunk['vault']}"

    # vault-b metadata only has vault-b entries
    for chunk in meta_b.values():
        assert chunk["vault"] == "vault-b", f"Expected vault-b, got {chunk['vault']}"

    # Indexes are different objects
    assert two_vault_indexes["vault-a"]["index"] is not two_vault_indexes["vault-b"]["index"]

    # Vault configs point to different paths
    assert (
        two_vault_indexes["vault-a"]["vault_config"].path
        != two_vault_indexes["vault-b"]["vault_config"].path
    )


# ---------------------------------------------------------------------------
# Test 2: Search scoped to vault-a only returns vault-a results
# ---------------------------------------------------------------------------


def test_search_scoped_to_vault_a(mock_ctx_two_vaults):
    """Search with vault_name='vault-a' only searches vault-a index (VAULT-03)."""
    _, registered = make_mcp_and_register(mock_ctx_two_vaults.lifespan_context["config"])
    search_fn = registered["search"]

    fake_embedding = [0.1] * 768
    captured_calls = []

    def fake_search(index, metadata, query_embedding, **kwargs):
        captured_calls.append({"index": index})
        return {
            "results": [
                {
                    "source_path": "notes/hello.md",
                    "heading_path": "# Hello",
                    "relevance_score": 0.90,
                    "snippet": "Hello world from vault A",
                    "vault_name": "vault-a",
                }
            ]
        }

    with (
        patch("obsidian_rag.tools.ollama.Client") as mock_client_cls,
        patch("obsidian_rag.tools.retriever_search", side_effect=fake_search),
    ):
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=[fake_embedding])
        mock_client_cls.return_value = mock_client

        result = search_fn(query="hello", vault_name="vault-a", ctx=mock_ctx_two_vaults)

    # Should only have called retriever_search once, against vault-a's index
    vault_a_index = mock_ctx_two_vaults.lifespan_context["vault_indexes"]["vault-a"]["index"]
    assert len(captured_calls) == 1
    assert captured_calls[0]["index"] is vault_a_index

    # All results should be from vault-a
    for r in result["results"]:
        assert r["vault_name"] == "vault-a"


# ---------------------------------------------------------------------------
# Test 3: Search scoped to vault-b only returns vault-b results
# ---------------------------------------------------------------------------


def test_search_scoped_to_vault_b(mock_ctx_two_vaults):
    """Search with vault_name='vault-b' only searches vault-b index (VAULT-03)."""
    _, registered = make_mcp_and_register(mock_ctx_two_vaults.lifespan_context["config"])
    search_fn = registered["search"]

    fake_embedding = [0.1] * 768
    captured_calls = []

    def fake_search(index, metadata, query_embedding, vault_name=None, **kwargs):
        captured_calls.append({"index": index, "vault_name": vault_name})
        return {
            "results": [
                {
                    "source_path": "vault-b-note.md",
                    "heading_path": "# Vault B Note",
                    "relevance_score": 0.88,
                    "snippet": "Distributed systems from vault B",
                    "vault_name": "vault-b",
                }
            ]
        }

    with (
        patch("obsidian_rag.tools.ollama.Client") as mock_client_cls,
        patch("obsidian_rag.tools.retriever_search", side_effect=fake_search),
    ):
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=[fake_embedding])
        mock_client_cls.return_value = mock_client

        result = search_fn(query="distributed", vault_name="vault-b", ctx=mock_ctx_two_vaults)

    # Should only have called retriever_search once (for vault-b only)
    assert len(captured_calls) == 1

    # All results should be from vault-b
    for r in result["results"]:
        assert r["vault_name"] == "vault-b"


# ---------------------------------------------------------------------------
# Test 4: Cross-vault search merges results sorted by relevance
# ---------------------------------------------------------------------------


def test_cross_vault_search_merges(mock_ctx_two_vaults):
    """Search with vault_name=None returns merged results from all vaults sorted by score (D-18)."""
    _, registered = make_mcp_and_register(mock_ctx_two_vaults.lifespan_context["config"])
    search_fn = registered["search"]

    fake_embedding = [0.1] * 768
    vault_calls: list[str] = []

    def fake_search(index, metadata, query_embedding, vault_name=None, **kwargs):
        # Identify which vault by checking which index was passed
        vi = mock_ctx_two_vaults.lifespan_context["vault_indexes"]
        if index is vi["vault-a"]["index"]:
            vault_calls.append("vault-a")
            return {
                "results": [
                    {
                        "source_path": "notes/hello.md",
                        "heading_path": "# Hello",
                        "relevance_score": 0.75,
                        "snippet": "Hello from vault A",
                        "vault_name": "vault-a",
                    }
                ]
            }
        else:
            vault_calls.append("vault-b")
            return {
                "results": [
                    {
                        "source_path": "vault-b-note.md",
                        "heading_path": "# Vault B Note",
                        "relevance_score": 0.90,
                        "snippet": "Distributed systems from vault B",
                        "vault_name": "vault-b",
                    }
                ]
            }

    with (
        patch("obsidian_rag.tools.ollama.Client") as mock_client_cls,
        patch("obsidian_rag.tools.retriever_search", side_effect=fake_search),
    ):
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=[fake_embedding])
        mock_client_cls.return_value = mock_client

        result = search_fn(query="test query", vault_name=None, ctx=mock_ctx_two_vaults)

    # Both vaults should have been searched
    assert "vault-a" in vault_calls
    assert "vault-b" in vault_calls
    assert len(vault_calls) == 2

    # Results should contain entries from both vaults
    result_vaults = {r["vault_name"] for r in result["results"]}
    assert "vault-a" in result_vaults
    assert "vault-b" in result_vaults

    # Results should be sorted by relevance_score descending
    scores = [r["relevance_score"] for r in result["results"]]
    assert scores == sorted(scores, reverse=True), f"Results not sorted: {scores}"


# ---------------------------------------------------------------------------
# Test 5: read_note respects vault_name
# ---------------------------------------------------------------------------


def test_read_note_respects_vault_name(mock_ctx_two_vaults):
    """read_note with vault_name='vault-b' reads from vault-b directory."""
    _, registered = make_mcp_and_register(mock_ctx_two_vaults.lifespan_context["config"])
    read_note_fn = registered["read_note"]

    result = read_note_fn(path="vault-b-note.md", vault_name="vault-b", ctx=mock_ctx_two_vaults)

    assert "error" not in result, f"Unexpected error: {result}"
    assert result["path"] == "vault-b-note.md"
    assert "content" in result
    assert "vault-b" in result["content"].lower() or "vault b" in result["content"].lower()


# ---------------------------------------------------------------------------
# Test 6: list_notes respects vault_name
# ---------------------------------------------------------------------------


def test_list_notes_respects_vault_name(mock_ctx_two_vaults):
    """list_notes with vault_name='vault-b' returns only vault-b files."""
    _, registered = make_mcp_and_register(mock_ctx_two_vaults.lifespan_context["config"])
    list_notes_fn = registered["list_notes"]

    result = list_notes_fn(vault_name="vault-b", ctx=mock_ctx_two_vaults)

    assert "notes" in result
    # vault-b only has vault-b-note.md
    paths = [n["path"] for n in result["notes"]]
    assert len(paths) >= 1
    for p in paths:
        assert "vault-b" in p or p.endswith(".md"), f"Unexpected file in vault-b: {p}"

    # vault-a files should not appear
    assert not any("hello" in p for p in paths), "vault-a files should not appear in vault-b listing"
    assert not any("projects" in p for p in paths), "vault-a files should not appear in vault-b listing"


# ---------------------------------------------------------------------------
# Test 7: find_notes searches all vaults when vault_name=None
# ---------------------------------------------------------------------------


def test_find_notes_searches_all_vaults(mock_ctx_two_vaults):
    """find_notes with vault_name=None returns matches from both vaults."""
    _, registered = make_mcp_and_register(mock_ctx_two_vaults.lifespan_context["config"])
    find_notes_fn = registered["find_notes"]

    # Use a common query that would match metadata entries in both vaults
    # The mock metadata has "notes/hello.md" in vault-a and "vault-b-note.md" in vault-b
    result = find_notes_fn(query="", vault_name=None, ctx=mock_ctx_two_vaults)  # empty = matches all

    assert "results" in result
    # With empty query all entries from both vaults match
    files = [r["file"] for r in result["results"]]
    # Should have files from vault-a
    assert any("hello" in f or "readme" in f for f in files), f"No vault-a files found: {files}"
    # Should have files from vault-b
    assert any("vault-b" in f for f in files), f"No vault-b files found: {files}"


# ---------------------------------------------------------------------------
# Test 8: Invalid vault_name returns error dict (D-20)
# ---------------------------------------------------------------------------


def test_invalid_vault_name_error(mock_ctx_two_vaults):
    """All file-access tools return error dict with 'Vault not found' for unknown vault (D-20)."""
    _, registered = make_mcp_and_register(mock_ctx_two_vaults.lifespan_context["config"])

    fake_embedding = [0.1] * 768

    # Test read_note
    read_note_fn = registered["read_note"]
    result = read_note_fn(path="some-note.md", vault_name="nonexistent", ctx=mock_ctx_two_vaults)
    assert result["error"] == "Vault not found"
    assert result["vault_name"] == "nonexistent"
    assert "suggestion" in result
    assert "vault-a" in result["suggestion"]
    assert "vault-b" in result["suggestion"]

    # Test list_notes
    list_notes_fn = registered["list_notes"]
    result = list_notes_fn(vault_name="nonexistent", ctx=mock_ctx_two_vaults)
    assert result["error"] == "Vault not found"
    assert result["vault_name"] == "nonexistent"
    assert "suggestion" in result

    # Test find_notes
    find_notes_fn = registered["find_notes"]
    result = find_notes_fn(query="test", vault_name="nonexistent", ctx=mock_ctx_two_vaults)
    assert result["error"] == "Vault not found"
    assert result["vault_name"] == "nonexistent"
    assert "suggestion" in result

    # Test search
    search_fn = registered["search"]
    with (
        patch("obsidian_rag.tools.ollama.Client") as mock_client_cls,
    ):
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=[fake_embedding])
        mock_client_cls.return_value = mock_client

        result = search_fn(query="test", vault_name="nonexistent", ctx=mock_ctx_two_vaults)

    assert result["error"] == "Vault not found"
    assert result["vault_name"] == "nonexistent"
    assert "suggestion" in result

    # Test note_context
    note_context_fn = registered["note_context"]
    result = note_context_fn(path="some-note.md", vault_name="nonexistent", ctx=mock_ctx_two_vaults)
    assert result["error"] == "Vault not found"
    assert result["vault_name"] == "nonexistent"
    assert "suggestion" in result


# ---------------------------------------------------------------------------
# Test 9: note_context with vault_name resolves within correct vault
# ---------------------------------------------------------------------------


def test_note_context_with_vault_name(mock_ctx_two_vaults):
    """note_context with vault_name='vault-b' resolves note within vault-b."""
    _, registered = make_mcp_and_register(mock_ctx_two_vaults.lifespan_context["config"])
    note_context_fn = registered["note_context"]

    result = note_context_fn(path="vault-b-note.md", vault_name="vault-b", ctx=mock_ctx_two_vaults)

    assert "error" not in result, f"Unexpected error: {result}"
    assert "note" in result
    assert "forward_links" in result
    assert "backlinks" in result
    assert result["note"]["path"] == "vault-b-note.md"


# ---------------------------------------------------------------------------
# Test 10: Single vault auto-infer when vault_name=None
# ---------------------------------------------------------------------------


def test_single_vault_auto_infer(tmp_path):
    """With one vault configured, vault_name=None auto-uses the single vault."""
    vault = tmp_path / "my_vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Single Vault Note\n\nContent here.", encoding="utf-8")

    config = AppConfig(vaults=[VaultConfig(name="only-vault", path=vault)])

    mock_index = MagicMock()
    mock_index.ntotal = 1

    vault_indexes = {
        "only-vault": {
            "index": mock_index,
            "metadata": {
                "0": _make_chunk_meta(0, "note.md", "only-vault", "Content here."),
            },
            "file_hashes": {},
            "vault_config": VaultConfig(name="only-vault", path=vault),
        }
    }

    ctx = MagicMock()
    ctx.lifespan_context = {
        "vault_indexes": vault_indexes,
        "config": config,
        "index_lock": threading.Lock(),
    }

    _, registered = make_mcp_and_register(config)
    read_note_fn = registered["read_note"]

    # vault_name=None should auto-infer the single vault
    result = read_note_fn(path="note.md", vault_name=None, ctx=ctx)

    assert "error" not in result, f"Unexpected error: {result}"
    assert result["path"] == "note.md"
    assert "Content here" in result["content"]


def test_find_notes_scoped_to_vault(mock_ctx_two_vaults):
    """find_notes with a valid vault_name returns only that vault's files."""
    _, registered = make_mcp_and_register(mock_ctx_two_vaults.lifespan_context["config"])
    find_notes_fn = registered["find_notes"]

    result = find_notes_fn(query="", vault_name="vault-b", ctx=mock_ctx_two_vaults)

    files = [r["file"] for r in result["results"]]
    assert files, "Expected vault-b notes"
    assert all("vault-b" in f or f == "vault-b-note.md" for f in files) or all(
        f not in ("notes/hello.md", "projects/readme.md") for f in files
    ), f"vault-a files leaked into scoped find_notes: {files}"
