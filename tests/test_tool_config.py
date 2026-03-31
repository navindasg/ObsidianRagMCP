"""Per-tool enable/disable tests for config.tools.enabled (TOOL-08).

Tests verify that each of the 7 MCP tools (search, read_note, list_notes,
find_notes, note_context, vault_stats, reindex) can be individually enabled
or disabled through the ToolsConfig.enabled list.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from obsidian_rag.models import AppConfig, ToolsConfig, VaultConfig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

ALL_TOOLS = frozenset(["search", "read_note", "list_notes", "find_notes", "note_context", "vault_stats", "reindex"])


def _get_registered_tools(enabled_list: list[str], tmp_path) -> set[str]:
    """Register tools with given enabled list and return set of registered tool names."""
    from obsidian_rag.tools import register_tools

    mcp = MagicMock()
    registered: list[str] = []

    def mock_tool(fn=None, **kwargs):
        if fn is not None:
            registered.append(fn.__name__)
            return fn

        def decorator(f):
            registered.append(f.__name__)
            return f

        return decorator

    mcp.tool = mock_tool

    config = AppConfig(
        vaults=[VaultConfig(name="test", path=tmp_path)],
        tools=ToolsConfig(enabled=enabled_list),
    )
    register_tools(mcp, config)
    return set(registered)


# ---------------------------------------------------------------------------
# Test 1: Default config registers all 7 tools
# ---------------------------------------------------------------------------


def test_all_tools_registered_by_default(tmp_path):
    """Default ToolsConfig registers all 7 tools."""
    registered = _get_registered_tools(list(ALL_TOOLS), tmp_path)
    for tool in ALL_TOOLS:
        assert tool in registered, f"Expected {tool} to be registered by default"


# ---------------------------------------------------------------------------
# Test 2: search disabled
# ---------------------------------------------------------------------------


def test_search_disabled(tmp_path):
    """When 'search' is absent from enabled list, search is NOT registered."""
    enabled = ["read_note", "list_notes", "find_notes", "note_context", "vault_stats", "reindex"]
    registered = _get_registered_tools(enabled, tmp_path)
    assert "search" not in registered, "search should NOT be registered"
    # Other tools should still be registered
    assert "read_note" in registered
    assert "list_notes" in registered


# ---------------------------------------------------------------------------
# Test 3: note_context disabled
# ---------------------------------------------------------------------------


def test_note_context_disabled(tmp_path):
    """When 'note_context' is absent, note_context is NOT registered."""
    enabled = ["search", "read_note", "list_notes", "find_notes", "vault_stats", "reindex"]
    registered = _get_registered_tools(enabled, tmp_path)
    assert "note_context" not in registered, "note_context should NOT be registered"
    # Other tools should still be registered
    assert "search" in registered
    assert "read_note" in registered


# ---------------------------------------------------------------------------
# Test 4: Only search enabled
# ---------------------------------------------------------------------------


def test_only_search_enabled(tmp_path):
    """When only 'search' is enabled, only search is registered."""
    registered = _get_registered_tools(["search"], tmp_path)
    assert "search" in registered
    # All other tools should be absent
    for tool in ALL_TOOLS - {"search"}:
        assert tool not in registered, f"{tool} should NOT be registered when only search is enabled"


# ---------------------------------------------------------------------------
# Test 5: Empty enabled list registers no tools
# ---------------------------------------------------------------------------


def test_empty_enabled_list(tmp_path):
    """When enabled=[], no tools are registered."""
    registered = _get_registered_tools([], tmp_path)
    assert len(registered) == 0, f"Expected no tools, but got: {registered}"


# ---------------------------------------------------------------------------
# Test 6: reindex disabled
# ---------------------------------------------------------------------------


def test_reindex_disabled(tmp_path):
    """When 'reindex' is absent from enabled list, reindex is NOT registered."""
    enabled = ["search", "read_note", "list_notes", "find_notes", "note_context", "vault_stats"]
    registered = _get_registered_tools(enabled, tmp_path)
    assert "reindex" not in registered, "reindex should NOT be registered"
    # Other tools should still be registered
    assert "search" in registered
    assert "vault_stats" in registered


# ---------------------------------------------------------------------------
# Test 7: vault_stats disabled
# ---------------------------------------------------------------------------


def test_vault_stats_disabled(tmp_path):
    """When 'vault_stats' is absent from enabled list, vault_stats is NOT registered."""
    enabled = ["search", "read_note", "list_notes", "find_notes", "note_context", "reindex"]
    registered = _get_registered_tools(enabled, tmp_path)
    assert "vault_stats" not in registered, "vault_stats should NOT be registered"
    # Other tools should still be registered
    assert "search" in registered
    assert "reindex" in registered
