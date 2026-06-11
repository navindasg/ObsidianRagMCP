"""End-to-end MCP integration tests through a real FastMCP client.

These tests exercise the full tool-call path (client -> FastMCP -> handler)
so context injection is tested for real, not simulated by passing ctx manually.
Regression coverage for the bug where ctx lacked a Context type annotation and
FastMCP never injected it, making every tool crash on real MCP calls.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastmcp import Client, FastMCP

from obsidian_rag.models import AppConfig, VaultConfig
from obsidian_rag.tools import register_tools


@pytest.fixture
def vault_dir(tmp_path):
    """A real vault directory with one markdown note."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "hello.md").write_text("# Hello\n\nA note about greetings.", encoding="utf-8")
    return vault


@pytest.fixture
def app_config(vault_dir):
    return AppConfig(vaults=[VaultConfig(name="test", path=vault_dir)])


def _make_metadata() -> dict:
    return {
        "0": {
            "chunk_id": 0,
            "file": "hello.md",
            "heading_path": "# Hello",
            "text": "A note about greetings.",
            "tags": [],
            "folder": "",
            "vault": "test",
            "modified_ts": 1700000000.0,
            "char_count": 24,
        }
    }


def _make_server(config: AppConfig) -> FastMCP:
    """Build a FastMCP server with a lifespan yielding a prebuilt context."""
    index = MagicMock()
    index.ntotal = 1

    vault_indexes = {
        "test": {
            "index": index,
            "metadata": _make_metadata(),
            "file_hashes": {},
            "vault_config": config.vaults[0],
        }
    }

    @asynccontextmanager
    async def lifespan(server: FastMCP):
        yield {
            "vault_indexes": vault_indexes,
            "config": config,
            "index_lock": threading.Lock(),
        }

    mcp = FastMCP("obsidian-rag-test", lifespan=lifespan)
    register_tools(mcp, config)
    return mcp


def _call_tool(config: AppConfig, tool: str, args: dict) -> object:
    """Call a tool through a real in-memory MCP client and return its data."""
    mcp = _make_server(config)

    async def run():
        async with Client(mcp) as client:
            result = await client.call_tool(tool, args)
            return result.data

    return asyncio.run(run())


def test_find_notes_via_real_mcp_call(app_config):
    """ctx must be injected by FastMCP — not crash with AttributeError on None."""
    data = _call_tool(app_config, "find_notes", {"query": "hello"})
    assert "error" not in data
    assert data["results"] == [{"file": "hello.md", "heading_path": "# Hello"}]


def test_list_notes_via_real_mcp_call(app_config):
    data = _call_tool(app_config, "list_notes", {})
    assert "error" not in data
    assert [n["path"] for n in data["notes"]] == ["hello.md"]


def test_read_note_via_real_mcp_call(app_config):
    data = _call_tool(app_config, "read_note", {"path": "hello.md"})
    assert "error" not in data
    assert "greetings" in data["content"]


def test_vault_stats_via_real_mcp_call(app_config):
    data = _call_tool(app_config, "vault_stats", {})
    assert data["total_notes"] == 1
    assert data["vaults"][0]["vault"] == "test"


def test_note_context_via_real_mcp_call(app_config):
    data = _call_tool(app_config, "note_context", {"path": "hello.md"})
    assert "error" not in data
    assert data["note"]["path"] == "hello.md"


def test_every_default_tool_is_callable_via_mcp(app_config):
    """All 7 default tools must be registered and reachable via a real client."""
    mcp = _make_server(app_config)

    async def run():
        async with Client(mcp) as client:
            tools = await client.list_tools()
            return sorted(t.name for t in tools)

    names = asyncio.run(run())
    assert names == sorted(app_config.tools.enabled)
