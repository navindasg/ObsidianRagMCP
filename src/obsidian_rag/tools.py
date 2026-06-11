"""MCP tool handlers for ObsidianRAG.

Exposes 7 tools: search, read_note, list_notes, find_notes, vault_stats,
reindex, note_context. Handlers are module-level functions registered by
register_tools() according to config.tools.enabled; each reads live state
from ctx.lifespan_context.

Locking: the watcher and background reindex mutate vault index state under
lifespan_context["index_lock"]. Every handler that reads the FAISS index or
chunk metadata takes the same lock so queries never observe a half-applied
update.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import frontmatter
import ollama
from fastmcp import Context

from obsidian_rag.indexer import build_index, is_excluded, vault_storage_dir
from obsidian_rag.models import AppConfig, VaultConfig
from obsidian_rag.retriever import search as retriever_search
from obsidian_rag.wikilinks import (
    build_note_index,
    find_backlinks,
    parse_wikilinks,
    resolve_wikilink,
)

logger = logging.getLogger(__name__)

# Tracker for running background reindex jobs: vault_name -> True.
# _reindex_guard makes the check-then-set atomic across tool-handler threads.
_reindex_locks: dict[str, bool] = {}
_reindex_guard = threading.Lock()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _vault_not_found(vault_indexes: dict, vault_name: str) -> dict:
    return {
        "error": "Vault not found",
        "vault_name": vault_name,
        "suggestion": f"Available vaults: {list(vault_indexes.keys())}",
    }


def _resolve_vault(
    vault_indexes: dict, vault_name: str | None
) -> tuple[dict | None, dict | None]:
    """Return (vault_data, error). Uses the first vault when vault_name is None."""
    if vault_name is not None:
        if vault_name not in vault_indexes:
            return None, _vault_not_found(vault_indexes, vault_name)
        return vault_indexes[vault_name], None
    return next(iter(vault_indexes.values())), None


def _select_vaults(
    vault_indexes: dict, vault_name: str | None
) -> tuple[dict | None, dict | None]:
    """Return ({name: vault_data}, error). Uses all vaults when vault_name is None."""
    if vault_name is not None:
        if vault_name not in vault_indexes:
            return None, _vault_not_found(vault_indexes, vault_name)
        return {vault_name: vault_indexes[vault_name]}, None
    return vault_indexes, None


def _resolve_note_path(
    vault_config: VaultConfig, path: str
) -> tuple[Path | None, dict | None]:
    """Resolve a vault-relative note path, rejecting traversal and non-notes.

    is_relative_to() gives true path containment — a plain string prefix
    check would accept sibling directories like <vault>-private.
    """
    vault_root = vault_config.path.resolve()
    resolved = (vault_config.path / path).resolve()

    if not resolved.is_relative_to(vault_root):
        return None, {
            "error": "Path outside vault",
            "path": path,
            "suggestion": "Use list_notes to browse available files",
        }

    rel = resolved.relative_to(vault_root)
    if resolved.suffix != ".md" or is_excluded(
        rel, vault_config.excluded_dirs, vault_config.excluded_patterns
    ):
        return None, {
            "error": "Not an accessible note",
            "path": path,
            "suggestion": "Only .md notes outside excluded directories can be read",
        }

    return resolved, None


def _read_note_text(resolved: Path, path: str) -> tuple[str | None, dict | None]:
    """Read note content, mapping I/O failures to the structured error dict."""
    try:
        return resolved.read_text(encoding="utf-8"), None
    except (OSError, UnicodeDecodeError):
        logger.exception("Could not read note %s", resolved)
        return None, {
            "error": "Could not read file",
            "path": path,
            "suggestion": "The file may not be valid UTF-8 text",
        }


# ---------------------------------------------------------------------------
# 1. search
# ---------------------------------------------------------------------------


def search(
    query: str,
    vault_name: str | None = None,
    tags: list[str] | None = None,
    folder: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """Semantic search across one or all Obsidian vaults.

    Args:
        query: Natural language search query.
        vault_name: Restrict search to this vault (None = all vaults).
        tags: Filter results to chunks with at least one of these tags.
        folder: Filter results to chunks whose file path starts with folder.
        ctx: FastMCP context (injected automatically).

    Returns:
        dict with "results" list of SearchResult dicts, and optional "message".
    """
    lifespan = ctx.lifespan_context
    vault_indexes: dict = lifespan["vault_indexes"]
    cfg: AppConfig = lifespan["config"]
    index_lock = lifespan["index_lock"]

    vaults_to_search, err = _select_vaults(vault_indexes, vault_name)
    if err is not None:
        return err

    try:
        client = ollama.Client(host=cfg.embedding.ollama_url)
        embed_response = client.embed(model=cfg.embedding.model, input=[query])
        query_embedding: list[float] = embed_response.embeddings[0]
    except Exception:
        logger.exception("Query embedding failed")
        return {
            "error": "Embedding failed — Ollama unreachable",
            "suggestion": (
                f"Ensure Ollama is running at {cfg.embedding.ollama_url} (ollama serve)"
            ),
        }

    all_results: list[dict] = []
    all_messages: list[str] = []

    for vault_data in vaults_to_search.values():
        with index_lock:
            result = retriever_search(
                vault_data["index"],
                vault_data["metadata"],
                query_embedding,
                top_k=cfg.retrieval.top_k,
                similarity_threshold=cfg.retrieval.similarity_threshold,
                max_context_tokens=cfg.retrieval.max_context_tokens,
                tags=tags,
                folder=folder,
                query_text=query,
                rerank_config=cfg.rerank,
                ollama_url=cfg.embedding.ollama_url,
            )
        all_results.extend(result.get("results", []))
        if "message" in result:
            all_messages.append(result["message"])

    # Sort by relevance_score descending, cap at top_k
    all_results.sort(key=lambda r: r.get("relevance_score", 0.0), reverse=True)
    all_results = all_results[: cfg.retrieval.top_k]

    merged: dict = {"results": all_results}
    if not all_results and all_messages:
        merged["message"] = all_messages[0]

    return merged


# ---------------------------------------------------------------------------
# 2. read_note
# ---------------------------------------------------------------------------


def read_note(
    path: str,
    vault_name: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """Read the full content of a note from the vault.

    Args:
        path: Relative path to the markdown file within the vault.
        vault_name: Target vault (uses first vault if None).
        ctx: FastMCP context (injected automatically).

    Returns:
        dict with "path", "content", "frontmatter" on success, or
        "error", "path", "suggestion" on failure.
    """
    vault_indexes: dict = ctx.lifespan_context["vault_indexes"]

    vault_data, err = _resolve_vault(vault_indexes, vault_name)
    if err is not None:
        return err

    resolved, err = _resolve_note_path(vault_data["vault_config"], path)
    if err is not None:
        return err

    if not resolved.exists():
        return {
            "error": "File not found",
            "path": path,
            "suggestion": "Use list_notes to browse available files",
        }

    content, err = _read_note_text(resolved, path)
    if err is not None:
        return err

    try:
        post = frontmatter.loads(content)
        fm = dict(post.metadata)
    except Exception:
        logger.warning("Failed to parse frontmatter for %s", path)
        fm = {}

    return {"path": path, "content": content, "frontmatter": fm}


# ---------------------------------------------------------------------------
# 3. list_notes
# ---------------------------------------------------------------------------


def list_notes(
    path_prefix: str | None = None,
    vault_name: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """List all markdown files in a vault with metadata.

    Args:
        path_prefix: Only return files whose relative path starts with this prefix.
        vault_name: Target vault (uses first vault if None).
        ctx: FastMCP context (injected automatically).

    Returns:
        dict with "notes" list; each entry has path, size, modified, tag_count.
    """
    lifespan = ctx.lifespan_context
    vault_indexes: dict = lifespan["vault_indexes"]
    index_lock = lifespan["index_lock"]

    vault_data, err = _resolve_vault(vault_indexes, vault_name)
    if err is not None:
        return err

    vault_config: VaultConfig = vault_data["vault_config"]

    # Build tag_count lookup under the lock: file -> set of unique tags
    file_tags: dict[str, set[str]] = {}
    with index_lock:
        for chunk_meta in vault_data["metadata"].values():
            file_path = chunk_meta.get("file", "")
            file_tags.setdefault(file_path, set()).update(chunk_meta.get("tags", []))

    notes: list[dict] = []
    for md_file in sorted(vault_config.path.rglob("*.md")):
        rel = md_file.relative_to(vault_config.path)
        if is_excluded(rel, vault_config.excluded_dirs, vault_config.excluded_patterns):
            continue

        rel_str = str(rel)
        if path_prefix is not None and not rel_str.startswith(path_prefix):
            continue

        stat = md_file.stat()
        notes.append(
            {
                "path": rel_str,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "tag_count": len(file_tags.get(rel_str, set())),
            }
        )

    return {"notes": notes}


# ---------------------------------------------------------------------------
# 4. find_notes
# ---------------------------------------------------------------------------


def find_notes(
    query: str,
    vault_name: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """Find notes by filename or heading substring (case-insensitive).

    Args:
        query: Substring to search for in file names and heading paths.
        vault_name: Target vault (uses all vaults if None).
        ctx: FastMCP context (injected automatically).

    Returns:
        dict with "results" list; each entry has file and heading_path.
    """
    lifespan = ctx.lifespan_context
    vault_indexes: dict = lifespan["vault_indexes"]
    index_lock = lifespan["index_lock"]

    vaults_to_search, err = _select_vaults(vault_indexes, vault_name)
    if err is not None:
        return err

    query_lower = query.lower()
    seen_files: set[str] = set()
    results: list[dict] = []

    with index_lock:
        for vault_data in vaults_to_search.values():
            for chunk_meta in vault_data["metadata"].values():
                file_path = chunk_meta.get("file", "")
                heading_path = chunk_meta.get("heading_path", "")

                matches = (
                    query_lower in file_path.lower()
                    or query_lower in heading_path.lower()
                )
                if matches and file_path not in seen_files:
                    seen_files.add(file_path)
                    results.append({"file": file_path, "heading_path": heading_path})

    return {"results": results}


# ---------------------------------------------------------------------------
# 5. vault_stats
# ---------------------------------------------------------------------------


def vault_stats(ctx: Context | None = None) -> dict:
    """Return statistics for each vault and aggregate totals.

    Returns:
        dict with "vaults" list (vault, note_count, chunk_count, index_age,
        embedding_model, last_reindex) and "total_notes", "total_chunks".
    """
    lifespan = ctx.lifespan_context
    vault_indexes: dict = lifespan["vault_indexes"]
    cfg: AppConfig = lifespan["config"]
    index_lock = lifespan["index_lock"]

    vaults_stats: list[dict] = []
    total_notes = 0
    total_chunks = 0

    for vault_name, vault_data in vault_indexes.items():
        with index_lock:
            index = vault_data["index"]
            note_count = len({m["file"] for m in vault_data["metadata"].values()})
            chunk_count = index.ntotal if index is not None else 0
            last_reindex = vault_data.get("last_reindex")

        # Compute index age from metadata.json mtime if available
        meta_path = vault_storage_dir(vault_name) / "metadata.json"
        index_age: str | None = None
        if meta_path.exists():
            index_age = datetime.fromtimestamp(
                meta_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()

        vaults_stats.append(
            {
                "vault": vault_name,
                "note_count": note_count,
                "chunk_count": chunk_count,
                "index_age": index_age,
                "embedding_model": cfg.embedding.model,
                "last_reindex": last_reindex,
            }
        )
        total_notes += note_count
        total_chunks += chunk_count

    return {
        "vaults": vaults_stats,
        "total_notes": total_notes,
        "total_chunks": total_chunks,
    }


# ---------------------------------------------------------------------------
# 6. reindex
# ---------------------------------------------------------------------------


def reindex(vault_name: str, ctx: Context | None = None) -> dict:
    """Trigger a background reindex of a vault.

    Returns immediately with status "started" or "already_running". The
    outcome of the last completed reindex is visible via vault_stats.

    Args:
        vault_name: Name of the vault to reindex.
        ctx: FastMCP context (injected automatically).

    Returns:
        dict with "status", "vault", and "message" keys.
    """
    lifespan = ctx.lifespan_context
    vault_indexes: dict = lifespan["vault_indexes"]
    cfg: AppConfig = lifespan["config"]
    index_lock = lifespan["index_lock"]

    if vault_name not in vault_indexes:
        return _vault_not_found(vault_indexes, vault_name)

    with _reindex_guard:
        if _reindex_locks.get(vault_name):
            return {
                "status": "already_running",
                "vault": vault_name,
                "message": "Reindex in progress",
            }
        _reindex_locks[vault_name] = True

    vault_data = vault_indexes[vault_name]
    with index_lock:
        note_count = len({m["file"] for m in vault_data["metadata"].values()})

    thread = threading.Thread(
        target=_reindex_worker,
        args=(vault_indexes, vault_name, cfg, vault_data["vault_config"], index_lock),
        daemon=True,
    )
    thread.start()

    return {
        "status": "started",
        "vault": vault_name,
        "note_count": note_count,
        "message": "Reindexing in background",
    }


# ---------------------------------------------------------------------------
# 7. note_context
# ---------------------------------------------------------------------------


def note_context(
    path: str,
    vault_name: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """Return a note plus its single-hop backlinks and forward wikilinks.

    Args:
        path: Relative path to the markdown file within the vault.
        vault_name: Target vault (uses first vault if None).
        ctx: FastMCP context (injected automatically).

    Returns:
        dict with "note", "forward_links", "backlinks" on success,
        or "error", "path", "suggestion" on failure.
    """
    lifespan = ctx.lifespan_context
    vault_indexes: dict = lifespan["vault_indexes"]
    index_lock = lifespan["index_lock"]

    vault_data, err = _resolve_vault(vault_indexes, vault_name)
    if err is not None:
        return err

    vault_config: VaultConfig = vault_data["vault_config"]
    vault_root = vault_config.path.resolve()

    resolved, err = _resolve_note_path(vault_config, path)
    if err is not None:
        return err

    if not resolved.exists():
        return {
            "error": "Note not found",
            "path": path,
            "suggestion": "Use find_notes to locate it",
        }

    content, err = _read_note_text(resolved, path)
    if err is not None:
        return err

    # Parse forward wikilinks (D-12); one note index serves all targets
    forward_targets = parse_wikilinks(content)
    note_index = build_note_index(vault_root) if forward_targets else {}
    forward_links: list[dict] = []
    seen_targets: set[str] = set()
    for target in forward_targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        matches = resolve_wikilink(target, vault_root, note_index=note_index)
        if matches:
            for match in matches:
                forward_links.append(
                    {"path": str(match.relative_to(vault_root)), "exists": True}
                )
        else:
            forward_links.append({"path": target, "exists": False})

    # Find backlinks (D-11) — scan metadata in memory, under the lock
    with index_lock:
        backlinks = find_backlinks(resolved.stem, vault_data["metadata"])

    return {
        "note": {"path": path, "content": content},
        "forward_links": forward_links,
        "backlinks": backlinks,
    }


# ---------------------------------------------------------------------------
# Background reindex worker
# ---------------------------------------------------------------------------


def _reindex_worker(
    vault_indexes: dict,
    vault_name: str,
    config: AppConfig,
    vault_config: VaultConfig,
    index_lock: threading.Lock,
) -> None:
    """Background worker that rebuilds the FAISS index for a vault.

    Acquires index_lock before mutating vault_indexes entries to prevent
    a race condition with the watcher's _flush() which also mutates the
    index. Records the outcome in vault_indexes[vault_name]["last_reindex"]
    so failures are observable via vault_stats.

    Args:
        vault_indexes: Shared vault index dict from lifespan context.
        vault_name: Name of vault to reindex.
        config: Full AppConfig.
        vault_config: VaultConfig for the target vault.
        index_lock: threading.Lock shared with the file watcher.
    """
    finished_at = lambda: datetime.now(timezone.utc).isoformat()  # noqa: E731
    try:
        new_index, new_metadata, new_file_hashes = build_index(config, vault_config)

        with index_lock:
            vault_indexes[vault_name]["index"] = new_index
            vault_indexes[vault_name]["metadata"] = new_metadata
            vault_indexes[vault_name]["file_hashes"] = new_file_hashes
            vault_indexes[vault_name]["last_reindex"] = {
                "status": "ok",
                "completed_at": finished_at(),
                "error": None,
            }

        logger.info("Reindex complete for vault '%s': %d chunks", vault_name, new_index.ntotal)
    except Exception as exc:
        logger.exception("Reindex failed for vault '%s'", vault_name)
        vault_indexes[vault_name]["last_reindex"] = {
            "status": "failed",
            "completed_at": finished_at(),
            "error": str(exc),
        }
    finally:
        with _reindex_guard:
            _reindex_locks.pop(vault_name, None)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_TOOL_HANDLERS = {
    "search": search,
    "read_note": read_note,
    "list_notes": list_notes,
    "find_notes": find_notes,
    "vault_stats": vault_stats,
    "reindex": reindex,
    "note_context": note_context,
}


def register_tools(mcp, config: AppConfig) -> None:
    """Register MCP tool handlers based on config.tools.enabled.

    Each tool is only registered if its name appears in config.tools.enabled;
    names without a matching handler are ignored.

    Args:
        mcp: FastMCP server instance.
        config: Full AppConfig with embedding, retrieval, and tools settings.
    """
    for name in config.tools.enabled:
        handler = _TOOL_HANDLERS.get(name)
        if handler is not None:
            mcp.tool(handler)
        else:
            logger.warning("Unknown tool name in config.tools.enabled: %r", name)
