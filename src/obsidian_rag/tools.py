"""MCP tool handlers for ObsidianRAG.

Exposes 7 tools: search, read_note, list_notes, find_notes, vault_stats, reindex, note_context.

Public API:
    register_tools(mcp: FastMCP, config: AppConfig) -> None
    _reindex_worker(vault_indexes, vault_name, config, vault_config, index_lock) -> None
    _reindex_locks: dict[str, bool]  -- module-level lock tracker for background reindex jobs
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import frontmatter
import ollama
from fastmcp import Context

from obsidian_rag.indexer import build_index
from obsidian_rag.models import AppConfig
from obsidian_rag.retriever import search as retriever_search
from obsidian_rag.wikilinks import find_backlinks, parse_wikilinks, resolve_wikilink

logger = logging.getLogger(__name__)

# Module-level tracker for running reindex jobs: vault_name -> bool
_reindex_locks: dict[str, bool] = {}


def register_tools(mcp, config: AppConfig) -> None:
    """Register MCP tool handlers based on config.tools.enabled.

    Each tool is only registered if its name appears in config.tools.enabled.
    Handlers close over ``config`` and read live state from ctx.lifespan_context.

    Args:
        mcp: FastMCP server instance.
        config: Full AppConfig with embedding, retrieval, and tools settings.
    """

    # -----------------------------------------------------------------------
    # 1. search
    # -----------------------------------------------------------------------

    if "search" in config.tools.enabled:

        @mcp.tool
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

            client = ollama.Client(host=cfg.embedding.ollama_url)
            embed_response = client.embed(model=cfg.embedding.model, input=[query])
            query_embedding: list[float] = embed_response.embeddings[0]

            # Determine which vaults to search
            if vault_name is not None and vault_name not in vault_indexes:
                return {
                    "error": "Vault not found",
                    "vault_name": vault_name,
                    "suggestion": f"Available vaults: {list(vault_indexes.keys())}",
                }

            if vault_name is not None:
                vaults_to_search = {vault_name: vault_indexes[vault_name]}
            else:
                vaults_to_search = vault_indexes

            all_results: list[dict] = []
            all_messages: list[str] = []

            for vname, vault_data in vaults_to_search.items():
                index = vault_data["index"]
                metadata = vault_data["metadata"]

                result = retriever_search(
                    index,
                    metadata,
                    query_embedding,
                    top_k=cfg.retrieval.top_k,
                    similarity_threshold=cfg.retrieval.similarity_threshold,
                    max_context_tokens=cfg.retrieval.max_context_tokens,
                    tags=tags,
                    folder=folder,
                    vault_name=vault_name,
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

    # -----------------------------------------------------------------------
    # 2. read_note
    # -----------------------------------------------------------------------

    if "read_note" in config.tools.enabled:

        @mcp.tool
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
            lifespan = ctx.lifespan_context
            vault_indexes: dict = lifespan["vault_indexes"]

            # Resolve vault
            if vault_name is not None and vault_name not in vault_indexes:
                return {
                    "error": "Vault not found",
                    "vault_name": vault_name,
                    "suggestion": f"Available vaults: {list(vault_indexes.keys())}",
                }

            if vault_name is not None:
                vault_data = vault_indexes[vault_name]
            else:
                vault_data = next(iter(vault_indexes.values()))

            vault_config = vault_data["vault_config"]
            vault_root = vault_config.path.resolve()

            # Resolve the requested path and check for traversal
            resolved = (vault_config.path / path).resolve()
            if not str(resolved).startswith(str(vault_root)):
                return {
                    "error": "Path outside vault",
                    "path": path,
                    "suggestion": "Use list_notes to browse available files",
                }

            if not resolved.exists():
                return {
                    "error": "File not found",
                    "path": path,
                    "suggestion": "Use list_notes to browse available files",
                }

            content = resolved.read_text(encoding="utf-8")

            try:
                post = frontmatter.loads(content)
                fm = dict(post.metadata)
            except Exception:
                logger.warning("Failed to parse frontmatter for %s", path)
                fm = {}

            return {"path": path, "content": content, "frontmatter": fm}

    # -----------------------------------------------------------------------
    # 3. list_notes
    # -----------------------------------------------------------------------

    if "list_notes" in config.tools.enabled:

        @mcp.tool
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

            # Resolve vault
            if vault_name is not None and vault_name not in vault_indexes:
                return {
                    "error": "Vault not found",
                    "vault_name": vault_name,
                    "suggestion": f"Available vaults: {list(vault_indexes.keys())}",
                }

            if vault_name is not None:
                vault_data = vault_indexes[vault_name]
            else:
                vault_data = next(iter(vault_indexes.values()))

            vault_config = vault_data["vault_config"]
            metadata: dict = vault_data["metadata"]
            excluded_dirs = set(vault_config.excluded_dirs)

            # Build tag_count lookup: file -> set of unique tags
            file_tags: dict[str, set[str]] = {}
            for chunk_meta in metadata.values():
                file_path = chunk_meta.get("file", "")
                tags_list = chunk_meta.get("tags", [])
                if file_path not in file_tags:
                    file_tags[file_path] = set()
                file_tags[file_path].update(tags_list)

            notes: list[dict] = []
            for md_file in sorted(vault_config.path.rglob("*.md")):
                rel = md_file.relative_to(vault_config.path)
                # Skip excluded directories
                if any(part in excluded_dirs for part in rel.parts[:-1]):
                    continue

                rel_str = str(rel)

                # Apply prefix filter
                if path_prefix is not None and not rel_str.startswith(path_prefix):
                    continue

                stat = md_file.stat()
                modified_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

                notes.append(
                    {
                        "path": rel_str,
                        "size": stat.st_size,
                        "modified": modified_dt.isoformat(),
                        "tag_count": len(file_tags.get(rel_str, set())),
                    }
                )

            return {"notes": notes}

    # -----------------------------------------------------------------------
    # 4. find_notes
    # -----------------------------------------------------------------------

    if "find_notes" in config.tools.enabled:

        @mcp.tool
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

            # Determine which vaults to search
            if vault_name is not None and vault_name not in vault_indexes:
                return {
                    "error": "Vault not found",
                    "vault_name": vault_name,
                    "suggestion": f"Available vaults: {list(vault_indexes.keys())}",
                }

            if vault_name is not None:
                vaults_to_search = {vault_name: vault_indexes[vault_name]}
            else:
                vaults_to_search = vault_indexes

            query_lower = query.lower()
            seen_files: set[str] = set()
            results: list[dict] = []

            for _vname, vault_data in vaults_to_search.items():
                metadata: dict = vault_data["metadata"]
                for chunk_meta in metadata.values():
                    file_path = chunk_meta.get("file", "")
                    heading_path = chunk_meta.get("heading_path", "")

                    file_matches = query_lower in file_path.lower()
                    heading_matches = query_lower in heading_path.lower()

                    if (file_matches or heading_matches) and file_path not in seen_files:
                        seen_files.add(file_path)
                        results.append({"file": file_path, "heading_path": heading_path})

            return {"results": results}

    # -----------------------------------------------------------------------
    # 5. vault_stats
    # -----------------------------------------------------------------------

    if "vault_stats" in config.tools.enabled:

        @mcp.tool
        def vault_stats(ctx: Context | None = None) -> dict:
            """Return statistics for each vault and aggregate totals.

            Returns:
                dict with "vaults" list (vault, note_count, chunk_count, index_age,
                embedding_model) and "total_notes", "total_chunks" aggregates.
            """
            lifespan = ctx.lifespan_context
            vault_indexes: dict = lifespan["vault_indexes"]
            cfg: AppConfig = lifespan["config"]

            vaults_stats: list[dict] = []
            total_notes = 0
            total_chunks = 0

            for vault_name, vault_data in vault_indexes.items():
                index = vault_data["index"]
                metadata: dict = vault_data["metadata"]

                note_count = len({m["file"] for m in metadata.values()})
                chunk_count = index.ntotal if index is not None else 0

                # Compute index age from metadata.json mtime if available
                vault_dir = Path.home() / ".obsidian-rag" / vault_name
                meta_path = vault_dir / "metadata.json"
                index_age: str | None = None
                if meta_path.exists():
                    mtime = meta_path.stat().st_mtime
                    age_dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
                    index_age = age_dt.isoformat()

                vaults_stats.append(
                    {
                        "vault": vault_name,
                        "note_count": note_count,
                        "chunk_count": chunk_count,
                        "index_age": index_age,
                        "embedding_model": cfg.embedding.model,
                    }
                )
                total_notes += note_count
                total_chunks += chunk_count

            return {
                "vaults": vaults_stats,
                "total_notes": total_notes,
                "total_chunks": total_chunks,
            }

    # -----------------------------------------------------------------------
    # 6. reindex
    # -----------------------------------------------------------------------

    if "reindex" in config.tools.enabled:

        @mcp.tool
        def reindex(vault_name: str, ctx: Context | None = None) -> dict:
            """Trigger a background reindex of a vault.

            Returns immediately with status "started" or "already_running".

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
                return {
                    "error": "Vault not found",
                    "vault": vault_name,
                    "suggestion": f"Available vaults: {list(vault_indexes.keys())}",
                }

            if _reindex_locks.get(vault_name):
                return {
                    "status": "already_running",
                    "vault": vault_name,
                    "message": "Reindex in progress",
                }

            _reindex_locks[vault_name] = True

            vault_data = vault_indexes[vault_name]
            metadata = vault_data["metadata"]
            vault_config = vault_data["vault_config"]
            note_count = len({m["file"] for m in metadata.values()})

            thread = threading.Thread(
                target=_reindex_worker,
                args=(vault_indexes, vault_name, cfg, vault_config, index_lock),
                daemon=True,
            )
            thread.start()

            return {
                "status": "started",
                "vault": vault_name,
                "note_count": note_count,
                "message": "Reindexing in background",
            }

    # -----------------------------------------------------------------------
    # 7. note_context
    # -----------------------------------------------------------------------

    if "note_context" in config.tools.enabled:

        @mcp.tool
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

            # Resolve vault (same pattern as read_note)
            if vault_name is not None and vault_name not in vault_indexes:
                return {
                    "error": "Vault not found",
                    "vault_name": vault_name,
                    "suggestion": f"Available vaults: {list(vault_indexes.keys())}",
                }

            if vault_name is not None:
                vault_data = vault_indexes[vault_name]
            else:
                vault_data = next(iter(vault_indexes.values()))

            vault_config = vault_data["vault_config"]
            metadata: dict = vault_data["metadata"]
            vault_root = vault_config.path.resolve()

            # Resolve path and check traversal
            resolved = (vault_config.path / path).resolve()
            if not str(resolved).startswith(str(vault_root)):
                return {
                    "error": "Path outside vault",
                    "path": path,
                    "suggestion": "Use list_notes to browse available files",
                }

            if not resolved.exists():
                return {
                    "error": "Note not found",
                    "path": path,
                    "suggestion": "Use find_notes to locate it",
                }

            content = resolved.read_text(encoding="utf-8")

            # Parse forward wikilinks (D-12)
            forward_targets = parse_wikilinks(content)
            forward_links: list[dict] = []
            seen_targets: set[str] = set()
            for target in forward_targets:
                if target in seen_targets:
                    continue
                seen_targets.add(target)
                matches = resolve_wikilink(target, vault_root)
                if matches:
                    for match in matches:
                        rel_path = str(match.relative_to(vault_root))
                        forward_links.append({"path": rel_path, "exists": True})
                else:
                    forward_links.append({"path": target, "exists": False})

            # Find backlinks (D-11) — scan metadata in memory
            note_basename = resolved.stem
            backlinks = find_backlinks(note_basename, metadata)

            return {
                "note": {"path": path, "content": content},
                "forward_links": forward_links,
                "backlinks": backlinks,
            }


def _reindex_worker(
    vault_indexes: dict,
    vault_name: str,
    config: AppConfig,
    vault_config,
    index_lock,
) -> None:
    """Background worker that rebuilds the FAISS index for a vault.

    Acquires index_lock before mutating vault_indexes entries to prevent
    a race condition with the watcher's _flush() which also mutates the index.

    Args:
        vault_indexes: Shared vault index dict from lifespan context.
        vault_name: Name of vault to reindex.
        config: Full AppConfig.
        vault_config: VaultConfig for the target vault.
        index_lock: threading.Lock shared with the file watcher.
    """
    try:
        new_index, new_metadata, new_file_hashes = build_index(config, vault_config)

        with index_lock:
            vault_indexes[vault_name]["index"] = new_index
            vault_indexes[vault_name]["metadata"] = new_metadata
            vault_indexes[vault_name]["file_hashes"] = new_file_hashes

        logger.info("Reindex complete for vault '%s': %d chunks", vault_name, new_index.ntotal)
    except Exception:
        logger.exception("Reindex failed for vault '%s'", vault_name)
    finally:
        _reindex_locks.pop(vault_name, None)
