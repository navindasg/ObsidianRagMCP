"""FastMCP server with Ollama health-check lifespan."""
import importlib.metadata
import logging
import sys
import threading
from contextlib import asynccontextmanager

import ollama as ollama_client
from fastmcp import FastMCP

import obsidian_rag
from obsidian_rag.indexer import build_index
from obsidian_rag.models import DEFAULT_RERANK_MODEL, AppConfig
from obsidian_rag.tools import register_tools
from obsidian_rag.watcher import VaultWatcher

logger = logging.getLogger(__name__)


def _check_ollama_health(config: AppConfig) -> None:
    """Verify Ollama is reachable and the configured embedding model is pulled.

    Raises SystemExit with an actionable error message on failure.
    """
    client = ollama_client.Client(host=config.embedding.ollama_url)
    try:
        response = client.list()
    except Exception as exc:
        raise SystemExit(
            f"Ollama is not reachable at {config.embedding.ollama_url}\n"
            "Fix: ensure Ollama is running (ollama serve)"
        ) from exc

    config_model_base = config.embedding.model.split(":")[0]
    pulled = {m.model.split(":")[0] for m in response.models}
    if config_model_base not in pulled:
        raise SystemExit(
            f"Embedding model '{config.embedding.model}' not found in Ollama.\n"
            f"Fix: run: ollama pull {config.embedding.model}"
        )

    # Validate rerank model when reranking is enabled
    if config.rerank.enabled:
        rerank_model = config.rerank.model or DEFAULT_RERANK_MODEL
        rerank_base = rerank_model.split(":")[0]
        if rerank_base not in pulled:
            raise SystemExit(
                f"Rerank model '{rerank_model}' not found in Ollama.\n"
                f"Fix: run: ollama pull {rerank_model}"
            )


def create_server(config: AppConfig) -> FastMCP:
    """Create a FastMCP server with a lifespan that validates Ollama health."""

    @asynccontextmanager
    async def lifespan(server: FastMCP):
        _check_ollama_health(config)

        vault_count = len(config.vaults)
        try:
            version_str = importlib.metadata.version("obsidian-rag")
        except importlib.metadata.PackageNotFoundError:
            version_str = obsidian_rag.__version__

        print(
            f"obsidian-rag v{version_str} | {vault_count} vault{'s' if vault_count != 1 else ''} | Ollama OK",
            file=sys.stderr,
        )

        # Index each vault (blocks startup per CONTEXT.md decision)
        vault_indexes = {}
        for vault_cfg in config.vaults:
            logger.info("Indexing vault: %s", vault_cfg.name)
            index, metadata, file_hashes = build_index(config, vault_cfg)
            vault_indexes[vault_cfg.name] = {
                "index": index,
                "metadata": metadata,
                "file_hashes": file_hashes,
                "vault_config": vault_cfg,
            }
            chunk_count = index.ntotal if index is not None else 0
            logger.info("Vault '%s' indexed: %d chunks", vault_cfg.name, chunk_count)

        print(
            f"Indexing complete | {sum(v['index'].ntotal for v in vault_indexes.values() if v['index'])} chunks",
            file=sys.stderr,
        )

        logger.info("Server started successfully")

        index_lock = threading.Lock()
        watcher = VaultWatcher(vault_indexes, config, index_lock=index_lock)
        watcher.start()  # no-op if watch_enabled=False

        try:
            yield {"vault_indexes": vault_indexes, "config": config, "index_lock": index_lock}
        finally:
            watcher.stop()
            logger.info("Server shutting down")

    mcp = FastMCP("obsidian-rag", lifespan=lifespan)
    register_tools(mcp, config)
    return mcp


def run_server(config: AppConfig) -> None:
    """Create and run the MCP server over stdio."""
    server = create_server(config)
    server.run()
