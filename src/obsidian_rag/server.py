"""FastMCP server with Ollama health-check lifespan."""
import importlib.metadata
import logging
import sys
from contextlib import asynccontextmanager

import ollama as ollama_client
from fastmcp import FastMCP

from obsidian_rag.models import AppConfig

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


def create_server(config: AppConfig) -> FastMCP:
    """Create a FastMCP server with a lifespan that validates Ollama health."""

    @asynccontextmanager
    async def lifespan(server: FastMCP):
        _check_ollama_health(config)

        vault_count = len(config.vaults)
        try:
            version_str = importlib.metadata.version("obsidian-rag")
        except Exception:
            version_str = "0.1.0"

        print(
            f"obsidian-rag v{version_str} | {vault_count} vault{'s' if vault_count != 1 else ''} | Ollama OK",
            file=sys.stderr,
        )
        logger.info("Server started successfully")
        yield {}
        logger.info("Server shutting down")

    mcp = FastMCP("obsidian-rag", lifespan=lifespan)
    return mcp


def run_server(config: AppConfig) -> None:
    """Create and run the MCP server over stdio."""
    server = create_server(config)
    server.run()
