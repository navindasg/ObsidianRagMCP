"""Click CLI entry point for the ObsidianRAG MCP server."""
import importlib.metadata
import logging
import sys

import click

from obsidian_rag.config import load_config
from obsidian_rag.server import run_server

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--config",
    "config_path",
    default="~/.obsidian-rag/config.yaml",
    help="Path to config file",
)
@click.option("--vault-path", default=None, help="Override vault path")
@click.option("--vault-name", default=None, help="Override vault name")
@click.option("--ollama-url", default=None, help="Override Ollama API URL")
@click.option("--verbose", is_flag=True, default=False, help="Set log level to INFO")
@click.option("--debug", is_flag=True, default=False, help="Set log level to DEBUG")
@click.version_option(
    importlib.metadata.version("obsidian-rag"),
    prog_name="obsidian-rag",
)
def cli(config_path, vault_path, vault_name, ollama_url, verbose, debug):
    """ObsidianRAG MCP server for Claude Desktop."""
    # Route logs to stderr (stdout carries the MCP stdio protocol). basicConfig
    # is a no-op when a handler already exists (e.g. the python -m entry point
    # configured one), but the console script lands here unconfigured — without
    # a handler, --verbose/--debug would change the level of a logger that
    # never emits anything below WARNING.
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)

    cfg = load_config(
        config_path,
        overrides={
            "vault_path": vault_path,
            "vault_name": vault_name,
            "ollama_url": ollama_url,
        },
    )

    vault_count = len(cfg.vaults)
    version = importlib.metadata.version("obsidian-rag")
    print(
        f"obsidian-rag v{version} | {vault_count} vault{'s' if vault_count != 1 else ''} | starting...",
        file=sys.stderr,
    )

    run_server(cfg)
