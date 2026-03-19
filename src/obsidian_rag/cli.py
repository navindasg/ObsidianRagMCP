import importlib.metadata
import logging

import click

from obsidian_rag.config import load_config


@click.command()
@click.option("--config", default="~/.obsidian-rag/config.yaml", help="Path to config file")
@click.option("--vault-path", default=None, help="Override vault path")
@click.option("--vault-name", default=None, help="Override vault name")
@click.option("--ollama-url", default=None, help="Override Ollama API URL")
@click.option("--verbose", is_flag=True, default=False, help="Set log level to INFO")
@click.option("--debug", is_flag=True, default=False, help="Set log level to DEBUG")
@click.version_option(
    importlib.metadata.version("obsidian-rag"),
    prog_name="obsidian-rag",
)
def cli(config, vault_path, vault_name, ollama_url, verbose, debug):
    """ObsidianRAG MCP server for Claude Desktop."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)

    cfg = load_config(
        config,
        overrides={
            "vault_path": vault_path,
            "vault_name": vault_name,
            "ollama_url": ollama_url,
        },
    )
    # Server startup deferred to plan 02
    _ = cfg
