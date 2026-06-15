"""Click CLI entry point for the ObsidianRAG MCP server.

Bare invocation starts the MCP server (unchanged behavior). Subcommands:
``format-daily`` runs one nightly daily-note formatting pass, and
``schedule install|uninstall|status`` manages the launchd LaunchAgent.
"""
import datetime
import importlib.metadata
import logging
import sys

import click

from obsidian_rag.config import load_config
from obsidian_rag.daily_format import launchd
from obsidian_rag.daily_format.runner import run_format_daily
from obsidian_rag.server import run_server

logger = logging.getLogger(__name__)

_CONFIG_OPTION_KWARGS = {
    "default": "~/.obsidian-rag/config.yaml",
    "help": "Path to config file",
}


def _setup_logging(verbose: bool, debug: bool) -> None:
    """Route logs to stderr (stdout carries the MCP stdio protocol).

    basicConfig is a no-op when a handler already exists (e.g. the python -m
    entry point configured one), but the console script lands here
    unconfigured — without a handler, --verbose/--debug would change the
    level of a logger that never emits anything below WARNING.
    """
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


@click.group(invoke_without_command=True)
@click.option("--config", "config_path", **_CONFIG_OPTION_KWARGS)
@click.option("--vault-path", default=None, help="Override vault path")
@click.option("--vault-name", default=None, help="Override vault name")
@click.option("--ollama-url", default=None, help="Override Ollama API URL")
@click.option("--verbose", is_flag=True, default=False, help="Set log level to INFO")
@click.option("--debug", is_flag=True, default=False, help="Set log level to DEBUG")
@click.version_option(
    importlib.metadata.version("obsidian-rag"),
    prog_name="obsidian-rag",
)
@click.pass_context
def cli(ctx, config_path, vault_path, vault_name, ollama_url, verbose, debug):
    """ObsidianRAG MCP server for Claude Desktop."""
    _setup_logging(verbose, debug)
    if ctx.invoked_subcommand is not None:
        return

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


def _parse_override_date(date_str: str | None, flag: str) -> datetime.date | None:
    """Parse a date-valued flag, exiting politely on bad input."""
    if date_str is None:
        return None
    try:
        return datetime.date.fromisoformat(date_str)
    except ValueError:
        raise SystemExit(
            f"Invalid {flag} '{date_str}': expected YYYY-MM-DD"
        ) from None


@cli.command("format-daily")
@click.option("--config", "config_path", **_CONFIG_OPTION_KWARGS)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Enqueue and report, but do not call Ollama or rewrite notes",
)
@click.option(
    "--tags-only",
    is_flag=True,
    default=False,
    help="Only pick up format-tagged notes (used by the background poll)",
)
@click.option(
    "--since",
    "since_str",
    default=None,
    metavar="YYYY-MM-DD",
    help="Backfill: format every daily note from this date on, including "
    "the most recent (lifts the latest-note hold)",
)
def format_daily(
    config_path: str,
    dry_run: bool,
    tags_only: bool,
    since_str: str | None,
) -> None:
    """Format raw daily notes (and format-tagged notes) via Ollama."""
    if tags_only and since_str is not None:
        raise SystemExit(
            "--since backfills daily notes; --tags-only skips them. "
            "Use one or the other."
        )
    cfg = load_config(config_path)
    if not cfg.daily_format.enabled:
        raise SystemExit(
            "Daily-note formatting is disabled.\n"
            "Fix: enable daily_format in config "
            "(set daily_format.enabled: true)"
        )
    since = _parse_override_date(since_str, "--since")
    summary = run_format_daily(
        cfg, dry_run=dry_run, tags_only=tags_only, since=since
    )
    rendered = " ".join(f"{key}={value}" for key, value in summary.items())
    click.echo(f"format-daily: {rendered}", err=True)
    sys.exit(1 if summary.get("failed", 0) > 0 else 0)


@cli.group()
def schedule() -> None:
    """Manage the launchd LaunchAgent for nightly formatting."""


@schedule.command("install")
@click.option("--config", "config_path", **_CONFIG_OPTION_KWARGS)
def schedule_install(config_path: str) -> None:
    """Install (or reinstall) the nightly and tag-poll LaunchAgents."""
    cfg = load_config(config_path)
    plists = launchd.install(cfg)
    daily = cfg.daily_format
    for plist in plists:
        click.echo(f"Installed LaunchAgent: {plist}", err=True)
    click.echo(
        f"Nightly run: daily at {daily.schedule_hour:02d}:{daily.schedule_minute:02d} "
        "(launchd fires missed runs on wake)",
        err=True,
    )
    click.echo(
        f"Tag poll: every {daily.poll_minutes} min in the background "
        "(low priority, picks up #!format tags)",
        err=True,
    )


@schedule.command("uninstall")
@click.option("--config", "config_path", **_CONFIG_OPTION_KWARGS)
def schedule_uninstall(config_path: str) -> None:
    """Remove the nightly LaunchAgent."""
    launchd.uninstall()
    click.echo("Uninstalled LaunchAgent (if it was installed).", err=True)


@schedule.command("status")
@click.option("--config", "config_path", **_CONFIG_OPTION_KWARGS)
def schedule_status(config_path: str) -> None:
    """Show the LaunchAgent's launchd status."""
    click.echo(launchd.status(), err=True)
