"""Tests for CLI flags, config override behavior, and subcommands."""
import datetime
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from obsidian_rag.cli import cli


@pytest.fixture
def config_file(tmp_path):
    """Write a minimal valid config YAML to a temp file."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    config = {"vaults": [{"name": "test-vault", "path": str(vault_dir)}]}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    return config_path


@pytest.fixture
def daily_config_file(tmp_path):
    """Write a config YAML with daily_format enabled to a temp file."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir(exist_ok=True)
    config = {
        "vaults": [{"name": "test-vault", "path": str(vault_dir)}],
        "daily_format": {"enabled": True},
    }
    config_path = tmp_path / "daily-config.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    return config_path


def test_cli_help():
    """--help exits 0 and shows all flags."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
    assert "--vault-path" in result.output
    assert "--vault-name" in result.output
    assert "--ollama-url" in result.output
    assert "--verbose" in result.output
    assert "--debug" in result.output


def test_cli_version():
    """--version prints program name and version."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "obsidian-rag" in result.output
    assert "0.1.0" in result.output


def test_cli_override_vault_path(tmp_path, config_file):
    """--vault-path overrides the vault path in config."""
    captured = {}

    def fake_run_server(cfg):
        captured["cfg"] = cfg

    runner = CliRunner()
    with patch("obsidian_rag.cli.run_server", side_effect=fake_run_server):
        result = runner.invoke(
            cli,
            ["--config", str(config_file), "--vault-path", str(tmp_path)],
        )

    assert result.exit_code == 0, result.output
    assert captured["cfg"].vaults[0].path == tmp_path


def test_cli_override_vault_name(tmp_path, config_file):
    """--vault-name overrides the vault name in config."""
    captured = {}

    def fake_run_server(cfg):
        captured["cfg"] = cfg

    runner = CliRunner()
    with patch("obsidian_rag.cli.run_server", side_effect=fake_run_server):
        result = runner.invoke(
            cli,
            ["--config", str(config_file), "--vault-name", "custom"],
        )

    assert result.exit_code == 0, result.output
    assert captured["cfg"].vaults[0].name == "custom"


def test_cli_override_ollama_url(tmp_path, config_file):
    """--ollama-url overrides the Ollama URL in config."""
    captured = {}

    def fake_run_server(cfg):
        captured["cfg"] = cfg

    runner = CliRunner()
    with patch("obsidian_rag.cli.run_server", side_effect=fake_run_server):
        result = runner.invoke(
            cli,
            ["--config", str(config_file), "--ollama-url", "http://custom:9999"],
        )

    assert result.exit_code == 0, result.output
    assert captured["cfg"].embedding.ollama_url == "http://custom:9999"


def test_cli_verbose_sets_info(config_file):
    """--verbose sets root logger level to INFO."""
    runner = CliRunner()
    with patch("obsidian_rag.cli.run_server"):
        result = runner.invoke(cli, ["--config", str(config_file), "--verbose"])

    assert result.exit_code == 0, result.output
    assert logging.getLogger().level == logging.INFO


def test_cli_debug_sets_debug(config_file):
    """--debug sets root logger level to DEBUG."""
    runner = CliRunner()
    with patch("obsidian_rag.cli.run_server"):
        result = runner.invoke(cli, ["--config", str(config_file), "--debug"])

    assert result.exit_code == 0, result.output
    assert logging.getLogger().level == logging.DEBUG


# ---------------------------------------------------------------------------
# python -m obsidian_rag entry point (regression: __main__ had 0% coverage)
# ---------------------------------------------------------------------------


def test_python_m_entry_point_runs():
    """python -m obsidian_rag --version executes __main__.py end to end."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "obsidian_rag", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0
    assert "obsidian-rag" in result.stdout


# ---------------------------------------------------------------------------
# format-daily subcommand
# ---------------------------------------------------------------------------


def test_help_lists_subcommands():
    """--help shows the new subcommands."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "format-daily" in result.output
    assert "schedule" in result.output


def test_subcommand_does_not_start_server(daily_config_file):
    """Invoking a subcommand never runs the MCP server."""
    runner = CliRunner()
    with (
        patch("obsidian_rag.cli.run_server") as mock_server,
        patch(
            "obsidian_rag.cli.run_format_daily",
            return_value={"enqueued": 0, "formatted": 0, "failed": 0, "skipped": 0},
        ),
    ):
        result = runner.invoke(
            cli, ["format-daily", "--config", str(daily_config_file)]
        )

    assert result.exit_code == 0, result.output
    mock_server.assert_not_called()


def test_format_daily_invokes_runner_and_exits_zero(daily_config_file):
    """format-daily loads config, calls the runner, prints a stderr summary."""
    summary = {"enqueued": 1, "formatted": 1, "failed": 0, "skipped": 0}
    runner = CliRunner()
    with patch("obsidian_rag.cli.run_format_daily", return_value=summary) as mock_run:
        result = runner.invoke(
            cli, ["format-daily", "--config", str(daily_config_file)]
        )

    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    assert kwargs["dry_run"] is False
    assert kwargs["since"] is None
    assert "formatted=1" in result.stderr


def test_format_daily_exits_one_on_failures(daily_config_file):
    """format-daily exits 1 when the summary reports failures."""
    summary = {"enqueued": 2, "formatted": 0, "failed": 2, "skipped": 0}
    runner = CliRunner()
    with patch("obsidian_rag.cli.run_format_daily", return_value=summary):
        result = runner.invoke(
            cli, ["format-daily", "--config", str(daily_config_file)]
        )

    assert result.exit_code == 1


def test_format_daily_passes_dry_run(daily_config_file):
    """--dry-run is forwarded to run_format_daily."""
    summary = {"enqueued": 0, "pending": [], "formatted": 0, "failed": 0}
    runner = CliRunner()
    with patch("obsidian_rag.cli.run_format_daily", return_value=summary) as mock_run:
        result = runner.invoke(
            cli,
            ["format-daily", "--config", str(daily_config_file), "--dry-run"],
        )

    assert result.exit_code == 0, result.output
    assert mock_run.call_args.kwargs["dry_run"] is True


def test_format_daily_disabled_config_exits_with_message(config_file):
    """format-daily refuses to run when daily_format.enabled is false."""
    runner = CliRunner()
    with patch("obsidian_rag.cli.run_format_daily") as mock_run:
        result = runner.invoke(cli, ["format-daily", "--config", str(config_file)])

    assert result.exit_code != 0
    assert "enable daily_format in config" in result.output
    mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# schedule subcommands
# ---------------------------------------------------------------------------


def test_schedule_install_prints_plist_path_and_next_run(daily_config_file):
    """schedule install delegates to launchd.install and reports to stderr."""
    runner = CliRunner()
    with patch(
        "obsidian_rag.cli.launchd.install",
        return_value=[
            Path("/fake/LaunchAgents/com.obsidian-rag.daily-format.plist"),
            Path("/fake/LaunchAgents/com.obsidian-rag.format-tag-poll.plist"),
        ],
    ) as mock_install:
        result = runner.invoke(
            cli, ["schedule", "install", "--config", str(daily_config_file)]
        )

    assert result.exit_code == 0, result.output
    mock_install.assert_called_once()
    cfg = mock_install.call_args.args[0]
    assert cfg.daily_format.enabled is True
    assert "com.obsidian-rag.daily-format.plist" in result.stderr
    assert "com.obsidian-rag.format-tag-poll.plist" in result.stderr
    assert "00:30" in result.stderr  # default schedule_hour=0, schedule_minute=30
    assert "5 min" in result.stderr  # default poll_minutes=5


def test_schedule_uninstall_delegates(daily_config_file):
    """schedule uninstall delegates to launchd.uninstall."""
    runner = CliRunner()
    with patch("obsidian_rag.cli.launchd.uninstall") as mock_uninstall:
        result = runner.invoke(
            cli, ["schedule", "uninstall", "--config", str(daily_config_file)]
        )

    assert result.exit_code == 0, result.output
    mock_uninstall.assert_called_once_with()


def test_schedule_status_prints_status(daily_config_file):
    """schedule status prints launchd.status() output to stderr."""
    runner = CliRunner()
    with patch(
        "obsidian_rag.cli.launchd.status", return_value="state = waiting"
    ) as mock_status:
        result = runner.invoke(
            cli, ["schedule", "status", "--config", str(daily_config_file)]
        )

    assert result.exit_code == 0, result.output
    mock_status.assert_called_once_with()
    assert "state = waiting" in result.stderr


def test_format_daily_passes_tags_only_and_since(daily_config_file):
    """--tags-only and --since are forwarded to run_format_daily."""
    summary = {"enqueued": 0, "formatted": 0, "failed": 0, "skipped": 0}
    runner = CliRunner()
    with patch("obsidian_rag.cli.run_format_daily", return_value=summary) as mock_run:
        result = runner.invoke(
            cli,
            ["format-daily", "--config", str(daily_config_file), "--tags-only"],
        )
    assert result.exit_code == 0, result.output
    assert mock_run.call_args.kwargs["tags_only"] is True

    with patch("obsidian_rag.cli.run_format_daily", return_value=summary) as mock_run:
        result = runner.invoke(
            cli,
            [
                "format-daily",
                "--config",
                str(daily_config_file),
                "--since",
                "2026-03-19",
            ],
        )
    assert result.exit_code == 0, result.output
    assert mock_run.call_args.kwargs["since"] == datetime.date(2026, 3, 19)


def test_format_daily_rejects_bad_since(daily_config_file):
    """An unparseable --since exits non-zero with a helpful message."""
    runner = CliRunner()
    with patch("obsidian_rag.cli.run_format_daily") as mock_run:
        result = runner.invoke(
            cli,
            ["format-daily", "--config", str(daily_config_file), "--since", "junk"],
        )

    assert result.exit_code != 0
    assert "YYYY-MM-DD" in result.output
    mock_run.assert_not_called()


def test_format_daily_since_conflicts_with_tags_only(daily_config_file):
    """--since (backfill dailies) and --tags-only are mutually exclusive."""
    runner = CliRunner()
    with patch("obsidian_rag.cli.run_format_daily") as mock_run:
        result = runner.invoke(
            cli,
            [
                "format-daily",
                "--config",
                str(daily_config_file),
                "--since",
                "2026-03-19",
                "--tags-only",
            ],
        )

    assert result.exit_code != 0
    assert "--since" in result.output and "--tags-only" in result.output
    mock_run.assert_not_called()
