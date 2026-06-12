"""Tests for CLI flags and config override behavior."""
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
