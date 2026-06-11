"""Tests for config loading, validation, default generation, and CLI overrides.

Covers requirements: INFRA-04 (defaults), INFRA-05 (required field validation,
collect-all-errors), and config bootstrapping (auto-generate).
"""
import yaml
import pytest

from obsidian_rag.config import load_config, generate_default_config
from obsidian_rag.models import AppConfig, VaultConfig


def _write_config(path, data):
    """Helper: write a dict as YAML to a config file path."""
    path.write_text(yaml.dump(data), encoding="utf-8")
    return str(path)


def test_valid_config_loads(tmp_path):
    """Valid config with name+path loads successfully and returns AppConfig."""
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, {"vaults": [{"name": "test", "path": str(tmp_path)}]})

    result = load_config(str(config_file))

    assert isinstance(result, AppConfig)
    assert result.vaults[0].name == "test"


def test_default_values(tmp_path):
    """All optional fields fall back to documented defaults."""
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, {"vaults": [{"name": "test", "path": str(tmp_path)}]})

    cfg = load_config(str(config_file))

    assert cfg.embedding.model == "nomic-embed-text"
    assert cfg.embedding.ollama_url == "http://localhost:11434"
    assert cfg.embedding.batch_size == 64
    assert cfg.indexing.chunk_strategy == "heading"
    assert cfg.retrieval.top_k == 5
    assert cfg.retrieval.similarity_threshold == 0.7
    assert cfg.rerank.enabled is False
    assert cfg.tools.enabled == [
        "search",
        "read_note",
        "list_notes",
        "find_notes",
        "note_context",
        "vault_stats",
        "reindex",
    ]


def test_required_field_vault_name(tmp_path):
    """Config missing vault name raises SystemExit containing 'name'."""
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, {"vaults": [{"path": str(tmp_path)}]})

    with pytest.raises(SystemExit) as exc_info:
        load_config(str(config_file))

    assert "name" in str(exc_info.value)


def test_required_field_vault_path(tmp_path):
    """Config missing vault path raises SystemExit containing 'path'."""
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, {"vaults": [{"name": "test"}]})

    with pytest.raises(SystemExit) as exc_info:
        load_config(str(config_file))

    assert "path" in str(exc_info.value)


def test_all_errors_reported(tmp_path):
    """Both name and path missing produces error listing BOTH fields."""
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, {"vaults": [{}]})

    with pytest.raises(SystemExit) as exc_info:
        load_config(str(config_file))

    error_msg = str(exc_info.value)
    assert "name" in error_msg
    assert "path" in error_msg


def test_nonexistent_vault_path(tmp_path):
    """Config with nonexistent vault path raises error containing 'does not exist'."""
    config_file = tmp_path / "config.yaml"
    _write_config(
        config_file,
        {"vaults": [{"name": "test", "path": "/nonexistent/path/abc123"}]},
    )

    with pytest.raises(SystemExit) as exc_info:
        load_config(str(config_file))

    assert "does not exist" in str(exc_info.value)


def test_tilde_expansion(tmp_path):
    """Tilde in config path is expanded to the user's home directory.

    We verify the expansion by writing a config with a tilde path pointing
    to a real directory relative to the actual home, and confirming the
    resulting Path has no tilde.
    """
    from pathlib import Path
    import os

    # Use the actual home directory; pick a subdir that exists (e.g. ~ itself)
    home = Path.home()
    # We only test that tilde expansion happens — use a config YAML approach
    # so we can pick a path we know exists (home directory itself)
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, {"vaults": [{"name": "test", "path": "~"}]})

    cfg = load_config(str(config_file))

    assert "~" not in str(cfg.vaults[0].path)
    assert cfg.vaults[0].path == home


def test_auto_generate_config(tmp_path):
    """Missing config file triggers generate_default_config and raises SystemExit."""
    config_path = tmp_path / "subdir" / "config.yaml"

    with pytest.raises(SystemExit) as exc_info:
        load_config(str(config_path))

    error_msg = str(exc_info.value)
    assert "created default" in error_msg
    assert config_path.exists()
    assert "YOUR_VAULT_NAME" in config_path.read_text(encoding="utf-8")


def test_cli_override_vault_path(tmp_path):
    """CLI vault_path override replaces config vault path."""
    new_vault = tmp_path / "new_vault"
    new_vault.mkdir()
    config_file = tmp_path / "config.yaml"
    # Config has a different vault path — override should win
    old_vault = tmp_path / "old_vault"
    old_vault.mkdir()
    _write_config(config_file, {"vaults": [{"name": "test", "path": str(old_vault)}]})

    cfg = load_config(str(config_file), overrides={"vault_path": str(new_vault)})

    assert cfg.vaults[0].path == new_vault


def test_cli_override_vault_name(tmp_path):
    """CLI vault_name override replaces config vault name."""
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, {"vaults": [{"name": "original", "path": str(tmp_path)}]})

    cfg = load_config(str(config_file), overrides={"vault_name": "overridden"})

    assert cfg.vaults[0].name == "overridden"


def test_cli_override_ollama_url(tmp_path):
    """CLI ollama_url override replaces config embedding URL."""
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, {"vaults": [{"name": "test", "path": str(tmp_path)}]})

    cfg = load_config(
        str(config_file), overrides={"ollama_url": "http://custom:1234"}
    )

    assert cfg.embedding.ollama_url == "http://custom:1234"


# ---------------------------------------------------------------------------
# Regression tests: null vaults override, invalid YAML, typo'd enum values
# ---------------------------------------------------------------------------


def test_override_with_null_vaults_key(tmp_path):
    """A bare 'vaults:' key (YAML null) must not crash _apply_overrides."""
    from obsidian_rag.config import load_config

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("vaults:\n")

    cfg = load_config(
        str(config_file),
        overrides={"vault_path": str(vault_dir), "vault_name": "cli-vault"},
    )

    assert cfg.vaults[0].name == "cli-vault"
    assert cfg.vaults[0].path == vault_dir


def test_invalid_yaml_exits_with_friendly_message(tmp_path):
    """Malformed YAML produces a SystemExit message, not a raw traceback."""
    from obsidian_rag.config import load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text("vaults: [unclosed\n  - oops")

    with pytest.raises(SystemExit) as exc_info:
        load_config(str(config_file))

    assert "not valid YAML" in str(exc_info.value)


def test_chunk_strategy_typo_rejected(tmp_path):
    """A typo'd chunk_strategy fails validation instead of silently defaulting."""
    from obsidian_rag.models import AppConfig
    from pydantic import ValidationError

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()

    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "vaults": [{"name": "v", "path": str(vault_dir)}],
                "indexing": {"chunk_strategy": "fiexd"},
            }
        )


def test_include_frontmatter_typo_rejected(tmp_path):
    """An invalid include_frontmatter mode fails validation at load time."""
    from obsidian_rag.models import AppConfig
    from pydantic import ValidationError

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()

    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "vaults": [{"name": "v", "path": str(vault_dir)}],
                "indexing": {"include_frontmatter": "everything"},
            }
        )
