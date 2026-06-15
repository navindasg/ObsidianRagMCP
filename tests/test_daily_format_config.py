"""Tests for DailyFormatConfig: defaults, validation, and config wiring.

Covers the daily_format configuration model in models.py, its wiring into
AppConfig, the commented daily_format section in DEFAULT_CONFIG, and YAML
round-tripping through load_config.
"""
import pytest
import yaml
from pydantic import ValidationError

from obsidian_rag.config import DEFAULT_CONFIG, load_config
from obsidian_rag.models import AppConfig, DailyFormatConfig


def _write_config(path, data):
    """Helper: write a dict as YAML to a config file path."""
    path.write_text(yaml.dump(data), encoding="utf-8")
    return str(path)


def _vaults_section(tmp_path):
    """Helper: a minimal valid vaults entry backed by a real directory."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir(exist_ok=True)
    return [{"name": "test", "path": str(vault_dir)}]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults():
    """DailyFormatConfig defaults match the documented values."""
    cfg = DailyFormatConfig()

    assert cfg.enabled is False
    assert cfg.daily_folder == ""
    assert cfg.filename_format == "%Y-%m-%d"
    assert cfg.model is None
    assert cfg.schedule_hour == 0
    assert cfg.schedule_minute == 30
    assert cfg.max_retries == 3
    assert cfg.blacklist == []


def test_blacklist_instances_do_not_share_state():
    """The blacklist default is a fresh list per instance (default_factory)."""
    first = DailyFormatConfig()
    second = DailyFormatConfig()
    assert first.blacklist is not second.blacklist


# ---------------------------------------------------------------------------
# Validation rejections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hour", [-1, 24, 100])
def test_schedule_hour_out_of_range_rejected(hour):
    """schedule_hour outside 0-23 fails validation."""
    with pytest.raises(ValidationError):
        DailyFormatConfig(schedule_hour=hour)


@pytest.mark.parametrize("minute", [-1, 60, 999])
def test_schedule_minute_out_of_range_rejected(minute):
    """schedule_minute outside 0-59 fails validation."""
    with pytest.raises(ValidationError):
        DailyFormatConfig(schedule_minute=minute)


@pytest.mark.parametrize("retries", [0, -3])
def test_max_retries_non_positive_rejected(retries):
    """max_retries must be strictly positive."""
    with pytest.raises(ValidationError):
        DailyFormatConfig(max_retries=retries)


@pytest.mark.parametrize("bad_format", [123, ["%Y-%m-%d"], {"fmt": "%Y"}])
def test_filename_format_non_string_rejected(bad_format):
    """filename_format must be a string; other types fail validation."""
    with pytest.raises(ValidationError):
        DailyFormatConfig(filename_format=bad_format)


# ---------------------------------------------------------------------------
# AppConfig wiring
# ---------------------------------------------------------------------------


def test_appconfig_has_daily_format_default(tmp_path):
    """AppConfig without a daily_format section gets defaults via default_factory."""
    cfg = AppConfig.model_validate({"vaults": _vaults_section(tmp_path)})

    assert isinstance(cfg.daily_format, DailyFormatConfig)
    assert cfg.daily_format.enabled is False


def test_appconfig_daily_format_uses_default_factory(tmp_path):
    """Each AppConfig instance gets its own DailyFormatConfig (default_factory)."""
    raw = {"vaults": _vaults_section(tmp_path)}
    first = AppConfig.model_validate(raw)
    second = AppConfig.model_validate(raw)

    assert first.daily_format is not second.daily_format
    assert AppConfig.model_fields["daily_format"].default_factory is not None


def test_empty_daily_format_section_uses_defaults(tmp_path):
    """A vault yaml with 'daily_format: {}' loads with all defaults."""
    config_file = tmp_path / "config.yaml"
    _write_config(
        config_file,
        {"vaults": _vaults_section(tmp_path), "daily_format": {}},
    )

    cfg = load_config(str(config_file))

    assert cfg.daily_format == DailyFormatConfig()


# ---------------------------------------------------------------------------
# DEFAULT_CONFIG documentation
# ---------------------------------------------------------------------------


def _uncommented_daily_format_block():
    """Extract the commented daily_format section from DEFAULT_CONFIG as live YAML."""
    lines = DEFAULT_CONFIG.splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith("# daily_format:")
    )
    block = []
    for line in lines[start:]:
        if not line.startswith("#"):
            break
        block.append(line[2:] if line.startswith("# ") else "")
    return "\n".join(block)


def test_default_config_parses_and_mentions_daily_format():
    """DEFAULT_CONFIG is valid YAML and documents a daily_format section."""
    parsed = yaml.safe_load(DEFAULT_CONFIG)

    assert isinstance(parsed, dict)
    assert "vaults" in parsed
    assert "# daily_format:" in DEFAULT_CONFIG
    assert "#   enabled: false" in DEFAULT_CONFIG


def test_default_config_daily_format_round_trips(tmp_path):
    """The documented daily_format defaults load back to model defaults."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    config_text = (
        f'vaults:\n  - name: "test"\n    path: "{vault_dir}"\n\n'
        + _uncommented_daily_format_block()
        + "\n"
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_text, encoding="utf-8")

    cfg = load_config(str(config_file))

    assert cfg.daily_format == DailyFormatConfig()


# ---------------------------------------------------------------------------
# Package skeleton
# ---------------------------------------------------------------------------


def test_daily_format_package_importable():
    """The daily_format package skeleton exists and has a docstring."""
    import obsidian_rag.daily_format as pkg

    assert pkg.__doc__


def test_poll_minutes_default_and_validation():
    """poll_minutes defaults to 5 and must be positive."""
    assert DailyFormatConfig().poll_minutes == 5
    with pytest.raises(ValidationError):
        DailyFormatConfig(poll_minutes=0)


def test_min_battery_percent_default_and_validation():
    """min_battery_percent defaults to 20 and is bounded 0..100."""
    assert DailyFormatConfig().min_battery_percent == 20
    with pytest.raises(ValidationError):
        DailyFormatConfig(min_battery_percent=-1)
    with pytest.raises(ValidationError):
        DailyFormatConfig(min_battery_percent=101)
