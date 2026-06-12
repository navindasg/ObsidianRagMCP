"""Tests for launchd LaunchAgent management (daily_format/launchd.py).

Tests:
  1. LABEL and plist_path live where launchd expects them
  2. generate_plist round-trips through plistlib with the exact keys
  3. install writes the plist, creates the log dir, bootouts then bootstraps
  4. install ignores bootout failure but raises SystemExit on bootstrap failure
  5. uninstall bootouts and deletes the plist (missing plist is fine)
  6. status returns launchctl output, or a "not installed" message
"""

from __future__ import annotations

import os
import plistlib
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.daily_format import launchd
from obsidian_rag.models import AppConfig

DOMAIN = f"gui/{os.getuid()}"


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _make_cfg(tmp_path: Path, hour: int = 3, minute: int = 5) -> AppConfig:
    vault = tmp_path / "vault"
    vault.mkdir(exist_ok=True)
    return AppConfig(
        vaults=[{"name": "v", "path": str(vault)}],
        daily_format={"enabled": True, "schedule_hour": hour, "schedule_minute": minute},
    )


def _result(returncode: int = 0, stdout: Any = b"", stderr: Any = b"") -> MagicMock:
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# Test 1 + 2: label, plist path, plist contents
# ---------------------------------------------------------------------------


def test_label_and_plist_path(fake_home: Path) -> None:
    assert launchd.LABEL == "com.obsidian-rag.daily-format"
    assert launchd.plist_path() == (
        fake_home / "Library" / "LaunchAgents" / "com.obsidian-rag.daily-format.plist"
    )


def test_generate_plist_contents(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "daily-format.log"

    xml = launchd.generate_plist(2, 45, log_path)

    payload = plistlib.loads(xml.encode("utf-8"))
    assert payload["Label"] == launchd.LABEL
    assert payload["ProgramArguments"] == [
        sys.executable,
        "-m",
        "obsidian_rag",
        "format-daily",
    ]
    assert payload["StartCalendarInterval"] == {"Hour": 2, "Minute": 45}
    assert payload["StandardOutPath"] == str(log_path)
    assert payload["StandardErrorPath"] == str(log_path)
    assert payload["RunAtLoad"] is False


# ---------------------------------------------------------------------------
# Test 3: install happy path (bootout failure is ignored)
# ---------------------------------------------------------------------------


def test_install_writes_plist_and_bootstraps(fake_home: Path) -> None:
    cfg = _make_cfg(fake_home, hour=3, minute=5)
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> MagicMock:
        calls.append(argv)
        if argv[1] == "bootout":
            return _result(returncode=3)  # not loaded yet -- must be ignored
        return _result(returncode=0)

    with patch(
        "obsidian_rag.daily_format.launchd.subprocess.run", side_effect=fake_run
    ):
        path = launchd.install(cfg)

    assert path == launchd.plist_path()
    payload = plistlib.loads(path.read_bytes())
    assert payload["StartCalendarInterval"] == {"Hour": 3, "Minute": 5}
    log_path = Path(payload["StandardOutPath"])
    assert log_path == fake_home / ".obsidian-rag" / "logs" / "daily-format.log"
    assert log_path.parent.is_dir()
    assert calls == [
        ["launchctl", "bootout", f"{DOMAIN}/{launchd.LABEL}"],
        ["launchctl", "bootstrap", DOMAIN, str(path)],
    ]


# ---------------------------------------------------------------------------
# Test 4: install raises SystemExit when bootstrap fails
# ---------------------------------------------------------------------------


def test_install_bootstrap_failure_raises_system_exit(fake_home: Path) -> None:
    cfg = _make_cfg(fake_home)

    def fake_run(argv: list[str], **kwargs: Any) -> MagicMock:
        if argv[1] == "bootstrap":
            return _result(returncode=5, stderr=b"Bootstrap failed: 5: I/O error")
        return _result(returncode=0)

    with patch(
        "obsidian_rag.daily_format.launchd.subprocess.run", side_effect=fake_run
    ):
        with pytest.raises(SystemExit, match="Bootstrap failed"):
            launchd.install(cfg)


# ---------------------------------------------------------------------------
# Test 5: uninstall
# ---------------------------------------------------------------------------


def test_uninstall_bootouts_and_removes_plist(fake_home: Path) -> None:
    plist = launchd.plist_path()
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text("placeholder", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> MagicMock:
        calls.append(argv)
        return _result(returncode=0)

    with patch(
        "obsidian_rag.daily_format.launchd.subprocess.run", side_effect=fake_run
    ):
        launchd.uninstall()
        launchd.uninstall()  # missing plist must not raise

    assert not plist.exists()
    assert calls == [["launchctl", "bootout", f"{DOMAIN}/{launchd.LABEL}"]] * 2


# ---------------------------------------------------------------------------
# Test 6: status
# ---------------------------------------------------------------------------


def test_status_returns_launchctl_output() -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> MagicMock:
        calls.append(argv)
        return _result(returncode=0, stdout="state = waiting\n")

    with patch(
        "obsidian_rag.daily_format.launchd.subprocess.run", side_effect=fake_run
    ):
        out = launchd.status()

    assert out == "state = waiting\n"
    assert calls == [["launchctl", "print", f"{DOMAIN}/{launchd.LABEL}"]]


def test_status_not_installed() -> None:
    with patch(
        "obsidian_rag.daily_format.launchd.subprocess.run",
        return_value=_result(returncode=113, stdout="", stderr="not found"),
    ):
        out = launchd.status()

    assert "not installed" in out
    assert launchd.LABEL in out
