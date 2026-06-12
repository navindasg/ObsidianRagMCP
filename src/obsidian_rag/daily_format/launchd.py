"""launchd LaunchAgent management for the nightly daily-note formatter.

Public API:
    LABEL
    plist_path() -> Path
    default_log_path() -> Path
    generate_plist(schedule_hour, schedule_minute, log_path) -> str
    install(cfg) -> Path
    uninstall() -> None
    status() -> str

The agent runs ``python -m obsidian_rag format-daily`` on a nightly
StartCalendarInterval; launchd fires missed runs when the machine wakes.
"""

from __future__ import annotations

import logging
import os
import plistlib
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from obsidian_rag.models import AppConfig

logger = logging.getLogger(__name__)

LABEL = "com.obsidian-rag.daily-format"


def plist_path() -> Path:
    """Location of the LaunchAgent plist in the user's LaunchAgents dir."""
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def default_log_path() -> Path:
    """Log file the agent's stdout/stderr are appended to."""
    return Path.home() / ".obsidian-rag" / "logs" / "daily-format.log"


def generate_plist(
    schedule_hour: int, schedule_minute: int, log_path: Path
) -> str:
    """Render the LaunchAgent plist XML via plistlib for correctness."""
    payload = {
        "Label": LABEL,
        "ProgramArguments": [sys.executable, "-m", "obsidian_rag", "format-daily"],
        "StartCalendarInterval": {"Hour": schedule_hour, "Minute": schedule_minute},
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(log_path),
        "RunAtLoad": False,
    }
    return plistlib.dumps(payload, sort_keys=False).decode("utf-8")


def _gui_domain() -> str:
    """The per-user launchd domain target, e.g. ``gui/501``."""
    return f"gui/{os.getuid()}"


def install(cfg: AppConfig) -> Path:
    """Write the plist and (re)register it with launchd.

    Any previous registration is booted out first (failure ignored: the
    agent may simply not be loaded yet). Raises SystemExit with launchctl's
    stderr when bootstrap fails.

    Returns:
        The plist path that was installed.
    """
    log_path = default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    path = plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        generate_plist(
            cfg.daily_format.schedule_hour,
            cfg.daily_format.schedule_minute,
            log_path,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        ["launchctl", "bootout", f"{_gui_domain()}/{LABEL}"], capture_output=True
    )
    result = subprocess.run(
        ["launchctl", "bootstrap", _gui_domain(), str(path)], capture_output=True
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(
            f"launchctl bootstrap failed (exit {result.returncode}): {stderr}"
        )
    logger.info("Installed LaunchAgent %s at %s", LABEL, path)
    return path


def uninstall() -> None:
    """Boot the agent out of launchd and delete the plist (missing is fine)."""
    subprocess.run(
        ["launchctl", "bootout", f"{_gui_domain()}/{LABEL}"], capture_output=True
    )
    plist_path().unlink(missing_ok=True)
    logger.info("Uninstalled LaunchAgent %s", LABEL)


def status() -> str:
    """Return ``launchctl print`` output, or a not-installed message."""
    result = subprocess.run(
        ["launchctl", "print", f"{_gui_domain()}/{LABEL}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return (
            f"{LABEL} is not installed "
            f"(launchctl print exited {result.returncode})"
        )
    return result.stdout
