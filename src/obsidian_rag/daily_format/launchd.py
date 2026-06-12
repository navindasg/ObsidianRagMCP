"""launchd LaunchAgent management for the daily-note formatter.

Two agents are managed together:

* The nightly agent runs ``python -m obsidian_rag format-daily`` on a
  StartCalendarInterval; launchd fires missed runs when the machine wakes
  from sleep (though not runs missed while powered off — the catch-up
  window in the runner covers those).
* The tag-poll agent runs ``format-daily --tags-only`` every
  ``poll_minutes`` on a StartInterval, niced and marked Background with
  low-priority IO, so format tags are picked up promptly without ever
  competing with foreground work.

Public API:
    LABEL, POLL_LABEL
    plist_path() / poll_plist_path() -> Path
    default_log_path() / poll_log_path() -> Path
    generate_plist(schedule_hour, schedule_minute, log_path) -> str
    generate_poll_plist(poll_minutes, log_path) -> str
    install(cfg) -> list[Path]
    uninstall() -> None
    status() -> str
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
POLL_LABEL = "com.obsidian-rag.format-tag-poll"


def plist_path() -> Path:
    """Location of the nightly LaunchAgent plist."""
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def poll_plist_path() -> Path:
    """Location of the tag-poll LaunchAgent plist."""
    return Path.home() / "Library" / "LaunchAgents" / f"{POLL_LABEL}.plist"


def default_log_path() -> Path:
    """Log file the nightly agent's stdout/stderr are appended to."""
    return Path.home() / ".obsidian-rag" / "logs" / "daily-format.log"


def poll_log_path() -> Path:
    """Log file the tag-poll agent's stdout/stderr are appended to."""
    return Path.home() / ".obsidian-rag" / "logs" / "tag-poll.log"


def generate_plist(
    schedule_hour: int, schedule_minute: int, log_path: Path
) -> str:
    """Render the nightly LaunchAgent plist XML via plistlib for correctness."""
    payload = {
        "Label": LABEL,
        "ProgramArguments": [sys.executable, "-m", "obsidian_rag", "format-daily"],
        "StartCalendarInterval": {"Hour": schedule_hour, "Minute": schedule_minute},
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(log_path),
        "RunAtLoad": False,
    }
    return plistlib.dumps(payload, sort_keys=False).decode("utf-8")


def generate_poll_plist(poll_minutes: int, log_path: Path) -> str:
    """Render the tag-poll LaunchAgent plist XML.

    Non-invasive by construction: ProcessType Background, niced, and
    low-priority IO, so the poll never competes with foreground work.
    RunAtLoad is True so tags dropped while the machine was off are picked
    up promptly after login.
    """
    payload = {
        "Label": POLL_LABEL,
        "ProgramArguments": [
            sys.executable,
            "-m",
            "obsidian_rag",
            "format-daily",
            "--tags-only",
        ],
        "StartInterval": poll_minutes * 60,
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(log_path),
        "RunAtLoad": True,
        "ProcessType": "Background",
        "Nice": 10,
        "LowPriorityBackgroundIO": True,
    }
    return plistlib.dumps(payload, sort_keys=False).decode("utf-8")


def _gui_domain() -> str:
    """The per-user launchd domain target, e.g. ``gui/501``."""
    return f"gui/{os.getuid()}"


def _register(label: str, path: Path, plist_xml: str) -> None:
    """Write one plist and (re)register it with launchd.

    Any previous registration is booted out first (failure ignored: the
    agent may simply not be loaded yet). Raises SystemExit with launchctl's
    stderr when bootstrap fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plist_xml, encoding="utf-8")
    subprocess.run(
        ["launchctl", "bootout", f"{_gui_domain()}/{label}"], capture_output=True
    )
    result = subprocess.run(
        ["launchctl", "bootstrap", _gui_domain(), str(path)], capture_output=True
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(
            f"launchctl bootstrap failed (exit {result.returncode}): {stderr}"
        )
    logger.info("Installed LaunchAgent %s at %s", label, path)


def install(cfg: AppConfig) -> list[Path]:
    """Install (or reinstall) both agents: nightly run and tag poll.

    Returns:
        The plist paths that were installed, nightly first.
    """
    daily = cfg.daily_format
    nightly_log = default_log_path()
    nightly_log.parent.mkdir(parents=True, exist_ok=True)

    _register(
        LABEL,
        plist_path(),
        generate_plist(daily.schedule_hour, daily.schedule_minute, nightly_log),
    )
    _register(
        POLL_LABEL,
        poll_plist_path(),
        generate_poll_plist(daily.poll_minutes, poll_log_path()),
    )
    return [plist_path(), poll_plist_path()]


def uninstall() -> None:
    """Boot both agents out of launchd and delete the plists (missing is fine)."""
    for label, path in ((LABEL, plist_path()), (POLL_LABEL, poll_plist_path())):
        subprocess.run(
            ["launchctl", "bootout", f"{_gui_domain()}/{label}"], capture_output=True
        )
        path.unlink(missing_ok=True)
        logger.info("Uninstalled LaunchAgent %s", label)


def status() -> str:
    """Return ``launchctl print`` output for both agents."""
    sections: list[str] = []
    for label in (LABEL, POLL_LABEL):
        result = subprocess.run(
            ["launchctl", "print", f"{_gui_domain()}/{label}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            sections.append(
                f"{label} is not installed "
                f"(launchctl print exited {result.returncode})"
            )
        else:
            sections.append(result.stdout)
    return "\n".join(sections)
