"""Battery gate for the daily-note formatter.

A formatting run can spend minutes in the model, so on a laptop running low
on battery it should wait rather than risk draining (or killing) the machine
mid-run. The gate defers only when the machine is *on battery power and below
a threshold*: on AC power (charging, no drain risk) or on a desktop with no
battery, work always proceeds. A battery state that cannot be read never
blocks formatting — the gate fails open.

Deferral leaves everything queued, so the next poll or nightly run picks the
work up once the battery recovers; there is no busy-waiting here.

Public API:
    PowerState
    read_power_state() -> PowerState
    should_defer(state, min_percent) -> bool
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_PERCENT_RE = re.compile(r"(\d+)%")


@dataclass(frozen=True)
class PowerState:
    """Snapshot of the machine's power source.

    Attributes:
        has_battery: Whether an internal battery is present at all.
        percent: Current charge 0-100, or None when it cannot be parsed.
        on_ac_power: Whether the machine is currently drawing from AC.
    """

    has_battery: bool
    percent: int | None
    on_ac_power: bool


_NO_GATE = PowerState(has_battery=False, percent=None, on_ac_power=True)


def read_power_state() -> PowerState:
    """Read battery state from ``pmset -g batt``.

    On any failure — pmset missing (non-macOS), a subprocess error, or a
    machine with no battery — returns a state that disables the gate, so a
    broken or absent battery check never blocks formatting.
    """
    pmset = shutil.which("pmset")
    if pmset is None:
        logger.debug("pmset not found; battery gate disabled")
        return _NO_GATE
    try:
        result = subprocess.run(
            [pmset, "-g", "batt"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("pmset failed (%s); battery gate disabled", exc)
        return _NO_GATE
    if result.returncode != 0:
        logger.debug(
            "pmset exited %s; battery gate disabled", result.returncode
        )
        return _NO_GATE
    return _parse_pmset(result.stdout)


def _parse_pmset(output: str) -> PowerState:
    """Parse ``pmset -g batt`` output into a PowerState."""
    lower = output.lower()
    if "internalbattery" not in lower and "%" not in output:
        # Desktop or otherwise no battery line.
        return _NO_GATE
    match = _PERCENT_RE.search(output)
    percent = int(match.group(1)) if match else None
    # pmset's header line reads "Now drawing from 'AC Power'" or
    # "'Battery Power'"; the per-battery line says "discharging" when on
    # battery. Treat either signal as on-battery so a missing header still
    # defers correctly.
    on_ac_power = "battery power" not in lower and "discharging" not in lower
    return PowerState(has_battery=True, percent=percent, on_ac_power=on_ac_power)


def should_defer(state: PowerState, min_percent: int) -> bool:
    """Whether a formatting run should be deferred for low battery.

    Defers only when on battery power below ``min_percent``. A threshold of
    0 (or less) disables the gate; AC power, no battery, and an unknown
    percent all proceed.
    """
    if min_percent <= 0:
        return False
    if not state.has_battery or state.on_ac_power or state.percent is None:
        return False
    return state.percent < min_percent
