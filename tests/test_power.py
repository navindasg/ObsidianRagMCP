"""Tests for the battery gate (daily_format/power.py).

Covers: parsing `pmset -g batt` across AC/battery/desktop/malformed cases,
the defer decision (only on battery below threshold), and the safe-default
behavior when battery state cannot be read.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from obsidian_rag.daily_format.power import (
    PowerState,
    read_power_state,
    should_defer,
)

# Real `pmset -g batt` samples.
AC_CHARGING_12 = (
    "Now drawing from 'AC Power'\n"
    " -InternalBattery-0 (id=22020195)\t12%; charging; 1:41 remaining present: true\n"
)
AC_CHARGED_100 = (
    "Now drawing from 'AC Power'\n"
    " -InternalBattery-0 (id=4128867)\t100%; charged; 0:00 remaining present: true\n"
)
BATTERY_DISCHARGING_62 = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=4128867)\t62%; discharging; 4:51 remaining present: true\n"
)
BATTERY_DISCHARGING_15 = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=4128867)\t15%; discharging; 0:48 remaining present: true\n"
)
DESKTOP_NO_BATTERY = "Now drawing from 'AC Power'\n"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse(output: str) -> PowerState:
    result = MagicMock(stdout=output, returncode=0)
    with (
        patch("obsidian_rag.daily_format.power.shutil.which", return_value="/usr/bin/pmset"),
        patch("obsidian_rag.daily_format.power.subprocess.run", return_value=result),
    ):
        return read_power_state()


def test_parse_ac_charging():
    state = _parse(AC_CHARGING_12)
    assert state == PowerState(has_battery=True, percent=12, on_ac_power=True)


def test_parse_ac_charged():
    state = _parse(AC_CHARGED_100)
    assert state == PowerState(has_battery=True, percent=100, on_ac_power=True)


def test_parse_battery_discharging():
    state = _parse(BATTERY_DISCHARGING_62)
    assert state == PowerState(has_battery=True, percent=62, on_ac_power=False)


def test_parse_desktop_without_battery():
    state = _parse(DESKTOP_NO_BATTERY)
    assert state.has_battery is False
    assert state.on_ac_power is True


def test_parse_malformed_output_has_no_percent():
    state = _parse("garbage that mentions internalbattery but no number")
    # has_battery may be True, but an unreadable percent must be None.
    assert state.percent is None


def test_pmset_missing_assumes_no_gate():
    with patch("obsidian_rag.daily_format.power.shutil.which", return_value=None):
        state = read_power_state()
    assert state == PowerState(has_battery=False, percent=None, on_ac_power=True)


def test_pmset_nonzero_exit_assumes_no_gate():
    result = MagicMock(stdout=BATTERY_DISCHARGING_15, returncode=1)
    with (
        patch("obsidian_rag.daily_format.power.shutil.which", return_value="/usr/bin/pmset"),
        patch("obsidian_rag.daily_format.power.subprocess.run", return_value=result),
    ):
        state = read_power_state()
    assert state.has_battery is False  # ignored despite low-battery stdout


def test_parse_battery_without_header_still_on_battery():
    """A discharging battery line with no 'drawing from' header reads as battery."""
    state = _parse(
        " -InternalBattery-0 (id=4128867)\t10%; discharging; 0:30 remaining present: true\n"
    )
    assert state.on_ac_power is False
    assert state.percent == 10


def test_pmset_failure_assumes_no_gate():
    with (
        patch("obsidian_rag.daily_format.power.shutil.which", return_value="/usr/bin/pmset"),
        patch(
            "obsidian_rag.daily_format.power.subprocess.run",
            side_effect=OSError("boom"),
        ),
    ):
        state = read_power_state()
    assert state.has_battery is False


# ---------------------------------------------------------------------------
# Defer decision
# ---------------------------------------------------------------------------


def test_defer_on_battery_below_threshold():
    state = PowerState(has_battery=True, percent=15, on_ac_power=False)
    assert should_defer(state, 20) is True


def test_no_defer_on_battery_at_or_above_threshold():
    state = PowerState(has_battery=True, percent=20, on_ac_power=False)
    assert should_defer(state, 20) is False


def test_no_defer_when_charging_even_if_low():
    """On AC power the battery is not at risk — proceed regardless of level."""
    state = PowerState(has_battery=True, percent=12, on_ac_power=True)
    assert should_defer(state, 20) is False


def test_no_defer_without_battery():
    state = PowerState(has_battery=False, percent=None, on_ac_power=True)
    assert should_defer(state, 20) is False


def test_no_defer_when_percent_unknown():
    state = PowerState(has_battery=True, percent=None, on_ac_power=False)
    assert should_defer(state, 20) is False


def test_threshold_zero_disables_gate():
    state = PowerState(has_battery=True, percent=1, on_ac_power=False)
    assert should_defer(state, 0) is False
