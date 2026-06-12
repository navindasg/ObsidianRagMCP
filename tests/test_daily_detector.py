"""Tests for obsidian_rag.daily_format.detector.

Covers: date parsing with strict round-trip validation, already-formatted
detection via frontmatter key and "## Original Notes" marker, and candidate
scanning with boundary dates, catch-up window, exclusions, and skip rules.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import pytest

from obsidian_rag.daily_format.detector import (
    find_candidates,
    is_already_formatted,
    is_blacklisted,
    parse_note_date,
)

TODAY = datetime.date(2026, 6, 12)
START = datetime.date(2026, 6, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_note(directory: Path, name: str, text: str = "raw daily note\n") -> Path:
    """Create a note file under directory (created if missing)."""
    directory.mkdir(parents=True, exist_ok=True)
    note = directory / name
    note.write_text(text, encoding="utf-8")
    return note


def _find(vault: Path, **overrides) -> list[Path]:
    """Call find_candidates with sensible defaults, overridable per test."""
    kwargs = {
        "daily_folder": "",
        "filename_format": "%Y-%m-%d",
        "today": TODAY,
        "start_date": START,
        "catchup_days": 14,
        "excluded_dirs": [".obsidian", ".trash"],
        "excluded_patterns": [],
        **overrides,
    }
    return find_candidates(vault, **kwargs)


# ---------------------------------------------------------------------------
# parse_note_date
# ---------------------------------------------------------------------------


def test_parse_valid_date():
    """A correctly formatted stem parses to the expected date."""
    result = parse_note_date(Path("2026-06-11.md"), "%Y-%m-%d")
    assert result == datetime.date(2026, 6, 11)


def test_parse_custom_format():
    """A non-default strftime format is honoured."""
    result = parse_note_date(Path("11-06-2026.md"), "%d-%m-%Y")
    assert result == datetime.date(2026, 6, 11)


@pytest.mark.parametrize(
    "name",
    [
        "meeting notes.md",
        "2026-06-11 extra.md",
        "not-a-date.md",
        "20260611.md",
    ],
)
def test_parse_non_matching_stem_returns_none(name):
    """Stems that strptime cannot fully consume return None."""
    assert parse_note_date(Path(name), "%Y-%m-%d") is None


@pytest.mark.parametrize("name", ["2026-6-1.md", "2026-06-1.md", "2026-6-11.md"])
def test_parse_rejects_round_trip_mismatch(name):
    """Stems strptime accepts leniently but that do not round-trip are rejected."""
    assert parse_note_date(Path(name), "%Y-%m-%d") is None


# ---------------------------------------------------------------------------
# is_already_formatted
# ---------------------------------------------------------------------------


def test_formatted_frontmatter_key_detected():
    """A 'formatted' key in frontmatter marks the note as formatted."""
    text = "---\nformatted: 2026-06-11T02:00:00\ntags: [a]\n---\nbody\n"
    assert is_already_formatted(text) is True


def test_frontmatter_without_formatted_key():
    """Frontmatter lacking the 'formatted' key is not formatted."""
    text = "---\ntags: [a, b]\ndate: 2026-06-11\n---\nbody\n"
    assert is_already_formatted(text) is False


def test_malformed_frontmatter_means_not_formatted():
    """YAML that fails to parse is treated leniently as not formatted."""
    text = "---\nformatted: [unclosed\n---\nbody\n"
    assert is_already_formatted(text) is False


def test_non_mapping_frontmatter_means_not_formatted():
    """Frontmatter that parses to a non-dict carries no 'formatted' key."""
    text = "---\n- just\n- a list\n---\nbody\n"
    assert is_already_formatted(text) is False


@pytest.mark.parametrize(
    "line",
    [
        "## Original Notes",
        "## Original Notes:",
        "## original notes",
        "## ORIGINAL NOTES:",
    ],
)
def test_mangled_formatted_frontmatter_with_heading_detected(line):
    """Malformed frontmatter with a 'formatted:' line plus the heading counts.

    This is the recovery path for formatter output whose YAML a user mangled:
    the heading (any case, optional colon) confirms it was formatter output.
    """
    text = f"---\nformatted: [unclosed\n---\nbody\n\n{line}\nraw text\n"
    assert is_already_formatted(text) is True


def test_bare_original_notes_heading_is_not_marker():
    """A raw note with a user-authored '## Original Notes' section is raw.

    Without a leading frontmatter block carrying a 'formatted' key the
    heading alone must not mark the note as formatted (review fix).
    """
    assert is_already_formatted("some body\n\n## Original Notes\nraw text\n") is False


def test_heading_with_plain_frontmatter_is_not_marker():
    """Frontmatter without a 'formatted' key does not arm the heading marker."""
    text = "---\ntags: [a]\n---\nbody\n\n## Original Notes\nraw\n"
    assert is_already_formatted(text) is False


@pytest.mark.parametrize(
    "tail",
    [
        "### Original Notes\n",
        "see ## Original Notes inline\n",
        "plain raw note text\n",
        "",
    ],
)
def test_non_marker_text_not_formatted(tail):
    """Other heading levels, inline mentions, and plain text are not markers."""
    text = f"---\nformatted: [unclosed\n---\n{tail}"
    assert is_already_formatted(text) is False
    assert is_already_formatted(tail) is False


# ---------------------------------------------------------------------------
# find_candidates — boundary dates
# ---------------------------------------------------------------------------


def test_yesterday_is_eligible(tmp_path):
    """A note dated yesterday is a candidate."""
    note = _make_note(tmp_path, "2026-06-11.md")
    assert _find(tmp_path) == [note]


def test_today_is_never_eligible(tmp_path):
    """Today's note is never a candidate (next-day rule)."""
    _make_note(tmp_path, "2026-06-12.md")
    assert _find(tmp_path) == []


def test_future_note_is_not_eligible(tmp_path):
    """A future-dated note is not a candidate."""
    _make_note(tmp_path, "2026-06-13.md")
    assert _find(tmp_path) == []


def test_start_date_itself_eligible_once_past(tmp_path):
    """A note dated exactly start_date is eligible once that date is past."""
    note = _make_note(tmp_path, "2026-06-01.md")
    assert _find(tmp_path) == [note]


def test_note_before_start_date_not_eligible(tmp_path):
    """A note dated before start_date is never a candidate (no backfill)."""
    _make_note(tmp_path, "2026-05-31.md")
    assert _find(tmp_path) == []


# ---------------------------------------------------------------------------
# find_candidates — catch-up window
# ---------------------------------------------------------------------------


def test_catchup_window_inclusive_lower_bound(tmp_path):
    """A note dated exactly today - catchup_days is still eligible."""
    note = _make_note(tmp_path, "2026-06-05.md")
    result = _find(tmp_path, start_date=datetime.date(2026, 1, 1), catchup_days=7)
    assert result == [note]


def test_note_older_than_catchup_window_skipped(tmp_path):
    """A note older than the catch-up window is skipped."""
    _make_note(tmp_path, "2026-06-04.md")
    result = _find(tmp_path, start_date=datetime.date(2026, 1, 1), catchup_days=7)
    assert result == []


# ---------------------------------------------------------------------------
# find_candidates — already formatted, non-date names, exclusions
# ---------------------------------------------------------------------------


def test_formatted_via_frontmatter_skipped(tmp_path):
    """A note with a 'formatted' frontmatter key is skipped."""
    _make_note(tmp_path, "2026-06-10.md", "---\nformatted: true\n---\nbody\n")
    assert _find(tmp_path) == []


def test_formatted_via_original_notes_marker_skipped(tmp_path):
    """Mangled formatter output (formatted: line + heading) is skipped."""
    text = "---\nformatted: [unclosed\n---\nbody\n\n## Original Notes\nraw\n"
    _make_note(tmp_path, "2026-06-10.md", text)
    assert _find(tmp_path) == []


def test_raw_note_with_original_notes_heading_is_candidate(tmp_path):
    """A raw note with a user-authored '## Original Notes' section is eligible."""
    note = _make_note(tmp_path, "2026-06-10.md", "body\n\n## Original Notes\nraw\n")
    assert _find(tmp_path) == [note]


def test_non_date_filenames_ignored(tmp_path):
    """Files whose stems are not dates are ignored."""
    _make_note(tmp_path, "shopping list.md")
    _make_note(tmp_path, "2026-06-11 standup.md")
    assert _find(tmp_path) == []


def test_excluded_patterns_respected(tmp_path):
    """A date-named note matching an excluded glob pattern is skipped."""
    _make_note(tmp_path, "2026-06-10.md")
    keep = _make_note(tmp_path, "2026-06-11.md")
    result = _find(tmp_path, excluded_patterns=["*-06-10.md"])
    assert result == [keep]


def test_excluded_dirs_respected(tmp_path):
    """A daily folder under an excluded directory yields no candidates."""
    _make_note(tmp_path / "archive" / "daily", "2026-06-11.md")
    result = _find(
        tmp_path,
        daily_folder="archive/daily",
        excluded_dirs=["archive"],
    )
    assert result == []


# ---------------------------------------------------------------------------
# find_candidates — folder handling
# ---------------------------------------------------------------------------


def test_empty_daily_folder_means_vault_root(tmp_path):
    """daily_folder='' scans the vault root itself."""
    note = _make_note(tmp_path, "2026-06-11.md")
    assert _find(tmp_path, daily_folder="") == [note]


def test_scan_is_non_recursive(tmp_path):
    """Notes in subdirectories of the daily folder are not scanned."""
    _make_note(tmp_path / "sub", "2026-06-11.md")
    assert _find(tmp_path) == []


def test_daily_folder_scoped_to_subdir(tmp_path):
    """With a named daily folder, only that folder is scanned."""
    inside = _make_note(tmp_path / "Daily", "2026-06-11.md")
    _make_note(tmp_path, "2026-06-10.md")
    assert _find(tmp_path, daily_folder="Daily") == [inside]


def test_missing_daily_folder_returns_empty(tmp_path):
    """A nonexistent daily folder yields no candidates without raising."""
    assert _find(tmp_path, daily_folder="does-not-exist") == []


# ---------------------------------------------------------------------------
# find_candidates — unreadable files and ordering
# ---------------------------------------------------------------------------


def test_unreadable_file_skipped_with_warning(tmp_path, caplog):
    """A file that cannot be decoded is skipped and a warning is logged."""
    bad = tmp_path / "2026-06-09.md"
    bad.write_bytes(b"\xff\xfe invalid \xff utf8")
    good = _make_note(tmp_path, "2026-06-10.md")

    with caplog.at_level(logging.WARNING, logger="obsidian_rag.daily_format.detector"):
        result = _find(tmp_path)

    assert result == [good]
    assert any("2026-06-09" in record.message for record in caplog.records)


def test_results_sorted_by_date_ascending(tmp_path):
    """Candidates are returned ordered by note date, oldest first."""
    n10 = _make_note(tmp_path, "2026-06-10.md")
    n08 = _make_note(tmp_path, "2026-06-08.md")
    n11 = _make_note(tmp_path, "2026-06-11.md")
    assert _find(tmp_path) == [n08, n10, n11]


# ---------------------------------------------------------------------------
# is_blacklisted / blacklist in find_candidates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    ["2026-06-10", "2026-06-10.md", "daily/2026-06-10", "daily/2026-06-10.md"],
)
def test_is_blacklisted_matches_stem_and_rel_path_forms(entry):
    """A blacklist entry matches by stem or relative path, .md optional."""
    assert is_blacklisted(Path("daily/2026-06-10.md"), [entry])


def test_is_blacklisted_no_match():
    """Unrelated entries and an empty blacklist never match."""
    rel = Path("daily/2026-06-10.md")
    assert not is_blacklisted(rel, [])
    assert not is_blacklisted(rel, ["2026-06-11", "other/2026-06-10.md"])


def test_find_candidates_skips_blacklisted_note(tmp_path):
    """A blacklisted daily note is never a candidate; siblings still are."""
    _make_note(tmp_path, "2026-06-10.md")
    kept = _make_note(tmp_path, "2026-06-11.md")

    assert _find(tmp_path, blacklist=["2026-06-10"]) == [kept]
