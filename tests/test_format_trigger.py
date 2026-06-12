"""Tests for the #!format tag trigger (daily_format/trigger.py).

Covers: recursive marker scanning with exclusion/blacklist rules, and
marker stripping (own-line removal, inline removal, atomic no-op when
the marker is absent).
"""

from __future__ import annotations

import logging
from pathlib import Path

from obsidian_rag.daily_format.trigger import scan_format_tags, strip_format_tag

TAG = "#!format"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_note(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    note = directory / name
    note.write_text(text, encoding="utf-8")
    return note


def _scan(vault: Path, **overrides) -> list[Path]:
    kwargs = {
        "format_tag": TAG,
        "excluded_dirs": [".obsidian", ".trash"],
        "excluded_patterns": [],
        "blacklist": (),
        **overrides,
    }
    return scan_format_tags(vault, **kwargs)


# ---------------------------------------------------------------------------
# scan_format_tags
# ---------------------------------------------------------------------------


def test_scan_finds_marker_recursively(tmp_path):
    """Notes carrying the marker are found anywhere in the vault tree."""
    top = _make_note(tmp_path, "draft.md", f"some text\n{TAG}\n")
    nested = _make_note(tmp_path / "sub" / "deep", "idea.md", f"inline {TAG} here\n")
    _make_note(tmp_path, "plain.md", "no marker here\n")

    assert _scan(tmp_path) == sorted([top, nested])


def test_scan_skips_excluded_and_blacklisted(tmp_path):
    """Excluded dirs and blacklisted notes never trigger, marker or not."""
    _make_note(tmp_path / ".obsidian", "config.md", f"{TAG}\n")
    _make_note(tmp_path, "blocked.md", f"{TAG}\n")
    kept = _make_note(tmp_path, "kept.md", f"{TAG}\n")

    assert _scan(tmp_path, blacklist=["blocked"]) == [kept]


def test_scan_skips_non_md_and_unreadable(tmp_path, caplog):
    """Only readable .md files are scanned; unreadable ones warn and skip."""
    _make_note(tmp_path, "note.txt", f"{TAG}\n")
    bad = tmp_path / "bad.md"
    bad.write_bytes(b"\xff\xfe broken " + TAG.encode())
    kept = _make_note(tmp_path, "good.md", f"{TAG}\n")

    with caplog.at_level(logging.WARNING, logger="obsidian_rag.daily_format.trigger"):
        result = _scan(tmp_path)

    assert result == [kept]
    assert any("bad.md" in record.message for record in caplog.records)


def test_scan_requires_exact_marker(tmp_path):
    """Supersets of the marker do not trigger (no prefix matching)."""
    _make_note(tmp_path, "note.md", "#!formatting is not the marker\n")

    assert _scan(tmp_path) == []


# ---------------------------------------------------------------------------
# strip_format_tag
# ---------------------------------------------------------------------------


def test_strip_removes_own_line_entirely(tmp_path):
    """A marker alone on a line vanishes with the whole line."""
    note = _make_note(tmp_path, "n.md", f"first\n{TAG}\nlast\n")

    strip_format_tag(note, TAG)

    assert note.read_text(encoding="utf-8") == "first\nlast\n"


def test_strip_removes_inline_marker_and_padding(tmp_path):
    """An inline marker disappears along with its leading spaces."""
    note = _make_note(tmp_path, "n.md", f"call Alice {TAG} tomorrow\n")

    strip_format_tag(note, TAG)

    assert note.read_text(encoding="utf-8") == "call Alice tomorrow\n"


def test_strip_removes_multiple_occurrences(tmp_path):
    """Every occurrence is consumed in one pass."""
    note = _make_note(tmp_path, "n.md", f"{TAG}\nbody {TAG}\n{TAG} trailing\n")

    strip_format_tag(note, TAG)

    text = note.read_text(encoding="utf-8")
    assert TAG not in text
    assert "body" in text and "trailing" in text


def test_strip_no_marker_leaves_file_untouched(tmp_path):
    """Without the marker the file is not rewritten at all."""
    note = _make_note(tmp_path, "n.md", "plain note\n")
    before = note.stat().st_mtime_ns

    strip_format_tag(note, TAG)

    assert note.read_text(encoding="utf-8") == "plain note\n"
    assert note.stat().st_mtime_ns == before
