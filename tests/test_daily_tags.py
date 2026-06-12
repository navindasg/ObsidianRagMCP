"""Tests for obsidian_rag.daily_format.tags module.

Covers: list-form and string-form tags, frequency ordering, case-insensitive
dedupe, malformed/missing frontmatter handling, excluded dirs/patterns,
limit capping, and unreadable-file tolerance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from obsidian_rag.daily_format.tags import collect_vault_tags

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _write_note(vault: Path, name: str, tags: list[str] | str | None) -> None:
    """Write a note with YAML frontmatter containing the given tags."""
    if tags is None:
        (vault / name).write_text("# Plain note\n\nNo frontmatter here.\n", encoding="utf-8")
        return
    if isinstance(tags, str):
        tags_block = f"tags: {tags}\n"
    else:
        tags_block = "tags:\n" + "".join(f"  - {tag}\n" for tag in tags)
    content = f"---\n{tags_block}---\n\n# {name}\n\nBody text.\n"
    (vault / name).write_text(content, encoding="utf-8")


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Empty temporary vault root."""
    root = tmp_path / "vault"
    root.mkdir()
    return root


# ---------------------------------------------------------------------------
# Tag extraction forms
# ---------------------------------------------------------------------------


def test_list_form_tags(vault: Path) -> None:
    """Tags declared as a YAML list are all collected."""
    _write_note(vault, "a.md", ["python", "testing"])
    tags = collect_vault_tags(vault, excluded_dirs=[], excluded_patterns=[])
    assert sorted(tags) == ["python", "testing"]


def test_string_form_tags(vault: Path) -> None:
    """A single string tags value is treated as one tag."""
    _write_note(vault, "a.md", "project")
    tags = collect_vault_tags(vault, excluded_dirs=[], excluded_patterns=[])
    assert tags == ["project"]


def test_tags_are_stripped_and_empties_dropped(vault: Path) -> None:
    """Whitespace is stripped and empty/blank tags are dropped."""
    (vault / "a.md").write_text(
        '---\ntags:\n  - "  spaced  "\n  - ""\n  - "   "\n---\nBody.\n',
        encoding="utf-8",
    )
    tags = collect_vault_tags(vault, excluded_dirs=[], excluded_patterns=[])
    assert tags == ["spaced"]


def test_non_string_tags_value_ignored(vault: Path) -> None:
    """A tags value that is neither list nor string contributes nothing."""
    (vault / "a.md").write_text("---\ntags: 42\n---\nBody.\n", encoding="utf-8")
    _write_note(vault, "b.md", ["real"])
    tags = collect_vault_tags(vault, excluded_dirs=[], excluded_patterns=[])
    assert tags == ["real"]


# ---------------------------------------------------------------------------
# Ordering and dedupe
# ---------------------------------------------------------------------------


def test_frequency_ordering_then_alphabetical(vault: Path) -> None:
    """Tags sort by frequency descending, ties broken alphabetically."""
    _write_note(vault, "a.md", ["zeta", "beta"])
    _write_note(vault, "b.md", ["zeta", "alpha"])
    _write_note(vault, "c.md", ["zeta"])
    tags = collect_vault_tags(vault, excluded_dirs=[], excluded_patterns=[])
    assert tags == ["zeta", "alpha", "beta"]


def test_case_insensitive_dedupe_keeps_first_seen_casing(vault: Path) -> None:
    """Differently-cased duplicates collapse, keeping first-seen casing."""
    _write_note(vault, "a.md", ["Python"])
    _write_note(vault, "b.md", ["python", "PYTHON"])
    tags = collect_vault_tags(vault, excluded_dirs=[], excluded_patterns=[])
    assert tags == ["Python"]


def test_case_insensitive_frequency_counts_merge(vault: Path) -> None:
    """Frequency counts merge across casings when ordering."""
    _write_note(vault, "a.md", ["Rare", "Common"])
    _write_note(vault, "b.md", ["common"])
    tags = collect_vault_tags(vault, excluded_dirs=[], excluded_patterns=[])
    assert tags == ["Common", "Rare"]


# ---------------------------------------------------------------------------
# Tolerant parsing
# ---------------------------------------------------------------------------


def test_malformed_frontmatter_skipped(vault: Path) -> None:
    """A file with invalid YAML frontmatter is skipped, others still count."""
    (vault / "bad.md").write_text(
        "---\ntags: [unclosed\n  bad: : yaml\n---\nBody.\n", encoding="utf-8"
    )
    _write_note(vault, "good.md", ["keep"])
    tags = collect_vault_tags(vault, excluded_dirs=[], excluded_patterns=[])
    assert tags == ["keep"]


def test_file_without_frontmatter_skipped(vault: Path) -> None:
    """A file with no frontmatter block contributes no tags."""
    _write_note(vault, "plain.md", None)
    _write_note(vault, "tagged.md", ["only"])
    tags = collect_vault_tags(vault, excluded_dirs=[], excluded_patterns=[])
    assert tags == ["only"]


def test_frontmatter_without_tags_key_skipped(vault: Path) -> None:
    """Frontmatter lacking a tags key contributes no tags."""
    (vault / "a.md").write_text("---\ntitle: Hello\n---\nBody.\n", encoding="utf-8")
    tags = collect_vault_tags(vault, excluded_dirs=[], excluded_patterns=[])
    assert tags == []


def test_unreadable_file_skipped_with_warning(
    vault: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unreadable file is skipped and a warning is logged."""
    _write_note(vault, "good.md", ["keep"])
    bad = vault / "locked.md"
    _write_note(vault, "locked.md", ["secret"])
    bad.chmod(0o000)
    try:
        with caplog.at_level(logging.WARNING, logger="obsidian_rag.daily_format.tags"):
            tags = collect_vault_tags(vault, excluded_dirs=[], excluded_patterns=[])
    finally:
        bad.chmod(0o644)
    assert tags == ["keep"]
    assert "locked.md" in caplog.text


# ---------------------------------------------------------------------------
# Exclusions and limit
# ---------------------------------------------------------------------------


def test_excluded_dirs_skipped(vault: Path) -> None:
    """Files under excluded directories are not scanned."""
    hidden = vault / ".obsidian"
    hidden.mkdir()
    _write_note(hidden, "config.md", ["internal"])
    _write_note(vault, "visible.md", ["public"])
    tags = collect_vault_tags(
        vault, excluded_dirs=[".obsidian"], excluded_patterns=[]
    )
    assert tags == ["public"]


def test_excluded_patterns_skipped(vault: Path) -> None:
    """Files matching excluded glob patterns are not scanned."""
    _write_note(vault, "daily-2024-01-01.md", ["daily"])
    _write_note(vault, "regular.md", ["normal"])
    tags = collect_vault_tags(
        vault, excluded_dirs=[], excluded_patterns=["daily-*.md"]
    )
    assert tags == ["normal"]


def test_limit_respected(vault: Path) -> None:
    """At most `limit` tags are returned, highest-frequency first."""
    _write_note(vault, "a.md", ["one", "two", "three"])
    _write_note(vault, "b.md", ["one", "two", "four"])
    _write_note(vault, "c.md", ["one", "five"])
    tags = collect_vault_tags(vault, excluded_dirs=[], excluded_patterns=[], limit=2)
    assert tags == ["one", "two"]


def test_empty_vault_returns_empty_list(vault: Path) -> None:
    """A vault with no markdown files yields an empty vocabulary."""
    tags = collect_vault_tags(vault, excluded_dirs=[], excluded_patterns=[])
    assert tags == []
