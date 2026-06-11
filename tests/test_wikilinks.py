"""Unit tests for obsidian_rag.wikilinks — parse, resolve, and backlink functions.

TDD RED phase: tests written before implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# parse_wikilinks tests
# ---------------------------------------------------------------------------


def test_parse_basic_wikilink():
    """parse_wikilinks extracts a simple note name."""
    from obsidian_rag.wikilinks import parse_wikilinks

    result = parse_wikilinks("See [[note-a]] here")
    assert result == ["note-a"]


def test_parse_alias_wikilink():
    """parse_wikilinks strips alias after pipe character."""
    from obsidian_rag.wikilinks import parse_wikilinks

    result = parse_wikilinks("Check [[2024-01-15|yesterday]]")
    assert result == ["2024-01-15"]


def test_parse_heading_wikilink():
    """parse_wikilinks strips heading fragment after hash."""
    from obsidian_rag.wikilinks import parse_wikilinks

    result = parse_wikilinks("See [[note-a#Section 2]]")
    assert result == ["note-a"]


def test_parse_heading_and_alias():
    """parse_wikilinks strips both heading and alias."""
    from obsidian_rag.wikilinks import parse_wikilinks

    result = parse_wikilinks("See [[note-a#Section|display]]")
    assert result == ["note-a"]


def test_parse_excludes_embeds():
    """parse_wikilinks ignores embed syntax (![[...]]) and returns only real links."""
    from obsidian_rag.wikilinks import parse_wikilinks

    result = parse_wikilinks("![[embedded-note]] and [[real-link]]")
    assert result == ["real-link"]


def test_parse_multiple_links():
    """parse_wikilinks returns all links in order."""
    from obsidian_rag.wikilinks import parse_wikilinks

    result = parse_wikilinks("[[a]] then [[b]] then [[c]]")
    assert result == ["a", "b", "c"]


def test_parse_empty_link():
    """parse_wikilinks filters out empty targets from [[]]."""
    from obsidian_rag.wikilinks import parse_wikilinks

    result = parse_wikilinks("[[]]")
    assert result == []


# ---------------------------------------------------------------------------
# resolve_wikilink tests
# ---------------------------------------------------------------------------


def test_resolve_finds_file(tmp_path):
    """resolve_wikilink returns matching Path for an existing note."""
    from obsidian_rag.wikilinks import resolve_wikilink

    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "wsn-pipeline.md").write_text("# WSN Pipeline", encoding="utf-8")

    result = resolve_wikilink("wsn-pipeline", tmp_path)

    assert len(result) == 1
    assert result[0] == notes_dir / "wsn-pipeline.md"


def test_resolve_case_insensitive(tmp_path):
    """resolve_wikilink matches case-insensitively by basename."""
    from obsidian_rag.wikilinks import resolve_wikilink

    (tmp_path / "WSN-Pipeline.md").write_text("# WSN Pipeline", encoding="utf-8")

    result = resolve_wikilink("wsn-pipeline", tmp_path)

    assert len(result) == 1
    assert result[0].name == "WSN-Pipeline.md"


def test_resolve_with_md_extension(tmp_path):
    """resolve_wikilink works when target already includes .md extension."""
    from obsidian_rag.wikilinks import resolve_wikilink

    (tmp_path / "wsn-pipeline.md").write_text("# WSN", encoding="utf-8")

    result = resolve_wikilink("wsn-pipeline.md", tmp_path)

    assert len(result) == 1
    assert result[0].name == "wsn-pipeline.md"


def test_resolve_no_match(tmp_path):
    """resolve_wikilink returns empty list when no note matches."""
    from obsidian_rag.wikilinks import resolve_wikilink

    result = resolve_wikilink("nonexistent", tmp_path)

    assert result == []


# ---------------------------------------------------------------------------
# find_backlinks tests
# ---------------------------------------------------------------------------


def test_find_backlinks_basic():
    """find_backlinks returns entries for chunks containing [[target-note]]."""
    from obsidian_rag.wikilinks import find_backlinks

    metadata = {
        "0": {
            "chunk_id": 0,
            "file": "notes/source.md",
            "heading_path": "# Source",
            "text": "See [[target-note]] for more info.",
            "tags": [],
            "vault": "test",
        }
    }

    result = find_backlinks("target-note", metadata)

    assert len(result) == 1
    assert result[0]["source_path"] == "notes/source.md"
    assert result[0]["heading_path"] == "# Source"
    assert "snippet" in result[0]


def test_find_backlinks_case_insensitive():
    """find_backlinks matches regardless of case in the wikilink text."""
    from obsidian_rag.wikilinks import find_backlinks

    metadata = {
        "0": {
            "chunk_id": 0,
            "file": "notes/source.md",
            "heading_path": "# Source",
            "text": "See [[Target-Note]] for more info.",
            "tags": [],
            "vault": "test",
        }
    }

    result = find_backlinks("target-note", metadata)

    assert len(result) == 1
    assert result[0]["source_path"] == "notes/source.md"


def test_find_backlinks_no_matches():
    """find_backlinks returns empty list when no chunks reference the target."""
    from obsidian_rag.wikilinks import find_backlinks

    metadata = {
        "0": {
            "chunk_id": 0,
            "file": "notes/other.md",
            "heading_path": "# Other",
            "text": "Completely unrelated content.",
            "tags": [],
            "vault": "test",
        }
    }

    result = find_backlinks("no-refs", metadata)

    assert result == []


def test_find_backlinks_deduplicates_by_source():
    """find_backlinks deduplicates multiple chunks from the same source file."""
    from obsidian_rag.wikilinks import find_backlinks

    metadata = {
        "0": {
            "chunk_id": 0,
            "file": "notes/source.md",
            "heading_path": "# Intro",
            "text": "See [[target-note]] here.",
            "tags": [],
            "vault": "test",
        },
        "1": {
            "chunk_id": 1,
            "file": "notes/source.md",
            "heading_path": "# Details",
            "text": "Also [[target-note]] again.",
            "tags": [],
            "vault": "test",
        },
    }

    result = find_backlinks("target-note", metadata)

    # Deduplicated: only one entry per source_path
    assert len(result) == 1
    assert result[0]["source_path"] == "notes/source.md"


def test_find_backlinks_alias_variant():
    """find_backlinks matches [[target-note|alias]] variant."""
    from obsidian_rag.wikilinks import find_backlinks

    metadata = {
        "0": {
            "chunk_id": 0,
            "file": "notes/src.md",
            "heading_path": "# Src",
            "text": "Refer to [[target-note|the target]] in this note.",
            "tags": [],
            "vault": "test",
        }
    }

    result = find_backlinks("target-note", metadata)

    assert len(result) == 1


def test_find_backlinks_heading_variant():
    """find_backlinks matches [[target-note#Section]] variant."""
    from obsidian_rag.wikilinks import find_backlinks

    metadata = {
        "0": {
            "chunk_id": 0,
            "file": "notes/src.md",
            "heading_path": "# Src",
            "text": "See [[target-note#Overview]] for the intro.",
            "tags": [],
            "vault": "test",
        }
    }

    result = find_backlinks("target-note", metadata)

    assert len(result) == 1


# ---------------------------------------------------------------------------
# Path-qualified wikilink tests (regression: [[folder/note]] never resolved)
# ---------------------------------------------------------------------------


def test_resolve_wikilink_path_qualified(tmp_path):
    """[[folder/note]] resolves to the note at that relative path."""
    from obsidian_rag.wikilinks import resolve_wikilink

    (tmp_path / "projects").mkdir()
    target_file = tmp_path / "projects" / "alpha.md"
    target_file.write_text("# Alpha")
    # Same basename elsewhere must NOT match a path-qualified link
    other = tmp_path / "alpha.md"
    other.write_text("# Other Alpha")

    result = resolve_wikilink("projects/Alpha", tmp_path)

    assert result == [target_file]


def test_resolve_wikilink_path_qualified_no_match(tmp_path):
    """[[wrong-folder/note]] returns no matches even when the basename exists."""
    from obsidian_rag.wikilinks import resolve_wikilink

    (tmp_path / "projects").mkdir()
    (tmp_path / "projects" / "alpha.md").write_text("# Alpha")

    result = resolve_wikilink("archive/alpha", tmp_path)

    assert result == []


def test_resolve_wikilink_with_prebuilt_index(tmp_path):
    """A prebuilt note index produces the same result as an on-the-fly walk."""
    from obsidian_rag.wikilinks import build_note_index, resolve_wikilink

    note = tmp_path / "beta.md"
    note.write_text("# Beta")
    index = build_note_index(tmp_path)

    assert resolve_wikilink("beta", tmp_path, note_index=index) == [note]


def test_find_backlinks_path_qualified():
    """find_backlinks detects [[folder/note]] references to the note."""
    from obsidian_rag.wikilinks import find_backlinks

    metadata = {
        "0": {
            "chunk_id": 0,
            "file": "notes/src.md",
            "heading_path": "# Src",
            "text": "Link here: [[projects/Alpha]] for details.",
            "tags": [],
            "vault": "test",
        }
    }

    result = find_backlinks("Alpha", metadata)

    assert len(result) == 1
    assert result[0]["source_path"] == "notes/src.md"


def test_find_backlinks_does_not_match_substring_names():
    """A note named 'alpha' must not match links to 'alphabet'."""
    from obsidian_rag.wikilinks import find_backlinks

    metadata = {
        "0": {
            "chunk_id": 0,
            "file": "notes/src.md",
            "heading_path": "# Src",
            "text": "See [[alphabet]] for the full list.",
            "tags": [],
            "vault": "test",
        }
    }

    assert find_backlinks("alpha", metadata) == []
