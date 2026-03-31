"""Wikilink parsing and resolution utilities for ObsidianRAG.

Provides three public functions:
    parse_wikilinks(text) -> list[str]
    resolve_wikilink(target, vault_root) -> list[Path]
    find_backlinks(note_name, metadata) -> list[dict]

Design decisions:
- D-09: Embed syntax (![[...]]) excluded from link parsing via negative lookbehind
- D-10: resolve_wikilink matches by case-insensitive basename, .md extension optional
- D-11: find_backlinks scans metadata text fields in memory (no disk reads)
- D-13: find_backlinks returns {source_path, heading_path, snippet} per entry
"""

from __future__ import annotations

import re
from pathlib import Path

# Matches [[target]] but NOT ![[target]] (embed syntax excluded via negative lookbehind)
WIKILINK_RE = re.compile(r"(?<!!)\[\[([^\]]+)\]\]")


def parse_wikilinks(text: str) -> list[str]:
    """Parse double-bracket wikilinks from markdown text, excluding embed syntax.

    Strips aliases (pipe) and heading fragments (hash). Filters empty targets.

    Args:
        text: Markdown text to parse.

    Returns:
        List of resolved wikilink target names (no aliases, no headings, no .md).
    """
    targets: list[str] = []
    for match in WIKILINK_RE.finditer(text):
        raw = match.group(1)
        # Strip alias: [[target|alias]] -> target
        raw = raw.split("|")[0]
        # Strip heading: [[target#section]] -> target
        raw = raw.split("#")[0]
        raw = raw.strip()
        if raw:
            targets.append(raw)
    return targets


def resolve_wikilink(target: str, vault_root: Path) -> list[Path]:
    """Find markdown files in vault_root matching the given wikilink target.

    Matching is case-insensitive by basename. Target may or may not include .md.

    Args:
        target: Wikilink target string (e.g. "wsn-pipeline" or "wsn-pipeline.md").
        vault_root: Root Path of the vault to search.

    Returns:
        List of matching Path objects (all matches for ambiguous cases).
    """
    # Normalize: lowercase, ensure .md extension
    target_lower = target.lower()
    if not target_lower.endswith(".md"):
        target_lower_md = target_lower + ".md"
    else:
        target_lower_md = target_lower

    matches: list[Path] = []
    for md_file in vault_root.rglob("*.md"):
        if md_file.name.lower() == target_lower_md:
            matches.append(md_file)
    return matches


def find_backlinks(note_name: str, metadata: dict[str, dict]) -> list[dict]:
    """Scan chunk metadata text fields for references to note_name.

    Matches [[note_name]], [[note_name|alias]], and [[note_name#heading]] variants
    using case-insensitive search. Deduplicates results by source_path.

    Args:
        note_name: The basename of the note to find backlinks for (no .md extension).
        metadata: Dict of chunk_id -> chunk metadata dicts (from vault_indexes).

    Returns:
        List of dicts with keys: source_path, heading_path, snippet (first 200 chars).
    """
    note_lower = note_name.lower()
    seen_paths: set[str] = set()
    results: list[dict] = []

    for chunk in metadata.values():
        text: str = chunk.get("text", "")
        text_lower = text.lower()

        # Check for any wikilink variant referencing this note:
        # [[note_name]], [[note_name|...]], [[note_name#...]]
        found = (
            f"[[{note_lower}]]" in text_lower
            or f"[[{note_lower}|" in text_lower
            or f"[[{note_lower}#" in text_lower
        )

        if found:
            source_path = chunk.get("file", "")
            if source_path in seen_paths:
                continue
            seen_paths.add(source_path)
            results.append(
                {
                    "source_path": source_path,
                    "heading_path": chunk.get("heading_path", ""),
                    "snippet": text[:200],
                }
            )

    return results
