"""Wikilink parsing and resolution utilities for ObsidianRAG.

Public API:
    parse_wikilinks(text) -> list[str]
    build_note_index(vault_root) -> dict[str, list[Path]]
    resolve_wikilink(target, vault_root, note_index=None) -> list[Path]
    find_backlinks(note_name, metadata) -> list[dict]

Design decisions:
- D-09: Embed syntax (![[...]]) excluded from link parsing via negative lookbehind
- D-10: resolve_wikilink matches by case-insensitive basename, .md extension
  optional; path-qualified targets ([[folder/note]]) match by relative path
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


def build_note_index(vault_root: Path) -> dict[str, list[Path]]:
    """Map lowercase .md basenames to their paths under vault_root.

    Build once per operation and pass to resolve_wikilink to avoid one full
    vault walk per link target.
    """
    note_index: dict[str, list[Path]] = {}
    for md_file in sorted(vault_root.rglob("*.md")):
        note_index.setdefault(md_file.name.lower(), []).append(md_file)
    return note_index


def resolve_wikilink(
    target: str,
    vault_root: Path,
    note_index: dict[str, list[Path]] | None = None,
) -> list[Path]:
    """Find markdown files in vault_root matching the given wikilink target.

    Matching is case-insensitive. Target may or may not include .md. A
    path-qualified target like "folder/note" matches by relative path suffix
    (Obsidian's disambiguation syntax), not by basename alone.

    Args:
        target: Wikilink target string (e.g. "wsn-pipeline" or "projects/wsn-pipeline").
        vault_root: Root Path of the vault to search.
        note_index: Optional prebuilt map from build_note_index; built on the
            fly when omitted.

    Returns:
        List of matching Path objects (all matches for ambiguous cases).
    """
    target_lower = target.lower()
    if not target_lower.endswith(".md"):
        target_lower += ".md"

    if note_index is None:
        note_index = build_note_index(vault_root)

    basename = target_lower.rsplit("/", 1)[-1]
    candidates = note_index.get(basename, [])

    if "/" not in target_lower:
        return list(candidates)

    # Path-qualified link: the relative path must equal or end with the target.
    matches: list[Path] = []
    for md_file in candidates:
        rel = str(md_file.relative_to(vault_root)).replace("\\", "/").lower()
        if rel == target_lower or rel.endswith("/" + target_lower):
            matches.append(md_file)
    return matches


def find_backlinks(note_name: str, metadata: dict[str, dict]) -> list[dict]:
    """Scan chunk metadata text fields for references to note_name.

    Matches [[note]], [[note|alias]], [[note#heading]], and path-qualified
    [[folder/note]] variants using case-insensitive search. Deduplicates
    results by source_path.

    Args:
        note_name: The basename of the note to find backlinks for (no .md extension).
        metadata: Dict of chunk_id -> chunk metadata dicts (from vault_indexes).

    Returns:
        List of dicts with keys: source_path, heading_path, snippet (first 200 chars).
    """
    # [[, optional "path/" prefix, the note name, optional .md, then an
    # alias pipe, heading hash, or closing brackets.
    backlink_re = re.compile(
        r"\[\[(?:[^\]|#]*/)?"
        + re.escape(note_name.lower())
        + r"(?:\.md)?\s*(?:[|#]|\]\])"
    )

    seen_paths: set[str] = set()
    results: list[dict] = []

    for chunk in metadata.values():
        text: str = chunk.get("text", "")

        if backlink_re.search(text.lower()):
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
