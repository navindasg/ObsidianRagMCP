"""Vault tag vocabulary collection for daily-note formatting.

Public API:
    collect_vault_tags(vault_root, *, excluded_dirs, excluded_patterns, limit)
        -> list[str]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from obsidian_rag.indexer import is_excluded

logger = logging.getLogger(__name__)

# Leading YAML frontmatter block: opening ``---`` on the first line, lazily
# matched body, closing ``---`` on its own line.
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*$", re.DOTALL | re.MULTILINE)


def _parse_frontmatter(text: str) -> dict | None:
    """Return the YAML frontmatter mapping, or None when absent or malformed."""
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return None
    try:
        parsed = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_tags(frontmatter: dict) -> list[str]:
    """Return normalized tag strings from a frontmatter mapping.

    The ``tags`` key may be a list of strings or a single string; any other
    shape contributes nothing. Tags are stripped and empties dropped.
    """
    raw = frontmatter.get("tags")
    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = [item for item in raw if isinstance(item, str)]
    else:
        candidates = []
    return [stripped for tag in candidates if (stripped := tag.strip())]


def _file_tags(md_file: Path) -> list[str]:
    """Return tags for one file; unreadable or malformed files yield []."""
    try:
        text = md_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        logger.warning("Could not read %s — skipping for tag collection", md_file)
        return []
    frontmatter = _parse_frontmatter(text)
    if frontmatter is None:
        return []
    return _extract_tags(frontmatter)


def collect_vault_tags(
    vault_root: Path,
    *,
    excluded_dirs: list[str],
    excluded_patterns: list[str],
    limit: int = 50,
) -> list[str]:
    """Collect the vault's existing tag vocabulary, most frequent first.

    Walks all .md files under vault_root (skipping excluded dirs/patterns via
    the same is_excluded rule the indexer uses), parses YAML frontmatter
    tolerantly, and aggregates the ``tags`` key. Tags are deduped
    case-insensitively keeping first-seen casing, ordered by frequency
    descending then alphabetically, and capped at ``limit``.

    Args:
        vault_root: Root directory of the Obsidian vault.
        excluded_dirs: Directory names to skip (e.g. [".obsidian", ".trash"]).
        excluded_patterns: Glob patterns for filenames to skip.
        limit: Maximum number of tags to return.

    Returns:
        Up to ``limit`` tag strings in first-seen casing.
    """
    md_files = sorted(
        md_file
        for md_file in vault_root.rglob("*.md")
        if not is_excluded(
            md_file.relative_to(vault_root), excluded_dirs, excluded_patterns
        )
    )

    counts: dict[str, int] = {}
    first_seen: dict[str, str] = {}
    for md_file in md_files:
        for tag in _file_tags(md_file):
            key = tag.lower()
            counts[key] = counts.get(key, 0) + 1
            first_seen.setdefault(key, tag)

    ordered = sorted(counts, key=lambda key: (-counts[key], key))
    return [first_seen[key] for key in ordered[:limit]]
