"""Daily-note candidate detection for the nightly formatter.

A daily note becomes eligible for formatting once a later-dated daily note
exists in the same folder — the single most recent note is always held back,
since it may still be in progress. Calendar time is irrelevant: a note from
years ago is formatted the moment any later note appears. The `since`
override is the manual escape hatch, formatting every note on/after a date
including the most recent.

Public API:
    parse_note_date(path, filename_format) -> datetime.date | None
    is_already_formatted(text) -> bool
    is_blacklisted(rel_path, blacklist) -> bool
    find_candidates(vault_root, *, daily_folder, filename_format,
        excluded_dirs, excluded_patterns, blacklist=(), since=None)
        -> list[Path]
"""

from __future__ import annotations

import datetime
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import yaml

from obsidian_rag.indexer import is_excluded

logger = logging.getLogger(__name__)

# Leading YAML frontmatter block: --- ... --- at the very start of the file.
_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)",
    re.DOTALL,
)

# A standalone "## Original Notes" heading (any case, optional trailing colon).
_ORIGINAL_NOTES_RE = re.compile(
    r"^##[ \t]+original notes:?[ \t]*\r?$",
    re.IGNORECASE | re.MULTILINE,
)

# A 'formatted' key line inside the raw frontmatter text; lets heading-based
# detection still work when a formatted note's YAML has been mangled.
_FORMATTED_KEY_LINE_RE = re.compile(r"^formatted[ \t]*:", re.MULTILINE)


def parse_note_date(path: Path, filename_format: str) -> datetime.date | None:
    """Parse a daily-note date from a filename stem, or None if it is not one.

    The stem must round-trip (strptime then strftime reproduces the stem
    exactly) so leniently parsed near-misses like "2026-6-1" are rejected.

    Args:
        path: Note path; only the stem is examined.
        filename_format: strftime/strptime format, e.g. "%Y-%m-%d".

    Returns:
        The parsed date, or None when the stem does not match the format.
    """
    stem = path.stem
    try:
        parsed = datetime.datetime.strptime(stem, filename_format)
    except ValueError:
        return None
    if parsed.strftime(filename_format) != stem:
        return None
    return parsed.date()


def is_already_formatted(text: str) -> bool:
    """Return True when a note has already been through the formatter.

    Formatter output always leads with a YAML frontmatter block carrying a
    'formatted' key, so that is the authoritative marker. As a recovery
    path for formatted notes whose YAML a user later mangled, a textual
    'formatted:' line in the frontmatter block combined with an
    "## Original Notes" heading (case-insensitive, optional trailing colon)
    also counts. A bare heading in raw content is NOT a marker, so raw
    notes with a user-authored "## Original Notes" section still format.

    Args:
        text: Full note text.

    Returns:
        True when a formatted marker is present.
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return False
    if _frontmatter_has_formatted_key(text):
        return True
    return (
        _FORMATTED_KEY_LINE_RE.search(match.group(1)) is not None
        and _ORIGINAL_NOTES_RE.search(text) is not None
    )


def _frontmatter_has_formatted_key(text: str) -> bool:
    """Return True when the leading frontmatter block has a 'formatted' key.

    Frontmatter is parsed leniently: malformed YAML or a non-mapping block
    simply means the note is not formatted.
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return False
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return False
    return isinstance(frontmatter, dict) and "formatted" in frontmatter


def is_blacklisted(rel_path: Path, blacklist: Sequence[str]) -> bool:
    """Return True when a vault-relative path matches a blacklist entry.

    An entry matches the note's filename stem ("2026-06-10") or its
    vault-relative path, with the .md suffix optional in either form.
    Stem entries therefore match any note of that name in any folder.
    """
    if not blacklist:
        return False
    posix = rel_path.as_posix()
    forms = {rel_path.stem, rel_path.name, posix, posix.removesuffix(".md")}
    return any(entry in forms for entry in blacklist)


def find_candidates(
    vault_root: Path,
    *,
    daily_folder: str,
    filename_format: str,
    excluded_dirs: list[str],
    excluded_patterns: list[str],
    blacklist: Sequence[str] = (),
    since: datetime.date | None = None,
) -> list[Path]:
    """Scan the daily folder for raw daily notes that need formatting.

    A note is eligible when its stem parses to a date, it is not excluded,
    not blacklisted, not already formatted, and it has a strictly later-dated
    sibling — the most recent note is held back. Calendar time never matters.
    When `since` is given, the latest-note hold is lifted and every note dated
    on or after `since` is eligible instead (manual backfill).

    Args:
        vault_root: Root directory of the Obsidian vault.
        daily_folder: Folder of daily notes relative to vault_root; "" means
            the vault root itself. Scanned non-recursively.
        filename_format: strftime format the daily-note stems follow.
        excluded_dirs: Directory names excluded from indexing.
        excluded_patterns: Filename globs excluded from indexing.
        blacklist: Notes never formatted (stems or vault-relative paths).
        since: Manual backfill floor; when set, format every note dated on or
            after it, including the most recent (overrides the latest hold).

    Returns:
        Eligible note paths sorted by note date ascending.
    """
    daily_dir = (vault_root / daily_folder) if daily_folder else vault_root
    if not daily_dir.is_dir():
        logger.warning("Daily-note folder does not exist: %s", daily_dir)
        return []

    dated = [
        (note_date, md_file)
        for md_file in daily_dir.glob("*.md")
        if md_file.is_file()
        and (note_date := parse_note_date(md_file, filename_format)) is not None
        and not is_excluded(
            md_file.relative_to(vault_root), excluded_dirs, excluded_patterns
        )
        and not is_blacklisted(md_file.relative_to(vault_root), blacklist)
    ]
    if not dated:
        return []

    if since is not None:
        in_window = [(d, f) for d, f in dated if d >= since]
    else:
        # Hold back the single most recent note; format every older one.
        latest = max(d for d, _ in dated)
        in_window = [(d, f) for d, f in dated if d < latest]

    eligible = [(d, f) for d, f in in_window if _is_readable_raw_note(f)]
    return [md_file for _, md_file in sorted(eligible)]


def _is_readable_raw_note(path: Path) -> bool:
    """Return True when the note can be read and is not already formatted.

    Unreadable files (OSError or UnicodeDecodeError) are skipped with a
    warning rather than failing the whole scan.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Skipping unreadable daily note %s: %s", path, exc)
        return False
    return not is_already_formatted(text)
