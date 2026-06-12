"""Daily-note candidate detection for the nightly formatter.

Public API:
    parse_note_date(path, filename_format) -> datetime.date | None
    is_already_formatted(text) -> bool
    find_candidates(vault_root, *, daily_folder, filename_format, today,
        start_date, catchup_days, excluded_dirs, excluded_patterns)
        -> list[Path]
"""

from __future__ import annotations

import datetime
import logging
import re
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

    A note counts as formatted when its YAML frontmatter contains a
    'formatted' key, or its body contains an "## Original Notes" heading
    (case-insensitive, optional trailing colon).

    Args:
        text: Full note text.

    Returns:
        True when either formatted marker is present.
    """
    if _frontmatter_has_formatted_key(text):
        return True
    return _ORIGINAL_NOTES_RE.search(text) is not None


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


def find_candidates(
    vault_root: Path,
    *,
    daily_folder: str,
    filename_format: str,
    today: datetime.date,
    start_date: datetime.date,
    catchup_days: int,
    excluded_dirs: list[str],
    excluded_patterns: list[str],
) -> list[Path]:
    """Scan the daily folder for raw daily notes that need formatting.

    A note is eligible when its stem parses to a date d with
    max(start_date, today - catchup_days) <= d < today, it is not excluded
    by the vault's exclusion rules, and it is not already formatted.
    Unreadable files are skipped with a warning.

    Args:
        vault_root: Root directory of the Obsidian vault.
        daily_folder: Folder of daily notes relative to vault_root; "" means
            the vault root itself. Scanned non-recursively.
        filename_format: strftime format the daily-note stems follow.
        today: Current date; notes dated today or later are never eligible.
        start_date: No-backfill floor; notes before this date are skipped.
        catchup_days: Maximum age in days of notes to pick up after downtime.
        excluded_dirs: Directory names excluded from indexing.
        excluded_patterns: Filename globs excluded from indexing.

    Returns:
        Eligible note paths sorted by note date ascending.
    """
    daily_dir = (vault_root / daily_folder) if daily_folder else vault_root
    if not daily_dir.is_dir():
        logger.warning("Daily-note folder does not exist: %s", daily_dir)
        return []

    earliest = max(start_date, today - datetime.timedelta(days=catchup_days))
    dated_notes = (
        (parse_note_date(md_file, filename_format), md_file)
        for md_file in daily_dir.glob("*.md")
        if md_file.is_file()
    )
    eligible = [
        (note_date, md_file)
        for note_date, md_file in dated_notes
        if note_date is not None
        and earliest <= note_date < today
        and not is_excluded(
            md_file.relative_to(vault_root), excluded_dirs, excluded_patterns
        )
        and _is_readable_raw_note(md_file)
    ]
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
