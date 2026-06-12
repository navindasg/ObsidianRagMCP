"""Opt-in format trigger: a marker tag queues any note for formatting.

Typing the format tag (default "#!format") anywhere in a note opts it in
to the next formatting run, daily or not. The runner scans for the marker,
enqueues the note, and strips the marker so the request is consumed even
when Ollama is down (the queued item survives until it can be processed).

Public API:
    scan_format_tags(vault_root, *, format_tag, excluded_dirs,
        excluded_patterns, blacklist=()) -> list[Path]
    strip_format_tag(path, format_tag) -> None
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

from obsidian_rag.daily_format.detector import is_blacklisted
from obsidian_rag.daily_format.formatter import write_atomically
from obsidian_rag.indexer import is_excluded

logger = logging.getLogger(__name__)


def _marker_re(format_tag: str) -> re.Pattern[str]:
    """Compile a pattern matching the exact marker, not supersets.

    The marker must not be followed by a word character or hyphen, so
    "#!format" never matches inside "#!formatting".
    """
    return re.compile(re.escape(format_tag) + r"(?![\w-])")


def scan_format_tags(
    vault_root: Path,
    *,
    format_tag: str,
    excluded_dirs: list[str],
    excluded_patterns: list[str],
    blacklist: Sequence[str] = (),
) -> list[Path]:
    """Find every note in the vault carrying the format tag.

    The whole vault tree is scanned recursively; excluded and blacklisted
    notes never trigger. Unreadable files are skipped with a warning.

    Args:
        vault_root: Root directory of the Obsidian vault.
        format_tag: Marker text that opts a note in, e.g. "#!format".
        excluded_dirs: Directory names excluded from indexing.
        excluded_patterns: Filename globs excluded from indexing.
        blacklist: Notes never formatted (stems or vault-relative paths).

    Returns:
        Marker-carrying note paths in sorted order.
    """
    marker = _marker_re(format_tag)
    tagged: list[Path] = []
    for md_file in sorted(vault_root.rglob("*.md")):
        if not md_file.is_file():
            continue
        rel_path = md_file.relative_to(vault_root)
        if is_excluded(rel_path, excluded_dirs, excluded_patterns):
            continue
        if is_blacklisted(rel_path, blacklist):
            continue
        try:
            text = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Skipping unreadable note %s: %s", md_file, exc)
            continue
        if marker.search(text):
            tagged.append(md_file)
    return tagged


def strip_format_tag(path: Path, format_tag: str) -> None:
    """Remove every occurrence of the format tag from a note, atomically.

    A marker alone on a line vanishes with the whole line; an inline
    marker disappears along with its leading whitespace. The file is not
    rewritten when no marker is present.
    """
    text = path.read_text(encoding="utf-8")
    escaped = re.escape(format_tag)
    boundary = r"(?![\w-])"
    own_line = re.compile(
        rf"^[ \t]*{escaped}{boundary}[ \t]*(?:\r?\n|\Z)", re.MULTILINE
    )
    inline = re.compile(rf"[ \t]*{escaped}{boundary}")
    stripped = inline.sub("", own_line.sub("", text))
    if stripped == text:
        return
    write_atomically(path, stripped)
    logger.info("Stripped %s from %s", format_tag, path)
