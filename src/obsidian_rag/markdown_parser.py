"""Markdown chunking and frontmatter parsing for Obsidian vault indexing.

Public API:
    parse_frontmatter(file_path, include_frontmatter) -> tuple[dict, str]
    chunk_by_headings(text, chunk_max_tokens, chunk_overlap) -> list[dict]
    recursive_split(text, chunk_max_tokens, chunk_overlap, separators) -> list[str]
    chunk_document(file_path, chunk_strategy, chunk_max_tokens, chunk_overlap, include_frontmatter) -> tuple[dict, list[dict]]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import frontmatter
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
MIN_CHUNK_CHARS = 200  # ~50 tokens; discard chunks smaller than this


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _token_estimate(text: str) -> int:
    """Approximate token count: ~4 characters per token."""
    return len(text) // 4


def _build_heading_path(matches: list) -> str:
    """Build breadcrumb from a list of regex match objects up to current heading.

    Keeps only the most recent heading at each level, discards deeper levels
    when a shallower heading appears.

    Args:
        matches: List of re.Match objects from HEADING_RE, ordered by position.

    Returns:
        Breadcrumb string like "# Title > ## Section > ### Subsection".
    """
    # Build list of (level, title) tuples from matches
    heading_list: list[tuple[int, str]] = []
    for m in matches:
        level = len(m.group(1))
        title = m.group(2).strip()
        heading_list.append((level, title))

    # Keep only the most recent heading at each level, using a stack approach.
    # When a heading appears that is <= the level of the last entry, pop deeper ones.
    stack: list[tuple[int, str]] = []
    for level, title in heading_list:
        # Remove any stack entries at same or deeper level
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

    return " > ".join(f"{'#' * lvl} {ttl}" for lvl, ttl in stack)


def _split_preserving_code_blocks(text: str, separator: str) -> list[str]:
    """Split text on separator without splitting inside fenced code blocks.

    Strategy: replace each code block with a unique placeholder, split on
    separator, then restore the placeholders in each resulting part.

    Args:
        text: Input text which may contain fenced code blocks.
        separator: The string separator to split on.

    Returns:
        List of text parts with code blocks intact.
    """
    placeholders: dict[str, str] = {}
    protected = text

    # Find all code fences and replace with placeholders (in reverse order to
    # preserve offsets).
    for i, match in enumerate(CODE_FENCE_RE.finditer(text)):
        key = f"\x00CODE{i}\x00"
        placeholders[key] = match.group(0)
        # Replace only the first occurrence of this specific match content;
        # use the protected string which has prior replacements already applied.
        protected = protected.replace(match.group(0), key, 1)

    parts = protected.split(separator) if separator else list(protected)

    # Restore code block content in each part.
    result = []
    for part in parts:
        restored = part
        for key, value in placeholders.items():
            if key in restored:
                restored = restored.replace(key, value)
        result.append(restored)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_frontmatter(
    file_path: Path,
    include_frontmatter: str = "metadata_only",
) -> tuple[dict, str]:
    """Parse YAML frontmatter from a markdown file.

    Args:
        file_path: Path to the markdown file.
        include_frontmatter: One of:
            - "metadata_only" (default): return metadata dict + body text (no frontmatter)
            - "embed": return metadata dict + full text including frontmatter
            - "ignore": return empty dict + raw file text as-is

    Returns:
        (metadata_dict, body_text) tuple.
        On malformed YAML: logs warning to stderr, returns ({}, raw_text).
    """
    raw = file_path.read_text(encoding="utf-8")

    if include_frontmatter == "ignore":
        return {}, raw

    try:
        post = frontmatter.loads(raw)
        metadata = dict(post.metadata)
        body = post.content

        if include_frontmatter == "embed":
            return metadata, raw

        # "metadata_only" mode: return body without frontmatter
        return metadata, body

    except yaml.YAMLError:
        logger.warning(
            "Malformed YAML frontmatter in %s - skipping frontmatter",
            file_path,
        )
        return {}, raw


def recursive_split(
    text: str,
    chunk_max_tokens: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """Split text using progressively finer separators until chunks fit within chunk_max_tokens.

    This implements the LangChain RecursiveCharacterTextSplitter algorithm directly,
    without LangChain as a dependency.

    Separator hierarchy (default): ["\n\n", "\n", " ", ""]
    Code blocks are kept atomic for all separators except "".

    Args:
        text: Input text to split.
        chunk_max_tokens: Maximum token count per chunk (~4 chars per token).
        chunk_overlap: Number of tokens of overlap between adjacent chunks.
        separators: Custom separator list (default hierarchy used if None).

    Returns:
        List of text strings, each within chunk_max_tokens. When the text
        splits into multiple chunks, fragments smaller than MIN_CHUNK_CHARS
        (200 chars, capped at chunk_max_tokens * 4) are discarded; a text
        that fits in a single chunk is always kept whole.
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    def _trim_to_tokens(parts: list[str], token_limit: int, sep: str) -> list[str]:
        """Return the tail of parts that fits within token_limit tokens.

        Counts characters (including separators) rather than per-part token
        estimates: short parts would otherwise round to 0 tokens and be kept
        without bound, making "zero overlap" duplicate entire chunks.
        """
        if token_limit <= 0:
            return []
        kept: list[str] = []
        total_chars = 0
        for part in reversed(parts):
            candidate_chars = total_chars + len(part) + (len(sep) if kept else 0)
            if candidate_chars // 4 <= token_limit:
                kept.insert(0, part)
                total_chars = candidate_chars
            else:
                break
        return kept

    def _joined_token_estimate(parts: list[str], sep: str) -> int:
        """Estimate tokens for the joined result of parts."""
        return _token_estimate(sep.join(parts))

    def _split_inner(text: str, seps: list[str]) -> list[str]:
        if not seps or _token_estimate(text) <= chunk_max_tokens:
            return [text] if text.strip() else []

        sep = seps[0]
        if sep:
            parts = _split_preserving_code_blocks(text, sep)
        else:
            parts = list(text)

        result: list[str] = []
        current_parts: list[str] = []

        for part in parts:
            # Use joined size to correctly estimate tokens (avoids zero-token words)
            candidate_parts = current_parts + [part]
            candidate_tokens = _joined_token_estimate(candidate_parts, sep)
            part_tokens = _token_estimate(part)

            if part_tokens > chunk_max_tokens:
                # This part alone exceeds the limit — flush current accumulation,
                # then recurse on the oversized part.
                if current_parts:
                    chunk_text = sep.join(current_parts)
                    result.append(chunk_text)
                    current_parts = []
                result.extend(_split_inner(part, seps[1:]))
            elif candidate_tokens > chunk_max_tokens and current_parts:
                # Adding this part would exceed the limit — emit current and start fresh.
                chunk_text = sep.join(current_parts)
                result.append(chunk_text)
                overlap_parts = _trim_to_tokens(current_parts, chunk_overlap, sep)
                current_parts = overlap_parts + [part]
            else:
                current_parts.append(part)

        if current_parts:
            chunk_text = sep.join(current_parts)
            result.append(chunk_text)

        return [r for r in result if r.strip()]

    raw_chunks = _split_inner(text, separators)

    # A document (or section) that fits in a single chunk is kept whole even
    # when short — discarding it would silently make the note unsearchable.
    if len(raw_chunks) <= 1:
        return raw_chunks

    # The minimum is moot when the configured max chunk size is itself smaller:
    # no chunk could ever reach MIN_CHUNK_CHARS, so everything would be dropped.
    min_chars = min(MIN_CHUNK_CHARS, chunk_max_tokens * 4)
    return [c for c in raw_chunks if len(c.strip()) >= min_chars]


def chunk_by_headings(
    text: str,
    chunk_max_tokens: int,
    chunk_overlap: int,
) -> list[dict]:
    """Split markdown text at heading boundaries.

    Each section (text between consecutive headings) becomes one chunk. Sections
    that exceed chunk_max_tokens fall back to recursive_split. Documents with no
    headings fall back to recursive_split on the entire text.

    Each result dict has keys:
        - "heading_path" (str): breadcrumb e.g. "# Title > ## Section"
        - "text" (str): chunk text

    Args:
        text: Markdown body text (frontmatter should already be stripped).
        chunk_max_tokens: Maximum tokens per chunk.
        chunk_overlap: Token overlap between chunks in the recursive fallback.

    Returns:
        List of chunk dicts.
    """
    # Ignore heading-like lines inside fenced code blocks (e.g. "# comment").
    fence_spans = [m.span() for m in CODE_FENCE_RE.finditer(text)]
    matches = [
        m
        for m in HEADING_RE.finditer(text)
        if not any(start <= m.start() < end for start, end in fence_spans)
    ]

    if not matches:
        # No headings — apply recursive splitter to the whole document.
        return [
            {"heading_path": "", "text": chunk}
            for chunk in recursive_split(text, chunk_max_tokens, chunk_overlap)
        ]

    sections = []

    # Content before the first heading (intro paragraphs) must not be lost.
    preamble = text[: matches[0].start()]
    if preamble.strip():
        for chunk in recursive_split(preamble, chunk_max_tokens, chunk_overlap):
            sections.append({"heading_path": "", "text": chunk})

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        heading_path = _build_heading_path(matches[: i + 1])

        if _token_estimate(section_text) <= chunk_max_tokens:
            sections.append({"heading_path": heading_path, "text": section_text})
        else:
            # Oversized — fall back to recursive splitter within this section.
            for sub_chunk in recursive_split(section_text, chunk_max_tokens, chunk_overlap):
                sections.append({"heading_path": heading_path, "text": sub_chunk})

    return sections


def chunk_document(
    file_path: Path,
    chunk_strategy: str = "heading",
    chunk_max_tokens: int = 512,
    chunk_overlap: int = 50,
    include_frontmatter: str = "metadata_only",
) -> tuple[dict, list[dict]]:
    """Orchestrate frontmatter parsing and chunking for a single document.

    Args:
        file_path: Path to the markdown file.
        chunk_strategy: "heading" (default) or "fixed".
        chunk_max_tokens: Maximum tokens per chunk.
        chunk_overlap: Token overlap between chunks.
        include_frontmatter: One of "metadata_only", "embed", "ignore".

    Returns:
        (metadata, chunks) where chunks is a list of dicts with
        keys "heading_path" (str) and "text" (str).
    """
    metadata, body = parse_frontmatter(file_path, include_frontmatter=include_frontmatter)

    if chunk_strategy == "fixed":
        chunks = [
            {"heading_path": "", "text": chunk}
            for chunk in recursive_split(body, chunk_max_tokens, chunk_overlap)
        ]
    else:
        # Default: heading-based chunking
        chunks = chunk_by_headings(body, chunk_max_tokens, chunk_overlap)

    return metadata, chunks
