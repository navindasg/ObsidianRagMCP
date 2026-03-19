"""Tests for markdown chunking and frontmatter parsing (Phase 02-01).

TDD RED phase: all tests import from obsidian_rag.markdown_parser which does not yet exist.
"""
from pathlib import Path

import pytest

# These imports are expected to fail in RED phase.
from obsidian_rag.markdown_parser import (
    chunk_by_headings,
    chunk_document,
    parse_frontmatter,
    recursive_split,
)


@pytest.fixture
def sample_vault_path() -> Path:
    """Path to the sample vault fixtures directory."""
    return Path(__file__).parent / "fixtures" / "sample_vault"


# ---------------------------------------------------------------------------
# test_heading_chunks_preserves_path
# ---------------------------------------------------------------------------

def test_heading_chunks_preserves_path():
    """chunk_by_headings preserves the full heading path on each chunk."""
    text = """# Project

Overview text.

## Goals

Some goals here.

### Q1

Q1 specific goals.
"""
    chunks = chunk_by_headings(text, chunk_max_tokens=512, chunk_overlap=0)
    assert len(chunks) >= 1

    # Find a chunk at the H3 level
    h3_chunks = [c for c in chunks if "### Q1" in c["heading_path"]]
    assert h3_chunks, f"No H3 chunk found in: {[c['heading_path'] for c in chunks]}"

    path = h3_chunks[0]["heading_path"]
    # Must contain all ancestor headings separated by " > "
    assert "# Project" in path
    assert "## Goals" in path
    assert "### Q1" in path
    assert " > " in path


# ---------------------------------------------------------------------------
# test_no_headings_uses_recursive_split
# ---------------------------------------------------------------------------

def test_no_headings_uses_recursive_split(sample_vault_path: Path):
    """Document with no headings returns chunks from recursive splitter (heading_path='')."""
    text = (sample_vault_path / "no-headings.md").read_text()
    # Strip frontmatter manually for this test — or use chunk_document
    # Use chunk_by_headings with the body (no headings)
    body = """This is a note without any headings. It contains multiple paragraphs of text that should be chunked using the recursive text splitter since there are no heading boundaries to split on.

Second paragraph with more content. This tests the fallback behavior when no markdown headings are present in the document.

Third paragraph ensures we have enough content to potentially trigger splitting if the chunk_max_tokens is set low enough for testing purposes here.
"""
    chunks = chunk_by_headings(body, chunk_max_tokens=512, chunk_overlap=0)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk["heading_path"] == ""


# ---------------------------------------------------------------------------
# test_recursive_fallback_on_oversized_section
# ---------------------------------------------------------------------------

def test_recursive_fallback_on_oversized_section():
    """A section exceeding chunk_max_tokens is split into multiple chunks carrying the same path."""
    # 100 tokens ~ 400 chars; build a section of ~600 chars to force splitting
    long_para = "This is a sentence that is moderately long. " * 15  # ~660 chars, ~165 tokens
    text = f"# Big Section\n\n{long_para}"
    chunks = chunk_by_headings(text, chunk_max_tokens=100, chunk_overlap=0)
    # Should have more than one chunk
    assert len(chunks) > 1
    for chunk in chunks:
        assert "# Big Section" in chunk["heading_path"]


# ---------------------------------------------------------------------------
# test_code_block_atomic
# ---------------------------------------------------------------------------

def test_code_block_atomic():
    """Fenced code block surrounded by text is never split mid-fence."""
    code = "x = 1\n" * 20  # code content
    text = f"# Examples\n\nSome intro text.\n\n```python\n{code}```\n\nSome trailing text."
    # chunk_max_tokens=100 so sections may be split, but code block must stay whole
    chunks = chunk_by_headings(text, chunk_max_tokens=100, chunk_overlap=0)

    for chunk in chunks:
        chunk_text = chunk["text"]
        # Count fence markers: each complete code block has exactly 2 (opening + closing) or 0
        fence_count = chunk_text.count("```")
        assert fence_count % 2 == 0, (
            f"Code block was split mid-fence in chunk (fence_count={fence_count}):\n{chunk_text!r}"
        )


# ---------------------------------------------------------------------------
# test_code_block_exceeding_max_is_split
# ---------------------------------------------------------------------------

def test_code_block_exceeding_max_is_split():
    """A code block alone exceeding chunk_max_tokens IS split (edge case)."""
    # Very small limit; large code block
    long_code = "x = " + "1" * 500  # ~500 chars of code content
    text = f"```python\n{long_code}\n```"
    # chunk_max_tokens=10 forces splitting even the code block
    chunks = recursive_split(text, chunk_max_tokens=10, chunk_overlap=0)
    assert len(chunks) > 1, "Expected large code block to be split when chunk_max_tokens is very small"


# ---------------------------------------------------------------------------
# test_fixed_window_strategy
# ---------------------------------------------------------------------------

def test_fixed_window_strategy(sample_vault_path: Path, tmp_path: Path):
    """chunk_document with strategy='fixed' splits entire document with overlap."""
    # Create a doc with enough content to produce multiple chunks.
    # Each paragraph is ~300 chars so chunks exceed MIN_CHUNK_CHARS; use chunk_max_tokens=100
    # so two paragraphs together (~600 chars = ~150 tokens) exceeds the limit.
    para = "word " * 60  # ~300 chars, ~75 tokens
    content = "\n\n".join([f"Paragraph {i}: {para}" for i in range(5)])
    doc = tmp_path / "test-fixed.md"
    doc.write_text(content)

    metadata, chunks = chunk_document(
        file_path=doc,
        chunk_strategy="fixed",
        chunk_max_tokens=100,
        chunk_overlap=10,
        include_frontmatter="ignore",
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert "text" in chunk
        assert "heading_path" in chunk


# ---------------------------------------------------------------------------
# test_frontmatter_metadata_only
# ---------------------------------------------------------------------------

def test_frontmatter_metadata_only(sample_vault_path: Path):
    """metadata_only mode: returns metadata dict; body text excludes frontmatter."""
    file = sample_vault_path / "projects" / "wsn-pipeline.md"
    metadata, body = parse_frontmatter(file, include_frontmatter="metadata_only")

    assert "tags" in metadata
    assert "project" in metadata["tags"]
    # Body should NOT contain the YAML frontmatter delimiters
    assert "---" not in body.strip()[:10]


# ---------------------------------------------------------------------------
# test_frontmatter_embed
# ---------------------------------------------------------------------------

def test_frontmatter_embed(sample_vault_path: Path):
    """embed mode: returns metadata dict AND includes frontmatter text in body."""
    file = sample_vault_path / "projects" / "wsn-pipeline.md"
    metadata, body = parse_frontmatter(file, include_frontmatter="embed")

    assert "tags" in metadata
    # Body SHOULD contain the frontmatter delimiters
    assert "---" in body


# ---------------------------------------------------------------------------
# test_frontmatter_ignore
# ---------------------------------------------------------------------------

def test_frontmatter_ignore(sample_vault_path: Path):
    """ignore mode: returns empty metadata and raw text including frontmatter delimiters."""
    file = sample_vault_path / "projects" / "wsn-pipeline.md"
    metadata, body = parse_frontmatter(file, include_frontmatter="ignore")

    assert metadata == {}
    assert "---" in body


# ---------------------------------------------------------------------------
# test_malformed_frontmatter_logs_warning
# ---------------------------------------------------------------------------

def test_malformed_frontmatter_logs_warning(sample_vault_path: Path, caplog):
    """Malformed YAML logs warning to stderr and returns empty metadata with full file text."""
    import logging

    file = sample_vault_path / "malformed-frontmatter.md"

    with caplog.at_level(logging.WARNING, logger="obsidian_rag.markdown_parser"):
        metadata, body = parse_frontmatter(file, include_frontmatter="metadata_only")

    assert metadata == {}, f"Expected empty metadata for malformed frontmatter, got: {metadata}"
    # Content should still be returned
    assert "Valid Content" in body
    # Warning should have been logged
    assert any("Malformed" in record.message or "malformed" in record.message.lower() for record in caplog.records), (
        f"Expected malformed frontmatter warning in logs, got: {[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# test_obsidian_syntax_preserved
# ---------------------------------------------------------------------------

def test_obsidian_syntax_preserved(sample_vault_path: Path):
    """Wikilinks, callouts, #tags, and embeds are preserved as-is in chunk text."""
    file = sample_vault_path / "wikilinks-callouts.md"
    _, body = parse_frontmatter(file, include_frontmatter="metadata_only")

    chunks = chunk_by_headings(body, chunk_max_tokens=512, chunk_overlap=0)
    all_text = " ".join(c["text"] for c in chunks)

    assert "[[wsn-pipeline]]" in all_text, "Wikilink should be preserved"
    assert "[[2024-01-15|yesterday's note]]" in all_text, "Aliased wikilink should be preserved"
    assert "> [!note]" in all_text, "Callout should be preserved"
    assert "#production" in all_text, "Inline tag should be preserved"
    assert "![[wsn-pipeline#Architecture]]" in all_text, "Embed should be preserved"


# ---------------------------------------------------------------------------
# test_to_float32_converts_correctly
# ---------------------------------------------------------------------------

def test_to_float32_converts_correctly():
    """to_float32 converts list[list[float]] to numpy float32 ndarray."""
    import numpy as np

    from obsidian_rag.models import to_float32

    vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = to_float32(vectors)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (2, 3)


# ---------------------------------------------------------------------------
# test_minimum_chunk_discarded
# ---------------------------------------------------------------------------

def test_minimum_chunk_discarded():
    """Chunks smaller than 50 tokens (~200 chars) are discarded."""
    # Build text with some very short paragraphs and one long enough paragraph
    short_para = "Short."  # <200 chars
    long_para = "This is a longer paragraph with sufficient content to meet the minimum chunk size requirement. " * 3
    text = f"{short_para}\n\n{long_para}\n\n{short_para}"

    chunks = recursive_split(text, chunk_max_tokens=512, chunk_overlap=0)

    for chunk in chunks:
        assert len(chunk) >= 200, (
            f"Chunk below minimum size (200 chars) was not discarded: {chunk!r}"
        )
