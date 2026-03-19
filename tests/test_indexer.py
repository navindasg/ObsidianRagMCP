"""Tests for obsidian_rag.indexer module.

Covers: vault scanning, batch embedding, FAISS index management,
atomic persistence, SHA-256 hash tracking, and the full build_index pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest

from obsidian_rag.models import AppConfig, ChunkMetadata, to_float32

# ---------------------------------------------------------------------------
# Import from indexer (does not exist yet — RED phase)
# ---------------------------------------------------------------------------
from obsidian_rag.indexer import (
    add_vectors,
    build_index,
    create_index,
    embed_batch,
    find_changed_files,
    load_index,
    persist_index_atomically,
    scan_vault,
    sha256_file,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_VAULT = Path(__file__).parent / "fixtures" / "sample_vault"


@pytest.fixture
def vault_with_obsidian(tmp_path):
    """Creates a tmp vault with .obsidian directory and some .md files."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note1.md").write_text("# Note 1\nContent here")
    (vault / "note2.md").write_text("# Note 2\nMore content")
    hidden = vault / ".obsidian"
    hidden.mkdir()
    (hidden / "config.md").write_text("# config")
    return vault


@pytest.fixture
def vault_with_daily(tmp_path):
    """Creates a tmp vault with daily notes that match a glob pattern."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "daily-2024-01-01.md").write_text("# Daily\nContent")
    (vault / "daily-2024-01-02.md").write_text("# Daily\nContent")
    (vault / "regular-note.md").write_text("# Regular\nContent")
    return vault


@pytest.fixture
def minimal_app_config(tmp_path):
    """Returns a minimal AppConfig pointing at the sample vault."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    (vault_dir / "test.md").write_text(
        "---\ntags:\n  - test\n---\n\n# Test Note\n\nThis is test content for embedding."
    )
    return AppConfig.model_validate({"vaults": [{"name": "test-vault", "path": str(vault_dir)}]})


@pytest.fixture
def mock_embed_client():
    """Returns a mock ollama.Client whose embed() returns 768-dim vectors."""
    client = MagicMock()
    client.embed.side_effect = lambda model, input: MagicMock(
        embeddings=[[0.1] * 768] * len(input)
    )
    return client


# ---------------------------------------------------------------------------
# Vault scanner tests
# ---------------------------------------------------------------------------


def test_scan_vault_finds_md_files():
    """scan_vault on sample_vault returns all .md files."""
    files = scan_vault(SAMPLE_VAULT, excluded_dirs=[], excluded_patterns=[])
    assert len(files) >= 1
    assert all(f.suffix == ".md" for f in files)


def test_scan_vault_excludes_dirs(vault_with_obsidian):
    """scan_vault with excluded_dirs=['.obsidian'] skips files in that directory."""
    files = scan_vault(vault_with_obsidian, excluded_dirs=[".obsidian"], excluded_patterns=[])
    # Should find note1.md and note2.md but NOT .obsidian/config.md
    file_names = [f.name for f in files]
    assert "config.md" not in file_names
    assert "note1.md" in file_names
    assert "note2.md" in file_names


def test_scan_vault_excludes_patterns(vault_with_daily):
    """scan_vault with excluded_patterns=['daily-*.md'] skips matching files."""
    files = scan_vault(vault_with_daily, excluded_dirs=[], excluded_patterns=["daily-*.md"])
    file_names = [f.name for f in files]
    assert "daily-2024-01-01.md" not in file_names
    assert "daily-2024-01-02.md" not in file_names
    assert "regular-note.md" in file_names


# ---------------------------------------------------------------------------
# FAISS index management tests
# ---------------------------------------------------------------------------


def test_create_index_returns_idmap():
    """create_index(768) returns faiss.IndexIDMap with dimension 768."""
    index = create_index(768)
    assert isinstance(index, faiss.IndexIDMap)
    assert index.d == 768


def test_add_vectors_normalizes_and_stores():
    """add_vectors stores normalized float32 vectors with correct IDs."""
    index = create_index(4)
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    ids = [10, 20]
    add_vectors(index, vectors, ids)
    assert index.ntotal == 2
    # Verify IDs are stored
    stored_ids = faiss.vector_to_array(index.id_map)
    assert 10 in stored_ids
    assert 20 in stored_ids


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------


def test_embed_batch_calls_ollama(mock_embed_client):
    """embed_batch with mocked ollama returns expected embeddings."""
    texts = ["text one", "text two", "text three"]
    result = embed_batch(mock_embed_client, "nomic-embed-text", texts, batch_size=64)
    assert len(result) == 3
    assert len(result[0]) == 768
    mock_embed_client.embed.assert_called_once()


def test_embed_batch_multiple_batches(mock_embed_client):
    """10 texts with batch_size=3 produces 4 Ollama calls."""
    texts = [f"text {i}" for i in range(10)]
    result = embed_batch(mock_embed_client, "nomic-embed-text", texts, batch_size=3)
    assert len(result) == 10
    assert mock_embed_client.embed.call_count == 4


# ---------------------------------------------------------------------------
# Atomic persistence tests
# ---------------------------------------------------------------------------


def test_persist_index_atomically_creates_files(tmp_path):
    """persist creates index.faiss, metadata.json, file_hashes.json."""
    vault_dir = tmp_path / "index"
    index = create_index(4)
    add_vectors(index, [[1.0, 0.0, 0.0, 0.0]], [0])
    metadata = {"0": {"chunk_id": 0, "file": "test.md", "heading_path": "# Test"}}
    file_hashes = {"test.md": "abc123"}

    persist_index_atomically(index, metadata, file_hashes, vault_dir)

    assert (vault_dir / "index.faiss").exists()
    assert (vault_dir / "metadata.json").exists()
    assert (vault_dir / "file_hashes.json").exists()


def test_persist_index_no_tmp_files_remain(tmp_path):
    """After persist, no .tmp files exist in the directory."""
    vault_dir = tmp_path / "index"
    index = create_index(4)
    add_vectors(index, [[1.0, 0.0, 0.0, 0.0]], [0])
    metadata = {}
    file_hashes = {}

    persist_index_atomically(index, metadata, file_hashes, vault_dir)

    tmp_files = list(vault_dir.glob("*.tmp"))
    assert len(tmp_files) == 0


# ---------------------------------------------------------------------------
# Load index tests
# ---------------------------------------------------------------------------


def test_load_index_reads_persisted(tmp_path):
    """load_index reads back the index and metadata written by persist."""
    vault_dir = tmp_path / "index"
    index = create_index(4)
    add_vectors(index, [[1.0, 0.0, 0.0, 0.0]], [0])
    metadata = {"0": {"chunk_id": 0, "file": "test.md", "heading_path": "# Test"}}
    file_hashes = {"test.md": "abc123"}

    persist_index_atomically(index, metadata, file_hashes, vault_dir)

    loaded_index, loaded_meta, loaded_hashes = load_index(vault_dir)
    assert loaded_index is not None
    assert loaded_index.ntotal == 1
    assert loaded_meta == metadata
    assert loaded_hashes == file_hashes


# ---------------------------------------------------------------------------
# Hash tracking tests
# ---------------------------------------------------------------------------


def test_sha256_file_deterministic(tmp_path):
    """sha256_file returns same hash for same content, different for different content."""
    file_a = tmp_path / "a.md"
    file_b = tmp_path / "b.md"
    file_a.write_text("same content")
    file_b.write_text("different content")
    file_a2 = tmp_path / "a2.md"
    file_a2.write_text("same content")

    hash_a = sha256_file(file_a)
    hash_a2 = sha256_file(file_a2)
    hash_b = sha256_file(file_b)

    assert hash_a == hash_a2
    assert hash_a != hash_b
    assert len(hash_a) == 64  # sha256 hex digest


def test_find_changed_files_detects_new(tmp_path):
    """New file not in stored_hashes appears in to_reindex."""
    vault = tmp_path / "vault"
    vault.mkdir()
    new_file = vault / "new.md"
    new_file.write_text("# New\nContent")

    to_reindex, deleted = find_changed_files(vault, [new_file], stored_hashes={})

    assert new_file in to_reindex
    assert len(deleted) == 0


def test_find_changed_files_detects_modified(tmp_path):
    """File with different hash appears in to_reindex."""
    vault = tmp_path / "vault"
    vault.mkdir()
    modified = vault / "modified.md"
    modified.write_text("# Modified\nUpdated content")

    # Store a different (stale) hash
    stored = {"modified.md": "stalehash000"}
    to_reindex, deleted = find_changed_files(vault, [modified], stored_hashes=stored)

    assert modified in to_reindex


def test_find_changed_files_detects_deleted(tmp_path):
    """File in stored_hashes but not on disk appears in deleted list."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # File exists in stored hashes but not on disk
    stored = {"ghost.md": "abc123"}
    to_reindex, deleted = find_changed_files(vault, [], stored_hashes=stored)

    assert "ghost.md" in deleted


def test_find_changed_files_skips_unchanged(tmp_path):
    """File with same hash is NOT in to_reindex."""
    vault = tmp_path / "vault"
    vault.mkdir()
    unchanged = vault / "unchanged.md"
    unchanged.write_text("# Unchanged\nSame content always")

    actual_hash = sha256_file(unchanged)
    stored = {"unchanged.md": actual_hash}

    to_reindex, deleted = find_changed_files(vault, [unchanged], stored_hashes=stored)

    assert unchanged not in to_reindex
    assert len(deleted) == 0


# ---------------------------------------------------------------------------
# Model utility test
# ---------------------------------------------------------------------------


def test_float32_utility_from_models():
    """to_float32 returns np.float32 array (verifies IDX-09 shared utility)."""
    vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = to_float32(vectors)
    assert result.dtype == np.float32
    assert result.shape == (2, 3)


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


def test_build_index_full_pipeline(tmp_path):
    """build_index with mocked Ollama processes sample vault, creates index with
    correct chunk count, persists all files."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    # Write two notes with headings so we get consistent chunks
    (vault_dir / "note-a.md").write_text(
        "---\ntags:\n  - test\n---\n\n# Note A\n\nThis is note A content with enough text to form a valid chunk."
    )
    (vault_dir / "note-b.md").write_text(
        "# Note B\n\nThis is note B content with enough text to form a valid chunk for the index."
    )

    config = AppConfig.model_validate({
        "vaults": [{"name": "test-vault", "path": str(vault_dir)}]
    })
    vault_config = config.vaults[0]

    with patch("obsidian_rag.indexer.ollama_client.Client") as mock_cls:
        mock_client = MagicMock()
        mock_client.embed.side_effect = lambda model, input: MagicMock(
            embeddings=[[0.1] * 768] * len(input)
        )
        mock_cls.return_value = mock_client

        # Override storage path to tmp_path
        storage_path = tmp_path / ".obsidian-rag" / "test-vault"
        with patch("obsidian_rag.indexer.Path.home", return_value=tmp_path):
            index, metadata, file_hashes = build_index(config, vault_config)

    assert index is not None
    assert index.ntotal > 0
    assert len(metadata) > 0
    assert len(file_hashes) > 0

    # Check that chunk text is stored in metadata (RET-01)
    for chunk_meta in metadata.values():
        assert "text" in chunk_meta
        assert isinstance(chunk_meta["text"], str)
