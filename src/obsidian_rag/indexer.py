"""Indexing engine for Obsidian vault files.

Public API:
    scan_vault(vault_path, excluded_dirs, excluded_patterns) -> list[Path]
    create_index(dimensions) -> faiss.IndexIDMap
    add_vectors(index, vectors, ids) -> None
    embed_batch(client, model, texts, batch_size) -> list[list[float]]
    sha256_file(path) -> str
    find_changed_files(vault_path, md_files, stored_hashes) -> tuple[list[Path], list[str]]
    persist_index_atomically(index, metadata, file_hashes, vault_dir) -> None
    load_index(vault_dir) -> tuple[faiss.IndexIDMap | None, dict, dict]
    build_index(config, vault_config, progress_callback) -> tuple[faiss.IndexIDMap, dict, dict]
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np
import ollama as ollama_client

from obsidian_rag.markdown_parser import chunk_document
from obsidian_rag.models import AppConfig, ChunkMetadata, to_float32

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vault scanner
# ---------------------------------------------------------------------------


def scan_vault(
    vault_path: Path,
    excluded_dirs: list[str],
    excluded_patterns: list[str],
) -> list[Path]:
    """Return all .md files not under excluded dirs or matching excluded patterns.

    Args:
        vault_path: Root directory of the Obsidian vault.
        excluded_dirs: Directory names to skip (e.g. [".obsidian", ".trash"]).
        excluded_patterns: Glob patterns for filenames to skip (e.g. ["daily-*.md"]).

    Returns:
        Sorted list of absolute paths to .md files.
    """
    excluded_dir_set = set(excluded_dirs)
    result: list[Path] = []

    for md_file in vault_path.rglob("*.md"):
        rel = md_file.relative_to(vault_path)
        # Skip if any ancestor directory is in excluded_dirs
        if any(part in excluded_dir_set for part in rel.parts[:-1]):
            continue
        # Skip if filename matches any excluded_patterns glob
        if any(fnmatch.fnmatch(md_file.name, pat) for pat in excluded_patterns):
            continue
        result.append(md_file)

    return sorted(result)


# ---------------------------------------------------------------------------
# FAISS index management
# ---------------------------------------------------------------------------


def create_index(dimensions: int = 768) -> faiss.IndexIDMap:
    """Create a new empty FAISS IndexIDMap wrapping IndexFlatL2.

    IndexIDMap enables custom int64 IDs and future remove_ids support.

    Args:
        dimensions: Embedding dimensions (768 for nomic-embed-text).

    Returns:
        Empty faiss.IndexIDMap ready for add_with_ids.
    """
    base_index = faiss.IndexFlatL2(dimensions)
    return faiss.IndexIDMap(base_index)


def add_vectors(
    index: faiss.IndexIDMap,
    vectors: list[list[float]],
    ids: list[int],
) -> None:
    """Add float32 L2-normalized vectors with custom IDs to the index.

    Vectors are cast to float32 via to_float32() then normalized in-place
    with faiss.normalize_L2() for cosine similarity via L2.

    Args:
        index: Target FAISS IndexIDMap.
        vectors: List of embedding vectors (any float type).
        ids: Parallel list of integer IDs for each vector.
    """
    arr = to_float32(vectors)
    faiss.normalize_L2(arr)
    id_arr = np.array(ids, dtype=np.int64)
    index.add_with_ids(arr, id_arr)


# ---------------------------------------------------------------------------
# Batch embedding
# ---------------------------------------------------------------------------


def embed_batch(
    client: ollama_client.Client,
    model: str,
    texts: list[str],
    batch_size: int = 64,
) -> list[list[float]]:
    """Embed texts in batches using Ollama client.

    Iterates over texts in slices of batch_size, calling client.embed()
    once per batch. Returns a flat list of all embedding vectors.

    Args:
        client: An ollama.Client instance.
        model: Embedding model name (e.g. "nomic-embed-text").
        texts: List of strings to embed.
        batch_size: Number of texts per Ollama API call.

    Returns:
        List of embedding vectors, one per input text.
    """
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embed(model=model, input=batch)
        all_embeddings.extend(response.embeddings)
    return all_embeddings


# ---------------------------------------------------------------------------
# SHA-256 hash tracking
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file's content.

    Reads in 64KB blocks to handle large files without loading all content
    into memory at once.

    Args:
        path: Path to the file.

    Returns:
        64-character lowercase hex digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def find_changed_files(
    vault_path: Path,
    md_files: list[Path],
    stored_hashes: dict[str, str],
) -> tuple[list[Path], list[str]]:
    """Compare current file hashes against stored hashes.

    Args:
        vault_path: Vault root for computing relative paths.
        md_files: Current list of .md files found by scan_vault.
        stored_hashes: Dict mapping relative path -> sha256 hex from previous run.

    Returns:
        (to_reindex, deleted_relative_paths) where:
          - to_reindex: files that are new or have changed content
          - deleted_relative_paths: relative paths in stored_hashes that no longer exist
    """
    current_rel: dict[str, Path] = {
        str(f.relative_to(vault_path)): f for f in md_files
    }
    to_reindex: list[Path] = []

    for rel_path, abs_path in current_rel.items():
        new_hash = sha256_file(abs_path)
        if stored_hashes.get(rel_path) != new_hash:
            to_reindex.append(abs_path)

    deleted = [k for k in stored_hashes if k not in current_rel]
    return to_reindex, deleted


# ---------------------------------------------------------------------------
# Atomic persistence
# ---------------------------------------------------------------------------


def persist_index_atomically(
    index: faiss.IndexIDMap,
    metadata: dict,
    file_hashes: dict,
    vault_dir: Path,
) -> None:
    """Write FAISS index, metadata, and hashes atomically.

    Uses temp-file + os.replace() pattern so a mid-write crash leaves the
    previous consistent state intact. Creation order: metadata first, then
    hashes, then index — so a partial state always has metadata for the
    existing index.

    Args:
        index: FAISS IndexIDMap to persist.
        metadata: Dict mapping str(chunk_id) -> chunk metadata dict.
        file_hashes: Dict mapping relative_path -> sha256 hex.
        vault_dir: Target directory (created if it doesn't exist).
    """
    vault_dir.mkdir(parents=True, exist_ok=True)

    # Metadata JSON
    tmp_meta = vault_dir / "metadata.json.tmp"
    tmp_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(tmp_meta, vault_dir / "metadata.json")

    # File hashes JSON
    tmp_hashes = vault_dir / "file_hashes.json.tmp"
    tmp_hashes.write_text(json.dumps(file_hashes, indent=2), encoding="utf-8")
    os.replace(tmp_hashes, vault_dir / "file_hashes.json")

    # FAISS binary index
    tmp_index = vault_dir / "index.faiss.tmp"
    faiss.write_index(index, str(tmp_index))
    os.replace(tmp_index, vault_dir / "index.faiss")


def load_index(
    vault_dir: Path,
) -> tuple[faiss.IndexIDMap | None, dict, dict]:
    """Load previously persisted FAISS index and metadata.

    Args:
        vault_dir: Directory containing index.faiss, metadata.json, file_hashes.json.

    Returns:
        (index, metadata, file_hashes) if all files exist, else (None, {}, {}).
    """
    index_path = vault_dir / "index.faiss"
    meta_path = vault_dir / "metadata.json"
    hashes_path = vault_dir / "file_hashes.json"

    if not (index_path.exists() and meta_path.exists() and hashes_path.exists()):
        return None, {}, {}

    try:
        index = faiss.read_index(str(index_path))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        file_hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        return index, metadata, file_hashes
    except Exception:
        logger.exception("Failed to load index from %s", vault_dir)
        return None, {}, {}


# ---------------------------------------------------------------------------
# Full indexing pipeline
# ---------------------------------------------------------------------------


def build_index(
    config: AppConfig,
    vault_config,
    progress_callback=None,
) -> tuple[faiss.IndexIDMap, dict, dict]:
    """Build or update the FAISS index for a single vault.

    Orchestrates: load existing index -> scan vault -> diff hashes ->
    remove deleted chunks -> embed changed files -> add to index -> persist.

    Chunk text is stored in ChunkMetadata.text for downstream snippet
    retrieval (RET-01).

    Args:
        config: Full AppConfig with embedding and indexing settings.
        vault_config: VaultConfig for the vault to index.
        progress_callback: Optional callable(current, total) for progress.

    Returns:
        (index, metadata, file_hashes) after indexing.
    """
    vault_path = vault_config.path
    storage_dir = Path.home() / ".obsidian-rag" / vault_config.name

    # Load any existing index
    existing_index, metadata, file_hashes = load_index(storage_dir)

    # Scan current vault
    md_files = scan_vault(
        vault_path,
        excluded_dirs=vault_config.excluded_dirs,
        excluded_patterns=vault_config.excluded_patterns,
    )

    # Find changed files
    to_reindex, deleted_rel_paths = find_changed_files(vault_path, md_files, file_hashes)

    # Remove deleted file chunks from existing index
    if existing_index is not None and deleted_rel_paths:
        ids_to_remove = [
            int(chunk_id)
            for chunk_id, meta in metadata.items()
            if meta.get("file") in deleted_rel_paths
        ]
        if ids_to_remove:
            existing_index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
        for rel_path in deleted_rel_paths:
            metadata = {k: v for k, v in metadata.items() if v.get("file") != rel_path}
            file_hashes.pop(rel_path, None)

    # Determine next chunk ID
    if metadata:
        next_id = max(int(k) for k in metadata.keys()) + 1
    else:
        next_id = 0

    # Collect new chunks from changed files
    new_chunks: list[dict] = []
    new_chunk_ids: list[int] = []
    new_chunk_meta: list[dict] = []

    total_files = len(to_reindex)
    for i, file_path in enumerate(to_reindex):
        pct = int((i + 1) / max(total_files, 1) * 100)
        logger.info("Indexing: %d/%d files (%d%%)", i + 1, total_files, pct)

        if progress_callback is not None:
            progress_callback(i + 1, total_files)

        try:
            file_metadata, chunks = chunk_document(
                file_path,
                chunk_strategy=config.indexing.chunk_strategy,
                chunk_max_tokens=config.indexing.chunk_max_tokens,
                chunk_overlap=config.indexing.chunk_overlap,
                include_frontmatter=config.indexing.include_frontmatter,
            )
        except Exception:
            logger.exception("Failed to chunk %s — skipping", file_path)
            continue

        rel_path = str(file_path.relative_to(vault_path))
        folder = rel_path.split("/")[0] if "/" in rel_path else ""
        mtime = os.path.getmtime(file_path)
        tags = file_metadata.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        for chunk in chunks:
            chunk_id = next_id
            next_id += 1

            new_chunks.append(chunk)
            new_chunk_ids.append(chunk_id)
            new_chunk_meta.append(
                ChunkMetadata(
                    chunk_id=chunk_id,
                    file=rel_path,
                    heading_path=chunk.get("heading_path", ""),
                    text=chunk.get("text", ""),
                    tags=tags,
                    folder=folder,
                    vault=vault_config.name,
                    modified_ts=mtime,
                    char_count=len(chunk.get("text", "")),
                ).model_dump()
            )

        # Update file hash
        file_hashes[rel_path] = sha256_file(file_path)

    # Embed all new chunks
    if new_chunks:
        client = ollama_client.Client(host=config.embedding.ollama_url)
        texts = [c.get("text", "") for c in new_chunks]
        embeddings = embed_batch(
            client,
            model=config.embedding.model,
            texts=texts,
            batch_size=config.embedding.batch_size,
        )

        dimensions = len(embeddings[0])

        if existing_index is not None:
            index = existing_index
        else:
            index = create_index(dimensions)

        add_vectors(index, embeddings, new_chunk_ids)

        for chunk_id, meta in zip(new_chunk_ids, new_chunk_meta):
            metadata[str(chunk_id)] = meta
    else:
        # No new chunks — use existing index or create empty one
        if existing_index is not None:
            index = existing_index
        else:
            index = create_index(768)

    persist_index_atomically(index, metadata, file_hashes, storage_dir)

    return index, metadata, file_hashes
