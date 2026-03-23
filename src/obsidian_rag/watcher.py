"""File system watcher for incremental Obsidian vault index updates.

Public API:
    VaultEventHandler: FileSystemEventHandler subclass with debounced batch flush.
    VaultWatcher: Lifecycle manager that schedules handlers and manages Observer.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np
import ollama
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from obsidian_rag.indexer import (
    embed_batch,
    add_vectors,
    persist_index_atomically,
    sha256_file,
)
from obsidian_rag.markdown_parser import chunk_document
from obsidian_rag.models import AppConfig, ChunkMetadata

logger = logging.getLogger(__name__)

# Debounce window: rapid saves within this many seconds are coalesced.
DEBOUNCE_SECONDS: float = 2.0


class VaultEventHandler(FileSystemEventHandler):
    """Handle file system events for a single vault with debounced batch processing.

    Events are buffered in pending sets. A timer fires DEBOUNCE_SECONDS after
    the last event, draining the sets and performing incremental FAISS updates.
    """

    def __init__(
        self,
        vault_name: str,
        vault_indexes: dict,
        index_lock: threading.Lock,
        config: AppConfig,
    ) -> None:
        super().__init__()
        self._vault_name = vault_name
        self._vault_indexes = vault_indexes
        self._index_lock = index_lock
        self._config = config
        self._debounce_seconds: float = DEBOUNCE_SECONDS

        self._pending_upserts: set[str] = set()
        self._pending_deletes: set[str] = set()
        self._timer: threading.Timer | None = None
        self._timer_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_created(self, event) -> None:
        if event.is_directory or not event.src_path.endswith(".md"):
            return
        self._pending_upserts.add(event.src_path)
        self._reset_timer()

    def on_modified(self, event) -> None:
        if event.is_directory or not event.src_path.endswith(".md"):
            return
        self._pending_upserts.add(event.src_path)
        self._reset_timer()

    def on_deleted(self, event) -> None:
        if event.is_directory or not event.src_path.endswith(".md"):
            return
        self._pending_deletes.add(event.src_path)
        self._reset_timer()

    def on_moved(self, event) -> None:
        if event.is_directory:
            return
        if event.src_path.endswith(".md"):
            self._pending_deletes.add(event.src_path)
        if event.dest_path.endswith(".md"):
            self._pending_upserts.add(event.dest_path)
        self._reset_timer()

    # ------------------------------------------------------------------
    # Debounce timer
    # ------------------------------------------------------------------

    def _reset_timer(self) -> None:
        """Cancel any existing timer and start a new one."""
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
            timer = threading.Timer(self._debounce_seconds, self._flush)
            timer.daemon = True
            timer.start()
            self._timer = timer

    # ------------------------------------------------------------------
    # Flush: incremental index update
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Drain pending sets and apply incremental FAISS updates."""
        with self._timer_lock:
            to_upsert = set(self._pending_upserts)
            to_delete = set(self._pending_deletes)
            self._pending_upserts.clear()
            self._pending_deletes.clear()
            self._timer = None

        if not to_upsert and not to_delete:
            return

        vi = self._vault_indexes[self._vault_name]
        vault_config = vi["vault_config"]
        index = vi["index"]
        metadata: dict = vi["metadata"]
        file_hashes: dict = vi["file_hashes"]
        vault_path: Path = vault_config.path

        # ------------------------------------------------------------------
        # Phase 1: Prepare deletions (compute IDs outside lock)
        # ------------------------------------------------------------------

        delete_id_sets: dict[str, list[int]] = {}

        for abs_path in to_delete:
            rel_path = str(Path(abs_path).relative_to(vault_path))
            ids = [int(k) for k, m in metadata.items() if m.get("file") == rel_path]
            delete_id_sets[rel_path] = ids

        # ------------------------------------------------------------------
        # Phase 2: Prepare upserts (chunk + embed OUTSIDE the index lock)
        # ------------------------------------------------------------------

        upsert_data: list[dict] = []  # {rel_path, abs_path, chunks, embeddings, hash}
        upsert_old_ids: dict[str, list[int]] = {}

        for abs_path in to_upsert:
            if not Path(abs_path).exists():
                # File was removed between event and flush
                logger.debug("Skipping upsert for missing file: %s", abs_path)
                continue

            rel_path = str(Path(abs_path).relative_to(vault_path))
            old_ids = [int(k) for k, m in metadata.items() if m.get("file") == rel_path]
            upsert_old_ids[rel_path] = old_ids

            try:
                file_metadata, chunks = chunk_document(
                    Path(abs_path),
                    chunk_strategy=self._config.indexing.chunk_strategy,
                    chunk_max_tokens=self._config.indexing.chunk_max_tokens,
                    chunk_overlap=self._config.indexing.chunk_overlap,
                    include_frontmatter=self._config.indexing.include_frontmatter,
                )
            except Exception:
                logger.exception("Failed to chunk %s — skipping", abs_path)
                continue

            if not chunks:
                logger.debug("No chunks produced for %s — skipping", abs_path)
                continue

            upsert_data.append(
                {
                    "abs_path": abs_path,
                    "rel_path": rel_path,
                    "file_metadata": file_metadata,
                    "chunks": chunks,
                }
            )

        # Embed all new chunks at once (outside the index lock)
        all_texts: list[str] = []
        for entry in upsert_data:
            all_texts.extend(c.get("text", "") for c in entry["chunks"])

        embeddings: list[list[float]] = []
        if all_texts:
            try:
                client = ollama.Client(host=self._config.embedding.ollama_url)
                embeddings = embed_batch(
                    client,
                    model=self._config.embedding.model,
                    texts=all_texts,
                    batch_size=self._config.embedding.batch_size,
                )
            except Exception:
                logger.exception("Embedding failed during incremental update — aborting flush")
                return

        # ------------------------------------------------------------------
        # Phase 3: Mutate index under lock
        # ------------------------------------------------------------------

        all_ids_to_remove: list[int] = []
        for ids in delete_id_sets.values():
            all_ids_to_remove.extend(ids)
        for ids in upsert_old_ids.values():
            all_ids_to_remove.extend(ids)

        with self._index_lock:
            # Remove stale and deleted vectors
            if all_ids_to_remove:
                remove_arr = np.array(all_ids_to_remove, dtype=np.int64)
                index.remove_ids(remove_arr)
                for chunk_id in all_ids_to_remove:
                    metadata.pop(str(chunk_id), None)

            # Remove hash entries for deleted files
            for rel_path in delete_id_sets:
                file_hashes.pop(rel_path, None)

            # Compute next_id
            next_id = (max(int(k) for k in metadata.keys()) + 1) if metadata else 0

            # Add new vectors and metadata
            embedding_offset = 0
            for entry in upsert_data:
                abs_path = entry["abs_path"]
                rel_path = entry["rel_path"]
                file_metadata = entry["file_metadata"]
                chunks = entry["chunks"]

                tags = file_metadata.get("tags", [])
                if isinstance(tags, str):
                    tags = [tags]

                folder = rel_path.split("/")[0] if "/" in rel_path else ""
                mtime = os.path.getmtime(abs_path)

                chunk_count = len(chunks)
                chunk_embeddings = embeddings[embedding_offset : embedding_offset + chunk_count]
                embedding_offset += chunk_count

                if not chunk_embeddings:
                    continue

                new_ids = list(range(next_id, next_id + chunk_count))
                next_id += chunk_count

                add_vectors(index, chunk_embeddings, new_ids)

                for chunk_id, chunk in zip(new_ids, chunks):
                    metadata[str(chunk_id)] = ChunkMetadata(
                        chunk_id=chunk_id,
                        file=rel_path,
                        heading_path=chunk.get("heading_path", ""),
                        text=chunk.get("text", ""),
                        tags=tags,
                        folder=folder,
                        vault=self._vault_name,
                        modified_ts=mtime,
                        char_count=len(chunk.get("text", "")),
                    ).model_dump()

                try:
                    file_hashes[rel_path] = sha256_file(Path(abs_path))
                except OSError:
                    logger.warning("Could not hash %s after indexing", abs_path)

        # ------------------------------------------------------------------
        # Phase 4: Persist after lock release (once per debounce batch)
        # ------------------------------------------------------------------

        storage_dir = Path.home() / ".obsidian-rag" / self._vault_name
        try:
            persist_index_atomically(index, metadata, file_hashes, storage_dir)
            logger.info(
                "Incremental update for vault '%s': %d upsert(s), %d delete(s)",
                self._vault_name,
                len(upsert_data),
                len(delete_id_sets),
            )
        except Exception:
            logger.exception("Failed to persist index after incremental update")


class VaultWatcher:
    """Lifecycle manager for file system watching across one or more vaults.

    Starts a single watchdog Observer and schedules one VaultEventHandler
    per vault. Respects config.indexing.watch_enabled.
    """

    def __init__(
        self,
        vault_indexes: dict,
        config: AppConfig,
        index_lock: threading.Lock | None = None,
    ) -> None:
        self._vault_indexes = vault_indexes
        self._config = config
        self._index_lock = index_lock or threading.Lock()
        self._observer: Observer | None = None
        self._handlers: list[VaultEventHandler] = []

    def start(self) -> None:
        """Start watching all vaults. No-op if watch_enabled is False."""
        if not self._config.indexing.watch_enabled:
            logger.info("File watcher disabled via config (watch_enabled=False)")
            return

        observer = Observer()
        for vault_name, vi in self._vault_indexes.items():
            handler = VaultEventHandler(
                vault_name,
                self._vault_indexes,
                self._index_lock,
                self._config,
            )
            observer.schedule(handler, str(vi["vault_config"].path), recursive=True)
            self._handlers.append(handler)

        observer.start()
        self._observer = observer
        logger.info(
            "File watcher started for %d vault(s)",
            len(self._vault_indexes),
        )

    def stop(self) -> None:
        """Stop the observer and cancel any pending debounce timers."""
        if self._observer is None:
            return

        # Cancel all pending timers before stopping the observer
        for handler in self._handlers:
            with handler._timer_lock:
                if handler._timer is not None:
                    handler._timer.cancel()
                    handler._timer = None

        self._observer.stop()
        self._observer.join(timeout=2.0)
        logger.info("File watcher stopped")
