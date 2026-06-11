"""File system watcher for incremental Obsidian vault index updates.

Public API:
    VaultEventHandler: FileSystemEventHandler subclass with debounced batch flush.
    VaultWatcher: Lifecycle manager that schedules handlers and manages Observer.

Threading model:
    - watchdog emitter threads queue events under _timer_lock (an RLock, so
      queueing and timer re-arming happen atomically).
    - a threading.Timer fires _flush after DEBOUNCE_SECONDS of quiet (capped
      at MAX_DEBOUNCE_SECONDS from the first buffered event so a continuous
      event stream cannot postpone flushing forever).
    - _flush serializes against itself via _flush_lock, does slow work
      (hashing, chunking, embedding) outside locks, then mutates and persists
      the index under the shared index_lock. Index/metadata references are
      re-fetched under the lock because a background reindex may have swapped
      them since the flush started.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import ollama
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from obsidian_rag.indexer import (
    embed_batch,
    add_vectors,
    is_excluded,
    persist_index_atomically,
    sha256_file,
    vault_storage_dir,
)
from obsidian_rag.markdown_parser import chunk_document
from obsidian_rag.models import AppConfig, ChunkMetadata

logger = logging.getLogger(__name__)

# Debounce window: rapid saves within this many seconds are coalesced.
DEBOUNCE_SECONDS: float = 2.0
# A continuous event stream resets the debounce timer; never postpone a flush
# by more than this many seconds past the first buffered event.
MAX_DEBOUNCE_SECONDS: float = 30.0


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

        self._vault_config = vault_indexes[vault_name]["vault_config"]
        self._vault_path: Path = self._vault_config.path
        # watchdog may report symlink-resolved paths (e.g. macOS FSEvents);
        # keep the resolved root as a fallback for relative_to.
        self._vault_path_resolved: Path = self._vault_path.resolve()

        self._pending_upserts: set[str] = set()
        self._pending_deletes: set[str] = set()
        self._timer: threading.Timer | None = None
        # RLock: event handlers queue + re-arm the timer in one atomic section.
        self._timer_lock = threading.RLock()
        self._first_event_at: float | None = None
        # Serializes flushes: a timer can fire while a previous flush is still
        # chunking/embedding.
        self._flush_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_created(self, event) -> None:
        if self._should_track(event.is_directory, event.src_path):
            self._queue(upsert=event.src_path)

    def on_modified(self, event) -> None:
        if self._should_track(event.is_directory, event.src_path):
            self._queue(upsert=event.src_path)

    def on_deleted(self, event) -> None:
        if self._should_track(event.is_directory, event.src_path):
            self._queue(delete=event.src_path)

    def on_moved(self, event) -> None:
        if event.is_directory:
            return
        if self._should_track(False, event.src_path):
            self._queue(delete=event.src_path)
        if self._should_track(False, event.dest_path):
            self._queue(upsert=event.dest_path)

    # ------------------------------------------------------------------
    # Event filtering and queueing
    # ------------------------------------------------------------------

    def _rel_path(self, abs_path: str) -> str | None:
        """Vault-relative path for an event path, or None if outside the vault."""
        path = Path(abs_path)
        for root in (self._vault_path, self._vault_path_resolved):
            try:
                return str(path.relative_to(root))
            except ValueError:
                continue
        try:
            return str(path.resolve().relative_to(self._vault_path_resolved))
        except (ValueError, OSError):
            logger.warning(
                "Event path %s is outside vault %s — ignoring", abs_path, self._vault_path
            )
            return None

    def _should_track(self, is_directory: bool, path: str) -> bool:
        """True when the event path is an includable .md file in this vault."""
        if is_directory or not path.endswith(".md"):
            return False
        rel = self._rel_path(path)
        if rel is None:
            return False
        # Same exclusion rules as scan_vault, so .trash/templates edits
        # cannot leak into the index through the watcher.
        return not is_excluded(
            Path(rel),
            self._vault_config.excluded_dirs,
            self._vault_config.excluded_patterns,
        )

    def _queue(self, upsert: str | None = None, delete: str | None = None) -> None:
        """Atomically buffer an event and re-arm the debounce timer."""
        with self._timer_lock:
            if upsert is not None:
                self._pending_upserts.add(upsert)
            if delete is not None:
                self._pending_deletes.add(delete)
            self._reset_timer()

    # ------------------------------------------------------------------
    # Debounce timer
    # ------------------------------------------------------------------

    def _reset_timer(self) -> None:
        """Cancel any existing timer and start a new one (with a max-wait cap)."""
        with self._timer_lock:
            now = time.monotonic()
            if self._first_event_at is None:
                self._first_event_at = now
            elif (
                self._timer is not None
                and now - self._first_event_at >= MAX_DEBOUNCE_SECONDS
            ):
                # Continuous events have postponed the flush long enough;
                # let the already-armed timer fire.
                return
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
        with self._flush_lock:
            self._flush_serialized()

    def _flush_serialized(self) -> None:
        with self._timer_lock:
            to_upsert = set(self._pending_upserts)
            to_delete = set(self._pending_deletes)
            self._pending_upserts.clear()
            self._pending_deletes.clear()
            self._timer = None
            self._first_event_at = None

        if not to_upsert and not to_delete:
            return

        # ------------------------------------------------------------------
        # Phase 1: Resolve deletions and prepare upserts (chunk + hash)
        # outside any lock — this is slow disk work.
        # ------------------------------------------------------------------

        delete_rel_paths: set[str] = set()
        for abs_path in to_delete:
            rel_path = self._rel_path(abs_path)
            if rel_path is not None:
                delete_rel_paths.add(rel_path)

        # Benign unlocked read: a stale hash only causes one redundant re-embed.
        stored_hashes: dict = self._vault_indexes[self._vault_name]["file_hashes"]

        upsert_data: list[dict] = []
        for abs_path in to_upsert:
            path = Path(abs_path)
            rel_path = self._rel_path(abs_path)
            if rel_path is None:
                continue

            try:
                if not path.exists():
                    # File was removed between event and flush
                    logger.debug("Skipping upsert for missing file: %s", abs_path)
                    continue
                content_hash = sha256_file(path)
                mtime = os.path.getmtime(abs_path)
            except OSError:
                logger.warning("Could not read %s — skipping", abs_path)
                continue

            if stored_hashes.get(rel_path) == content_hash:
                # mtime-only touches and sync-tool rewrites change nothing;
                # don't re-chunk and re-embed identical content.
                logger.debug("Content unchanged for %s — skipping", rel_path)
                continue

            try:
                file_metadata, chunks = chunk_document(
                    path,
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
                    "rel_path": rel_path,
                    "file_metadata": file_metadata,
                    "chunks": chunks,
                    "content_hash": content_hash,
                    "mtime": mtime,
                }
            )

        # ------------------------------------------------------------------
        # Phase 2: Embed all new chunks at once (outside the index lock)
        # ------------------------------------------------------------------

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
                # Re-queue the upserts and retry after the debounce window —
                # dropping them would silently desynchronize the index until
                # the file is touched again. Deletions need no embedding and
                # proceed below.
                logger.exception(
                    "Embedding failed during incremental update — will retry %d file(s)",
                    len(upsert_data),
                )
                with self._timer_lock:
                    self._pending_upserts |= to_upsert
                    self._reset_timer()
                upsert_data = []
                if not delete_rel_paths:
                    return

        # ------------------------------------------------------------------
        # Phase 3: Mutate and persist under the shared index lock.
        # References are re-fetched here: a background reindex may have
        # swapped vi["index"]/["metadata"]/["file_hashes"] for new objects
        # since this flush started.
        # ------------------------------------------------------------------

        with self._index_lock:
            vi = self._vault_indexes[self._vault_name]
            index = vi["index"]
            metadata: dict = vi["metadata"]
            file_hashes: dict = vi["file_hashes"]

            # One pass over metadata: file -> chunk ids
            ids_by_file: dict[str, list[int]] = {}
            for chunk_id, chunk_meta in metadata.items():
                ids_by_file.setdefault(chunk_meta.get("file", ""), []).append(int(chunk_id))

            all_ids_to_remove: list[int] = []
            for rel_path in delete_rel_paths:
                all_ids_to_remove.extend(ids_by_file.get(rel_path, []))
            for entry in upsert_data:
                all_ids_to_remove.extend(ids_by_file.get(entry["rel_path"], []))

            if all_ids_to_remove:
                index.remove_ids(np.array(all_ids_to_remove, dtype=np.int64))
                for chunk_id in all_ids_to_remove:
                    metadata.pop(str(chunk_id), None)

            for rel_path in delete_rel_paths:
                file_hashes.pop(rel_path, None)

            next_id = (max(int(k) for k in metadata.keys()) + 1) if metadata else 0

            embedding_offset = 0
            for entry in upsert_data:
                rel_path = entry["rel_path"]
                chunks = entry["chunks"]

                tags = entry["file_metadata"].get("tags", [])
                if isinstance(tags, str):
                    tags = [tags]

                folder = rel_path.split("/")[0] if "/" in rel_path else ""

                chunk_count = len(chunks)
                chunk_embeddings = embeddings[
                    embedding_offset : embedding_offset + chunk_count
                ]
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
                        modified_ts=entry["mtime"],
                        char_count=len(chunk.get("text", "")),
                    ).model_dump()

                file_hashes[rel_path] = entry["content_hash"]

            # Persist while still holding the lock: the index must be
            # quiescent while faiss serializes it, and a concurrent reindex
            # swap must not be clobbered with stale objects.
            try:
                persist_index_atomically(
                    index, metadata, file_hashes, vault_storage_dir(self._vault_name)
                )
                logger.info(
                    "Incremental update for vault '%s': %d upsert(s), %d delete(s)",
                    self._vault_name,
                    len(upsert_data),
                    len(delete_rel_paths),
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

        self._handlers = []
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
        """Stop the observer, then flush any buffered changes synchronously."""
        if self._observer is None:
            return

        self._observer.stop()
        self._observer.join(timeout=2.0)
        self._observer = None

        # Cancel pending debounce timers, then flush what they had buffered
        # so edits made just before shutdown are applied and persisted.
        for handler in self._handlers:
            with handler._timer_lock:
                if handler._timer is not None:
                    handler._timer.cancel()
                    handler._timer = None
            try:
                handler._flush()
            except Exception:
                logger.exception("Final flush failed during watcher shutdown")

        self._handlers = []
        logger.info("File watcher stopped")
