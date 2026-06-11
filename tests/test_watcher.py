"""Tests for VaultWatcher and VaultEventHandler.

TDD RED phase: All tests written before implementation.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from obsidian_rag.indexer import create_index
from obsidian_rag.models import AppConfig, EmbeddingConfig, IndexingConfig, VaultConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vault_dir(tmp_path):
    """Create a real temporary vault directory."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture
def vault_config(vault_dir):
    """Minimal VaultConfig pointing at tmp vault dir."""
    return VaultConfig(name="test-vault", path=vault_dir)


@pytest.fixture
def watcher_config(vault_config):
    """Minimal AppConfig with watch_enabled=True."""
    return AppConfig(
        vaults=[vault_config],
        embedding=EmbeddingConfig(model="nomic-embed-text", ollama_url="http://localhost:11434"),
        indexing=IndexingConfig(watch_enabled=True),
    )


@pytest.fixture
def vault_index_entry(vault_config):
    """Returns a vault_indexes-compatible entry with real 4D FAISS index."""
    index = create_index(4)
    return {
        "index": index,
        "metadata": {},
        "file_hashes": {},
        "vault_config": vault_config,
    }


@pytest.fixture
def vault_indexes(vault_index_entry, vault_config):
    """Dict of vault_indexes with single vault entry."""
    return {vault_config.name: vault_index_entry}


@pytest.fixture
def index_lock():
    """Shared threading lock for index access."""
    return threading.Lock()


# ---------------------------------------------------------------------------
# VaultWatcher lifecycle tests
# ---------------------------------------------------------------------------


def test_observer_started(watcher_config, vault_indexes, index_lock):
    """VaultWatcher.start() creates and starts a watchdog Observer."""
    from obsidian_rag.watcher import VaultWatcher

    with patch("obsidian_rag.watcher.Observer") as MockObserver:
        mock_obs = MockObserver.return_value

        watcher = VaultWatcher(vault_indexes, watcher_config, index_lock)
        watcher.start()

        assert MockObserver.called
        assert mock_obs.schedule.called
        assert mock_obs.start.called

        # Verify scheduled with recursive=True for vault path
        schedule_kwargs = mock_obs.schedule.call_args
        assert schedule_kwargs[1].get("recursive") is True or schedule_kwargs[0][2] is True


def test_observer_stopped(watcher_config, vault_indexes, index_lock):
    """VaultWatcher.stop() stops the observer and joins with timeout."""
    from obsidian_rag.watcher import VaultWatcher

    with patch("obsidian_rag.watcher.Observer") as MockObserver:
        mock_obs = MockObserver.return_value
        mock_obs.is_alive.return_value = True

        watcher = VaultWatcher(vault_indexes, watcher_config, index_lock)
        watcher.start()
        watcher.stop()

        mock_obs.stop.assert_called_once()
        mock_obs.join.assert_called_once()


def test_watch_disabled(watcher_config, vault_indexes, index_lock):
    """VaultWatcher.start() is a no-op when config.indexing.watch_enabled is False."""
    from obsidian_rag.watcher import VaultWatcher

    watcher_config.indexing.watch_enabled = False

    with patch("obsidian_rag.watcher.Observer") as MockObserver:
        watcher = VaultWatcher(vault_indexes, watcher_config, index_lock)
        watcher.start()

        # Observer should never be created or started
        assert not MockObserver.called


def test_stop_when_not_started(watcher_config, vault_indexes, index_lock):
    """VaultWatcher.stop() is a no-op when never started."""
    from obsidian_rag.watcher import VaultWatcher

    watcher = VaultWatcher(vault_indexes, watcher_config, index_lock)
    # Should not raise
    watcher.stop()


# ---------------------------------------------------------------------------
# VaultEventHandler filtering tests
# ---------------------------------------------------------------------------


def test_non_md_ignored(watcher_config, vault_indexes, index_lock, vault_dir):
    """on_created for a .txt file does not schedule any processing."""
    from obsidian_rag.watcher import VaultEventHandler

    handler = VaultEventHandler("test-vault", vault_indexes, index_lock, watcher_config)

    with patch.object(handler, "_reset_timer") as mock_reset:
        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = str(vault_dir / "notes.txt")

        handler.on_created(mock_event)
        mock_reset.assert_not_called()


def test_directory_events_ignored(watcher_config, vault_indexes, index_lock, vault_dir):
    """on_created with is_directory=True does not schedule processing."""
    from obsidian_rag.watcher import VaultEventHandler

    handler = VaultEventHandler("test-vault", vault_indexes, index_lock, watcher_config)

    with patch.object(handler, "_reset_timer") as mock_reset:
        mock_event = MagicMock()
        mock_event.is_directory = True
        mock_event.src_path = str(vault_dir / "subfolder")

        handler.on_created(mock_event)
        mock_reset.assert_not_called()


# ---------------------------------------------------------------------------
# Debounce tests
# ---------------------------------------------------------------------------


def test_debounce_coalesces_events(watcher_config, vault_indexes, index_lock, vault_dir):
    """Two on_modified events within 2 seconds result in only one _flush call."""
    from obsidian_rag.watcher import VaultEventHandler

    handler = VaultEventHandler("test-vault", vault_indexes, index_lock, watcher_config)

    flush_call_count = []

    def fake_flush():
        flush_call_count.append(1)

    # Use a short debounce for testing by patching the timer interval
    with patch("obsidian_rag.watcher.DEBOUNCE_SECONDS", 0.3):
        handler._debounce_seconds = 0.3

        # Monkey-patch _flush on the instance
        handler._flush = fake_flush

        # Create a test file
        md_file = vault_dir / "test.md"
        md_file.write_text("# Test\n\nContent here for testing.\n", encoding="utf-8")

        event1 = MagicMock()
        event1.is_directory = False
        event1.src_path = str(md_file)

        event2 = MagicMock()
        event2.is_directory = False
        event2.src_path = str(md_file)

        handler.on_modified(event1)
        time.sleep(0.05)  # small gap between events
        handler.on_modified(event2)

        # Wait for debounce to fire
        time.sleep(0.6)

        assert len(flush_call_count) == 1, f"Expected 1 flush call, got {len(flush_call_count)}"


# ---------------------------------------------------------------------------
# Incremental index update tests
# ---------------------------------------------------------------------------


def test_file_created_indexed(watcher_config, vault_indexes, index_lock, vault_dir, vault_config):
    """on_created for a new .md file results in new chunks added to index after debounce."""
    from obsidian_rag.watcher import VaultEventHandler

    # Create the markdown file
    md_file = vault_dir / "note.md"
    md_file.write_text("# Note\n\nThis is a test note with sufficient content for chunking.\n", encoding="utf-8")

    handler = VaultEventHandler("test-vault", vault_indexes, index_lock, watcher_config)

    mock_embeddings = [[0.1, 0.2, 0.3, 0.4]]
    mock_response = MagicMock()
    mock_response.embeddings = mock_embeddings

    mock_client = MagicMock()
    mock_client.embed.return_value = mock_response

    chunk_result = ({}, [{"text": "test chunk with enough content for indexing", "heading_path": "# Note"}])

    with patch("obsidian_rag.watcher.ollama.Client", return_value=mock_client), \
         patch("obsidian_rag.watcher.chunk_document", return_value=chunk_result), \
         patch("obsidian_rag.watcher.persist_index_atomically") as mock_persist:

        # Directly call _flush after staging the file
        with handler._timer_lock:
            handler._pending_upserts.add(str(md_file))

        handler._flush()

        index = vault_indexes["test-vault"]["index"]
        metadata = vault_indexes["test-vault"]["metadata"]

        assert index.ntotal == 1, f"Expected 1 vector in index, got {index.ntotal}"
        assert len(metadata) == 1
        assert mock_persist.called


def test_file_modified_rechunked(watcher_config, vault_indexes, index_lock, vault_dir, vault_config):
    """on_modified for existing file removes old chunks then adds new chunks."""
    from obsidian_rag.watcher import VaultEventHandler
    import numpy as np

    vi = vault_indexes["test-vault"]
    index = vi["index"]
    metadata = vi["metadata"]
    file_hashes = vi["file_hashes"]

    # Pre-seed index with old chunk for this file
    old_vector = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    import faiss as faiss_lib
    faiss_lib.normalize_L2(old_vector)
    index.add_with_ids(old_vector, np.array([0], dtype=np.int64))
    metadata["0"] = {"file": "note.md", "heading_path": "# Old", "text": "old content"}
    file_hashes["note.md"] = "oldhash"

    # Create updated file
    md_file = vault_dir / "note.md"
    md_file.write_text("# Note Updated\n\nUpdated content here.\n", encoding="utf-8")

    handler = VaultEventHandler("test-vault", vault_indexes, index_lock, watcher_config)

    mock_embeddings = [[0.1, 0.2, 0.3, 0.4]]
    mock_response = MagicMock()
    mock_response.embeddings = mock_embeddings
    mock_client = MagicMock()
    mock_client.embed.return_value = mock_response

    chunk_result = ({}, [{"text": "updated chunk content for testing", "heading_path": "# Note Updated"}])

    with patch("obsidian_rag.watcher.ollama.Client", return_value=mock_client), \
         patch("obsidian_rag.watcher.chunk_document", return_value=chunk_result), \
         patch("obsidian_rag.watcher.persist_index_atomically"):

        with handler._timer_lock:
            handler._pending_upserts.add(str(md_file))

        handler._flush()

    # Old chunk content should be replaced with new chunk content
    metadata = vault_indexes["test-vault"]["metadata"]
    assert len(metadata) == 1, f"Expected 1 new chunk, got {len(metadata)}"
    # New chunk should reference updated file with new heading
    new_meta = list(metadata.values())[0]
    assert new_meta["file"] == "note.md"
    assert new_meta["heading_path"] == "# Note Updated", "Should have new heading, not old"


def test_file_deleted_removed(watcher_config, vault_indexes, index_lock, vault_dir):
    """on_deleted removes all chunks for that file from index and metadata."""
    from obsidian_rag.watcher import VaultEventHandler
    import numpy as np

    vi = vault_indexes["test-vault"]
    index = vi["index"]
    metadata = vi["metadata"]
    file_hashes = vi["file_hashes"]

    # Pre-seed with a chunk for a file we'll "delete"
    old_vector = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    import faiss as faiss_lib
    faiss_lib.normalize_L2(old_vector)
    index.add_with_ids(old_vector, np.array([0], dtype=np.int64))
    metadata["0"] = {"file": "deleted.md", "heading_path": "# Delete Me", "text": "content"}
    file_hashes["deleted.md"] = "somehash"

    handler = VaultEventHandler("test-vault", vault_indexes, index_lock, watcher_config)

    with patch("obsidian_rag.watcher.persist_index_atomically") as mock_persist:
        with handler._timer_lock:
            handler._pending_deletes.add(str(vault_dir / "deleted.md"))

        handler._flush()

    metadata = vault_indexes["test-vault"]["metadata"]
    file_hashes = vault_indexes["test-vault"]["file_hashes"]
    index = vault_indexes["test-vault"]["index"]

    assert len(metadata) == 0, "All metadata for deleted file should be removed"
    assert "deleted.md" not in file_hashes, "Hash entry for deleted file should be removed"
    assert index.ntotal == 0, "All vectors for deleted file should be removed from index"
    assert mock_persist.called


def test_file_renamed(watcher_config, vault_indexes, index_lock, vault_dir):
    """on_moved results in old path chunks removed and new path chunks added."""
    from obsidian_rag.watcher import VaultEventHandler
    import numpy as np

    vi = vault_indexes["test-vault"]
    index = vi["index"]
    metadata = vi["metadata"]
    file_hashes = vi["file_hashes"]

    # Pre-seed old file chunk
    old_vector = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    import faiss as faiss_lib
    faiss_lib.normalize_L2(old_vector)
    index.add_with_ids(old_vector, np.array([0], dtype=np.int64))
    metadata["0"] = {"file": "old_name.md", "heading_path": "# Old Name", "text": "content"}
    file_hashes["old_name.md"] = "hash_old"

    # Create new file at destination
    new_md_file = vault_dir / "new_name.md"
    new_md_file.write_text("# New Name\n\nRenamed content here.\n", encoding="utf-8")

    handler = VaultEventHandler("test-vault", vault_indexes, index_lock, watcher_config)

    mock_embeddings = [[0.1, 0.2, 0.3, 0.4]]
    mock_response = MagicMock()
    mock_response.embeddings = mock_embeddings
    mock_client = MagicMock()
    mock_client.embed.return_value = mock_response

    chunk_result = ({}, [{"text": "renamed chunk content for testing", "heading_path": "# New Name"}])

    with patch("obsidian_rag.watcher.ollama.Client", return_value=mock_client), \
         patch("obsidian_rag.watcher.chunk_document", return_value=chunk_result), \
         patch("obsidian_rag.watcher.persist_index_atomically"):

        mock_move_event = MagicMock()
        mock_move_event.is_directory = False
        mock_move_event.src_path = str(vault_dir / "old_name.md")
        mock_move_event.dest_path = str(new_md_file)

        handler.on_moved(mock_move_event)

        # Wait for debounce then manually flush
        with handler._timer_lock:
            if handler._timer is not None:
                handler._timer.cancel()

        handler._flush()

    metadata = vault_indexes["test-vault"]["metadata"]
    file_hashes = vault_indexes["test-vault"]["file_hashes"]

    # Old chunk removed
    assert not any(m.get("file") == "old_name.md" for m in metadata.values()), \
        "Old path chunks should be removed"
    # New chunk added
    assert any(m.get("file") == "new_name.md" for m in metadata.values()), \
        "New path chunks should be added"
    assert "old_name.md" not in file_hashes


# ---------------------------------------------------------------------------
# Exclusion filtering (regression: watcher indexed .trash/templates edits)
# ---------------------------------------------------------------------------


def test_excluded_dir_events_ignored(watcher_config, vault_indexes, index_lock, vault_dir):
    """Events under excluded_dirs (.trash, .obsidian, templates) are not tracked."""
    from obsidian_rag.watcher import VaultEventHandler

    handler = VaultEventHandler("test-vault", vault_indexes, index_lock, watcher_config)

    with patch.object(handler, "_reset_timer") as mock_reset:
        for excluded in (".trash", ".obsidian", "templates"):
            event = MagicMock()
            event.is_directory = False
            event.src_path = str(vault_dir / excluded / "note.md")
            handler.on_created(event)
            handler.on_modified(event)
            handler.on_deleted(event)

        mock_reset.assert_not_called()

    assert handler._pending_upserts == set()
    assert handler._pending_deletes == set()


def test_excluded_pattern_events_ignored(watcher_config, vault_indexes, index_lock, vault_dir):
    """Events matching excluded_patterns globs are not tracked."""
    from obsidian_rag.watcher import VaultEventHandler

    vault_indexes["test-vault"]["vault_config"] = vault_indexes["test-vault"][
        "vault_config"
    ].model_copy(update={"excluded_patterns": ["daily-*.md"]})
    handler = VaultEventHandler("test-vault", vault_indexes, index_lock, watcher_config)

    event = MagicMock()
    event.is_directory = False
    event.src_path = str(vault_dir / "daily-2024-01-15.md")

    with patch.object(handler, "_reset_timer") as mock_reset:
        handler.on_modified(event)
        mock_reset.assert_not_called()


# ---------------------------------------------------------------------------
# Event routing through the real watchdog entry points
# ---------------------------------------------------------------------------


def test_event_routing_to_pending_sets(watcher_config, vault_indexes, index_lock, vault_dir):
    """on_created/on_modified queue upserts; on_deleted queues deletes."""
    from obsidian_rag.watcher import VaultEventHandler

    handler = VaultEventHandler("test-vault", vault_indexes, index_lock, watcher_config)
    md_path = str(vault_dir / "note.md")

    with patch.object(handler, "_reset_timer") as mock_reset:
        created = MagicMock(is_directory=False, src_path=md_path)
        handler.on_created(created)
        assert md_path in handler._pending_upserts

        handler.on_deleted(created)
        assert md_path in handler._pending_deletes

        assert mock_reset.call_count == 2


# ---------------------------------------------------------------------------
# Unchanged-content skip (regression: every save re-embedded via Ollama)
# ---------------------------------------------------------------------------


def test_unchanged_content_not_reembedded(watcher_config, vault_indexes, index_lock, vault_dir):
    """A file whose stored hash matches its content is not re-chunked or re-embedded."""
    from obsidian_rag.indexer import sha256_file
    from obsidian_rag.watcher import VaultEventHandler

    md_file = vault_dir / "note.md"
    md_file.write_text("# Note\n\nStable content.\n", encoding="utf-8")
    vault_indexes["test-vault"]["file_hashes"]["note.md"] = sha256_file(md_file)

    handler = VaultEventHandler("test-vault", vault_indexes, index_lock, watcher_config)

    with patch("obsidian_rag.watcher.ollama.Client") as mock_client_cls, \
         patch("obsidian_rag.watcher.chunk_document") as mock_chunk, \
         patch("obsidian_rag.watcher.persist_index_atomically"):

        with handler._timer_lock:
            handler._pending_upserts.add(str(md_file))
        handler._flush()

        mock_chunk.assert_not_called()
        mock_client_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Embed-failure recovery (regression: drained events were silently lost)
# ---------------------------------------------------------------------------


def test_flush_requeues_upserts_on_embed_failure(
    watcher_config, vault_indexes, index_lock, vault_dir
):
    """Embedding failure re-queues upserts for retry and still applies deletions."""
    import faiss as faiss_lib
    import numpy as np
    from obsidian_rag.watcher import VaultEventHandler

    vi = vault_indexes["test-vault"]
    old_vector = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    faiss_lib.normalize_L2(old_vector)
    vi["index"].add_with_ids(old_vector, np.array([0], dtype=np.int64))
    vi["metadata"]["0"] = {"file": "gone.md", "heading_path": "# Gone", "text": "x"}
    vi["file_hashes"]["gone.md"] = "hash"

    md_file = vault_dir / "note.md"
    md_file.write_text("# Note\n\nNew content to embed.\n", encoding="utf-8")

    handler = VaultEventHandler("test-vault", vault_indexes, index_lock, watcher_config)

    chunk_result = ({}, [{"text": "chunk", "heading_path": "# Note"}])
    mock_client = MagicMock()
    mock_client.embed.side_effect = ConnectionError("ollama down")

    with patch("obsidian_rag.watcher.ollama.Client", return_value=mock_client), \
         patch("obsidian_rag.watcher.chunk_document", return_value=chunk_result), \
         patch("obsidian_rag.watcher.persist_index_atomically") as mock_persist:

        with handler._timer_lock:
            handler._pending_upserts.add(str(md_file))
            handler._pending_deletes.add(str(vault_dir / "gone.md"))
        handler._flush()

        # Upsert re-queued for retry; nothing for it was indexed
        assert str(md_file) in handler._pending_upserts
        assert not any(
            m.get("file") == "note.md" for m in vi["metadata"].values()
        )
        # Deletion still applied
        assert vi["index"].ntotal == 0
        assert "gone.md" not in vi["file_hashes"]
        assert mock_persist.called

        # Cancel the retry timer so it cannot fire after the test exits
        with handler._timer_lock:
            if handler._timer is not None:
                handler._timer.cancel()
                handler._timer = None


def test_flush_skips_missing_file(watcher_config, vault_indexes, index_lock, vault_dir):
    """An upsert for a file deleted between event and flush is skipped quietly."""
    from obsidian_rag.watcher import VaultEventHandler

    handler = VaultEventHandler("test-vault", vault_indexes, index_lock, watcher_config)

    with patch("obsidian_rag.watcher.ollama.Client") as mock_client_cls, \
         patch("obsidian_rag.watcher.persist_index_atomically"):
        with handler._timer_lock:
            handler._pending_upserts.add(str(vault_dir / "vanished.md"))
        handler._flush()

        mock_client_cls.assert_not_called()

    assert vault_indexes["test-vault"]["metadata"] == {}


# ---------------------------------------------------------------------------
# Shutdown flush (regression: stop() silently discarded buffered changes)
# ---------------------------------------------------------------------------


def test_stop_flushes_pending_changes(watcher_config, vault_indexes, index_lock):
    """VaultWatcher.stop() flushes buffered events instead of dropping them."""
    from obsidian_rag.watcher import VaultWatcher

    with patch("obsidian_rag.watcher.Observer"):
        watcher = VaultWatcher(vault_indexes, watcher_config, index_lock)
        watcher.start()

        handler = watcher._handlers[0]
        flushed = []
        handler._flush = lambda: flushed.append(True)

        watcher.stop()

    assert flushed == [True]
    assert watcher._handlers == []
