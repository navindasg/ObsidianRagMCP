"""Tests for the persistent daily-format queue.

Covers round-trip persistence, corrupt-file recovery, dedupe on
(vault, rel_path), the no-backfill start_date cutoff, retry counting and
parking, atomic writes (temp-then-replace), and the default queue path.
"""
import dataclasses
import datetime
import json
import logging

import pytest

from obsidian_rag.daily_format.queue import (
    FormatQueue,
    QueueItem,
    default_queue_path,
)

LOGGER_NAME = "obsidian_rag.daily_format.queue"


def _item(
    vault: str = "main",
    rel_path: str = "2026-06-11.md",
    note_date: str = "2026-06-11",
    attempts: int = 0,
) -> QueueItem:
    """Helper: build a QueueItem with sensible defaults."""
    return QueueItem(
        vault=vault, rel_path=rel_path, note_date=note_date, attempts=attempts
    )


# ---------------------------------------------------------------------------
# QueueItem
# ---------------------------------------------------------------------------


def test_queue_item_is_frozen():
    """QueueItem is immutable: attribute assignment raises."""
    item = _item()
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.attempts = 5  # type: ignore[misc]


def test_queue_item_default_attempts():
    """attempts defaults to 0."""
    item = QueueItem(vault="main", rel_path="a.md", note_date="2026-06-11")
    assert item.attempts == 0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_missing_file_starts_fresh(tmp_path):
    """A missing queue file yields an empty queue with no start_date."""
    queue = FormatQueue.load(tmp_path / "format_queue.json")

    assert queue.items == ()
    assert queue.start_date is None


def test_load_corrupt_json_starts_fresh(tmp_path, caplog):
    """Corrupt JSON is tolerated: warn and start fresh."""
    path = tmp_path / "format_queue.json"
    path.write_text("{not valid json!!", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        queue = FormatQueue.load(path)

    assert queue.items == ()
    assert queue.start_date is None
    assert any("format_queue" in r.message or "queue" in r.message.lower()
               for r in caplog.records)


def test_load_wrong_structure_starts_fresh(tmp_path, caplog):
    """Valid JSON with the wrong shape is tolerated: warn and start fresh."""
    path = tmp_path / "format_queue.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        queue = FormatQueue.load(path)

    assert queue.items == ()
    assert queue.start_date is None
    assert len(caplog.records) >= 1


# ---------------------------------------------------------------------------
# Round-trip persistence
# ---------------------------------------------------------------------------


def test_round_trip_persistence(tmp_path):
    """Items and start_date survive save() + load()."""
    path = tmp_path / "format_queue.json"
    queue = FormatQueue.load(path)
    queue.ensure_start_date(datetime.date(2026, 6, 10))
    assert queue.enqueue(_item(rel_path="2026-06-10.md", note_date="2026-06-10"))
    assert queue.enqueue(
        _item(rel_path="2026-06-11.md", note_date="2026-06-11", attempts=2)
    )
    queue.save()

    reloaded = FormatQueue.load(path)

    assert reloaded.start_date == datetime.date(2026, 6, 10)
    assert reloaded.items == (
        QueueItem(vault="main", rel_path="2026-06-10.md", note_date="2026-06-10"),
        QueueItem(
            vault="main", rel_path="2026-06-11.md", note_date="2026-06-11", attempts=2
        ),
    )


def test_disk_format_shape(tmp_path):
    """On-disk state is {"start_date": str|null, "items": [...]}."""
    path = tmp_path / "format_queue.json"
    queue = FormatQueue.load(path)
    queue.enqueue(_item())
    queue.save()

    raw = json.loads(path.read_text(encoding="utf-8"))

    assert set(raw) == {"start_date", "items"}
    assert raw["start_date"] is None
    assert raw["items"] == [
        {
            "vault": "main",
            "rel_path": "2026-06-11.md",
            "note_date": "2026-06-11",
            "attempts": 0,
            "kind": "daily",
        }
    ]


def test_save_creates_parent_dirs_and_leaves_no_temp_files(tmp_path):
    """save() creates missing parents and replaces atomically (no temp debris)."""
    path = tmp_path / "deep" / "nested" / "format_queue.json"
    queue = FormatQueue.load(path)
    queue.enqueue(_item())
    queue.save()

    assert path.exists()
    leftovers = [p for p in path.parent.iterdir() if p != path]
    assert leftovers == []
    # Final file is complete, valid JSON (temp-then-replace, never partial).
    assert json.loads(path.read_text(encoding="utf-8"))["items"]


# ---------------------------------------------------------------------------
# start_date (no-backfill cutoff)
# ---------------------------------------------------------------------------


def test_ensure_start_date_records_today_and_persists(tmp_path):
    """First call records today, persists it, and returns it."""
    path = tmp_path / "format_queue.json"
    queue = FormatQueue.load(path)
    today = datetime.date(2026, 6, 12)

    assert queue.ensure_start_date(today) == today
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["start_date"] == "2026-06-12"


def test_ensure_start_date_stable_across_calls_and_loads(tmp_path):
    """start_date is recorded once and never moves on later calls or reloads."""
    path = tmp_path / "format_queue.json"
    queue = FormatQueue.load(path)
    first = queue.ensure_start_date(datetime.date(2026, 6, 12))

    assert queue.ensure_start_date(datetime.date(2026, 7, 1)) == first

    reloaded = FormatQueue.load(path)
    assert reloaded.ensure_start_date(datetime.date(2026, 8, 1)) == first


# ---------------------------------------------------------------------------
# Enqueue / dedupe
# ---------------------------------------------------------------------------


def test_enqueue_dedupes_on_vault_and_rel_path(tmp_path):
    """A second item with the same (vault, rel_path) is rejected."""
    queue = FormatQueue.load(tmp_path / "q.json")

    assert queue.enqueue(_item()) is True
    assert queue.enqueue(_item(note_date="2099-01-01", attempts=2)) is False
    assert len(queue.items) == 1


def test_enqueue_allows_same_rel_path_in_different_vault(tmp_path):
    """Dedupe key is (vault, rel_path), not rel_path alone."""
    queue = FormatQueue.load(tmp_path / "q.json")

    assert queue.enqueue(_item(vault="work")) is True
    assert queue.enqueue(_item(vault="personal")) is True
    assert len(queue.items) == 2


# ---------------------------------------------------------------------------
# pending / mark_done / mark_failed
# ---------------------------------------------------------------------------


def test_pending_filters_by_attempts(tmp_path):
    """pending() returns only items with attempts < max_retries."""
    queue = FormatQueue.load(tmp_path / "q.json")
    fresh = _item(rel_path="a.md")
    tried = _item(rel_path="b.md", attempts=2)
    parked = _item(rel_path="c.md", attempts=3)
    for item in (fresh, tried, parked):
        queue.enqueue(item)

    assert queue.pending(max_retries=3) == [fresh, tried]


def test_mark_done_removes_item(tmp_path):
    """mark_done() removes the item; persisted state reflects removal."""
    path = tmp_path / "q.json"
    queue = FormatQueue.load(path)
    keep = _item(rel_path="keep.md")
    done = _item(rel_path="done.md")
    queue.enqueue(keep)
    queue.enqueue(done)

    queue.mark_done(done)
    queue.save()

    assert queue.items == (keep,)
    assert FormatQueue.load(path).items == (keep,)


def test_mark_done_unknown_item_is_noop(tmp_path):
    """mark_done() on an item not in the queue does nothing."""
    queue = FormatQueue.load(tmp_path / "q.json")
    queue.enqueue(_item(rel_path="a.md"))

    queue.mark_done(_item(rel_path="ghost.md"))

    assert len(queue.items) == 1


def test_mark_failed_increments_attempts_with_new_item(tmp_path):
    """mark_failed() swaps in a new QueueItem with attempts+1."""
    queue = FormatQueue.load(tmp_path / "q.json")
    item = _item()
    queue.enqueue(item)

    queue.mark_failed(item, max_retries=3)

    (updated,) = queue.items
    assert updated == dataclasses.replace(item, attempts=1)
    assert updated is not item


def test_mark_failed_parks_item_at_max_retries(tmp_path, caplog):
    """Reaching max_retries logs a parking warning and drops it from pending()."""
    queue = FormatQueue.load(tmp_path / "q.json")
    item = _item(attempts=2)
    queue.enqueue(item)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        queue.mark_failed(item, max_retries=3)

    (updated,) = queue.items
    assert updated.attempts == 3
    assert queue.pending(max_retries=3) == []
    assert any("park" in r.message.lower() for r in caplog.records)


def test_mark_failed_below_max_retries_does_not_warn(tmp_path, caplog):
    """A retry that is still below max_retries logs no parking warning."""
    queue = FormatQueue.load(tmp_path / "q.json")
    item = _item()
    queue.enqueue(item)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        queue.mark_failed(item, max_retries=3)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == []
    assert queue.pending(max_retries=3) == list(queue.items)


# ---------------------------------------------------------------------------
# Default path
# ---------------------------------------------------------------------------


def test_default_queue_path(tmp_path, monkeypatch):
    """default_queue_path() lives under ~/.obsidian-rag/format_queue.json."""
    monkeypatch.setenv("HOME", str(tmp_path))

    assert default_queue_path() == tmp_path / ".obsidian-rag" / "format_queue.json"


# ---------------------------------------------------------------------------
# Tagged items: kind field, dateless notes, legacy state files
# ---------------------------------------------------------------------------


def test_tagged_item_round_trips(tmp_path):
    """A tagged item (kind='tagged', no date) survives save/load intact."""
    path = tmp_path / "queue.json"
    queue = FormatQueue.load(path)
    item = QueueItem(vault="main", rel_path="ideas/draft.md", note_date=None, kind="tagged")
    queue.enqueue(item)
    queue.save()

    reloaded = FormatQueue.load(path)

    assert reloaded.items == (item,)
    assert reloaded.items[0].kind == "tagged"
    assert reloaded.items[0].note_date is None


def test_item_kind_defaults_to_daily():
    """Items constructed without a kind are daily notes."""
    item = QueueItem(vault="main", rel_path="2026-06-11.md", note_date="2026-06-11")
    assert item.kind == "daily"


def test_legacy_state_without_kind_loads_as_daily(tmp_path):
    """Queue files written before the kind field default to daily items."""
    path = tmp_path / "queue.json"
    state = {
        "start_date": "2026-06-01",
        "items": [
            {"vault": "main", "rel_path": "2026-06-11.md", "note_date": "2026-06-11", "attempts": 1}
        ],
    }
    path.write_text(json.dumps(state), encoding="utf-8")

    queue = FormatQueue.load(path)

    assert queue.items[0].kind == "daily"
    assert queue.items[0].attempts == 1
