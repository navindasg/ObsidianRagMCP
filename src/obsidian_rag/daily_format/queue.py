"""Persistent JSON queue for the nightly daily-note formatter.

Public API:
    QueueItem: frozen dataclass identifying one note to format.
    FormatQueue.load(path) -> FormatQueue
    FormatQueue.ensure_start_date(today) -> datetime.date
    FormatQueue.enqueue(item) -> bool
    FormatQueue.pending(max_retries) -> list[QueueItem]
    FormatQueue.mark_done(item) -> None
    FormatQueue.mark_failed(item, max_retries) -> QueueItem
    FormatQueue.save() -> None
    default_queue_path() -> Path

On-disk state: ``{"start_date": str|null, "items": [...]}``. The file is
written atomically (temp file in the same directory + ``os.replace``) so a
crash or sleep mid-write never leaves a partial queue. Missing or corrupt
files are tolerated: the queue starts fresh and logs what happened.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueueItem:
    """One daily note awaiting formatting.

    Attributes:
        vault: Name of the vault containing the note.
        rel_path: Path of the note relative to the vault root.
        note_date: ISO date (YYYY-MM-DD) the note covers.
        attempts: Number of failed formatting attempts so far.
    """

    vault: str
    rel_path: str
    note_date: str
    attempts: int = 0

    @property
    def key(self) -> tuple[str, str]:
        """Dedupe identity: same vault + relative path means same note."""
        return (self.vault, self.rel_path)


def default_queue_path() -> Path:
    """Default location of the persistent format queue."""
    return Path.home() / ".obsidian-rag" / "format_queue.json"


def _parse_state(raw: object) -> tuple[datetime.date | None, tuple[QueueItem, ...]]:
    """Parse the on-disk JSON state, raising ValueError on any bad shape."""
    if not isinstance(raw, dict):
        raise ValueError(f"expected a JSON object, got {type(raw).__name__}")

    raw_start = raw.get("start_date")
    start_date = (
        datetime.date.fromisoformat(raw_start) if raw_start is not None else None
    )

    raw_items = raw.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("'items' is not a list")
    items = tuple(
        QueueItem(
            vault=str(entry["vault"]),
            rel_path=str(entry["rel_path"]),
            note_date=str(entry["note_date"]),
            attempts=int(entry.get("attempts", 0)),
        )
        for entry in raw_items
    )
    return start_date, items


class FormatQueue:
    """Persistent, dedup-ing queue of daily notes awaiting formatting.

    Mutating methods update in-memory state only; call :meth:`save` to
    persist. The one exception is :meth:`ensure_start_date`, which persists
    immediately so the no-backfill cutoff survives a first run that crashes
    before any item completes.
    """

    def __init__(
        self,
        path: Path,
        start_date: datetime.date | None = None,
        items: tuple[QueueItem, ...] = (),
    ) -> None:
        self._path = path
        self._start_date = start_date
        self._items = items

    @classmethod
    def load(cls, path: Path) -> FormatQueue:
        """Load the queue from path, starting fresh if missing or corrupt."""
        if not path.exists():
            logger.debug("Queue file %s not found; starting fresh", path)
            return cls(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            start_date, items = _parse_state(raw)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning(
                "Corrupt or unreadable queue file %s (%s); starting fresh",
                path,
                exc,
            )
            return cls(path)
        return cls(path, start_date=start_date, items=items)

    @property
    def start_date(self) -> datetime.date | None:
        """The recorded no-backfill cutoff, if any."""
        return self._start_date

    @property
    def items(self) -> tuple[QueueItem, ...]:
        """All queued items, including parked ones."""
        return self._items

    def ensure_start_date(self, today: datetime.date) -> datetime.date:
        """Return the stored start_date, recording and persisting today if absent.

        This implements the no-backfill cutoff: only notes dated on or after
        the first ever run are eligible for formatting.
        """
        if self._start_date is None:
            self._start_date = today
            self.save()
            logger.info("Recorded daily-format start date %s", today.isoformat())
        return self._start_date

    def enqueue(self, item: QueueItem) -> bool:
        """Add item unless one with the same (vault, rel_path) is queued."""
        if any(existing.key == item.key for existing in self._items):
            logger.debug("Already queued: %s/%s", item.vault, item.rel_path)
            return False
        self._items = (*self._items, item)
        return True

    def pending(self, max_retries: int) -> list[QueueItem]:
        """Items still eligible for a formatting attempt."""
        return [item for item in self._items if item.attempts < max_retries]

    def mark_done(self, item: QueueItem) -> None:
        """Remove item from the queue (no-op if absent)."""
        self._items = tuple(
            existing for existing in self._items if existing.key != item.key
        )

    def mark_failed(self, item: QueueItem, max_retries: int) -> QueueItem:
        """Replace item with a copy whose attempt count is incremented.

        Once attempts reaches max_retries the item is parked: it stays in
        the queue (visible in :attr:`items`) but no longer appears in
        :meth:`pending`.
        """
        updated = dataclasses.replace(item, attempts=item.attempts + 1)
        self._items = tuple(
            updated if existing.key == item.key else existing
            for existing in self._items
        )
        if updated.attempts >= max_retries:
            logger.warning(
                "Parking %s/%s after %d failed attempts (max_retries=%d)",
                updated.vault,
                updated.rel_path,
                updated.attempts,
                max_retries,
            )
        return updated

    def save(self) -> None:
        """Persist the queue atomically: temp file + os.replace.

        Mirrors ``_replace_atomically`` in indexer.py so a crash mid-write
        never leaves a truncated queue file behind.
        """
        state = {
            "start_date": (
                self._start_date.isoformat() if self._start_date is not None else None
            ),
            "items": [dataclasses.asdict(item) for item in self._items],
        }
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=parent, prefix=f"{self._path.name}.", suffix=".tmp"
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            os.replace(tmp_path, self._path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
