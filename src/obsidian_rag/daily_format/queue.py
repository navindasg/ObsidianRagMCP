"""Persistent JSON queue for the nightly daily-note formatter.

Public API:
    QueueItem: frozen dataclass identifying one note to format.
    FormatQueue.load(path) -> FormatQueue
    FormatQueue.enqueue(item) -> bool
    FormatQueue.pending(max_retries) -> list[QueueItem]
    FormatQueue.mark_done(item) -> None
    FormatQueue.mark_failed(item, max_retries) -> QueueItem
    FormatQueue.save() -> None
    default_queue_path() -> Path

On-disk state: ``{"items": [...]}``. The file is written atomically (temp
file in the same directory + ``os.replace``) so a crash or sleep mid-write
never leaves a partial queue. Missing or corrupt files are tolerated: the
queue starts fresh and logs what happened. A legacy ``start_date`` key from
older versions is ignored on load.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueueItem:
    """One note awaiting formatting.

    Attributes:
        vault: Name of the vault containing the note.
        rel_path: Path of the note relative to the vault root.
        note_date: ISO date (YYYY-MM-DD) the note covers; None for notes
            queued via the format tag, which have no date of their own.
        attempts: Number of failed formatting attempts so far.
        kind: "daily" for scheduled daily notes (successor rule applies),
            "tagged" for notes opted in via the format tag (formatted on
            the next run).
    """

    vault: str
    rel_path: str
    note_date: str | None
    attempts: int = 0
    kind: str = "daily"

    @property
    def key(self) -> tuple[str, str]:
        """Dedupe identity: same vault + relative path means same note."""
        return (self.vault, self.rel_path)


def default_queue_path() -> Path:
    """Default location of the persistent format queue."""
    return Path.home() / ".obsidian-rag" / "format_queue.json"


def _parse_state(raw: object) -> tuple[QueueItem, ...]:
    """Parse the on-disk JSON state, raising ValueError on any bad shape.

    A legacy ``start_date`` key from older versions is ignored.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"expected a JSON object, got {type(raw).__name__}")

    raw_items = raw.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("'items' is not a list")
    return tuple(
        QueueItem(
            vault=str(entry["vault"]),
            rel_path=str(entry["rel_path"]),
            note_date=(
                str(entry["note_date"])
                if entry.get("note_date") is not None
                else None
            ),
            attempts=int(entry.get("attempts", 0)),
            # State files written before the format-tag trigger have no kind.
            kind=str(entry.get("kind", "daily")),
        )
        for entry in raw_items
    )


class FormatQueue:
    """Persistent, dedup-ing queue of notes awaiting formatting.

    Mutating methods update in-memory state only; call :meth:`save` to persist.
    """

    def __init__(
        self,
        path: Path,
        items: tuple[QueueItem, ...] = (),
    ) -> None:
        self._path = path
        self._items = items

    @classmethod
    def load(cls, path: Path) -> FormatQueue:
        """Load the queue from path, starting fresh if missing or corrupt."""
        if not path.exists():
            logger.debug("Queue file %s not found; starting fresh", path)
            return cls(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            items = _parse_state(raw)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning(
                "Corrupt or unreadable queue file %s (%s); starting fresh",
                path,
                exc,
            )
            return cls(path)
        return cls(path, items=items)

    @property
    def items(self) -> tuple[QueueItem, ...]:
        """All queued items, including parked ones."""
        return self._items

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
        state = {"items": [dataclasses.asdict(item) for item in self._items]}
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
