"""Tests for the nightly daily-format runner (daily_format/runner.py).

Tests:
  1. happy path: yesterday's note formatted, today's note untouched
  2. start_date honored: notes before start_date are never enqueued
  3. first run records start_date == today, so nothing is backfilled
  4. ollama down: items stay queued, summary carries ollama_down=True
  5. per-item failure increments attempts and the run continues
  6. dry-run reports but formats nothing and never builds a client
  7. re-run after success enqueues nothing (idempotent)
  8. stale queue item for an already-formatted note is marked done
  9. queue item for an unknown vault is left parked with a warning
 10. queue item whose rel_path escapes the vault is dropped
 11. queue item dated today (not yet past) is skipped and marked done
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.daily_format.queue import FormatQueue, QueueItem
from obsidian_rag.daily_format.runner import run_format_daily
from obsidian_rag.models import AppConfig

TODAY = datetime.date(2026, 6, 12)
YESTERDAY = datetime.date(2026, 6, 11)
START = datetime.date(2026, 6, 1)

RAW_NOTE = "- [ ] call [[Alice]]\nidea about the garden\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(vault_dir: Path, **daily_overrides: Any) -> AppConfig:
    daily: dict[str, Any] = {"enabled": True, "start_date": START, **daily_overrides}
    return AppConfig(
        vaults=[{"name": "vault", "path": str(vault_dir)}],
        daily_format=daily,
    )


def _reply(tags: list[str], body: str) -> str:
    return json.dumps({"tags": tags, "formatted_markdown": body})


def _client_with_replies(*contents: str) -> MagicMock:
    """Mock ollama client whose chat() returns each content in sequence."""
    responses = []
    for content in contents:
        response = MagicMock()
        response.message.content = content
        responses.append(response)
    client = MagicMock()
    client.chat.side_effect = responses
    return client


def _run(
    cfg: AppConfig,
    queue_path: Path,
    *,
    client: MagicMock | None = None,
    today: datetime.date = TODAY,
    dry_run: bool = False,
    select_model_error: Exception | None = None,
) -> dict:
    """Invoke run_format_daily with ollama and select_model mocked out.

    With client=None, constructing ollama.Client raises AssertionError so a
    test can prove Ollama is never touched.
    """
    mock_ollama = MagicMock()
    if client is None:
        mock_ollama.Client.side_effect = AssertionError(
            "ollama.Client must not be constructed in this test"
        )
    else:
        mock_ollama.Client.return_value = client
    select_kwargs: dict[str, Any] = (
        {"side_effect": select_model_error}
        if select_model_error is not None
        else {"return_value": "llama3.2"}
    )
    with (
        patch("obsidian_rag.daily_format.runner.ollama", mock_ollama),
        patch("obsidian_rag.daily_format.runner.select_model", **select_kwargs),
    ):
        return run_format_daily(cfg, queue_path=queue_path, today=today, dry_run=dry_run)


def _seed_queue(queue_path: Path, *items: QueueItem) -> None:
    queue = FormatQueue(queue_path, start_date=START)
    for item in items:
        queue.enqueue(item)
    queue.save()


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    return vault_dir


@pytest.fixture
def queue_path(tmp_path: Path) -> Path:
    return tmp_path / "queue.json"


# ---------------------------------------------------------------------------
# Test 1: happy path
# ---------------------------------------------------------------------------


def test_yesterday_formatted_today_untouched(vault: Path, queue_path: Path) -> None:
    (vault / "2026-06-11.md").write_text(RAW_NOTE, encoding="utf-8")
    (vault / "2026-06-12.md").write_text("today raw\n", encoding="utf-8")
    cfg = _make_cfg(vault)
    client = _client_with_replies(_reply(["garden"], "## Tasks\n- [ ] call [[Alice]]"))

    summary = _run(cfg, queue_path, client=client)

    assert summary == {"enqueued": 1, "formatted": 1, "failed": 0, "skipped": 0}
    formatted = (vault / "2026-06-11.md").read_text(encoding="utf-8")
    assert formatted.startswith("---\n")
    assert "## Original Notes" in formatted
    assert RAW_NOTE.strip() in formatted
    assert (vault / "2026-06-12.md").read_text(encoding="utf-8") == "today raw\n"
    assert FormatQueue.load(queue_path).items == ()


# ---------------------------------------------------------------------------
# Test 2: start_date floor
# ---------------------------------------------------------------------------


def test_note_before_start_date_never_enqueued(vault: Path, queue_path: Path) -> None:
    (vault / "2026-06-11.md").write_text(RAW_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault, start_date=TODAY)

    summary = _run(cfg, queue_path, client=None)

    assert summary == {"enqueued": 0, "formatted": 0, "failed": 0, "skipped": 0}
    assert (vault / "2026-06-11.md").read_text(encoding="utf-8") == RAW_NOTE


# ---------------------------------------------------------------------------
# Test 3: first run records start_date (no backfill)
# ---------------------------------------------------------------------------


def test_first_run_records_start_date_and_backfills_nothing(
    vault: Path, queue_path: Path
) -> None:
    (vault / "2026-06-11.md").write_text(RAW_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault, start_date=None)

    summary = _run(cfg, queue_path, client=None)

    assert summary == {"enqueued": 0, "formatted": 0, "failed": 0, "skipped": 0}
    assert FormatQueue.load(queue_path).start_date == TODAY


# ---------------------------------------------------------------------------
# Test 4: Ollama down leaves items queued
# ---------------------------------------------------------------------------


def test_ollama_down_leaves_items_queued(vault: Path, queue_path: Path) -> None:
    (vault / "2026-06-11.md").write_text(RAW_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault)

    summary = _run(
        cfg,
        queue_path,
        client=MagicMock(),
        select_model_error=ConnectionError("Ollama is not reachable."),
    )

    assert summary["ollama_down"] is True
    assert summary["failed"] == 0
    assert summary["queued"] == 1
    assert summary["formatted"] == 0
    reloaded = FormatQueue.load(queue_path)
    assert len(reloaded.items) == 1
    assert reloaded.items[0].attempts == 0
    assert (vault / "2026-06-11.md").read_text(encoding="utf-8") == RAW_NOTE


# ---------------------------------------------------------------------------
# Test 5: per-item failure continues the run
# ---------------------------------------------------------------------------


def test_per_item_failure_increments_attempts_and_continues(
    vault: Path, queue_path: Path
) -> None:
    (vault / "2026-06-10.md").write_text("older note\n", encoding="utf-8")
    (vault / "2026-06-11.md").write_text(RAW_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault)
    # First reply (for 2026-06-10) is invalid JSON -> FormatError; second is valid.
    client = _client_with_replies("not json at all", _reply(["x"], "## Clean"))

    summary = _run(cfg, queue_path, client=client)

    assert summary == {"enqueued": 2, "formatted": 1, "failed": 1, "skipped": 0}
    assert (vault / "2026-06-10.md").read_text(encoding="utf-8") == "older note\n"
    assert "## Original Notes" in (vault / "2026-06-11.md").read_text(encoding="utf-8")
    reloaded = FormatQueue.load(queue_path)
    assert len(reloaded.items) == 1
    assert reloaded.items[0].rel_path == "2026-06-10.md"
    assert reloaded.items[0].attempts == 1


# ---------------------------------------------------------------------------
# Test 6: dry-run reports without touching Ollama or files
# ---------------------------------------------------------------------------


def test_dry_run_enqueues_and_reports_but_formats_nothing(
    vault: Path, queue_path: Path
) -> None:
    (vault / "2026-06-11.md").write_text(RAW_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault)

    summary = _run(cfg, queue_path, client=None, dry_run=True)

    assert summary == {
        "enqueued": 1,
        "pending": ["2026-06-11.md"],
        "formatted": 0,
        "failed": 0,
    }
    assert (vault / "2026-06-11.md").read_text(encoding="utf-8") == RAW_NOTE
    reloaded = FormatQueue.load(queue_path)
    assert len(reloaded.items) == 1


# ---------------------------------------------------------------------------
# Test 7: idempotent re-run
# ---------------------------------------------------------------------------


def test_rerun_after_success_enqueues_nothing(vault: Path, queue_path: Path) -> None:
    (vault / "2026-06-11.md").write_text(RAW_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault)
    client = _client_with_replies(_reply(["x"], "## Clean"))
    first = _run(cfg, queue_path, client=client)
    assert first["formatted"] == 1

    second = _run(cfg, queue_path, client=None)

    assert second == {"enqueued": 0, "formatted": 0, "failed": 0, "skipped": 0}


# ---------------------------------------------------------------------------
# Test 8: stale queue item for an already-formatted note
# ---------------------------------------------------------------------------


def test_stale_item_for_formatted_note_marked_done(
    vault: Path, queue_path: Path
) -> None:
    (vault / "2026-06-11.md").write_text(
        "---\nformatted: 2026-06-12T02:00:00\n---\n\nbody\n\n## Original Notes\n\nraw\n",
        encoding="utf-8",
    )
    _seed_queue(
        queue_path,
        QueueItem(vault="vault", rel_path="2026-06-11.md", note_date="2026-06-11"),
    )
    cfg = _make_cfg(vault)
    client = MagicMock()

    summary = _run(cfg, queue_path, client=client)

    assert summary == {"enqueued": 0, "formatted": 0, "failed": 0, "skipped": 1}
    client.chat.assert_not_called()
    assert FormatQueue.load(queue_path).items == ()


# ---------------------------------------------------------------------------
# Test 9: unknown vault is parked, not formatted, not dropped
# ---------------------------------------------------------------------------


def test_unknown_vault_item_left_parked(vault: Path, queue_path: Path) -> None:
    _seed_queue(
        queue_path,
        QueueItem(vault="ghost", rel_path="2026-06-11.md", note_date="2026-06-11"),
    )
    cfg = _make_cfg(vault)
    client = MagicMock()

    summary = _run(cfg, queue_path, client=client)

    assert summary == {"enqueued": 0, "formatted": 0, "failed": 0, "skipped": 1}
    client.chat.assert_not_called()
    reloaded = FormatQueue.load(queue_path)
    assert len(reloaded.items) == 1
    assert reloaded.items[0].attempts == 0


# ---------------------------------------------------------------------------
# Test 10: rel_path escaping the vault is dropped
# ---------------------------------------------------------------------------


def test_traversal_rel_path_dropped(
    vault: Path, queue_path: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "evil.md"
    outside.write_text("outside the vault\n", encoding="utf-8")
    _seed_queue(
        queue_path,
        QueueItem(vault="vault", rel_path="../evil.md", note_date="2026-06-11"),
    )
    cfg = _make_cfg(vault)
    client = MagicMock()

    summary = _run(cfg, queue_path, client=client)

    assert summary == {"enqueued": 0, "formatted": 0, "failed": 0, "skipped": 1}
    client.chat.assert_not_called()
    assert outside.read_text(encoding="utf-8") == "outside the vault\n"
    assert FormatQueue.load(queue_path).items == ()


# ---------------------------------------------------------------------------
# Test 11: item dated today is not yet eligible -> skipped + done
# ---------------------------------------------------------------------------


def test_item_dated_today_marked_done(vault: Path, queue_path: Path) -> None:
    (vault / "2026-06-12.md").write_text("today raw\n", encoding="utf-8")
    _seed_queue(
        queue_path,
        QueueItem(vault="vault", rel_path="2026-06-12.md", note_date="2026-06-12"),
    )
    cfg = _make_cfg(vault)
    client = MagicMock()

    summary = _run(cfg, queue_path, client=client)

    assert summary == {"enqueued": 0, "formatted": 0, "failed": 0, "skipped": 1}
    client.chat.assert_not_called()
    assert (vault / "2026-06-12.md").read_text(encoding="utf-8") == "today raw\n"
    assert FormatQueue.load(queue_path).items == ()
