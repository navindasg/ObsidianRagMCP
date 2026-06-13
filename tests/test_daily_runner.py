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
    tags_only: bool = False,
    since: datetime.date | None = None,
    select_model_error: Exception | None = None,
    power_state: Any = None,
) -> dict:
    """Invoke run_format_daily with ollama and select_model mocked out.

    With client=None, constructing ollama.Client raises AssertionError so a
    test can prove Ollama is never touched. read_power_state is always
    patched (default: charged on AC) so tests never shell out to pmset.
    """
    from obsidian_rag.daily_format.power import PowerState

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
    state = (
        power_state
        if power_state is not None
        else PowerState(has_battery=False, percent=None, on_ac_power=True)
    )
    with (
        patch("obsidian_rag.daily_format.runner.ollama", mock_ollama),
        patch("obsidian_rag.daily_format.runner.select_model", **select_kwargs),
        patch(
            "obsidian_rag.daily_format.runner.read_power_state", return_value=state
        ),
    ):
        return run_format_daily(
            cfg,
            queue_path=queue_path,
            today=today,
            dry_run=dry_run,
            tags_only=tags_only,
            since=since,
        )


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


# ---------------------------------------------------------------------------
# Test 12: blacklisted daily note is never enqueued
# ---------------------------------------------------------------------------


def test_blacklisted_note_never_enqueued(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "2026-06-11.md").write_text(RAW_NOTE, encoding="utf-8")
    (vault / "2026-06-10.md").write_text(RAW_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault, blacklist=["2026-06-10"])

    summary = _run(cfg, tmp_path / "queue.json", dry_run=True)

    assert summary["enqueued"] == 1
    assert summary["pending"] == ["2026-06-11.md"]


# ---------------------------------------------------------------------------
# Tests 13-17: #!format tag trigger
# ---------------------------------------------------------------------------

TAGGED_NOTE = "Dear Alice,\ndraft of my message\n#!format\n"


def test_tagged_note_formatted_same_run(tmp_path: Path) -> None:
    """A tagged non-daily note is enqueued, stripped, and formatted now."""
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "message draft.md"
    note.write_text(TAGGED_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault)
    client = _client_with_replies(_reply(["draft"], "## Draft: to Alice\ndraft of my message"))

    summary = _run(cfg, tmp_path / "queue.json", client=client)

    assert summary["enqueued"] == 1
    assert summary["formatted"] == 1
    text = note.read_text(encoding="utf-8")
    assert "#!format" not in text
    assert "## Original Notes" in text
    assert "draft of my message" in text
    # Non-daily notes carry no date key, only the formatted timestamp.
    assert "\ndate:" not in text
    assert "formatted:" in text


def test_dry_run_reports_tagged_note_but_keeps_marker(tmp_path: Path) -> None:
    """Dry runs never modify notes: the marker survives, the report shows it."""
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "idea.md"
    note.write_text(TAGGED_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault)

    summary = _run(cfg, tmp_path / "queue.json", dry_run=True)

    assert summary["pending"] == ["idea.md"]
    assert "#!format" in note.read_text(encoding="utf-8")


def test_tagged_daily_note_stripped_but_not_fast_tracked(tmp_path: Path) -> None:
    """The marker on a daily note is consumed; dailies keep the next-day rule."""
    vault = tmp_path / "vault"
    vault.mkdir()
    today_note = vault / "2026-06-12.md"
    today_note.write_text(TAGGED_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault)

    summary = _run(cfg, tmp_path / "queue.json")

    assert summary["enqueued"] == 0
    assert summary.get("formatted", 0) == 0
    assert "#!format" not in today_note.read_text(encoding="utf-8")


def test_tagged_already_formatted_note_skipped(tmp_path: Path) -> None:
    """Tagging an already-formatted note consumes the marker but never
    double-wraps the note in a second Original Notes section."""
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "done.md"
    note.write_text(
        "---\nformatted: '2026-06-01T00:00:00'\n---\nbody\n\n## Original Notes\n\nraw\n#!format\n",
        encoding="utf-8",
    )
    cfg = _make_cfg(vault)
    client = _client_with_replies()

    summary = _run(cfg, tmp_path / "queue.json", client=client)

    text = note.read_text(encoding="utf-8")
    assert "#!format" not in text
    assert text.count("## Original Notes") == 1
    assert summary["formatted"] == 0


def test_format_tag_none_disables_scan(tmp_path: Path) -> None:
    """With format_tag null the trigger scan never runs."""
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "idea.md"
    note.write_text(TAGGED_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault, format_tag=None)

    summary = _run(cfg, tmp_path / "queue.json", dry_run=True)

    assert summary["enqueued"] == 0
    assert "#!format" in note.read_text(encoding="utf-8")


def test_strip_failure_does_not_abort_run(tmp_path: Path, caplog) -> None:
    """A note that cannot be stripped (vanished, perms) warns and the run
    continues: every other note is still enqueued and formatted."""
    import logging
    from unittest.mock import patch as mock_patch

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a draft.md").write_text(TAGGED_NOTE, encoding="utf-8")
    (vault / "b idea.md").write_text(TAGGED_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault)
    client = _client_with_replies(
        _reply(["draft"], "## Draft\nbody"), _reply(["idea"], "## Idea\nbody")
    )

    with (
        mock_patch(
            "obsidian_rag.daily_format.runner.strip_format_tag",
            side_effect=[OSError("vanished"), None],
        ),
        caplog.at_level(logging.WARNING, logger="obsidian_rag.daily_format.runner"),
    ):
        summary = _run(cfg, tmp_path / "queue.json", client=client)

    assert summary["enqueued"] == 2
    assert summary["formatted"] == 2
    assert any("a draft.md" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Tests 18-20: --tags-only poll mode and --since backfill
# ---------------------------------------------------------------------------


def test_tags_only_formats_tagged_but_leaves_dailies(tmp_path: Path) -> None:
    """tags_only drains tagged notes now; dailies wait for the nightly run."""
    vault = tmp_path / "vault"
    vault.mkdir()
    daily = vault / "2026-06-11.md"
    daily.write_text(RAW_NOTE, encoding="utf-8")
    tagged = vault / "draft.md"
    tagged.write_text(TAGGED_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault)
    client = _client_with_replies(_reply(["draft"], "## Draft\nbody"))

    summary = _run(cfg, tmp_path / "queue.json", client=client, tags_only=True)

    assert summary["enqueued"] == 1
    assert summary["formatted"] == 1
    assert "## Original Notes" in tagged.read_text(encoding="utf-8")
    # The eligible daily was neither enqueued nor formatted.
    assert daily.read_text(encoding="utf-8") == RAW_NOTE


def test_tags_only_without_tags_never_touches_ollama(tmp_path: Path) -> None:
    """A poll with nothing tagged exits without constructing a client."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "2026-06-11.md").write_text(RAW_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault)

    # client=None asserts if ollama.Client is ever constructed.
    summary = _run(cfg, tmp_path / "queue.json", tags_only=True)

    assert summary["formatted"] == 0


def test_tags_only_leaves_queued_daily_items_untouched(tmp_path: Path) -> None:
    """Daily items already in the queue are not drained by a tag poll."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "2026-06-11.md").write_text(RAW_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault)
    queue_path = tmp_path / "queue.json"
    _seed_queue(
        queue_path,
        QueueItem(vault="vault", rel_path="2026-06-11.md", note_date="2026-06-11"),
    )

    summary = _run(cfg, queue_path, tags_only=True)

    assert summary["formatted"] == 0
    state = json.loads(queue_path.read_text(encoding="utf-8"))
    assert [item["rel_path"] for item in state["items"]] == ["2026-06-11.md"]


def test_since_backfills_past_start_date_and_catchup(tmp_path: Path) -> None:
    """--since formats notes older than both start_date and catchup_days."""
    vault = tmp_path / "vault"
    vault.mkdir()
    old = vault / "2026-03-20.md"  # before START (06-01) and outside catchup 14
    old.write_text(RAW_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault)
    client = _client_with_replies(_reply(["notes"], "## Notes\nbody"))

    summary = _run(
        cfg,
        tmp_path / "queue.json",
        client=client,
        since=datetime.date(2026, 3, 19),
    )

    assert summary["enqueued"] == 1
    assert summary["formatted"] == 1
    assert "## Original Notes" in old.read_text(encoding="utf-8")


def test_since_still_respects_blacklist_and_next_day(tmp_path: Path) -> None:
    """A backfill never touches blacklisted notes or today's note."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "2026-06-10.md").write_text(RAW_NOTE, encoding="utf-8")
    today_note = vault / "2026-06-12.md"
    today_note.write_text(RAW_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault, blacklist=["2026-06-10"])

    summary = _run(
        cfg, tmp_path / "queue.json", dry_run=True, since=datetime.date(2026, 3, 19)
    )

    assert summary["enqueued"] == 0
    assert summary["pending"] == []


# ---------------------------------------------------------------------------
# Tests 21-23: battery gate
# ---------------------------------------------------------------------------

from obsidian_rag.daily_format.power import PowerState  # noqa: E402


def test_low_battery_defers_without_touching_ollama(tmp_path: Path) -> None:
    """On battery below the threshold, drain is deferred and items stay queued."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "2026-06-11.md").write_text(RAW_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault)
    queue_path = tmp_path / "queue.json"

    # client=None makes ollama.Client construction assert; proving we never reach it.
    summary = _run(
        cfg,
        queue_path,
        power_state=PowerState(has_battery=True, percent=15, on_ac_power=False),
    )

    assert summary["battery_deferred"] is True
    assert summary["battery_percent"] == 15
    assert summary["formatted"] == 0
    assert summary["queued"] == 1
    # The note is still raw and still queued for a later run.
    assert (vault / "2026-06-11.md").read_text(encoding="utf-8") == RAW_NOTE
    state = json.loads(queue_path.read_text(encoding="utf-8"))
    assert [item["rel_path"] for item in state["items"]] == ["2026-06-11.md"]


def test_low_battery_on_ac_proceeds(tmp_path: Path) -> None:
    """Charging below the threshold still formats — no drain risk on AC."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "2026-06-11.md").write_text(RAW_NOTE, encoding="utf-8")
    cfg = _make_cfg(vault)
    client = _client_with_replies(_reply(["t"], "## B"))

    summary = _run(
        cfg,
        tmp_path / "queue.json",
        client=client,
        power_state=PowerState(has_battery=True, percent=12, on_ac_power=True),
    )

    assert summary["formatted"] == 1
    assert "battery_deferred" not in summary


def test_battery_gate_skipped_when_nothing_pending(tmp_path: Path) -> None:
    """An empty run returns the normal summary, not a battery deferral."""
    vault = tmp_path / "vault"
    vault.mkdir()  # no eligible notes
    cfg = _make_cfg(vault)

    summary = _run(
        cfg,
        tmp_path / "queue.json",
        power_state=PowerState(has_battery=True, percent=5, on_ac_power=False),
    )

    assert "battery_deferred" not in summary
    assert summary == {"enqueued": 0, "formatted": 0, "failed": 0, "skipped": 0}
