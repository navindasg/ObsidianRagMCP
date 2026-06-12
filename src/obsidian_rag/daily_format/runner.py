"""Orchestrates one nightly daily-note formatting run.

Public API:
    run_format_daily(cfg, *, queue_path=None, today=None, dry_run=False) -> dict

Flow: scan every vault for raw daily notes, enqueue them in the persistent
format queue, then drain the queue against a local Ollama chat model. The
queue survives sleep and failures; an unreachable Ollama simply leaves items
queued for the next run, and one item's failure never aborts the run.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import ollama

from obsidian_rag.daily_format.detector import (
    find_candidates,
    is_already_formatted,
    parse_note_date,
)
from obsidian_rag.daily_format.formatter import FormatError, format_file
from obsidian_rag.daily_format.model_select import select_model
from obsidian_rag.daily_format.queue import FormatQueue, QueueItem, default_queue_path
from obsidian_rag.daily_format.tags import collect_vault_tags
from obsidian_rag.daily_format.trigger import scan_format_tags, strip_format_tag
from obsidian_rag.models import AppConfig, VaultConfig

logger = logging.getLogger(__name__)


def run_format_daily(
    cfg: AppConfig,
    *,
    queue_path: Path | None = None,
    today: datetime.date | None = None,
    dry_run: bool = False,
) -> dict:
    """Run one nightly formatting pass: enqueue candidates, then drain.

    Args:
        cfg: Validated application config (daily_format section drives this).
        queue_path: Queue file location; defaults to default_queue_path().
        today: Current date override for testing; defaults to date.today().
        dry_run: When True, enqueue and report but never touch Ollama or
            rewrite any note (the queue itself is still persisted).

    Returns:
        Summary counts. Normal runs: {"enqueued", "formatted", "failed",
        "skipped"}. Dry runs: {"enqueued", "pending", "formatted", "failed"}.
        When Ollama is unreachable: {"enqueued", "formatted", "failed",
        "queued", "ollama_down"} with everything left queued.
    """
    today = today if today is not None else datetime.date.today()
    queue = FormatQueue.load(queue_path if queue_path is not None else default_queue_path())
    start_date = cfg.daily_format.start_date or queue.ensure_start_date(today)

    enqueued = _enqueue_candidates(cfg, queue, today=today, start_date=start_date)
    enqueued += _enqueue_tagged(cfg, queue, dry_run=dry_run)
    queue.save()
    pending = queue.pending(cfg.daily_format.max_retries)

    if dry_run:
        return {
            "enqueued": enqueued,
            "pending": [item.rel_path for item in pending],
            "formatted": 0,
            "failed": 0,
        }
    if not pending:
        logger.info("No pending daily notes to format")
        return {"enqueued": enqueued, "formatted": 0, "failed": 0, "skipped": 0}

    client = ollama.Client(host=cfg.embedding.ollama_url)
    try:
        model = select_model(client, cfg.daily_format.model)
    except ConnectionError as exc:
        logger.warning(
            "Ollama is unreachable (%s); left %d items queued", exc, len(pending)
        )
        return {
            "enqueued": enqueued,
            "formatted": 0,
            "failed": 0,
            "queued": len(pending),
            "ollama_down": True,
        }

    counts = _drain(cfg, queue, pending, client=client, model=model, today=today)
    queue.save()
    return {"enqueued": enqueued, **counts}


def _enqueue_candidates(
    cfg: AppConfig,
    queue: FormatQueue,
    *,
    today: datetime.date,
    start_date: datetime.date,
) -> int:
    """Scan every vault for eligible raw daily notes and enqueue them."""
    daily = cfg.daily_format
    enqueued = 0
    for vault in cfg.vaults:
        candidates = find_candidates(
            vault.path,
            daily_folder=daily.daily_folder,
            filename_format=daily.filename_format,
            today=today,
            start_date=start_date,
            catchup_days=daily.catchup_days,
            excluded_dirs=vault.excluded_dirs,
            excluded_patterns=vault.excluded_patterns,
            blacklist=daily.blacklist,
        )
        for path in candidates:
            note_date = parse_note_date(path, daily.filename_format)
            if note_date is None:  # find_candidates guarantees a date; defensive
                continue
            item = QueueItem(
                vault=vault.name,
                rel_path=str(path.relative_to(vault.path)),
                note_date=note_date.isoformat(),
            )
            if queue.enqueue(item):
                enqueued += 1
    return enqueued


def _enqueue_tagged(cfg: AppConfig, queue: FormatQueue, *, dry_run: bool) -> int:
    """Scan every vault for format-tagged notes and enqueue them.

    The marker is stripped as soon as the note is queued (the queue, not
    the marker, now carries the request) — except on dry runs, which never
    modify notes. Daily-pattern notes only have their marker consumed:
    dailies are auto-scheduled and keep the next-day rule.
    """
    daily = cfg.daily_format
    if daily.format_tag is None:
        return 0

    enqueued = 0
    for vault in cfg.vaults:
        tagged = scan_format_tags(
            vault.path,
            format_tag=daily.format_tag,
            excluded_dirs=vault.excluded_dirs,
            excluded_patterns=vault.excluded_patterns,
            blacklist=daily.blacklist,
        )
        for path in tagged:
            rel_path = str(path.relative_to(vault.path))
            if parse_note_date(path, daily.filename_format) is not None:
                logger.info(
                    "Daily note %s carries %s; dailies are auto-scheduled — "
                    "consuming the tag",
                    rel_path,
                    daily.format_tag,
                )
                if not dry_run:
                    strip_format_tag(path, daily.format_tag)
                continue
            item = QueueItem(
                vault=vault.name, rel_path=rel_path, note_date=None, kind="tagged"
            )
            if queue.enqueue(item):
                enqueued += 1
            if not dry_run:
                strip_format_tag(path, daily.format_tag)
    return enqueued


def _drain(
    cfg: AppConfig,
    queue: FormatQueue,
    pending: list[QueueItem],
    *,
    client: ollama.Client,
    model: str,
    today: datetime.date,
) -> dict[str, int]:
    """Format every pending item; one item's failure never aborts the run."""
    vaults = {vault.name: vault for vault in cfg.vaults}
    vocab_cache: dict[str, list[str]] = {}
    now = datetime.datetime.now()
    formatted = failed = skipped = 0

    for item in pending:
        vault = vaults.get(item.vault)
        if vault is None:
            logger.warning(
                "Unknown vault '%s' for queued note %s; leaving it parked",
                item.vault,
                item.rel_path,
            )
            skipped += 1
            continue

        path = _resolve_in_vault(vault.path, item.rel_path)
        note_date = (
            datetime.date.fromisoformat(item.note_date)
            if item.note_date is not None
            else None
        )
        if path is None or not _item_still_eligible(item, path, note_date, today):
            logger.info(
                "Skipping %s/%s: no longer eligible", item.vault, item.rel_path
            )
            queue.mark_done(item)
            skipped += 1
            continue

        try:
            format_file(
                path,
                client=client,
                model=model,
                tag_vocab=_vault_tag_vocab(vocab_cache, vault),
                note_date=note_date,
                now=now,
            )
        except (FormatError, ConnectionError) as exc:
            logger.error(
                "Failed to format %s/%s: %s", item.vault, item.rel_path, exc
            )
            queue.mark_failed(item, cfg.daily_format.max_retries)
            failed += 1
            continue
        queue.mark_done(item)
        formatted += 1

    return {"formatted": formatted, "failed": failed, "skipped": skipped}


def _resolve_in_vault(vault_root: Path, rel_path: str) -> Path | None:
    """Resolve a queued rel_path inside its vault, or None if it escapes."""
    candidate = (vault_root / rel_path).resolve()
    if not candidate.is_relative_to(vault_root.resolve()):
        logger.warning(
            "Queued path %s escapes vault root %s; dropping it", rel_path, vault_root
        )
        return None
    return candidate


def _item_still_eligible(
    item: QueueItem,
    path: Path,
    note_date: datetime.date | None,
    today: datetime.date,
) -> bool:
    """Re-check eligibility right before formatting (queue may be stale).

    A note is no longer eligible when the file vanished or it was formatted
    in the meantime. Daily items additionally require their date to be in
    the past (next-day rule); a daily item missing its date can only come
    from a hand-edited queue file and is treated as ineligible. Tagged items
    were opted in explicitly and have no date gate. Read errors other than
    a missing file return True so format_file can surface them as a proper
    FormatError.
    """
    if item.kind != "tagged":
        if note_date is None or note_date >= today:
            return False
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    except (OSError, UnicodeDecodeError):
        return True
    return not is_already_formatted(text)


def _vault_tag_vocab(cache: dict[str, list[str]], vault: VaultConfig) -> list[str]:
    """Collect a vault's tag vocabulary once per run, then reuse it."""
    if vault.name not in cache:
        cache[vault.name] = collect_vault_tags(
            vault.path,
            excluded_dirs=vault.excluded_dirs,
            excluded_patterns=vault.excluded_patterns,
        )
    return cache[vault.name]
