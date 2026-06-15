"""Orchestrates one nightly daily-note formatting run.

Public API:
    run_format_daily(cfg, *, queue_path=None, dry_run=False, tags_only=False,
        since=None) -> dict

Flow: scan every vault for raw daily notes (those with a later-dated
successor), enqueue them in the persistent format queue, then drain the
queue against a local Ollama chat model. The queue survives sleep and
failures; an unreachable Ollama simply leaves items queued for the next
run, and one item's failure never aborts the run.
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
from obsidian_rag.daily_format.power import read_power_state, should_defer
from obsidian_rag.daily_format.queue import FormatQueue, QueueItem, default_queue_path
from obsidian_rag.daily_format.tags import collect_vault_tags
from obsidian_rag.daily_format.trigger import scan_format_tags, strip_format_tag
from obsidian_rag.models import AppConfig, VaultConfig

logger = logging.getLogger(__name__)


def run_format_daily(
    cfg: AppConfig,
    *,
    queue_path: Path | None = None,
    dry_run: bool = False,
    tags_only: bool = False,
    since: datetime.date | None = None,
) -> dict:
    """Run one formatting pass: enqueue candidates, then drain.

    Eligibility is successor-based: a daily note is formatted once a
    later-dated daily note exists, and the most recent note is held back.
    Calendar time never matters.

    Args:
        cfg: Validated application config (daily_format section drives this).
        queue_path: Queue file location; defaults to default_queue_path().
        dry_run: When True, enqueue and report but never touch Ollama or
            rewrite any note (the queue itself is still persisted).
        tags_only: When True (the background poll), skip the daily-note
            scan and drain only tagged items; queued daily items wait for
            the nightly run.
        since: Manual backfill — formats every daily note dated on or after
            this date, including the most recent (lifts the latest-note
            hold). The blacklist still applies.

    Returns:
        Summary counts. Normal runs: {"enqueued", "formatted", "failed",
        "skipped"}. Dry runs: {"enqueued", "pending", "formatted", "failed"}.
        When Ollama is unreachable: {"enqueued", "formatted", "failed",
        "queued", "ollama_down"} with everything left queued. When deferred
        for low battery: {"enqueued", "formatted", "failed", "queued",
        "battery_deferred", "battery_percent"} with everything left queued.
    """
    queue = FormatQueue.load(queue_path if queue_path is not None else default_queue_path())

    enqueued = 0
    if not tags_only:
        enqueued += _enqueue_candidates(cfg, queue, since=since)
    enqueued += _enqueue_tagged(cfg, queue, dry_run=dry_run)
    pending = queue.pending(cfg.daily_format.max_retries)
    if tags_only:
        pending = [item for item in pending if item.kind == "tagged"]

    if dry_run:
        # A dry run is read-only: report what would happen without persisting
        # the queue or touching notes (so a dry --since never leaks the
        # latest note into a later automatic run).
        return {
            "enqueued": enqueued,
            "pending": [item.rel_path for item in pending],
            "formatted": 0,
            "failed": 0,
        }
    queue.save()
    if not pending:
        logger.info("No pending daily notes to format")
        return {"enqueued": enqueued, "formatted": 0, "failed": 0, "skipped": 0}

    power = read_power_state()
    if should_defer(power, cfg.daily_format.min_battery_percent):
        logger.warning(
            "Battery at %s%% on battery power (threshold %s%%); deferring "
            "%d item(s) until the battery recovers",
            power.percent,
            cfg.daily_format.min_battery_percent,
            len(pending),
        )
        return {
            "enqueued": enqueued,
            "formatted": 0,
            "failed": 0,
            "queued": len(pending),
            "battery_deferred": True,
            "battery_percent": power.percent,
        }

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

    counts = _drain(cfg, queue, pending, client=client, model=model)
    queue.save()
    return {"enqueued": enqueued, **counts}


def _enqueue_candidates(
    cfg: AppConfig,
    queue: FormatQueue,
    *,
    since: datetime.date | None,
) -> int:
    """Scan every vault for eligible raw daily notes and enqueue them."""
    daily = cfg.daily_format
    enqueued = 0
    for vault in cfg.vaults:
        candidates = find_candidates(
            vault.path,
            daily_folder=daily.daily_folder,
            filename_format=daily.filename_format,
            excluded_dirs=vault.excluded_dirs,
            excluded_patterns=vault.excluded_patterns,
            blacklist=daily.blacklist,
            since=since,
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
    dailies are auto-scheduled by the successor rule.
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
                    _strip_tag_safely(path, daily.format_tag)
                continue
            item = QueueItem(
                vault=vault.name, rel_path=rel_path, note_date=None, kind="tagged"
            )
            if queue.enqueue(item):
                enqueued += 1
            if not dry_run:
                _strip_tag_safely(path, daily.format_tag)
    return enqueued


def _strip_tag_safely(path: Path, format_tag: str) -> None:
    """Strip the marker, never letting one note's I/O failure abort the run.

    The note may vanish or change permissions between scan and strip; the
    queued item already carries the request, and a marker left behind is
    simply re-found (and deduped) on the next run.
    """
    try:
        strip_format_tag(path, format_tag)
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not strip %s from %s: %s", format_tag, path, exc)


def _drain(
    cfg: AppConfig,
    queue: FormatQueue,
    pending: list[QueueItem],
    *,
    client: ollama.Client,
    model: str,
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
        if path is None or not _still_eligible(path):
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


def _still_eligible(path: Path) -> bool:
    """Re-check eligibility right before formatting (queue may be stale).

    A note is no longer eligible when the file vanished or it was formatted
    in the meantime (the successor check that admitted it cannot change in
    the seconds between scan and drain). Read errors other than a missing
    file return True so format_file can surface them as a proper FormatError.
    """
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
