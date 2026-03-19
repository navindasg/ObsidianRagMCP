"""Retrieval engine: FAISS similarity search with metadata filtering, score conversion, token cap."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import faiss
import numpy as np

from obsidian_rag.models import SearchResult, to_float32

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def l2_to_cosine(l2_distance: float) -> float:
    """Convert L2 distance to cosine similarity.

    Valid only for pre-normalized vectors (unit norm).
    Formula: cosine = 1.0 - (l2_distance ** 2) / 2.0
    Result clamped to [0.0, 1.0] and rounded to 2 decimal places.
    """
    cosine = 1.0 - (l2_distance ** 2) / 2.0
    cosine = max(0.0, min(1.0, cosine))
    return round(cosine, 2)


def filter_results(
    candidates: list[tuple[int, float]],
    metadata: dict[str, dict],
    tags: list[str] | None = None,
    folder: str | None = None,
    modified_after: datetime | None = None,
    modified_before: datetime | None = None,
    vault_name: str | None = None,
) -> list[tuple[int, float]]:
    """Filter candidate (chunk_id, score) pairs using metadata predicates.

    Filter logic:
    - Tags: OR within tags list (chunk must match at least one tag)
    - Folder: prefix match on the file path
    - modified_after/modified_before: comparison against modified_ts unix timestamp
    - vault_name: exact match on vault field
    - AND between different filter types (all active filters must pass)
    """
    filtered: list[tuple[int, float]] = []

    for chunk_id, score in candidates:
        meta = metadata.get(str(chunk_id))
        if meta is None:
            logger.warning("No metadata for chunk_id %s — skipping", chunk_id)
            continue

        # Tags filter: OR logic — chunk must have at least one of the required tags
        if tags is not None:
            chunk_tags = meta.get("tags", [])
            if not any(t in chunk_tags for t in tags):
                continue

        # Folder filter: prefix match on file path
        if folder is not None:
            file_path = meta.get("file", "")
            if not file_path.startswith(folder):
                continue

        # Date range filters
        modified_ts = meta.get("modified_ts", 0.0)
        if modified_after is not None and modified_ts < modified_after.timestamp():
            continue
        if modified_before is not None and modified_ts > modified_before.timestamp():
            continue

        # Vault name filter
        if vault_name is not None and meta.get("vault") != vault_name:
            continue

        filtered.append((chunk_id, score))

    return filtered


def search(
    index: faiss.IndexIDMap,
    metadata: dict[str, dict],
    query_embedding: list[float],
    top_k: int = 5,
    similarity_threshold: float = 0.0,
    max_context_tokens: int = 4000,
    tags: list[str] | None = None,
    folder: str | None = None,
    modified_after: datetime | None = None,
    modified_before: datetime | None = None,
    vault_name: str | None = None,
) -> dict:
    """Search the FAISS index and return ranked, filtered, token-capped results.

    Returns a dict with:
    - "results": list of SearchResult dicts (empty list if no matches)
    - "message": explanatory string (only present when results is empty)
    """
    # 1. Convert query to float32 and normalize
    query_vec = to_float32([query_embedding])
    faiss.normalize_L2(query_vec)

    # 2. Over-fetch to account for post-search filtering (3x top_k, min 1)
    fetch_k = max(1, min(top_k * 3, index.ntotal))
    distances, ids = index.search(query_vec, fetch_k)

    # 3. Build candidates, skipping FAISS sentinel -1 values
    candidates: list[tuple[int, float]] = [
        (int(chunk_id), l2_to_cosine(float(dist)))
        for chunk_id, dist in zip(ids[0], distances[0])
        if chunk_id != -1
    ]

    # 4. Apply metadata filters
    candidates = filter_results(
        candidates,
        metadata,
        tags=tags,
        folder=folder,
        modified_after=modified_after,
        modified_before=modified_before,
        vault_name=vault_name,
    )

    # 5. Apply similarity threshold
    candidates = [(cid, score) for cid, score in candidates if score >= similarity_threshold]

    # 6. Sort by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)

    # 7. Apply token cap and build SearchResult objects
    results: list[SearchResult] = []
    token_budget_used = 0

    for chunk_id, score in candidates:
        if len(results) >= top_k:
            break

        meta = metadata.get(str(chunk_id))
        if meta is None:
            logger.warning("No metadata for chunk_id %s after filtering — skipping", chunk_id)
            continue

        # Approximate token count: char_count / 4 (rough but consistent)
        chunk_tokens = meta.get("char_count", 0) // 4
        if token_budget_used + chunk_tokens > max_context_tokens and results:
            # Stop once budget is exceeded (always allow at least 1 result)
            break

        token_budget_used += chunk_tokens

        results.append(
            SearchResult(
                source_path=meta["file"],
                heading_path=meta["heading_path"],
                relevance_score=score,
                snippet=meta.get("text", ""),
                vault_name=meta["vault"],
            )
        )

    if not results:
        return {
            "results": [],
            "message": (
                "No matching results found. "
                "Try broadening your search or adjusting filters."
            ),
        }

    return {"results": [r.model_dump() for r in results]}
