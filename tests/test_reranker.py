"""Tests for reranker — rerank() function and search() integration with reranking.

Tests:
  1. rerank_resorts_candidates — rerank reorders by LLM score
  2. rerank_fallback_on_model_error — ConnectionError falls back gracefully
  3. rerank_fallback_on_parse_error — unparseable LLM output assigns 0.0
  4. rerank_disabled_by_default — search() unchanged without rerank params
  5. search_with_rerank_enabled — search() re-sorts by rerank score
  6. search_fetch_k_changes_with_rerank — fetch_k becomes rerank_config.top_n
  7. rerank_score_replaces_cosine — relevance_score is rerank score not cosine
  8. existing_tests_still_pass — existing search() behavior unchanged
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import faiss
import numpy as np
import pytest

from obsidian_rag.models import RerankConfig, to_float32
from obsidian_rag.retriever import rerank, search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_response(content: str) -> MagicMock:
    """Create a mock ollama.Client.chat() response with the given message content."""
    response = MagicMock()
    response.message.content = content
    return response


def _make_metadata(*chunk_ids: int) -> dict[str, dict]:
    """Build a minimal metadata dict for the given chunk IDs."""
    meta: dict[str, dict] = {}
    for i, cid in enumerate(chunk_ids):
        meta[str(cid)] = {
            "chunk_id": cid,
            "file": f"notes/chunk_{cid}.md",
            "heading_path": f"# Chunk {cid}",
            "text": f"This is the text for chunk {cid}.",
            "tags": [],
            "folder": "notes",
            "vault": "test",
            "modified_ts": float(1700000000 + i * 1000),
            "char_count": len(f"This is the text for chunk {cid}."),
        }
    return meta


# ---------------------------------------------------------------------------
# Test 1: rerank resorts candidates by LLM score
# ---------------------------------------------------------------------------


def test_rerank_resorts_candidates():
    """Given 3 candidates with cosine scores [0.9, 0.8, 0.7], mock ollama to return
    rerank scores [0.3, 0.9, 0.6], verify output order is [cand2, cand3, cand1] with
    rerank scores replacing cosine scores."""
    candidates = [(1, 0.9), (2, 0.8), (3, 0.7)]
    metadata = _make_metadata(1, 2, 3)
    rerank_cfg = RerankConfig(enabled=True, model="llama3.2", top_n=20)

    # LLM returns scores [0.3, 0.9, 0.6] for chunks 1, 2, 3 respectively
    side_effects = [
        _make_chat_response("0.3"),
        _make_chat_response("0.9"),
        _make_chat_response("0.6"),
    ]
    mock_client = MagicMock()
    mock_client.chat.side_effect = side_effects

    with patch("obsidian_rag.retriever.ollama.Client", return_value=mock_client):
        result = rerank(
            candidates=candidates,
            metadata=metadata,
            query="test query",
            rerank_config=rerank_cfg,
            ollama_url="http://localhost:11434",
        )

    # Expected order: chunk 2 (0.9), chunk 3 (0.6), chunk 1 (0.3)
    assert len(result) == 3
    ids = [chunk_id for chunk_id, _ in result]
    scores = [score for _, score in result]

    assert ids == [2, 3, 1], f"Expected order [2, 3, 1] but got {ids}"
    assert scores[0] == pytest.approx(0.9)
    assert scores[1] == pytest.approx(0.6)
    assert scores[2] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Test 2: rerank fallback on model/connection error
# ---------------------------------------------------------------------------


def test_rerank_fallback_on_model_error(caplog):
    """When ollama.Client.chat raises ConnectionError, rerank returns candidates
    unchanged and logs a warning."""
    import logging

    candidates = [(1, 0.9), (2, 0.8), (3, 0.7)]
    metadata = _make_metadata(1, 2, 3)
    rerank_cfg = RerankConfig(enabled=True, model="llama3.2", top_n=20)

    mock_client = MagicMock()
    mock_client.chat.side_effect = ConnectionError("Connection refused")

    with patch("obsidian_rag.retriever.ollama.Client", return_value=mock_client):
        with caplog.at_level(logging.WARNING, logger="obsidian_rag.retriever"):
            result = rerank(
                candidates=candidates,
                metadata=metadata,
                query="test query",
                rerank_config=rerank_cfg,
                ollama_url="http://localhost:11434",
            )

    # Candidates returned unchanged
    assert result == candidates
    # Warning was logged
    assert any("Reranking failed" in record.message for record in caplog.records), (
        "Expected warning about reranking failure"
    )


# ---------------------------------------------------------------------------
# Test 3: rerank fallback on unparseable output
# ---------------------------------------------------------------------------


def test_rerank_fallback_on_parse_error(caplog):
    """When LLM returns unparseable text, rerank assigns 0.0 score and logs warning."""
    import logging

    candidates = [(1, 0.9), (2, 0.8)]
    metadata = _make_metadata(1, 2)
    rerank_cfg = RerankConfig(enabled=True, model="llama3.2", top_n=20)

    # First response is parseable, second is not
    mock_client = MagicMock()
    mock_client.chat.side_effect = [
        _make_chat_response("0.7"),
        _make_chat_response("I cannot rate this"),
    ]

    with patch("obsidian_rag.retriever.ollama.Client", return_value=mock_client):
        with caplog.at_level(logging.WARNING, logger="obsidian_rag.retriever"):
            result = rerank(
                candidates=candidates,
                metadata=metadata,
                query="test query",
                rerank_config=rerank_cfg,
                ollama_url="http://localhost:11434",
            )

    # chunk 1 gets 0.7, chunk 2 gets 0.0 — sorted descending: [1, 2]
    assert len(result) == 2
    ids = [chunk_id for chunk_id, _ in result]
    scores = [score for _, score in result]
    assert ids[0] == 1  # higher score first
    assert scores[0] == pytest.approx(0.7)
    assert scores[1] == pytest.approx(0.0)

    # Warning was logged for the parse failure
    assert any("parse" in record.message.lower() or "0.0" in record.message for record in caplog.records), (
        "Expected warning about parse failure"
    )


# ---------------------------------------------------------------------------
# Test 4: rerank disabled by default (search unchanged)
# ---------------------------------------------------------------------------


def test_rerank_disabled_by_default():
    """Call search() without rerank params — no ollama.chat calls are made."""
    dim = 4
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((3, dim)).astype(np.float32)
    faiss.normalize_L2(vectors)
    ids = np.array([10, 20, 30], dtype=np.int64)
    index.add_with_ids(vectors, ids)

    metadata = _make_metadata(10, 20, 30)
    query = vectors[0].tolist()

    with patch("obsidian_rag.retriever.ollama.Client") as MockClient:
        result = search(index, metadata, query, top_k=3)
        # No ollama.Client.chat calls — rerank is disabled
        MockClient.return_value.chat.assert_not_called()

    assert "results" in result


# ---------------------------------------------------------------------------
# Test 5: search with rerank enabled re-sorts by rerank score
# ---------------------------------------------------------------------------


def test_search_with_rerank_enabled():
    """search() with rerank_config(enabled=True) re-sorts results by rerank score."""
    dim = 4
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((3, dim)).astype(np.float32)
    faiss.normalize_L2(vectors)
    ids = np.array([10, 20, 30], dtype=np.int64)
    index.add_with_ids(vectors, ids)

    metadata = _make_metadata(10, 20, 30)
    query_vec = vectors[0].tolist()

    rerank_cfg = RerankConfig(enabled=True, model="llama3.2", top_n=3)

    # Mock LLM to return inverted relevance: chunk with worst cosine score gets highest rerank score
    # We let all 3 chunks pass, then assign rerank scores
    call_count = [0]
    scores_by_call = [0.1, 0.9, 0.5]  # LLM scores for each candidate in order

    def chat_side_effect(**kwargs):
        idx = call_count[0]
        call_count[0] += 1
        return _make_chat_response(str(scores_by_call[idx]))

    mock_client = MagicMock()
    mock_client.chat.side_effect = chat_side_effect

    with patch("obsidian_rag.retriever.ollama.Client", return_value=mock_client):
        result = search(
            index,
            metadata,
            query_vec,
            top_k=3,
            similarity_threshold=0.0,
            query_text="test query",
            rerank_config=rerank_cfg,
            ollama_url="http://localhost:11434",
        )

    assert "results" in result
    results = result["results"]
    assert len(results) >= 1

    # Scores should be in descending order (rerank scores, not cosine)
    scores = [r["relevance_score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results should be sorted by rerank score"


# ---------------------------------------------------------------------------
# Test 6: search_fetch_k_changes_with_rerank
# ---------------------------------------------------------------------------


def test_search_fetch_k_changes_with_rerank():
    """When rerank enabled with top_n=20, fetch_k should be rerank_config.top_n
    rather than top_k*3."""
    dim = 4
    n_vectors = 10
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    rng = np.random.default_rng(2)
    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    faiss.normalize_L2(vectors)
    ids = np.arange(100, 100 + n_vectors, dtype=np.int64)
    index.add_with_ids(vectors, ids)

    metadata = _make_metadata(*list(range(100, 100 + n_vectors)))
    query_vec = vectors[0].tolist()

    # top_n=5 — we expect 5 candidates fetched (not top_k*3=15)
    rerank_cfg = RerankConfig(enabled=True, model="llama3.2", top_n=5)

    chat_calls: list = []

    def chat_side_effect(**kwargs):
        chat_calls.append(kwargs)
        return _make_chat_response("0.5")

    mock_client = MagicMock()
    mock_client.chat.side_effect = chat_side_effect

    with patch("obsidian_rag.retriever.ollama.Client", return_value=mock_client):
        result = search(
            index,
            metadata,
            query_vec,
            top_k=2,  # top_k=2, so top_k*3=6 without rerank; with rerank top_n=5
            similarity_threshold=0.0,
            query_text="test query",
            rerank_config=rerank_cfg,
            ollama_url="http://localhost:11434",
        )

    # With top_n=5, reranker should have been called at most 5 times
    # (it might be fewer if metadata filter or threshold removes candidates)
    assert len(chat_calls) <= 5, (
        f"Expected at most 5 rerank calls (top_n=5), but got {len(chat_calls)}"
    )


# ---------------------------------------------------------------------------
# Test 7: rerank score replaces cosine score (D-03)
# ---------------------------------------------------------------------------


def test_rerank_score_replaces_cosine():
    """SearchResult.relevance_score uses rerank score, not cosine similarity."""
    dim = 4
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((2, dim)).astype(np.float32)
    faiss.normalize_L2(vectors)
    ids = np.array([10, 20], dtype=np.int64)
    index.add_with_ids(vectors, ids)

    metadata = _make_metadata(10, 20)
    # Use a fixed query so cosine scores are deterministic
    query_vec = vectors[0].tolist()

    rerank_cfg = RerankConfig(enabled=True, model="llama3.2", top_n=2)

    # LLM returns exactly 0.42 for all candidates
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_chat_response("0.42")

    with patch("obsidian_rag.retriever.ollama.Client", return_value=mock_client):
        result = search(
            index,
            metadata,
            query_vec,
            top_k=2,
            similarity_threshold=0.0,
            query_text="test query",
            rerank_config=rerank_cfg,
            ollama_url="http://localhost:11434",
        )

    assert "results" in result
    for r in result["results"]:
        # Cosine of identical vectors is 1.0 — if score is 0.42 it's the rerank score
        # (The first vector's cosine score with itself should be ~1.0, not 0.42)
        assert r["relevance_score"] == pytest.approx(0.42), (
            f"Expected rerank score 0.42 but got {r['relevance_score']}"
        )


# ---------------------------------------------------------------------------
# Test 8: Existing search tests still pass (backward-compatible new params)
# ---------------------------------------------------------------------------


def test_existing_search_backward_compatible():
    """All new search() params have safe defaults — calling without them works as before."""
    dim = 4
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    rng = np.random.default_rng(4)
    vectors = rng.standard_normal((3, dim)).astype(np.float32)
    faiss.normalize_L2(vectors)
    ids = np.array([10, 20, 30], dtype=np.int64)
    index.add_with_ids(vectors, ids)

    metadata = _make_metadata(10, 20, 30)
    query = vectors[0].tolist()

    # Call with ONLY the original positional args — no new params
    result = search(index, metadata, query, top_k=3)

    assert "results" in result
    results = result["results"]
    assert len(results) >= 1

    # Scores should still be sorted descending
    scores = [r["relevance_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
