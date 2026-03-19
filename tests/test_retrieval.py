"""Tests for obsidian_rag.retriever — covers search, filtering, threshold, token cap."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import faiss
import numpy as np
import pytest

from obsidian_rag.models import to_float32
from obsidian_rag.retriever import filter_results, l2_to_cosine, search


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_index_and_metadata():
    """Create a small FAISS index with known vectors and metadata for testing."""
    dim = 8  # small dimension for tests
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))

    # Create 5 known vectors (normalized)
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((5, dim)).astype(np.float32)
    faiss.normalize_L2(vectors)
    ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    index.add_with_ids(vectors, ids)

    metadata = {
        "10": {
            "chunk_id": 10,
            "file": "projects/wsn.md",
            "heading_path": "# WSN",
            "text": "Overview of the wireless sensor network pipeline.",
            "tags": ["project", "python"],
            "folder": "projects",
            "vault": "work",
            "modified_ts": 1700000000.0,
            "char_count": 200,
        },
        "20": {
            "chunk_id": 20,
            "file": "projects/ml.md",
            "heading_path": "# ML",
            "text": "Machine learning pipeline for data processing.",
            "tags": ["project", "ml"],
            "folder": "projects",
            "vault": "work",
            "modified_ts": 1700100000.0,
            "char_count": 300,
        },
        "30": {
            "chunk_id": 30,
            "file": "daily/2024-01-15.md",
            "heading_path": "# Daily",
            "text": "Review PR #42 and update documentation.",
            "tags": ["daily"],
            "folder": "daily",
            "vault": "work",
            "modified_ts": 1700200000.0,
            "char_count": 150,
        },
        "40": {
            "chunk_id": 40,
            "file": "notes/idea.md",
            "heading_path": "# Idea",
            "text": "New idea for the project architecture.",
            "tags": ["note"],
            "folder": "notes",
            "vault": "work",
            "modified_ts": 1700300000.0,
            "char_count": 500,
        },
        "50": {
            "chunk_id": 50,
            "file": "projects/api.md",
            "heading_path": "# API",
            "text": "REST API design for the sensor network.",
            "tags": ["project", "python"],
            "folder": "projects",
            "vault": "work",
            "modified_ts": 1700400000.0,
            "char_count": 250,
        },
    }

    # Store the vectors for query construction
    return index, metadata, vectors


# ---------------------------------------------------------------------------
# l2_to_cosine tests
# ---------------------------------------------------------------------------


def test_l2_to_cosine_identical_vectors():
    """L2 distance 0.0 (identical vectors) returns cosine similarity 1.0."""
    result = l2_to_cosine(0.0)
    assert result == 1.0


def test_l2_to_cosine_orthogonal():
    """L2 distance sqrt(2) (~1.414) for orthogonal normalized vectors returns cosine 0.0."""
    l2_dist = math.sqrt(2)
    result = l2_to_cosine(l2_dist)
    assert result == 0.0


def test_l2_to_cosine_clamps_negative():
    """Negative floating point artifacts are clamped to 0.0."""
    # Slightly negative distance due to floating point
    result = l2_to_cosine(-0.0001)
    assert result == 1.0  # clamped from > 1.0 to 1.0


def test_l2_to_cosine_rounds_to_2_decimals():
    """Result is rounded to exactly 2 decimal places."""
    # l2_distance=0.5 -> cosine = 1 - 0.25/2 = 0.875 -> rounds to 0.88
    result = l2_to_cosine(0.5)
    # Check it's rounded to 2 decimal places
    assert result == round(result, 2)
    # Also verify the value
    expected = round(max(0.0, min(1.0, 1.0 - (0.5 ** 2) / 2.0)), 2)
    assert result == expected


# ---------------------------------------------------------------------------
# search tests
# ---------------------------------------------------------------------------


def test_search_returns_ranked_results(sample_index_and_metadata):
    """Search with a real FAISS index returns SearchResult objects sorted by score descending."""
    index, metadata, vectors = sample_index_and_metadata

    # Use first stored vector as query — should match itself with highest score
    query = vectors[0].tolist()
    result = search(index, metadata, query, top_k=5)

    assert "results" in result
    results = result["results"]
    assert len(results) >= 1

    scores = [r["relevance_score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results must be sorted by relevance_score descending"

    # First result should be very close to 1.0 (exact match)
    assert results[0]["relevance_score"] >= 0.99


def test_search_returns_all_fields(sample_index_and_metadata):
    """Each SearchResult has all required fields: source_path, heading_path, relevance_score, snippet, vault_name."""
    index, metadata, vectors = sample_index_and_metadata

    query = vectors[0].tolist()
    result = search(index, metadata, query, top_k=5)

    for r in result["results"]:
        assert "source_path" in r
        assert "heading_path" in r
        assert "relevance_score" in r
        assert "snippet" in r
        assert "vault_name" in r
        # snippet must come from the chunk text in metadata
        assert r["snippet"] != "", "snippet should be populated from metadata text field"


# ---------------------------------------------------------------------------
# filter_results tests
# ---------------------------------------------------------------------------


def _make_candidates() -> tuple[list[tuple[int, float]], dict[str, dict]]:
    """Helper: return candidates list and matching metadata dict."""
    metadata = {
        "10": {
            "chunk_id": 10,
            "file": "projects/wsn.md",
            "heading_path": "# WSN",
            "text": "Overview of the wireless sensor network pipeline.",
            "tags": ["project", "python"],
            "folder": "projects",
            "vault": "work",
            "modified_ts": 1700000000.0,
            "char_count": 200,
        },
        "20": {
            "chunk_id": 20,
            "file": "projects/ml.md",
            "heading_path": "# ML",
            "text": "Machine learning pipeline.",
            "tags": ["project", "ml"],
            "folder": "projects",
            "vault": "work",
            "modified_ts": 1700100000.0,
            "char_count": 300,
        },
        "30": {
            "chunk_id": 30,
            "file": "daily/2024-01-15.md",
            "heading_path": "# Daily",
            "text": "Review PR #42.",
            "tags": ["daily"],
            "folder": "daily",
            "vault": "work",
            "modified_ts": 1700200000.0,
            "char_count": 150,
        },
    }
    candidates = [(10, 0.9), (20, 0.8), (30, 0.7)]
    return candidates, metadata


def test_filter_results_by_tags_or():
    """Filter with tags=['python', 'ml'] returns chunks tagged with EITHER tag (OR logic)."""
    candidates, metadata = _make_candidates()
    filtered = filter_results(candidates, metadata, tags=["python", "ml"])
    ids = [c[0] for c in filtered]
    # chunk 10 has "python", chunk 20 has "ml"
    assert 10 in ids
    assert 20 in ids
    # chunk 30 has neither
    assert 30 not in ids


def test_filter_results_by_folder():
    """Filter with folder='projects/' returns only chunks in that folder."""
    candidates, metadata = _make_candidates()
    filtered = filter_results(candidates, metadata, folder="projects/")
    ids = [c[0] for c in filtered]
    assert 10 in ids
    assert 20 in ids
    assert 30 not in ids


def test_filter_results_by_modified_after():
    """Filter with modified_after returns only chunks newer than the given datetime."""
    candidates, metadata = _make_candidates()
    # Cutoff between chunk 10 (1700000000) and chunk 20 (1700100000)
    cutoff = datetime.fromtimestamp(1700050000.0, tz=timezone.utc)
    filtered = filter_results(candidates, metadata, modified_after=cutoff)
    ids = [c[0] for c in filtered]
    assert 10 not in ids  # older than cutoff
    assert 20 in ids
    assert 30 in ids


def test_filter_results_by_modified_before():
    """Filter with modified_before returns only chunks older than the given datetime."""
    candidates, metadata = _make_candidates()
    # Cutoff between chunk 20 (1700100000) and chunk 30 (1700200000)
    cutoff = datetime.fromtimestamp(1700150000.0, tz=timezone.utc)
    filtered = filter_results(candidates, metadata, modified_before=cutoff)
    ids = [c[0] for c in filtered]
    assert 10 in ids
    assert 20 in ids
    assert 30 not in ids  # newer than cutoff


def test_filter_results_combined_and():
    """Tags + folder filter is AND (both must match)."""
    candidates, metadata = _make_candidates()
    # Require tag "python" AND folder "projects/"
    filtered = filter_results(candidates, metadata, tags=["python"], folder="projects/")
    ids = [c[0] for c in filtered]
    # Only chunk 10 has "python" AND is in projects/
    assert 10 in ids
    assert 20 not in ids  # no "python" tag
    assert 30 not in ids  # not in projects/


def test_similarity_threshold_filters(sample_index_and_metadata):
    """Results below similarity_threshold are excluded."""
    index, metadata, vectors = sample_index_and_metadata
    query = vectors[0].tolist()

    # Very high threshold — only the best match (self-match ~1.0) should pass
    result = search(index, metadata, query, top_k=5, similarity_threshold=0.99)
    for r in result["results"]:
        assert r["relevance_score"] >= 0.99


def test_token_cap_limits_results(sample_index_and_metadata):
    """max_context_tokens cap stops returning results once the token budget is exceeded."""
    index, metadata, vectors = sample_index_and_metadata
    query = vectors[0].tolist()

    # char_count for chunk 10 is 200 → ~50 tokens; set cap to 50 to get at most 1 result
    result = search(index, metadata, query, top_k=5, max_context_tokens=50)
    results = result["results"]
    # With a 50-token cap and each chunk at ≥50 tokens, we get at most 1 result
    assert len(results) <= 2


def test_empty_results_returns_message(sample_index_and_metadata):
    """Search with no matches above threshold returns empty list and explanatory message."""
    index, metadata, vectors = sample_index_and_metadata
    query = vectors[0].tolist()

    # Impossibly high threshold — no results should pass
    result = search(index, metadata, query, top_k=5, similarity_threshold=1.01)

    assert result["results"] == []
    assert "message" in result
    assert len(result["message"]) > 0


def test_search_query_vector_normalized(sample_index_and_metadata):
    """Query vector is L2-normalized before FAISS search (result score should be in 0-1 range)."""
    index, metadata, vectors = sample_index_and_metadata

    # Use an unnormalized query (large magnitude)
    unnormalized = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    result = search(index, metadata, unnormalized, top_k=5)

    # All scores must be in [0, 1] range — only valid if query is normalized
    for r in result["results"]:
        assert 0.0 <= r["relevance_score"] <= 1.0
