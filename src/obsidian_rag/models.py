from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

# Fallback model used for reranking when rerank.model is not configured.
# Shared by the health check and the reranker so they can never validate
# a different model than the one actually used.
DEFAULT_RERANK_MODEL = "llama3.2"


class VaultConfig(BaseModel):
    name: str
    path: Path
    excluded_dirs: list[str] = Field(
        default_factory=lambda: [".obsidian", ".trash", "templates"]
    )
    excluded_patterns: list[str] = Field(default_factory=list)

    @field_validator("path", mode="before")
    @classmethod
    def expand_tilde(cls, v: str | Path) -> Path:
        return Path(str(v)).expanduser()

    @field_validator("path")
    @classmethod
    def path_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Vault path does not exist: {v}")
        return v


class EmbeddingConfig(BaseModel):
    model: str = Field(default="nomic-embed-text")
    ollama_url: str = Field(default="http://localhost:11434")
    batch_size: int = Field(default=64)


class IndexingConfig(BaseModel):
    # Literal types reject config typos like "fiexd" at load time instead of
    # silently falling back to the default behavior.
    chunk_strategy: Literal["heading", "fixed"] = Field(default="heading")
    chunk_max_tokens: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    include_frontmatter: Literal["metadata_only", "embed", "ignore"] = Field(
        default="metadata_only"
    )
    watch_enabled: bool = Field(default=True)


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=5, gt=0)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_context_tokens: int = Field(default=4000, gt=0)


class RerankConfig(BaseModel):
    enabled: bool = Field(default=False)
    model: str | None = Field(default=None)
    top_n: int = Field(default=20, gt=0)


class ToolsConfig(BaseModel):
    enabled: list[str] = Field(
        default_factory=lambda: [
            "search",
            "read_note",
            "list_notes",
            "find_notes",
            "note_context",
            "vault_stats",
            "reindex",
        ]
    )


class AppConfig(BaseModel):
    vaults: list[VaultConfig]
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)


class ChunkMetadata(BaseModel):
    """Metadata for a single chunk stored alongside the FAISS index."""

    chunk_id: int
    file: str  # relative path from vault root
    heading_path: str  # e.g. "# Project > ## Goals"
    text: str = ""  # chunk text stored for snippet retrieval (RET-01)
    tags: list[str] = Field(default_factory=list)
    folder: str = ""  # top-level folder from vault root
    vault: str = ""
    modified_ts: float = 0.0  # filesystem mtime as unix timestamp
    char_count: int = 0  # character count of chunk text


class SearchResult(BaseModel):
    """A single search result returned to the user."""

    source_path: str
    heading_path: str
    relevance_score: float  # 0.0-1.0 cosine similarity, 2 decimal places
    snippet: str
    vault_name: str


def to_float32(vectors: list[list[float]]) -> np.ndarray:
    """Shared utility: cast vectors to float32 for FAISS operations (IDX-09)."""
    return np.array(vectors, dtype=np.float32)
