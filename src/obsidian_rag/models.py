from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class VaultConfig(BaseModel):
    name: str
    path: Path
    excluded_dirs: list[str] = Field(default=[".obsidian", ".trash", "templates"])
    excluded_patterns: list[str] = Field(default=[])

    @field_validator("path", mode="before")
    @classmethod
    def expand_tilde(cls, v) -> Path:
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
    chunk_strategy: str = Field(default="heading")
    chunk_max_tokens: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    include_frontmatter: str = Field(default="metadata_only")
    watch_enabled: bool = Field(default=True)


class RetrievalConfig(BaseModel):
    enabled: bool = Field(default=True)
    top_k: int = Field(default=5)
    similarity_threshold: float = Field(default=0.7)
    max_context_tokens: int = Field(default=4000)


class RerankConfig(BaseModel):
    enabled: bool = Field(default=False)
    model: str | None = Field(default=None)
    top_n: int = Field(default=20)


class ToolsConfig(BaseModel):
    enabled: list[str] = Field(
        default=["search", "read_note", "list_notes", "find_notes", "note_context", "vault_stats", "reindex"]
    )


class AppConfig(BaseModel):
    vaults: list[VaultConfig]
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
