# Product Requirements Document
## ObsidianRAG MCP Server

| | |
|---|---|
| **Version** | 1.0 |
| **Author** | Navin |
| **Date** | March 19, 2026 |
| **Status** | Draft |
| **Stack** | Python, FastMCP, FAISS-cpu, Ollama, watchdog |
| **Target Client** | Claude Desktop (stdio transport) |

---

## 1. Overview

ObsidianRAG is a local MCP server that gives Claude Desktop semantic search and file access over one or more Obsidian vaults. It indexes markdown notes into a FAISS vector store using locally-hosted embeddings via Ollama, watches for file changes in real time, and exposes a set of MCP tools through the stdio transport. The entire system runs on the user's machine with no cloud dependencies.

### 1.1 Goals

- Provide Claude with intelligent, embedding-based retrieval over Obsidian notes
- Run fully local on Apple Silicon (M-series Macs) with zero cloud dependencies
- Support multiple vaults with independent indexes and configurations
- Offer a configurable tool surface so users can toggle semantic search on/off and fall back to raw file access
- Keep the codebase small, dependency-light, and easy to extend

### 1.2 Non-Goals

- Cloud deployment or remote MCP transport (stdio only for v1)
- GUI or web dashboard for configuration
- Obsidian plugin development (this is a standalone MCP server)
- Real-time collaboration or multi-user access
- Write-back to vault (read-only for v1; append may follow in v2)

---

## 2. Architecture

### 2.1 System Components

The server is composed of five core modules that operate as a single Python process spawned by Claude Desktop via stdio:

- **MCP Server Layer (FastMCP):** Handles stdio transport, tool registration, request/response serialization. This is the entry point.
- **Indexing Engine:** Reads markdown files from vault(s), parses frontmatter, chunks content by heading boundaries, generates embeddings via Ollama, and stores vectors in FAISS with a sidecar metadata dict.
- **Retrieval Engine:** Accepts natural language queries, embeds them, performs FAISS similarity search, applies optional metadata filters and reranking, and returns ranked chunks with source context.
- **File Watcher (watchdog):** Monitors vault directories for create/modify/delete events and triggers incremental re-indexing of affected files.
- **Configuration Manager:** Loads and validates config.yaml, merges CLI overrides, and exposes typed config to all modules.

### 2.2 Data Flow

**Indexing flow (startup + incremental):**

1. Scan vault directory, filter by excluded_dirs/patterns
2. For each .md file: parse frontmatter (tags, aliases, dates), extract heading tree
3. Chunk by heading boundaries (or fixed-window fallback). Each chunk retains its heading path, source file, and frontmatter metadata.
4. Embed chunks via Ollama (batched). Store vectors in FAISS index, metadata in sidecar dict.
5. Persist index + metadata to disk (`~/.obsidian-rag/<vault-name>/`).

**Query flow (tool invocation):**

1. Claude calls search tool with query string and optional filters (tags, folders)
2. Embed query via Ollama
3. FAISS returns top-N candidates by L2 distance
4. Post-filter by metadata (tags, folders) if specified
5. Optional rerank pass (Ollama LLM-based relevance scoring)
6. Return top-K chunks with source path, heading path, relevance score, and snippet

### 2.3 Storage Layout

```
~/.obsidian-rag/
  config.yaml                    # global config
  <vault-name>/
    index.faiss                  # FAISS flat L2 index
    metadata.json                # chunk_id -> {file, tags, folder,
                                 #   heading_path, created, modified}
    file_hashes.json             # file path -> content hash
                                 #   (for incremental re-index)
```

---

## 3. MCP Tools

The server exposes the following tools to Claude. Each tool can be individually enabled or disabled via the enabled_tools config list.

| Tool | Description | Requires Index |
|------|-------------|----------------|
| `search` | Semantic similarity search. Returns top-K chunks with heading path, source file, and relevance score. Accepts optional tag/folder filters. | Yes |
| `read_note` | Read the full contents of a note by file path. Returns raw markdown with frontmatter. | No |
| `list_notes` | List vault structure: files, folders, note count. Supports optional path prefix filter. | No |
| `find_notes` | Keyword/filename substring search across vault. No embeddings — pure text matching. | No |
| `note_context` | Returns a note plus its backlinks and forward links (parsed from `[[wikilinks]]`). | No |
| `vault_stats` | Returns index health: total notes, total chunks, index age, embedding model, vault name. | No |
| `reindex` | Force a full re-index of the vault. Useful after bulk edits or config changes. | Yes |

### 3.1 Search Tool Response Format

Each search result returns the matched chunk with surrounding context so Claude knows where it sits within the document:

```json
{
  "results": [
    {
      "source": "projects/wsn-pipeline.md",
      "heading_path": "## Architecture > ### Email Ingestion",
      "chunk": "The M365 Group mailbox uses Graph API...",
      "relevance_score": 0.82,
      "tags": ["project", "wsn", "active"],
      "modified": "2026-03-15T10:30:00Z"
    }
  ],
  "query": "email ingestion architecture",
  "total_results": 3,
  "vault": "work"
}
```

Claude can then call `read_note` to get the full document if a chunk looks relevant but needs more context.

---

## 4. Configuration

All configuration lives in a single `config.yaml` file, defaulting to `~/.obsidian-rag/config.yaml`. CLI flags can override any setting. The config is validated at startup with clear error messages for missing required fields.

### 4.1 Vault Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `vaults[].name` | string | (required) | Human-readable vault identifier |
| `vaults[].path` | string | (required) | Absolute path to Obsidian vault directory |
| `vaults[].excluded_dirs` | list | `[.obsidian, .trash, templates]` | Directories to skip during indexing |
| `vaults[].excluded_patterns` | list | `[]` | Glob patterns to ignore (e.g., `"draft-*"`) |

### 4.2 Embedding Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `embedding.model` | string | `nomic-embed-text` | Ollama model name for embeddings |
| `embedding.dimensions` | int | (auto) | Embedding vector dimensions; auto-detected from model if omitted |
| `embedding.ollama_url` | string | `http://localhost:11434` | Ollama API base URL |
| `embedding.batch_size` | int | `64` | Number of chunks to embed per Ollama request |

### 4.3 Indexing Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `indexing.chunk_strategy` | string | `heading` | `"heading"` (split on `##` boundaries) or `"fixed"` (token window) |
| `indexing.chunk_max_tokens` | int | `512` | Max tokens per chunk (fixed strategy only) |
| `indexing.chunk_overlap` | int | `50` | Overlap tokens between chunks (fixed strategy only) |
| `indexing.include_frontmatter` | string | `metadata_only` | `"metadata_only"` (parse as filters), `"embed"` (include in chunk text), `"ignore"` |
| `indexing.watch_enabled` | bool | `true` | Auto-reindex on file system changes via watchdog |

### 4.4 Retrieval Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `retrieval.enabled` | bool | `true` | Toggle semantic search on/off. When off, only file-access tools are available. |
| `retrieval.top_k` | int | `5` | Number of results to return (range 1–20) |
| `retrieval.similarity_threshold` | float | `0.7` | Minimum similarity score cutoff (0.0–1.0) |
| `retrieval.max_context_tokens` | int | `4000` | Cap on total tokens returned across all chunks |

### 4.5 Reranking Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rerank.enabled` | bool | `false` | Toggle reranking pass on retrieved results |
| `rerank.model` | string | `null` | Ollama model for relevance scoring (e.g., `"llama3.2"`) |
| `rerank.top_n` | int | `20` | Candidates to pull from FAISS before reranking down to top_k |

### 4.6 Tool Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tools.enabled` | list | `[search, read_note, list_notes, find_notes, note_context, vault_stats, reindex]` | MCP tools to register with Claude |

### 4.7 Full Config Example

```yaml
vaults:
  - name: work
    path: ~/obsidian/work
    excluded_dirs: [.obsidian, .trash, templates]
    excluded_patterns: ["draft-*"]
  - name: personal
    path: ~/obsidian/personal
    excluded_dirs: [.obsidian, .trash]

embedding:
  model: nomic-embed-text
  ollama_url: http://localhost:11434
  batch_size: 64

indexing:
  chunk_strategy: heading
  include_frontmatter: metadata_only
  watch_enabled: true

retrieval:
  enabled: true
  top_k: 5
  similarity_threshold: 0.7
  max_context_tokens: 4000

rerank:
  enabled: false
  model: null
  top_n: 20

tools:
  enabled:
    - search
    - read_note
    - list_notes
    - find_notes
    - note_context
    - vault_stats
    - reindex
```

---

## 5. Claude Desktop Integration

The server is registered in Claude Desktop's `claude_desktop_config.json` as a stdio MCP server. Claude Desktop spawns the Python process directly.

### 5.1 Config Entry

```json
{
  "mcpServers": {
    "obsidian-rag": {
      "command": "python",
      "args": ["-m", "obsidian_rag", "--config", "~/.obsidian-rag/config.yaml"],
      "env": {}
    }
  }
}
```

### 5.2 Startup Sequence

1. Claude Desktop spawns the process via stdio
2. Config is loaded and validated
3. For each vault: load existing FAISS index from disk (or build if first run)
4. Incremental re-index: compare file hashes, re-embed only changed/new files
5. Start watchdog file watchers on all vault paths
6. Register enabled MCP tools
7. Begin accepting tool calls from Claude via stdio

---

## 6. Chunking Strategy

### 6.1 Heading-Based (Default)

The heading strategy splits markdown at heading boundaries (`#`, `##`, `###`, etc.). Each chunk includes:

- The heading path (e.g., `"## Architecture > ### Email Ingestion"`)
- All body text under that heading until the next heading of equal or higher level
- Frontmatter metadata attached as structured metadata (not embedded in chunk text by default)

If a section exceeds `chunk_max_tokens`, it falls back to fixed-window splitting within that section to prevent oversized chunks.

### 6.2 Fixed-Window (Fallback)

Splits text into fixed token-length windows with configurable overlap. Less semantically coherent but useful for vaults with flat, unstructured notes that lack heading hierarchy.

### 6.3 Frontmatter Handling

- **metadata_only (default):** Parse YAML frontmatter into structured metadata (tags, aliases, created, modified). Used for filtering but not embedded into chunk text.
- **embed:** Prepend frontmatter key-value pairs to chunk text before embedding. Useful if your frontmatter contains rich descriptions.
- **ignore:** Strip frontmatter entirely.

---

## 7. Metadata Filtering

Metadata filtering operates as a post-filter on FAISS results. The retrieval engine pulls top-N candidates (controlled by `rerank.top_n` when reranking is enabled, or a 4x multiplier on `top_k` when disabled), then filters by metadata predicates before returning the final top-K.

### 7.1 Supported Filters

- **tags:** Match notes containing any of the specified Obsidian tags (OR logic)
- **folders:** Restrict to notes within specific vault subdirectories
- **modified_after / modified_before:** Date range filters on file modification time
- **vault:** When multi-vault, restrict search to a specific vault by name

### 7.2 Metadata Store

Each chunk's metadata is stored in a Python dict serialized as `metadata.json` alongside the FAISS index. The dict maps chunk_id (int, matching FAISS vector ID) to a metadata object:

```json
{
  "chunk_id": 42,
  "file": "projects/wsn-pipeline.md",
  "heading_path": "## Architecture > ### Email Ingestion",
  "tags": ["project", "wsn", "active"],
  "folder": "projects",
  "created": "2026-01-15T09:00:00Z",
  "modified": "2026-03-15T10:30:00Z",
  "vault": "work",
  "char_count": 847
}
```

---

## 8. Reranking

When enabled, reranking adds a second pass after FAISS retrieval to improve result quality. The reranker pulls `rerank.top_n` candidates from FAISS, then uses an Ollama model to score each chunk's relevance to the original query.

### 8.1 Approach

The reranker sends a prompt to the specified Ollama model for each candidate chunk, asking it to score relevance on a 0–1 scale. Chunks are then re-sorted by this score and the top-K are returned. This is a simple pointwise reranking approach — effective and easy to implement.

### 8.2 Performance Considerations

- Reranking adds latency proportional to top_n (one Ollama inference per candidate)
- For a top_n of 20 with a small model like llama3.2, expect ~2–4 seconds on M3 Max
- Reranking is disabled by default; enable it when retrieval quality matters more than speed
- Future optimization: batch scoring or use a dedicated cross-encoder model via sentence-transformers

---

## 9. File Watching

The watchdog library monitors vault directories for filesystem events. On detecting a change, the server performs incremental re-indexing:

- **File created:** Parse, chunk, embed, and add to FAISS index + metadata store
- **File modified:** Remove old chunks for that file from index, re-parse and re-embed
- **File deleted:** Remove all chunks for that file from index and metadata store
- **File renamed:** Treat as delete + create

A debounce window of 2 seconds is applied to coalesce rapid edits (e.g., Obsidian auto-save) into a single re-index operation. File content hashes are stored in `file_hashes.json` to avoid redundant re-embedding when content hasn't actually changed.

---

## 10. Tech Stack & Dependencies

### 10.1 Core Dependencies

- **fastmcp:** MCP server framework (stdio transport)
- **faiss-cpu:** Vector similarity search (flat L2 index)
- **ollama (Python client):** Embedding generation via local Ollama instance
- **watchdog:** Filesystem event monitoring
- **python-frontmatter:** YAML frontmatter parsing from markdown files
- **pyyaml:** Config file parsing

### 10.2 Runtime Requirements

- Python 3.11+
- Ollama installed and running locally with `nomic-embed-text` (or configured model) pulled
- macOS with Apple Silicon (M1/M2/M3) — optimized for but not exclusive to
- Obsidian vault(s) on local filesystem

### 10.3 Explicitly Excluded

- No LangChain, LlamaIndex, or heavy orchestration frameworks
- No database server (SQLite, Postgres, etc.) — FAISS + JSON sidecar only
- No async complexity — stdio MCP is synchronous request/response
- No web server or HTTP layer — stdio transport only

---

## 11. Project Structure

```
obsidian-rag/
  pyproject.toml
  README.md
  src/
    obsidian_rag/
      __init__.py
      __main__.py          # CLI entry point
      server.py            # FastMCP server + tool registration
      config.py            # Config loading, validation, defaults
      indexer.py           # Chunking, embedding, FAISS index mgmt
      retriever.py         # Search, filtering, reranking
      watcher.py           # Watchdog file monitoring
      markdown_parser.py   # Frontmatter + heading-based chunking
      models.py            # Pydantic models for config, metadata, results
  tests/
    test_chunking.py
    test_retrieval.py
    test_config.py
    test_watcher.py
    fixtures/
      sample_vault/        # Test vault with sample notes
```

---

## 12. Milestones

### Phase 1: Core (MVP)

- Single vault support with heading-based chunking
- FAISS flat L2 index with Ollama embeddings
- MCP tools: search, read_note, list_notes, find_notes, vault_stats
- Config.yaml loading with sensible defaults
- Claude Desktop stdio integration working end-to-end

### Phase 2: Multi-Vault & Watching

- Multi-vault support with per-vault indexes
- Watchdog file watcher with debounced incremental re-indexing
- File hash tracking for skip-if-unchanged optimization
- note_context tool (backlink/forward link parsing)
- reindex tool for manual trigger

### Phase 3: Advanced Retrieval

- Metadata filtering (tags, folders, date ranges)
- Reranking via Ollama
- Configurable tool surface (enable/disable per tool)
- Fixed-window chunking fallback strategy

### Phase 4: Polish

- Comprehensive test suite with sample vault fixtures
- pip-installable package (pyproject.toml, entry point)
- README with setup guide and config reference
- Performance benchmarks on M3 Max (index build time, query latency)
- Potential: write-back tools (append_note, create_note)

---

## 13. Open Questions

- Should the reranker use a structured prompt with JSON output, or is a simple relevance score prompt sufficient?
- Should wikilink resolution for note_context be full graph traversal or single-hop only?
- Is there value in exposing an MCP resource (not tool) for vault metadata, so Claude can see vault structure without a tool call?
- Should the similarity_threshold use cosine similarity (normalized) or raw L2 distance? L2 is native to FAISS flat but less intuitive to configure.
- For fixed chunking: should overlap be token-based or sentence-boundary-based?
