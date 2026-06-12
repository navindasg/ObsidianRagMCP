# ObsidianRAG

A local MCP server that gives Claude Desktop semantic search and file access over Obsidian vaults. It indexes markdown notes into a FAISS vector store using locally-hosted embeddings via Ollama, watches for file changes in real time, and exposes MCP tools through the stdio transport. The entire system runs on your machine with zero cloud dependencies.

**Key features:**
- Semantic search over vault notes using FAISS and Ollama embeddings
- Heading-based chunking that preserves the semantic structure of Obsidian notes
- Optional LLM reranking via Ollama for more relevant results
- Wikilink context: follow forward links and discover backlinks from any note
- Multi-vault support with independent indexes per vault
- Real-time file watching with debounced incremental re-indexing
- Configurable tool surface — enable or disable individual tools via config

---

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai) installed and running (`ollama serve`)
- An embedding model pulled: `ollama pull nomic-embed-text`
- (Optional) A rerank model: `ollama pull llama3.2`

> **macOS ARM64 note:** `faiss-cpu` requires macOS 14+ for the ARM64 pip wheel. If you are on macOS 13 (Ventura) with Apple Silicon, install via conda instead:
> ```bash
> conda install -c conda-forge faiss-cpu
> ```

---

## Installation

Install from source:

```bash
git clone https://github.com/navindasg/ObsidianRagMCP.git
cd ObsidianRagMCP
pip install .
```

For development (editable install with dev dependencies):

```bash
pip install -e ".[dev]"
# or, with uv (a uv.lock is checked in):
uv sync
```

---

## Quick Start

### 1. Create a config file

Create `~/.obsidian-rag/config.yaml` with the path to your vault:

```yaml
vaults:
  - name: my-vault
    path: ~/Documents/ObsidianVault
```

Alternatively, run `python -m obsidian_rag` once — when no config exists it
generates a commented default at `~/.obsidian-rag/config.yaml` and exits with
instructions to edit it.

### 2. Verify the server starts

```bash
python -m obsidian_rag
```

You should see startup messages on stderr confirming Ollama connectivity and index build progress. The server then waits for MCP requests on stdin.

### 3. Add to Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` and add:

```json
{
  "mcpServers": {
    "obsidian-rag": {
      "command": "python",
      "args": ["-m", "obsidian_rag"]
    }
  }
}
```

Restart Claude Desktop. The ObsidianRAG tools will be available in your conversations.

---

## Claude Desktop Integration

The Claude Desktop configuration file lives at:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add the following entry under `"mcpServers"`:

```json
{
  "mcpServers": {
    "obsidian-rag": {
      "command": "python",
      "args": ["-m", "obsidian_rag"]
    }
  }
}
```

If you installed into a virtual environment, use the full path to the Python executable:

```json
{
  "mcpServers": {
    "obsidian-rag": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "obsidian_rag"]
    }
  }
}
```

Claude Desktop will spawn the server as a subprocess and communicate via stdio. No network ports are opened.

---

## Configuration Reference

The default config file location is `~/.obsidian-rag/config.yaml`. A minimal config requires only the `vaults` section; all other sections have sensible defaults.

```yaml
vaults:
  - name: my-vault
    path: ~/Documents/ObsidianVault
    excluded_dirs: [".obsidian", ".trash", "templates"]
    excluded_patterns: []

embedding:
  model: nomic-embed-text        # Ollama model name for embeddings
  ollama_url: http://localhost:11434
  batch_size: 64                 # Chunks embedded per Ollama request

indexing:
  chunk_strategy: heading        # "heading" (default) or "fixed"
  chunk_max_tokens: 512          # Max tokens per chunk
  chunk_overlap: 50              # Token overlap between consecutive chunks
  include_frontmatter: metadata_only  # "metadata_only", "embed", or "ignore"
  watch_enabled: true            # Watch for file changes and reindex automatically

retrieval:
  top_k: 5                       # Number of results to return
  similarity_threshold: 0.7      # Minimum cosine similarity (0.0–1.0)
  max_context_tokens: 4000       # Total token budget across all results

rerank:
  enabled: false                 # Set true to enable LLM reranking
  model: null                    # Reranking model (defaults to llama3.2 when enabled)
  top_n: 20                      # Candidates fetched from FAISS before reranking

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

### Section descriptions

| Section | Purpose | Notable defaults |
|---------|---------|-----------------|
| `vaults` | One or more vault definitions | `excluded_dirs` hides `.obsidian`, `.trash`, `templates` |
| `embedding` | Ollama embedding model settings | `nomic-embed-text` at `localhost:11434` |
| `indexing` | Chunking strategy and file watching | Heading-based chunking, watching enabled |
| `retrieval` | Search result count and quality thresholds | `top_k=5`, `similarity_threshold=0.7` |
| `rerank` | Optional LLM reranking pass | Disabled by default; requires `llama3.2` or similar |
| `tools` | Which MCP tools are exposed to Claude | All 7 tools enabled by default |

---

## Multi-Vault Setup

Each vault in `config.vaults` gets its own independent index stored under `~/.obsidian-rag/<vault-name>/`.

```yaml
vaults:
  - name: personal
    path: ~/Documents/PersonalVault
    excluded_dirs: [".obsidian", ".trash"]

  - name: work
    path: ~/Documents/WorkVault
    excluded_dirs: [".obsidian", ".trash", "archive"]
```

When using the `search` tool without a `vault_name` argument, results are merged across all vaults and sorted by relevance score. Each result includes a `vault_name` field so Claude can identify provenance.

---

## Available Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `search` | Semantic similarity search across vault notes | `query` (required), `vault_name`, `tags`, `folder` |
| `read_note` | Read the full content of a single note | `path` (required), `vault_name` |
| `list_notes` | List markdown files in a vault with metadata | `path_prefix`, `vault_name` |
| `find_notes` | Keyword/filename search (case-insensitive) | `query` (required), `vault_name` |
| `note_context` | Note content plus wikilink forward/back links | `path` (required), `vault_name` |
| `vault_stats` | Index health: note count, chunk count, index age | (none) |
| `reindex` | Trigger background rebuild of a vault's index | `vault_name` (required) |

### Tool return formats

**`search`** returns `{"results": [...]}` where each result has `source_path`, `heading_path`, `relevance_score` (0.0–1.0), `snippet`, and `vault_name`.

**`read_note`** returns `{"path": "...", "content": "...", "frontmatter": {...}}` on success, or `{"error": "...", "suggestion": "..."}` on failure. Only `.md` files inside the vault (and outside `excluded_dirs`) are accessible; the same applies to `note_context`.

**`list_notes`** returns `{"notes": [...]}` where each entry has `path`, `size`, `modified` (ISO 8601), and `tag_count`.

**`find_notes`** returns `{"results": [...]}` where each entry has `file` and `heading_path`.

**`note_context`** returns `{"note": {path, content}, "forward_links": [{path, exists}], "backlinks": [{source_path, heading_path, snippet}]}`.

**`vault_stats`** returns `{"vaults": [...], "total_notes": N, "total_chunks": N}` where each vault entry includes `vault`, `note_count`, `chunk_count`, `index_age`, `embedding_model`, and `last_reindex` (the outcome of the most recent background reindex, or `null`).

**`reindex`** returns `{"status": "started" | "already_running", "vault": "...", "message": "..."}` immediately without blocking. Check `vault_stats.last_reindex` for the outcome.

---

## CLI Reference

The package installs an `obsidian-rag` console script (equivalent to
`python -m obsidian_rag`):

```
obsidian-rag [OPTIONS]

  --config PATH       Path to config file (default: ~/.obsidian-rag/config.yaml)
  --vault-path PATH   Override the first vault's path
  --vault-name NAME   Override the first vault's name
  --ollama-url URL    Override the Ollama API URL
  --verbose           Log at INFO level (shows per-file indexing progress)
  --debug             Log at DEBUG level
  --version           Print the version and exit
```

All logs go to stderr; stdout is reserved for the MCP stdio protocol.

---

## Troubleshooting

**"Ollama is not reachable"**
Ensure Ollama is running: `ollama serve`. By default the server listens on `http://localhost:11434`.

**"Embedding model not found"**
Pull the required model: `ollama pull nomic-embed-text`. The model name must match `embedding.model` in your config.

**"Rerank model not found"**
Either pull the model (`ollama pull llama3.2`) or disable reranking in your config:
```yaml
rerank:
  enabled: false
```

**No search results returned**
- Lower `retrieval.similarity_threshold` (e.g., `0.5`) to allow less similar matches through.
- Verify the vault path in your config is correct and the directory contains `.md` files.
- Check that `vault_stats` reports a non-zero chunk count — if zero, the index build may have failed.

**`faiss-cpu` install fails on macOS 13 (Ventura)**
The ARM64 pip wheel for `faiss-cpu` requires macOS 14+. On macOS 13 with Apple Silicon, install via conda:
```bash
conda install -c conda-forge faiss-cpu
```

**Server not appearing in Claude Desktop**
- Verify the `claude_desktop_config.json` JSON is valid (no trailing commas).
- Check that `python -m obsidian_rag` works from your terminal with the same Python that Claude Desktop will use.
- Restart Claude Desktop after editing the config file.

**File changes not being picked up**
File watching is enabled by default (`indexing.watch_enabled: true`). If changes aren't reflected, use the `reindex` tool to force a rebuild, or restart the server.

---

## Development

```bash
pytest                   # run all tests
pytest --cov             # with coverage report
python -m obsidian_rag   # run server locally (reads ~/.obsidian-rag/config.yaml)
```

The package uses `src/` layout. Source lives in `src/obsidian_rag/`. Tests live in `tests/`.

---

## License

MIT
