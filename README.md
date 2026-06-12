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
- Optional nightly daily-note formatting: a local LLM tags and cleans up raw daily notes while preserving the original text

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

daily_format:
  enabled: false                 # Master switch for the nightly formatting job
  daily_folder: ""               # Daily-notes folder relative to vault root ("" = vault root)
  filename_format: "%Y-%m-%d"    # strptime pattern matched against note filename stems
  model: null                    # Ollama chat model (null = auto-select from pulled models)
  schedule_hour: 0               # Hour of the nightly launchd run (0–23)
  schedule_minute: 30            # Minute of the nightly launchd run (0–59)
  start_date: null               # No-backfill cutoff (null = recorded on first run)
  catchup_days: 14               # Max age in days of notes picked up after downtime
  max_retries: 3                 # Attempts per note before it is parked in the queue
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
| `daily_format` | Nightly daily-note formatting job | Disabled by default; runs at 00:30 when installed |

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

## Daily Note Formatting

An optional nightly job that cleans up raw Obsidian daily notes (files whose stem matches `daily_format.filename_format`, e.g. `2026-06-11.md`, in the vault root or a configured `daily_folder`). A local Ollama chat model suggests tags and a reorganized markdown body; the model is told to prefer tags from your vault's existing tag vocabulary and to invent a new lowercase-kebab-case tag only when nothing fits. Code — not the model — assembles the final file:

1. YAML frontmatter: merged `tags`, the note's `date`, and a `formatted` timestamp (any other frontmatter keys from the original are preserved)
2. The model's formatted body
3. A verbatim `## Original Notes` section containing the untouched original text

The `formatted` frontmatter key marks a note as done, so a note is never formatted twice. Disabled by default — set `daily_format.enabled: true` to use it.

### Eligibility and the no-backfill rule

- **Next-day rule:** a note is only formatted once its date is in the past. Today's note is never touched; yesterday's note is picked up by tonight's run.
- **No backfill:** only notes dated on or after `start_date` are eligible. When `start_date` is `null` (the default), the first run records its own date into the queue file — so daily notes that existed before you enabled the feature are never reformatted. Set `start_date` explicitly in the config to override.
- **Catch-up window:** after downtime, at most the last `catchup_days` days of notes are picked up.

### Running it

```bash
obsidian-rag format-daily              # one formatting pass now
obsidian-rag format-daily --dry-run    # enqueue and report; never calls Ollama or rewrites notes
obsidian-rag format-daily --date 2026-06-12   # override "today" (for testing)
```

`format-daily` exits non-zero if any note failed to format. Failed notes stay in the queue and are retried on the next run, up to `max_retries` attempts each, after which they are parked.

### Scheduling (macOS launchd)

```bash
obsidian-rag schedule install      # install (or reinstall) the nightly LaunchAgent
obsidian-rag schedule status       # show launchd's view of the agent
obsidian-rag schedule uninstall    # remove the agent
```

`schedule install` writes `~/Library/LaunchAgents/com.obsidian-rag.daily-format.plist`, which runs `format-daily` every night at `schedule_hour:schedule_minute` (default 00:30). If the machine is asleep at the scheduled time, launchd fires the missed run when it wakes — no run is ever silently dropped. The agent's output is appended to `~/.obsidian-rag/logs/daily-format.log`.

### The persistent queue

Work is tracked in a JSON queue at `~/.obsidian-rag/format_queue.json`, which also stores the recorded `start_date`. The queue survives sleep, crashes, and failures: if Ollama is unreachable, everything stays queued for the next run, and one note's failure never aborts the rest of the run.

### Model selection

When `daily_format.model` is set, it is validated against your pulled Ollama models (with an `ollama pull` hint if missing). When it is `null`, the first pulled model from this priority list is used:

1. `gemma4:26b-mlx`
2. `gemma4:12b-mlx`
3. `qwen3.5:9b`
4. `ministral-3:8b`
5. `llama3.2`

If none of those are pulled, the first pulled non-embedding model is used; if no chat model is available at all, the run fails with a suggestion to `ollama pull llama3.2`.

---

## CLI Reference

The package installs an `obsidian-rag` console script (equivalent to
`python -m obsidian_rag`). Bare invocation starts the MCP server:

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

Subcommands (see [Daily Note Formatting](#daily-note-formatting)):

```
obsidian-rag format-daily [OPTIONS]

  --config PATH       Path to config file (default: ~/.obsidian-rag/config.yaml)
  --dry-run           Enqueue and report, but do not call Ollama or rewrite notes
  --date YYYY-MM-DD   Override today's date (for testing)

obsidian-rag schedule install   [--config PATH]   Install (or reinstall) the nightly LaunchAgent
obsidian-rag schedule uninstall [--config PATH]   Remove the nightly LaunchAgent
obsidian-rag schedule status    [--config PATH]   Show the LaunchAgent's launchd status
```

`format-daily` exits with status 1 when any note failed to format. All logs
and command output go to stderr; stdout is reserved for the MCP stdio protocol.

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
