# ObsidianRAG MCP Server

## What This Is

A local MCP server that gives Claude Desktop semantic search and file access over Obsidian vaults. It indexes markdown notes into a FAISS vector store using locally-hosted embeddings via Ollama, watches for file changes in real time, and exposes MCP tools through stdio transport. The entire system runs on the user's machine with zero cloud dependencies.

## Core Value

Claude can intelligently search and retrieve relevant content from Obsidian vaults using semantic similarity — turning a passive file store into an active knowledge base.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Single and multi-vault support with independent indexes
- [ ] Heading-based markdown chunking with fixed-window fallback
- [ ] FAISS flat L2 vector index with Ollama embeddings (nomic-embed-text)
- [ ] Semantic search with metadata filtering (tags, folders, dates)
- [ ] MCP tools: search, read_note, list_notes, find_notes, note_context, vault_stats, reindex
- [ ] Config.yaml with vault, embedding, indexing, retrieval, rerank, and tool settings
- [ ] Claude Desktop stdio integration (spawned as subprocess)
- [ ] Watchdog file watcher with debounced incremental re-indexing
- [ ] File hash tracking to skip unchanged files
- [ ] Optional reranking via Ollama LLM-based relevance scoring
- [ ] Configurable tool surface (enable/disable per tool)
- [ ] Comprehensive test suite with sample vault fixtures
- [ ] pip-installable package with CLI entry point

### Out of Scope

- Cloud deployment or remote MCP transport — stdio only for v1
- GUI or web dashboard — configuration is YAML only
- Obsidian plugin — this is a standalone MCP server
- Real-time collaboration or multi-user access — single-user local tool
- Write-back to vault — read-only for v1 (append may follow in v2)
- LangChain, LlamaIndex, or heavy orchestration frameworks — keep it lean
- Database server (SQLite, Postgres) — FAISS + JSON sidecar only
- Async complexity — stdio MCP is synchronous request/response

## Context

- Target platform: macOS Apple Silicon (M-series), though not exclusive
- Runtime: Python 3.12+, Ollama running locally
- Available embedding model: nomic-embed-text (274MB, already pulled)
- Available rerank model: llama3.2 (2.0GB, already pulled)
- Storage: ~/.obsidian-rag/<vault-name>/ for indexes and metadata
- Transport: stdio only — Claude Desktop spawns the Python process directly
- Stack: FastMCP, faiss-cpu, ollama (Python client), watchdog, python-frontmatter, pyyaml
- Project structure: src/obsidian_rag/ package layout per PRD section 11

## Constraints

- **Tech stack**: Python + FastMCP + FAISS + Ollama — no heavy frameworks, no database servers
- **Transport**: stdio only — no HTTP, no web server
- **Dependencies**: Minimal — explicitly excludes LangChain/LlamaIndex
- **Storage**: FAISS flat L2 index + JSON sidecar metadata — no external DB
- **Platform**: Optimized for Apple Silicon but not exclusive

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Heading-based chunking as default | Preserves semantic coherence of Obsidian notes organized by headings | — Pending |
| FAISS flat L2 over approximate NN | Vault sizes are small enough that exact search is fast; simpler implementation | — Pending |
| Ollama for all ML inference | Fully local, no cloud API keys, consistent interface for embed + rerank | — Pending |
| JSON sidecar over SQLite for metadata | Simpler, no DB dependency, easy to inspect/debug | — Pending |
| Simple pointwise reranking | One Ollama call per candidate — effective and easy to implement | — Pending |
| Wikilink resolution: single-hop only | Full graph traversal is complex; single-hop covers the common case for note_context | — Pending |
| Similarity threshold: cosine similarity (normalized) | More intuitive 0-1 scale for users, convert L2 internally | — Pending |
| Fixed chunking overlap: sentence-boundary-based | More natural chunk boundaries than mid-token splits | — Pending |
| Reranker: simple relevance score prompt | Structured JSON output adds complexity without proportional quality gain | — Pending |

---
*Last updated: 2026-03-19 after initialization*
