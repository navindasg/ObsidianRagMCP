from pathlib import Path

import yaml
from pydantic import ValidationError

from obsidian_rag.models import AppConfig


DEFAULT_CONFIG = """\
# ObsidianRAG configuration
# Required fields have no default — you MUST set these.

vaults:
  - name: "YOUR_VAULT_NAME"     # Required: human-readable name
    path: "~/obsidian/vault"    # Required: absolute path to vault directory
    # excluded_dirs: [.obsidian, .trash, templates]
    # excluded_patterns: []

# embedding:
#   model: nomic-embed-text
#   ollama_url: http://localhost:11434
#   batch_size: 64

# indexing:
#   chunk_strategy: heading
#   include_frontmatter: metadata_only
#   watch_enabled: true

# retrieval:
#   top_k: 5
#   similarity_threshold: 0.7
#   max_context_tokens: 4000

# rerank:
#   enabled: false
#   model: null
#   top_n: 20

# tools:
#   enabled:
#     - search
#     - read_note
#     - list_notes
#     - find_notes
#     - note_context
#     - vault_stats
#     - reindex

# daily_format:
#   enabled: false
#   daily_folder: ""
#   filename_format: "%Y-%m-%d"
#   model: null
#   schedule_hour: 0
#   schedule_minute: 30
#   start_date: null
#   catchup_days: 14
#   max_retries: 3
#   blacklist: []
"""


def generate_default_config(path: Path) -> None:
    """Generate a self-documenting default config file at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG, encoding="utf-8")


def _apply_overrides(raw: dict, overrides: dict) -> None:
    """Apply CLI overrides into the raw config dict (in place)."""
    if overrides.get("vault_path") is not None or overrides.get("vault_name") is not None:
        # A bare "vaults:" key in YAML loads as None; normalize before indexing.
        vaults = raw.get("vaults") or []
        if not vaults:
            vaults.append({})
        raw["vaults"] = vaults

        if overrides.get("vault_path") is not None:
            vaults[0]["path"] = overrides["vault_path"]
        if overrides.get("vault_name") is not None:
            vaults[0]["name"] = overrides["vault_name"]

    if overrides.get("ollama_url") is not None:
        embedding = raw.get("embedding") or {}
        embedding["ollama_url"] = overrides["ollama_url"]
        raw["embedding"] = embedding


def load_config(config_path: str, overrides: dict | None = None) -> AppConfig:
    """Load, validate, and return AppConfig from a YAML file.

    If the file does not exist, a default config is generated and SystemExit
    is raised with instructions to edit it. If the file contains validation
    errors, all errors are reported at once and SystemExit is raised.
    """
    path = Path(config_path).expanduser()

    if not path.exists():
        generate_default_config(path)
        raise SystemExit(
            f"Config not found -- created default at {path}\n"
            "Edit it to set your vault name and path, then restart."
        )

    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise SystemExit(f"Config file {path} is not valid YAML:\n{e}") from None

    if overrides:
        _apply_overrides(raw, overrides)

    try:
        return AppConfig(**raw)
    except ValidationError as e:
        lines = ["Config validation failed:"]
        for err in e.errors():
            field = " > ".join(str(loc) for loc in err["loc"])
            lines.append(f"  {field}: {err['msg']}")
        raise SystemExit("\n".join(lines)) from None
