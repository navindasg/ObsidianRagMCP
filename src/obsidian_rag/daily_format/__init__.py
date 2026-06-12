"""Nightly formatting of raw Obsidian daily notes.

A scheduled job reads raw daily notes (files named after their date, e.g.
``2026-06-12.md``), asks a local Ollama chat model for suggested tags and a
cleaned-up markdown body, then assembles the final file in code: YAML
frontmatter (tags, date, formatted timestamp) + formatted body + a verbatim
"## Original Notes" section. A persistent JSON queue survives sleep and
failures, and a launchd LaunchAgent schedules the nightly runs.
"""
