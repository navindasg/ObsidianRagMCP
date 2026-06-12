"""Tests for the daily-note formatter core (daily_format/formatter.py).

Tests:
  1. happy path produces the exact document structure
  2. chat is called with FORMAT_SCHEMA, temperature 0.2, delimited note text
  3. existing frontmatter merges tags case-insensitively, preserves keys
  4. invalid JSON from the model -> FormatError, file untouched
  5. bad reply shapes -> FormatError (tags not list, non-str tag, empty body)
  6. original text always present even when the model drops content
  7. empty tags omits the tags key entirely
  8. atomicity: chat failure propagates and leaves the file unchanged
  9. truncation warning for huge notes; assembly still uses full original
 10. round-trip guard: assembled output is detected as already formatted
 11. hallucinated frontmatter in the model body is stripped
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from unittest.mock import MagicMock

import json

import pytest
import yaml

from obsidian_rag.daily_format.detector import is_already_formatted
from obsidian_rag.daily_format.formatter import (
    FORMAT_SCHEMA,
    MAX_PROMPT_CHARS,
    FormatError,
    assemble_note,
    format_file,
    format_with_model,
)

NOTE_DATE = datetime.date(2026, 6, 11)
NOW = datetime.datetime(2026, 6, 12, 2, 0, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(reply: str) -> MagicMock:
    """Mock ollama client whose chat() returns the given message content."""
    response = MagicMock()
    response.message.content = reply
    client = MagicMock()
    client.chat.return_value = response
    return client


def _json_reply(tags: list, body: str) -> str:
    return json.dumps({"tags": tags, "formatted_markdown": body})


def _frontmatter_of(document: str) -> dict:
    """Parse the leading YAML frontmatter block of an assembled document."""
    assert document.startswith("---\n")
    closing = document.index("\n---\n", 4)
    return yaml.safe_load(document[4:closing])


# ---------------------------------------------------------------------------
# Test 1: happy path — exact document structure
# ---------------------------------------------------------------------------


def test_happy_path_exact_document(tmp_path: Path) -> None:
    original = "- [ ] call [[Alice]]\nsome idea about the garden\n"
    note = tmp_path / "2026-06-11.md"
    note.write_text(original, encoding="utf-8")

    body = "## Tasks\n- [ ] call [[Alice]]\n\n## Ideas\nsome idea about the garden"
    client = _make_client(_json_reply(["work", "ideas"], body))

    format_file(
        note,
        client=client,
        model="llama3.2",
        tag_vocab=["work", "ideas"],
        note_date=NOTE_DATE,
        now=NOW,
    )

    expected = (
        "---\n"
        "tags:\n"
        "  - work\n"
        "  - ideas\n"
        "date: '2026-06-11'\n"
        "formatted: '2026-06-12T02:00:00'\n"
        "---\n"
        "\n"
        "## Tasks\n"
        "- [ ] call [[Alice]]\n"
        "\n"
        "## Ideas\n"
        "some idea about the garden\n"
        "\n"
        "---\n"
        "\n"
        "## Original Notes\n"
        "\n"
        "- [ ] call [[Alice]]\n"
        "some idea about the garden\n"
    )
    assert note.read_text(encoding="utf-8") == expected


# ---------------------------------------------------------------------------
# Test 2: chat call contract — schema, temperature, delimited data
# ---------------------------------------------------------------------------


def test_chat_called_with_schema_and_options() -> None:
    client = _make_client(_json_reply(["work"], "body text"))

    tags, body = format_with_model(
        client, "llama3.2", "raw note text", ["work", "ideas"]
    )

    assert tags == ["work"]
    assert body == "body text"
    kwargs = client.chat.call_args.kwargs
    assert kwargs["model"] == "llama3.2"
    assert kwargs["format"] == FORMAT_SCHEMA
    assert kwargs["options"] == {"temperature": 0.2}

    messages = kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    system = messages[0]["content"]
    user = messages[1]["content"]
    # Rules live in the system prompt.
    assert "- [ ]" in system and "- [x]" in system
    assert "[[" in system  # wikilinks rule
    assert "lowercase-kebab-case" in system
    # Note text is delimited as data in the user message, with the vocab.
    assert '"""\nraw note text\n"""' in user
    assert "EXISTING VAULT TAGS" in user
    assert "work, ideas" in user


# ---------------------------------------------------------------------------
# Test 2b: system prompt covers mixed-content classification
# ---------------------------------------------------------------------------


def test_system_prompt_covers_mixed_content_classification() -> None:
    """Dailies hold prompts, logins, and drafts — the model must be told.

    The prompt has to ask for classification under contextual headings
    (e.g. "## Draft: ..."), matching tags, and verbatim preservation of
    credentials, prompt text, code, URLs, and draft wording.
    """
    client = _make_client(_json_reply(["work"], "body text"))

    format_with_model(client, "llama3.2", "raw note text", ["work"])

    system = client.chat.call_args.kwargs["messages"][0]["content"]
    lowered = system.lower()
    # The mixed content types the model must classify.
    for content_type in ("draft", "prompt", "login"):
        assert content_type in lowered
    # Contextual headings naming type and subject.
    assert "## Draft:" in system
    # Verbatim preservation of sensitive/fragile content.
    assert "verbatim" in lowered
    for preserved in ("credential", "code", "url"):
        assert preserved in lowered


# ---------------------------------------------------------------------------
# Test 2c: dateless (tagged, non-daily) notes omit the date key
# ---------------------------------------------------------------------------


def test_dateless_note_omits_date_key() -> None:
    """Tag-triggered non-daily notes have no date; the key is omitted."""
    document = assemble_note("raw text\n", "## Notes\nraw text", ["idea"], None, NOW)

    frontmatter = _frontmatter_of(document)
    assert "date" not in frontmatter
    assert frontmatter["formatted"] == "2026-06-12T02:00:00"
    assert is_already_formatted(document)


# ---------------------------------------------------------------------------
# Test 3: existing frontmatter — tag union, preserved keys, moved block
# ---------------------------------------------------------------------------


def test_existing_frontmatter_merges_tags_and_preserves_keys() -> None:
    original = (
        "---\n"
        "tags:\n"
        "  - Work\n"
        "  - personal\n"
        "mood: happy\n"
        "---\n"
        "body line one\nbody line two\n"
    )

    document = assemble_note(
        original, "formatted body", ["work", "new-tag"], NOTE_DATE, NOW
    )

    frontmatter = _frontmatter_of(document)
    # Union: existing first (original casing), then new non-duplicate tags.
    assert frontmatter["tags"] == ["Work", "personal", "new-tag"]
    # Other existing keys preserved; ours win for date/formatted.
    assert frontmatter["mood"] == "happy"
    assert frontmatter["date"] == "2026-06-11"
    assert frontmatter["formatted"] == "2026-06-12T02:00:00"
    # The frontmatter block moved out of Original Notes; body kept verbatim.
    original_section = document.split("## Original Notes\n\n", 1)[1]
    assert original_section == "body line one\nbody line two\n"
    assert "mood: happy" not in original_section


# ---------------------------------------------------------------------------
# Test 4: invalid JSON -> FormatError, file untouched
# ---------------------------------------------------------------------------


def test_invalid_json_raises_and_leaves_file_untouched(tmp_path: Path) -> None:
    original = "raw daily note\n"
    note = tmp_path / "2026-06-11.md"
    note.write_text(original, encoding="utf-8")

    client = _make_client("this is not json {")

    with pytest.raises(FormatError):
        format_file(
            note,
            client=client,
            model="llama3.2",
            tag_vocab=[],
            note_date=NOTE_DATE,
            now=NOW,
        )

    assert note.read_text(encoding="utf-8") == original
    # No stray temp files left behind.
    assert list(tmp_path.iterdir()) == [note]


# ---------------------------------------------------------------------------
# Test 5: bad reply shapes -> FormatError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        json.dumps({"tags": "work", "formatted_markdown": "body"}),  # tags not list
        json.dumps({"tags": ["ok", 7], "formatted_markdown": "body"}),  # non-str tag
        json.dumps({"tags": ["work"], "formatted_markdown": ""}),  # empty body
        json.dumps({"tags": ["work"], "formatted_markdown": "   \n"}),  # blank body
        json.dumps({"tags": ["work"]}),  # missing body
        json.dumps(["not", "an", "object"]),  # not a dict
    ],
)
def test_bad_reply_shape_raises_format_error(reply: str) -> None:
    client = _make_client(reply)
    with pytest.raises(FormatError):
        format_with_model(client, "llama3.2", "note", [])


def test_none_content_raises_format_error() -> None:
    client = _make_client(None)  # type: ignore[arg-type]
    with pytest.raises(FormatError):
        format_with_model(client, "llama3.2", "note", [])


# ---------------------------------------------------------------------------
# Test 6: original always appended even when the model drops content
# ---------------------------------------------------------------------------


def test_original_always_present_when_model_drops_content(tmp_path: Path) -> None:
    original = "unique-marker-alpha\nunique-marker-beta\n"
    note = tmp_path / "2026-06-11.md"
    note.write_text(original, encoding="utf-8")

    client = _make_client(_json_reply([], "the model dropped everything"))

    format_file(
        note,
        client=client,
        model="llama3.2",
        tag_vocab=[],
        note_date=NOTE_DATE,
        now=NOW,
    )

    result = note.read_text(encoding="utf-8")
    assert "unique-marker-alpha" in result
    assert "unique-marker-beta" in result
    assert result.endswith(original)


# ---------------------------------------------------------------------------
# Test 7: empty tags omits the tags key
# ---------------------------------------------------------------------------


def test_empty_tags_omits_tags_key() -> None:
    document = assemble_note("raw note\n", "formatted body", [], NOTE_DATE, NOW)

    frontmatter = _frontmatter_of(document)
    assert "tags" not in frontmatter
    assert "tags:" not in document.split("\n---\n", 1)[0]
    assert frontmatter["date"] == "2026-06-11"
    assert frontmatter["formatted"] == "2026-06-12T02:00:00"


# ---------------------------------------------------------------------------
# Test 8: atomicity — chat failure propagates, file unchanged
# ---------------------------------------------------------------------------


def test_chat_failure_leaves_file_unchanged(tmp_path: Path) -> None:
    original = "raw daily note\n"
    note = tmp_path / "2026-06-11.md"
    note.write_text(original, encoding="utf-8")

    client = MagicMock()
    client.chat.side_effect = ConnectionError("Connection refused")

    with pytest.raises(ConnectionError):
        format_file(
            note,
            client=client,
            model="llama3.2",
            tag_vocab=[],
            note_date=NOTE_DATE,
            now=NOW,
        )

    assert note.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [note]


def test_unreadable_file_raises_format_error(tmp_path: Path) -> None:
    missing = tmp_path / "2026-06-11.md"
    client = MagicMock()

    with pytest.raises(FormatError):
        format_file(
            missing,
            client=client,
            model="llama3.2",
            tag_vocab=[],
            note_date=NOTE_DATE,
            now=NOW,
        )
    client.chat.assert_not_called()


# ---------------------------------------------------------------------------
# Test 9: truncation warning for huge notes; full original still assembled
# ---------------------------------------------------------------------------


def test_truncation_warns_and_keeps_full_original(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    huge = ("x" * 100 + "\n") * 300  # 30300 chars > MAX_PROMPT_CHARS
    original = huge + "FINAL-SENTINEL\n"
    note = tmp_path / "2026-06-11.md"
    note.write_text(original, encoding="utf-8")

    client = _make_client(_json_reply(["work"], "formatted body"))

    with caplog.at_level(logging.WARNING, logger="obsidian_rag.daily_format.formatter"):
        format_file(
            note,
            client=client,
            model="llama3.2",
            tag_vocab=[],
            note_date=NOTE_DATE,
            now=NOW,
        )

    assert any("runcat" in record.message for record in caplog.records), (
        "Expected a truncation warning"
    )
    # Prompt copy was capped: sentinel (at the very end) never reached the model.
    user_content = client.chat.call_args.kwargs["messages"][1]["content"]
    assert "FINAL-SENTINEL" not in user_content
    assert len(user_content) < len(original)
    # Assembly used the FULL original, sentinel included.
    assert "FINAL-SENTINEL" in note.read_text(encoding="utf-8")


def test_short_note_not_truncated_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _make_client(_json_reply([], "body"))
    text = "short note"
    assert len(text) < MAX_PROMPT_CHARS

    with caplog.at_level(logging.WARNING, logger="obsidian_rag.daily_format.formatter"):
        format_with_model(client, "llama3.2", text, [])

    assert not any("runcat" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Test 10: round-trip guard — assembled output is detected as formatted
# ---------------------------------------------------------------------------


def test_assembled_output_detected_as_already_formatted() -> None:
    document = assemble_note(
        "raw note\n", "formatted body", ["work"], NOTE_DATE, NOW
    )
    assert is_already_formatted(document) is True


# ---------------------------------------------------------------------------
# Test 11: hallucinated model frontmatter is stripped from the body
# ---------------------------------------------------------------------------


def test_model_hallucinated_frontmatter_stripped() -> None:
    body = "---\ntags: [sneaky]\n---\nactual formatted body"
    document = assemble_note("raw note\n", body, ["work"], NOTE_DATE, NOW)

    frontmatter = _frontmatter_of(document)
    assert frontmatter["tags"] == ["work"]
    assert "sneaky" not in document
    formatted_section = document.split("---\n\n", 1)[1]
    assert formatted_section.startswith("actual formatted body")


# ---------------------------------------------------------------------------
# Thinking models and lenient reply parsing (gemma4-mlx compatibility)
# ---------------------------------------------------------------------------


def test_chat_disables_thinking() -> None:
    """Thinking models must answer in content, not think forever."""
    client = _make_client(_json_reply(["work"], "body"))

    format_with_model(client, "gemma4:26b-mlx", "raw", [])

    assert client.chat.call_args.kwargs["think"] is False


def test_think_unsupported_retries_without() -> None:
    """Models without a thinking toggle get a second call sans think."""
    import ollama as ollama_pkg

    good = MagicMock()
    good.message.content = _json_reply(["work"], "body")
    client = MagicMock()
    client.chat.side_effect = [
        ollama_pkg.ResponseError('"llama3.2" does not support thinking'),
        good,
    ]

    tags, body = format_with_model(client, "llama3.2", "raw", [])

    assert (tags, body) == (["work"], "body")
    assert client.chat.call_count == 2
    assert "think" not in client.chat.call_args.kwargs


def test_fenced_json_reply_parsed() -> None:
    """MLX models ignore the schema and fence their JSON; parse it anyway."""
    fenced = f'```json\n{_json_reply(["work"], "## Body")}\n```'
    client = _make_client(fenced)

    tags, body = format_with_model(client, "gemma4:26b-mlx", "raw", [])

    assert tags == ["work"]
    assert body == "## Body"


def test_json_with_prose_around_it_parsed() -> None:
    """Prose before/after the JSON object is tolerated."""
    reply = f'Here is the JSON you asked for:\n{_json_reply([], "## B")}\nHope it helps!'
    client = _make_client(reply)

    tags, body = format_with_model(client, "m", "raw", [])

    assert body == "## B"


def test_empty_reply_mentions_thinking_budget() -> None:
    """An empty reply gets a diagnosis, not just 'invalid JSON'."""
    client = _make_client("")

    with pytest.raises(FormatError, match="empty"):
        format_with_model(client, "m", "raw", [])


def test_format_file_logs_duration(tmp_path: Path, caplog) -> None:
    """Per-note INFO log carries elapsed seconds for throughput tailing."""
    import re as re_mod

    note = tmp_path / "2026-06-11.md"
    note.write_text("raw\n", encoding="utf-8")
    client = _make_client(_json_reply(["t"], "## B"))

    with caplog.at_level(logging.INFO, logger="obsidian_rag.daily_format.formatter"):
        format_file(
            note, client=client, model="m", tag_vocab=[], note_date=NOTE_DATE, now=NOW
        )

    assert any(
        re_mod.search(r"Formatted .+ in \d+\.\d+s", record.message)
        for record in caplog.records
    )


def test_json_with_trailing_second_object_parsed() -> None:
    """A valid object followed by extra data parses to the first object."""
    reply = _json_reply(["work"], "## B") + '\n{"stray": true}\ntrailing prose'
    client = _make_client(reply)

    tags, body = format_with_model(client, "m", "raw", [])

    assert (tags, body) == (["work"], "## B")
