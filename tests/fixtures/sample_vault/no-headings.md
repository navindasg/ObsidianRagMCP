---
tags:
  - note
---

This is a note without any headings. It contains multiple paragraphs of text that should be chunked using the recursive text splitter since there are no heading boundaries to split on.

Second paragraph with more content. This tests the fallback behavior when no markdown headings are present in the document.

Third paragraph ensures we have enough content to potentially trigger splitting if the chunk_max_tokens is set low enough for testing.
