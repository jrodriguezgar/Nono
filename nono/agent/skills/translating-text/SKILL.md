---
name: translating-text
description: >
  Translate text between languages accurately.
  Use when the user asks to translate content, convert text
  to another language, or localize content.
license: MIT
metadata:
  author: DatamanEdge
  version: "1.0.0"
tags:
  - text
  - translation
  - language
version: "1.0.0"
temperature: 0.3
---

# Translate Text

Translate text accurately while preserving the original meaning,
tone, and style.

## Guidelines

- Maintain technical terms when appropriate.
- Preserve formatting (bullet points, paragraphs, emphasis).
- Use natural phrasing in the target language — avoid literal translations.
- If the source language is ambiguous, auto-detect it.
- Default target language: English.

Return **only** the translated text, nothing else.

## Additional resources

- For language codes and translation quality guidelines, see [references/REFERENCE.md](references/REFERENCE.md)
