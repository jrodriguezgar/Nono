---
name: extracting-data
description: >
  Extract structured data from unstructured text.
  Use when the user asks to extract entities, fields, values,
  or structured information from text, documents, or logs.
license: MIT
metadata:
  author: DatamanEdge
  version: "1.0.0"
tags:
  - text
  - extraction
  - data
version: "1.0.0"
temperature: 0.1
output_format: json
---

# Extract Structured Data

Extract structured information from unstructured text and return
it in a clean JSON format.

## Output format

See [assets/output_schema.json](assets/output_schema.json) for the full JSON schema.

```json
{
  "entities": [
    {"text": "extracted value", "type": "entity_type", "context": "surrounding text"}
  ],
  "summary": "Brief description of what was extracted"
}
```

## Guidelines

- Extract all relevant entities: names, dates, numbers, locations.
- Use `null` for fields that cannot be determined from the text.
- Preserve original values — do not paraphrase or interpret.
- Group related entities when possible.

## Additional resources

- For supported entity types and extraction guidelines, see [references/REFERENCE.md](references/REFERENCE.md)
