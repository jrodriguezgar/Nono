---
name: classifying-data
description: >
  Classify text into categories with confidence scores.
  Use when the user asks to classify, categorize, label, tag,
  or route content based on its type or topic.
license: MIT
metadata:
  author: DatamanEdge
  version: "1.0.0"
tags:
  - text
  - classification
  - routing
version: "1.0.0"
temperature: 0.1
output_format: json
---

# Classify Data

Analyze inputs and determine the most appropriate category based on
content, intent, and context.

## Output format

See [assets/output_schema.json](assets/output_schema.json) for the full JSON schema.

Always respond in JSON:

```json
{
  "classification": "primary_category",
  "confidence": 0.95,
  "secondary_labels": ["optional", "labels"],
  "reasoning": "Brief explanation"
}
```

## Guidelines

- Assign exactly one primary classification.
- Include confidence score between 0.0 and 1.0.
- Add secondary labels only when clearly applicable.
- Keep reasoning concise (one sentence).

## Additional resources

- For confidence scoring guide and common domains, see [references/REFERENCE.md](references/REFERENCE.md)
