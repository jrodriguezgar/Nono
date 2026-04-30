# Data Extraction Reference

## Supported Entity Types

| Type | Examples | Notes |
|------|----------|-------|
| `person` | John Smith, María López | First + last name, honorifics |
| `organization` | OpenAI, United Nations | Companies, institutions, agencies |
| `location` | Paris, 123 Main St | Cities, countries, addresses |
| `date` | 2024-01-15, January 15th | ISO 8601 preferred in output |
| `time` | 14:30, 2:30 PM | 24-hour preferred in output |
| `money` | $1,500, €200.00 | Include currency code |
| `percentage` | 15%, 0.15 | Normalize to percentage format |
| `email` | user@example.com | Validate format |
| `phone` | +1-555-0123 | Include country code when available |
| `url` | https://example.com | Full URL with protocol |
| `code` | ERR-404, PO-12345 | Product codes, error codes, IDs |
| `quantity` | 42 units, 3.5 kg | Include unit of measurement |

## Extraction Guidelines

- **Be exhaustive**: Extract all instances, not just the first occurrence.
- **Preserve original text**: Use `null` for values that cannot be determined.
- **Provide context**: Include surrounding text that gives meaning to the entity.
- **Handle ambiguity**: When a value could be multiple types, prefer the most specific.
- **Normalize dates**: Convert to ISO 8601 when possible (keep original in `text` field).
