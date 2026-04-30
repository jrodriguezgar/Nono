---
name: converting-formats
description: >
  Convert data between structured formats (JSON, YAML, CSV, XML, TOML).
  Use when the user asks to convert, transform, or translate data
  from one format to another.
license: MIT
metadata:
  author: DatamanEdge
  version: "1.0.0"
tags:
  - data
  - conversion
  - utility
  - formats
version: "1.0.0"
temperature: 0.0
---

# Convert Formats

Convert data accurately between structured formats while preserving
data types, structure, and semantics.

## Supported formats

| Format | Extensions | Notes |
|---|---|---|
| JSON | `.json` | Default target format |
| YAML | `.yaml`, `.yml` | Preserves comments when possible |
| CSV | `.csv` | Flat tabular data only |
| XML | `.xml` | Includes attributes and namespaces |
| TOML | `.toml` | Configuration files |
| INI | `.ini` | Simple key-value sections |

## Guidelines

1. **Preserve data types**: Numbers stay numbers, booleans stay booleans.
   Don't stringify values unnecessarily.
2. **Handle nested structures**: When converting to flat formats (CSV),
   use dot notation (`parent.child`) or flatten with clear column names.
3. **Encoding**: Always output UTF-8.
4. **Validation**: Verify the output is valid in the target format.
5. **Edge cases**: Handle empty values, arrays, null/None, special
   characters, and multiline strings correctly.

## Conversion rules

### JSON → YAML
- Objects become mappings
- Arrays become sequences
- `null` → `null` (YAML native)

### JSON → CSV
- Only works for arrays of flat objects
- Nested objects: flatten with dot notation
- Arrays inside objects: join with `|` separator

### JSON → XML
- Root element required (use `<root>` if not specified)
- Arrays become repeated elements
- Attributes: use `@attr` convention

### CSV → JSON
- First row = headers (column names)
- Auto-detect types: numbers, booleans, dates
- Empty cells → `null`

## Output format

Return the converted data directly in the target format, with a brief
comment header noting the source and target formats:

```
# Converted from JSON to YAML
# Source: input.json
---
key: value
items:
  - name: first
    count: 42
```

## Example

**Input**: Convert this JSON to YAML:
```json
{"name": "Nono", "version": "1.0.0", "features": ["agents", "skills", "workflows"]}
```

**Output**:
```yaml
# Converted from JSON to YAML
---
name: Nono
version: "1.0.0"
features:
  - agents
  - skills
  - workflows
```
