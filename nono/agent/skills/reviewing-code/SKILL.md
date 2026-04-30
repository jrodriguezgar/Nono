---
name: reviewing-code
description: >
  Review code for quality, security, and best practices.
  Use when the user asks to review, audit, or analyze code
  for bugs, security issues, performance, or style.
license: MIT
metadata:
  author: DatamanEdge
  version: "1.0.0"
tags:
  - code
  - review
  - security
version: "1.0.0"
temperature: 0.2
output_format: json
tools:
  - name: score_review
    script: scripts/score.py
    description: Compute a numeric quality score from review issues.
---

# Review Code

Analyze code for quality, correctness, performance, security,
and best practices.

## Review checklist

1. **Correctness**: logic errors, edge cases, off-by-one
2. **Security**: injection, path traversal, credential leaks (OWASP Top 10)
3. **Performance**: unnecessary allocations, O(n²) where O(n) is possible
4. **Style**: naming, structure, DRY violations
5. **Best practices**: error handling, type hints, documentation

## Output format

See [assets/report_schema.json](assets/report_schema.json) for the full JSON schema.

```json
{
  "overall_score": 8,
  "issues": [
    {
      "severity": "high|medium|low",
      "line": 42,
      "description": "Issue description",
      "suggestion": "How to fix"
    }
  ],
  "strengths": ["Well-structured code", "Good error handling"],
  "summary": "Brief overall assessment"
}
```

## Additional resources

- For OWASP Top 10 checklist and severity levels, see [references/REFERENCE.md](references/REFERENCE.md)
