---
name: researching-topics
description: >
  Research a topic and synthesize findings into a structured report.
  Use when the user asks to research, investigate, explore, or
  compile information about a subject.
license: MIT
metadata:
  author: DatamanEdge
  version: "1.0.0"
tags:
  - research
  - knowledge
  - synthesis
  - analysis
version: "1.0.0"
temperature: 0.4
---

# Research Topics

Investigate a topic in depth and synthesize findings into a clear,
structured, and actionable report.

## Research approach

1. **Scope**: Define the boundaries of the topic clearly.
2. **Sources**: Use available knowledge; cite when possible.
3. **Structure**: Organize from broad overview to specific details.
4. **Balance**: Present multiple perspectives for debatable topics.
5. **Actionability**: End with concrete takeaways or next steps.

## Guidelines

- Start with a one-paragraph executive summary.
- Use headers to organize sections logically.
- Include data, numbers, and examples to support claims.
- Distinguish between established facts and opinions/predictions.
- Flag areas of uncertainty with "Note:" callouts.
- For technology topics, include version dates and maturity level.

## Output format

```markdown
## Executive Summary

[One paragraph capturing the main finding and recommendation]

## Background

[Context and history — why this topic matters]

## Key Findings

### Finding 1: [Title]
[Details, evidence, examples]

### Finding 2: [Title]
[Details, evidence, examples]

### Finding 3: [Title]
[Details, evidence, examples]

## Comparison (if applicable)

| Aspect | Option A | Option B | Option C |
|---|---|---|---|
| Feature 1 | ✅ | ❌ | ✅ |
| Feature 2 | ❌ | ✅ | ✅ |

## Risks and Considerations

- Risk 1: Description and mitigation
- Risk 2: Description and mitigation

## Recommendations

1. Primary recommendation with justification
2. Alternative approach if primary is not feasible

## Further Reading

- Reference 1
- Reference 2
```

## Example

**Input**: "Research the current state of WebAssembly for server-side applications"

**Output** (abbreviated):
```markdown
## Executive Summary

WebAssembly (Wasm) has matured beyond browser use with WASI enabling
server-side workloads. Key runtimes (Wasmtime, WasmEdge, Wasmer) now
support production deployments for edge computing and plugin systems.

## Key Findings

### 1. WASI standard is stabilizing
The WebAssembly System Interface reached Preview 2 with component model
support, enabling portable server-side modules...

### 2. Edge computing is the primary adoption vector
Cloudflare Workers, Fastly Compute, and Fermyon Spin run Wasm at the edge
with sub-millisecond cold starts...

## Recommendations

1. Evaluate Wasm for plugin/extension systems where sandboxing is critical.
2. Consider edge-Wasm for latency-sensitive endpoints.
```
