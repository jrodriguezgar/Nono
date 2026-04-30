---
name: explaining-code
description: >
  Explain code in clear, plain language with examples.
  Use when the user asks to explain, describe, walk through,
  or help understand a piece of code.
license: MIT
metadata:
  author: DatamanEdge
  version: "1.0.0"
tags:
  - code
  - education
  - explanation
version: "1.0.0"
temperature: 0.3
---

# Explain Code

Break down code into clear, understandable explanations for any
audience level.

## Explanation levels

| Level | Audience | Approach |
|---|---|---|
| **Beginner** | New to programming | Analogies, no jargon, step-by-step |
| **Intermediate** | Knows basics | Focus on patterns and "why" |
| **Advanced** | Experienced dev | Performance, edge cases, internals |

Default to **intermediate** unless the user specifies otherwise.

## Guidelines

1. **Start with purpose**: Always begin with *what* the code does
   and *why* it exists — before explaining *how*.
2. **Top-down**: Explain the overall flow first, then drill into details.
3. **Annotate**: Reference specific line numbers or sections.
4. **Jargon**: Define technical terms on first use.
5. **Examples**: Show input/output examples to illustrate behavior.
6. **Pitfalls**: Mention common mistakes or gotchas related to the code.

## Output format

```markdown
## Purpose

[One-sentence summary of what the code does and why.]

## How It Works

### Overview
[High-level flow described in 2-3 sentences.]

### Step-by-Step

1. **[Step name]** (lines X-Y): What happens and why.
2. **[Step name]** (lines X-Y): What happens and why.
3. ...

## Key Concepts

- **[Concept]**: Brief definition relevant to this code.

## Example

**Input**: `function_call("example")`
**Output**: `"expected result"`

## Gotchas

- [Common mistake or subtle behavior to watch for.]
```

## Example

**Input**: Explain this code:
```python
def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper
```

**Output**:
```markdown
## Purpose

A decorator that caches function results to avoid repeated computation
(memoization pattern).

## How It Works

### Overview
`memoize` wraps any function so that when called with the same arguments
twice, the second call returns the cached result instantly instead of
re-executing the function.

### Step-by-Step

1. **Cache initialization** (line 2): Creates an empty dictionary `cache`
   that persists across calls via closure.
2. **Wrapper function** (lines 3-6): Intercepts every call to the
   original function.
3. **Cache lookup** (line 4): Checks if the arguments have been seen before.
4. **Compute and store** (line 5): If not cached, calls the original
   function and stores the result.
5. **Return** (line 6): Returns the cached (or just-computed) result.

## Key Concepts

- **Closure**: `wrapper` captures `cache` from the enclosing scope.
- **Decorator**: `@memoize` syntax replaces a function with the wrapper.

## Gotchas

- Only works with **hashable** arguments (no lists or dicts as params).
- Cache grows unboundedly — consider `functools.lru_cache` for size limits.
```
