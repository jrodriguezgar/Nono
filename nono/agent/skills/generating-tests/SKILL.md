---
name: generating-tests
description: >
  Generate unit tests and test cases for code.
  Use when the user asks to write tests, create test cases,
  generate test suites, or add test coverage for functions or classes.
license: MIT
metadata:
  author: DatamanEdge
  version: "1.0.0"
tags:
  - code
  - testing
  - quality
version: "1.0.0"
temperature: 0.2
---

# Generate Tests

Create comprehensive, idiomatic unit tests for code following
best practices and the project's testing conventions.

## Supported frameworks

| Language | Framework | Style |
|---|---|---|
| Python | **pytest** (default) | Functions with `test_` prefix |
| Python | unittest | Class-based with `setUp/tearDown` |
| JavaScript | Jest | `describe/it` blocks |
| TypeScript | Vitest / Jest | `describe/it` with types |

Default to **pytest** unless the user specifies otherwise.

## Test categories

1. **Happy path**: Normal inputs, expected behavior.
2. **Edge cases**: Empty inputs, boundary values, None/null.
3. **Error cases**: Invalid inputs, exceptions, error messages.
4. **Integration hints**: Interactions between components (mocked).

## Guidelines

1. **AAA pattern**: Arrange → Act → Assert in every test.
2. **One assertion per concept**: Each test verifies one behavior.
3. **Descriptive names**: `test_function_name_when_condition_then_result`.
4. **Fixtures**: Use `@pytest.fixture` for shared setup (Python).
5. **Mocking**: Mock external dependencies (APIs, databases, filesystem).
   Use `unittest.mock.patch` or `pytest-mock`.
6. **No network calls**: Tests must run offline and fast.
7. **Parametrize**: Use `@pytest.mark.parametrize` for similar test cases.

## Output format

```python
"""Tests for module_name."""

import pytest
from unittest.mock import patch, MagicMock

from module_name import function_to_test


class TestFunctionToTest:
    """Tests for function_to_test."""

    def test_returns_expected_result_for_valid_input(self):
        """Happy path: valid input produces expected output."""
        result = function_to_test("valid_input")
        assert result == "expected_output"

    def test_raises_value_error_for_empty_input(self):
        """Error case: empty string raises ValueError."""
        with pytest.raises(ValueError, match="Input cannot be empty"):
            function_to_test("")

    @pytest.mark.parametrize("input_val,expected", [
        ("a", 1),
        ("bb", 2),
        ("ccc", 3),
    ])
    def test_returns_length_for_various_strings(self, input_val, expected):
        """Parametrized: verify length computation."""
        assert function_to_test(input_val) == expected

    def test_handles_none_input_gracefully(self):
        """Edge case: None input returns default."""
        result = function_to_test(None)
        assert result is None
```

## Example

**Input**: Write tests for this function:
```python
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b
```

**Output**:
```python
"""Tests for divide function."""

import pytest

from math_utils import divide


class TestDivide:

    def test_divides_two_positive_numbers(self):
        assert divide(10, 2) == 5.0

    def test_divides_negative_numbers(self):
        assert divide(-10, 2) == -5.0

    def test_returns_float_for_integer_inputs(self):
        result = divide(7, 2)
        assert result == 3.5
        assert isinstance(result, float)

    def test_raises_zero_division_error(self):
        with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
            divide(10, 0)

    @pytest.mark.parametrize("a,b,expected", [
        (0, 5, 0.0),
        (1, 1, 1.0),
        (-6, -3, 2.0),
        (0.5, 0.25, 2.0),
    ])
    def test_various_division_cases(self, a, b, expected):
        assert divide(a, b) == pytest.approx(expected)
```
