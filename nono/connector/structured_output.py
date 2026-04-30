"""Structured output parsing, validation, and retry for LLM responses.

Provides Pydantic-model-based structured output across all providers,
reusable output parsers (JSON, regex, CSV), and automatic retry when
the LLM returns malformed output.

Example:
    >>> from pydantic import BaseModel
    >>> from nono.connector.structured_output import StructuredGenerator
    >>>
    >>> class Sentiment(BaseModel):
    ...     label: str
    ...     score: float
    ...
    >>> gen = StructuredGenerator(service, model=Sentiment)
    >>> result = gen.generate([{"role": "user", "content": "Analyze: Great!"}])
    >>> result.label
    'positive'
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raised when an output parser cannot parse the LLM response."""

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


class MaxRetriesExceededError(Exception):
    """Raised when all retry attempts fail to produce valid output."""

    def __init__(self, attempts: int, last_error: Exception, raw: str) -> None:
        super().__init__(
            f"Failed to produce valid output after {attempts} attempts: {last_error}"
        )
        self.attempts = attempts
        self.last_error = last_error
        self.raw = raw


# ---------------------------------------------------------------------------
# Output Parsers
# ---------------------------------------------------------------------------

class OutputParser(ABC, Generic[T]):
    """Abstract base for all output parsers.

    Subclasses implement ``parse`` to convert raw LLM text into a typed
    result, raising ``ParseError`` on failure.
    """

    @abstractmethod
    def parse(self, text: str) -> T:
        """Parse raw LLM output into the target type.

        Args:
            text: Raw string from the LLM.

        Returns:
            Parsed result.

        Raises:
            ParseError: When *text* cannot be parsed.
        """

    @abstractmethod
    def format_instructions(self) -> str:
        """Return instructions appended to the prompt so the LLM produces
        output in the expected format.
        """

    def repair_prompt(self, raw: str, error: Exception) -> str:
        """Build a retry prompt asking the LLM to fix its output.

        Args:
            raw: The malformed LLM response.
            error: The parse/validation error.

        Returns:
            A user-message string to send as a follow-up.
        """
        return (
            f"Your previous response could not be parsed.\n"
            f"Error: {error}\n\n"
            f"Original response (first 500 chars):\n"
            f"{raw[:500]}\n\n"
            f"Please respond again following these instructions exactly:\n"
            f"{self.format_instructions()}"
        )


class JsonOutputParser(OutputParser[dict]):
    """Parse LLM output as JSON, optionally validating against a schema.

    Args:
        schema: Optional JSON Schema dict.  When provided, the parsed
            dict is validated with ``jsonschema.validate``.
    """

    def __init__(self, schema: dict | None = None) -> None:
        self.schema = schema

    def parse(self, text: str) -> dict:
        """Parse JSON from text, extracting from code fences if needed.

        Args:
            text: Raw LLM response.

        Returns:
            Parsed dict.

        Raises:
            ParseError: On invalid JSON or schema mismatch.
        """
        cleaned = _extract_json(text)

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ParseError(f"Invalid JSON: {exc}", text) from exc

        if not isinstance(result, dict):
            raise ParseError(
                f"Expected a JSON object, got {type(result).__name__}", text,
            )

        if self.schema is not None:
            self._validate_schema(result, text)

        return result

    def format_instructions(self) -> str:
        base = "Respond with valid JSON only. No markdown, no explanation."

        if self.schema:
            try:
                schema_str = json.dumps(self.schema, indent=2)
                return (
                    f"{base}\n"
                    f"Your response must conform to this JSON Schema:\n"
                    f"```json\n{schema_str}\n```"
                )
            except TypeError:
                pass

        return base

    # -- internal --

    def _validate_schema(self, data: dict, raw: str) -> None:
        try:
            import jsonschema  # noqa: delayed import — optional dep
        except ImportError:
            logger.debug("jsonschema not installed; skipping schema validation")
            return

        try:
            jsonschema.validate(instance=data, schema=self.schema)
        except jsonschema.ValidationError as exc:
            raise ParseError(
                f"JSON schema validation failed: {exc.message}", raw,
            ) from exc


class PydanticOutputParser(OutputParser[Any]):
    """Parse LLM output into a Pydantic model instance.

    Args:
        model: A Pydantic ``BaseModel`` subclass.

    Example:
        >>> class City(BaseModel):
        ...     name: str
        ...     population: int
        >>> parser = PydanticOutputParser(City)
        >>> parser.parse('{"name": "Madrid", "population": 3300000}')
        City(name='Madrid', population=3300000)
    """

    def __init__(self, model: type) -> None:
        self.model = model
        self._schema: dict = self._build_schema()

    def parse(self, text: str) -> Any:
        """Parse and validate text into a Pydantic model.

        Args:
            text: Raw LLM response.

        Returns:
            Validated Pydantic model instance.

        Raises:
            ParseError: On invalid JSON or validation failure.
        """
        cleaned = _extract_json(text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ParseError(f"Invalid JSON: {exc}", text) from exc

        try:
            return self.model.model_validate(data)
        except Exception as exc:
            raise ParseError(
                f"Pydantic validation failed: {exc}", text,
            ) from exc

    def format_instructions(self) -> str:
        schema_str = json.dumps(self._schema, indent=2)
        return (
            f"Respond with valid JSON only. No markdown, no explanation.\n"
            f"Your response must conform to this JSON Schema:\n"
            f"```json\n{schema_str}\n```"
        )

    def json_schema(self) -> dict:
        """Return the JSON Schema dict derived from the Pydantic model."""
        return self._schema

    # -- internal --

    def _build_schema(self) -> dict:
        try:
            return self.model.model_json_schema()
        except AttributeError:
            # Pydantic v1 fallback
            return self.model.schema()  # type: ignore[union-attr]


class RegexOutputParser(OutputParser[str]):
    """Extract text matching a regex pattern from the LLM response.

    Args:
        pattern: Regex pattern with at least one capturing group.
        group: Which group to return (default 1).
        description: Human-readable description of the expected format
            (used in ``format_instructions``).
    """

    def __init__(
        self,
        pattern: str,
        group: int = 1,
        description: str = "",
    ) -> None:
        self._pattern = re.compile(pattern)
        self._group = group
        self._description = description or f"Text matching pattern: {pattern}"

    def parse(self, text: str) -> str:
        """Extract the first match from text.

        Args:
            text: Raw LLM response.

        Returns:
            Captured group string.

        Raises:
            ParseError: When no match is found.
        """
        match = self._pattern.search(text)

        if match is None:
            raise ParseError(
                f"No match for pattern {self._pattern.pattern!r}", text,
            )

        try:
            return match.group(self._group)
        except IndexError as exc:
            raise ParseError(
                f"Group {self._group} not found in match", text,
            ) from exc

    def format_instructions(self) -> str:
        return self._description


class CsvOutputParser(OutputParser[list[dict[str, str]]]):
    """Parse LLM output as CSV with a header row.

    Args:
        delimiter: Column delimiter (default ``','``).
        expected_columns: Optional list of required column names.
    """

    def __init__(
        self,
        delimiter: str = ",",
        expected_columns: list[str] | None = None,
    ) -> None:
        self._delimiter = delimiter
        self._expected_columns = expected_columns

    def parse(self, text: str) -> list[dict[str, str]]:
        """Parse CSV text into a list of row dicts.

        Args:
            text: Raw LLM response.

        Returns:
            List of dicts keyed by header columns.

        Raises:
            ParseError: On empty data or missing columns.
        """
        cleaned = _extract_code_block(text, "csv") or text.strip()

        try:
            reader = csv.DictReader(
                io.StringIO(cleaned), delimiter=self._delimiter,
            )
            rows = list(reader)
        except csv.Error as exc:
            raise ParseError(f"CSV parse error: {exc}", text) from exc

        if not rows:
            raise ParseError("CSV output is empty (no data rows)", text)

        if self._expected_columns:
            actual = set(rows[0].keys())
            missing = set(self._expected_columns) - actual

            if missing:
                raise ParseError(
                    f"Missing CSV columns: {sorted(missing)}", text,
                )

        return rows

    def format_instructions(self) -> str:
        base = (
            "Respond with CSV data only. "
            "The first line must be the header row. "
            "No markdown, no explanation."
        )

        if self._expected_columns:
            cols = self._delimiter.join(self._expected_columns)
            return f"{base}\nRequired columns: {cols}"

        return base


# ---------------------------------------------------------------------------
# Structured Generator (validation + retry)
# ---------------------------------------------------------------------------

class StructuredGenerator(Generic[T]):
    """Generate structured LLM output with automatic validation and retry.

    Wraps any ``GenerativeAIService`` and an ``OutputParser`` to:
    1. Inject format instructions into the prompt.
    2. Parse the raw response.
    3. On parse failure, send a repair prompt and retry.

    Args:
        service: A Nono ``GenerativeAIService`` (any provider).
        parser: An ``OutputParser`` instance. When *model* is given,
            a ``PydanticOutputParser`` is created automatically.
        model: A Pydantic ``BaseModel`` subclass — shortcut for
            ``PydanticOutputParser(model)``.
        max_retries: Maximum retry attempts on parse failure (default 2).
        inject_instructions: Whether to append format instructions
            to the last user message (default ``True``).

    Example:
        >>> from pydantic import BaseModel
        >>> class Movie(BaseModel):
        ...     title: str
        ...     year: int
        >>> gen = StructuredGenerator(service, model=Movie, max_retries=3)
        >>> movie = gen.generate([
        ...     {"role": "user", "content": "Best sci-fi film?"}
        ... ])
        >>> movie.title
        'Blade Runner'
    """

    def __init__(
        self,
        service: Any,
        parser: OutputParser[T] | None = None,
        model: type | None = None,
        max_retries: int = 2,
        inject_instructions: bool = True,
    ) -> None:
        if parser is None and model is None:
            raise ValueError("Provide either 'parser' or 'model'.")

        if parser is not None:
            self.parser: OutputParser[T] = parser
        else:
            self.parser = PydanticOutputParser(model)  # type: ignore[arg-type]

        self.service = service
        self.max_retries = max_retries
        self.inject_instructions = inject_instructions

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | str = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> T:
        """Generate and parse structured output with retry.

        Args:
            messages: Chat messages in OpenAI format.
            temperature: LLM temperature.
            max_tokens: Optional token limit.
            **kwargs: Extra arguments forwarded to
                ``service.generate_completion``.

        Returns:
            Parsed and validated result of type *T*.

        Raises:
            MaxRetriesExceededError: When all retries are exhausted.
        """
        work_messages = self._inject(messages)
        response_format, json_schema = self._resolve_format_params()

        raw = ""
        last_error: Exception = ParseError("No attempt made", "")

        for attempt in range(1, self.max_retries + 2):  # 1 initial + retries
            raw = self.service.generate_completion(
                messages=work_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                json_schema=json_schema,
                **kwargs,
            )

            try:
                result = self.parser.parse(raw)
                logger.debug(
                    "Structured output parsed on attempt %d/%d",
                    attempt, self.max_retries + 1,
                )
                return result
            except ParseError as exc:
                last_error = exc
                logger.warning(
                    "Parse failed (attempt %d/%d): %s",
                    attempt, self.max_retries + 1, exc,
                )

                if attempt <= self.max_retries:
                    repair = self.parser.repair_prompt(raw, exc)
                    work_messages = work_messages + [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": repair},
                    ]

        raise MaxRetriesExceededError(
            attempts=self.max_retries + 1,
            last_error=last_error,
            raw=raw,
        )

    # -- internal helpers --

    def _inject(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Append format instructions to the last user message."""
        if not self.inject_instructions:
            return [m.copy() for m in messages]

        result = [m.copy() for m in messages]
        instructions = self.parser.format_instructions()

        for msg in reversed(result):
            if msg.get("role") == "user":
                msg["content"] += f"\n\n{instructions}"
                return result

        # No user message — prepend as system
        result.insert(0, {"role": "system", "content": instructions})
        return result

    def _resolve_format_params(self) -> tuple[Any, dict | None]:
        """Determine ResponseFormat and json_schema for the service call."""
        # Import here to avoid circular dependency
        try:
            from nono.connector.connector_genai import ResponseFormat
        except ImportError:
            from connector_genai import ResponseFormat  # type: ignore[no-redef]

        if isinstance(self.parser, (PydanticOutputParser, JsonOutputParser)):
            schema = None

            if isinstance(self.parser, PydanticOutputParser):
                schema = self.parser.json_schema()
            elif isinstance(self.parser, JsonOutputParser) and self.parser.schema:
                schema = self.parser.schema

            return ResponseFormat.JSON, schema

        return ResponseFormat.TEXT, None


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def parse_json(
    text: str,
    schema: dict | None = None,
) -> dict:
    """One-shot JSON parse with optional schema validation.

    Args:
        text: Raw LLM output.
        schema: Optional JSON Schema dict.

    Returns:
        Parsed dict.

    Raises:
        ParseError: On invalid JSON or schema mismatch.

    Example:
        >>> parse_json('{"name": "Nono", "version": 1}')
        {'name': 'Nono', 'version': 1}
    """
    return JsonOutputParser(schema).parse(text)


def parse_pydantic(text: str, model: type[T]) -> T:
    """One-shot parse into a Pydantic model.

    Args:
        text: Raw LLM output.
        model: Pydantic BaseModel subclass.

    Returns:
        Validated model instance.

    Raises:
        ParseError: On invalid JSON or validation failure.

    Example:
        >>> class Item(BaseModel):
        ...     name: str
        >>> parse_pydantic('{"name": "Widget"}', Item).name
        'Widget'
    """
    return PydanticOutputParser(model).parse(text)


def parse_csv(
    text: str,
    delimiter: str = ",",
    expected_columns: list[str] | None = None,
) -> list[dict[str, str]]:
    """One-shot CSV parse.

    Args:
        text: Raw LLM output.
        delimiter: Column delimiter.
        expected_columns: Required column names.

    Returns:
        List of row dicts.

    Raises:
        ParseError: On parse error or missing columns.
    """
    return CsvOutputParser(delimiter, expected_columns).parse(text)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n(.*?)\n\s*```", re.DOTALL,
)

_CODE_FENCE_RE = re.compile(
    r"```(?:{lang})?\s*\n(.*?)\n\s*```", re.DOTALL,
)


def _extract_json(text: str) -> str:
    """Extract JSON from text, stripping code fences if present.

    Args:
        text: Raw LLM response.

    Returns:
        Cleaned string ready for ``json.loads``.
    """
    stripped = text.strip()

    # Try code-fenced JSON first
    match = _JSON_FENCE_RE.search(stripped)

    if match:
        return match.group(1).strip()

    # Try to find a raw JSON object/array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = stripped.find(start_char)
        end = stripped.rfind(end_char)

        if start != -1 and end > start:
            return stripped[start:end + 1]

    return stripped


def _extract_code_block(text: str, lang: str) -> str | None:
    """Extract content from a code fence with a specific language tag.

    Args:
        text: Raw LLM response.
        lang: Language identifier (e.g. ``"csv"``).

    Returns:
        Extracted content or ``None`` if no fence found.
    """
    pattern = re.compile(
        rf"```{re.escape(lang)}\s*\n(.*?)\n\s*```", re.DOTALL,
    )
    match = pattern.search(text)

    if match:
        return match.group(1).strip()

    return None
