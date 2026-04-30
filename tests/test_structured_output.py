"""Tests for nono.connector.structured_output module.

Covers output parsers (JSON, Pydantic, Regex, CSV), the StructuredGenerator
retry loop, convenience functions, and LlmAgent integration.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from nono.connector.structured_output import (
    CsvOutputParser,
    JsonOutputParser,
    MaxRetriesExceededError,
    OutputParser,
    ParseError,
    PydanticOutputParser,
    RegexOutputParser,
    StructuredGenerator,
    parse_csv,
    parse_json,
    parse_pydantic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pydantic_available() -> bool:
    try:
        import pydantic  # noqa: F401
        return True
    except ImportError:
        return False


requires_pydantic = pytest.mark.skipif(
    not _pydantic_available(), reason="pydantic not installed",
)


# ---------------------------------------------------------------------------
# ParseError
# ---------------------------------------------------------------------------

class TestParseError:
    def test_stores_raw(self):
        err = ParseError("bad json", "raw text")
        assert err.raw == "raw text"
        assert "bad json" in str(err)


class TestMaxRetriesExceededError:
    def test_stores_metadata(self):
        inner = ParseError("oops", "raw")
        err = MaxRetriesExceededError(attempts=3, last_error=inner, raw="raw")
        assert err.attempts == 3
        assert err.last_error is inner
        assert err.raw == "raw"
        assert "3 attempts" in str(err)


# ---------------------------------------------------------------------------
# JsonOutputParser
# ---------------------------------------------------------------------------

class TestJsonOutputParser:
    def test_parse_plain_json(self):
        parser = JsonOutputParser()
        result = parser.parse('{"name": "Nono", "version": 1}')
        assert result == {"name": "Nono", "version": 1}

    def test_parse_json_in_code_fence(self):
        parser = JsonOutputParser()
        text = '```json\n{"key": "value"}\n```'
        assert parser.parse(text) == {"key": "value"}

    def test_parse_json_with_surrounding_text(self):
        parser = JsonOutputParser()
        text = 'Here is the result:\n{"a": 1}\nThat was it.'
        assert parser.parse(text) == {"a": 1}

    def test_parse_invalid_json_raises(self):
        parser = JsonOutputParser()
        with pytest.raises(ParseError, match="Invalid JSON"):
            parser.parse("not json at all")

    def test_parse_non_object_raises(self):
        parser = JsonOutputParser()
        with pytest.raises(ParseError, match="Expected a JSON object"):
            parser.parse("[1, 2, 3]")

    def test_schema_validation_passes(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        parser = JsonOutputParser(schema=schema)
        result = parser.parse('{"name": "test"}')
        assert result["name"] == "test"

    def test_schema_validation_fails(self):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        parser = JsonOutputParser(schema=schema)
        with pytest.raises(ParseError, match="schema validation failed"):
            parser.parse('{"count": "not_int"}')

    def test_format_instructions_with_schema(self):
        parser = JsonOutputParser(schema={"type": "object"})
        instructions = parser.format_instructions()
        assert "JSON Schema" in instructions

    def test_format_instructions_without_schema(self):
        parser = JsonOutputParser()
        instructions = parser.format_instructions()
        assert "valid JSON" in instructions

    def test_repair_prompt_contains_error(self):
        parser = JsonOutputParser()
        prompt = parser.repair_prompt("bad data", ValueError("nope"))
        assert "nope" in prompt
        assert "bad data" in prompt


# ---------------------------------------------------------------------------
# PydanticOutputParser
# ---------------------------------------------------------------------------

@requires_pydantic
class TestPydanticOutputParser:
    def _make_model(self):
        from pydantic import BaseModel

        class City(BaseModel):
            name: str
            population: int

        return City

    def test_parse_valid_json(self):
        City = self._make_model()
        parser = PydanticOutputParser(City)
        result = parser.parse('{"name": "Madrid", "population": 3300000}')
        assert result.name == "Madrid"
        assert result.population == 3300000

    def test_parse_from_code_fence(self):
        City = self._make_model()
        parser = PydanticOutputParser(City)
        text = '```json\n{"name": "Tokyo", "population": 14000000}\n```'
        result = parser.parse(text)
        assert result.name == "Tokyo"

    def test_parse_invalid_json_raises(self):
        City = self._make_model()
        parser = PydanticOutputParser(City)
        with pytest.raises(ParseError, match="Invalid JSON"):
            parser.parse("not json")

    def test_parse_validation_failure_raises(self):
        City = self._make_model()
        parser = PydanticOutputParser(City)
        with pytest.raises(ParseError, match="Pydantic validation failed"):
            parser.parse('{"name": "Paris"}')  # missing required population

    def test_json_schema_returns_dict(self):
        City = self._make_model()
        parser = PydanticOutputParser(City)
        schema = parser.json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema

    def test_format_instructions_include_schema(self):
        City = self._make_model()
        parser = PydanticOutputParser(City)
        instructions = parser.format_instructions()
        assert "JSON Schema" in instructions
        assert "population" in instructions


# ---------------------------------------------------------------------------
# RegexOutputParser
# ---------------------------------------------------------------------------

class TestRegexOutputParser:
    def test_parse_match(self):
        parser = RegexOutputParser(r"Score:\s*(\d+)", description="Score line")
        result = parser.parse("The result is Score: 42 points")
        assert result == "42"

    def test_parse_no_match_raises(self):
        parser = RegexOutputParser(r"Score:\s*(\d+)")
        with pytest.raises(ParseError, match="No match"):
            parser.parse("No scores here")

    def test_custom_group(self):
        parser = RegexOutputParser(r"(name):\s*(\w+)", group=2)
        result = parser.parse("name: Alice")
        assert result == "Alice"

    def test_format_instructions(self):
        parser = RegexOutputParser(r"\d+", description="A number")
        assert parser.format_instructions() == "A number"


# ---------------------------------------------------------------------------
# CsvOutputParser
# ---------------------------------------------------------------------------

class TestCsvOutputParser:
    def test_parse_simple_csv(self):
        parser = CsvOutputParser()
        text = "name,age\nAlice,30\nBob,25"
        rows = parser.parse(text)
        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"
        assert rows[1]["age"] == "25"

    def test_parse_csv_in_code_fence(self):
        parser = CsvOutputParser()
        text = "```csv\nname,age\nAlice,30\n```"
        rows = parser.parse(text)
        assert len(rows) == 1
        assert rows[0]["name"] == "Alice"

    def test_parse_empty_csv_raises(self):
        parser = CsvOutputParser()
        with pytest.raises(ParseError, match="empty"):
            parser.parse("name,age\n")

    def test_expected_columns_pass(self):
        parser = CsvOutputParser(expected_columns=["name", "age"])
        text = "name,age\nAlice,30"
        rows = parser.parse(text)
        assert len(rows) == 1

    def test_expected_columns_missing_raises(self):
        parser = CsvOutputParser(expected_columns=["name", "email"])
        text = "name,age\nAlice,30"
        with pytest.raises(ParseError, match="Missing CSV columns"):
            parser.parse(text)

    def test_custom_delimiter(self):
        parser = CsvOutputParser(delimiter="\t")
        text = "name\tage\nAlice\t30"
        rows = parser.parse(text)
        assert rows[0]["name"] == "Alice"

    def test_format_instructions_with_columns(self):
        parser = CsvOutputParser(expected_columns=["x", "y"])
        instructions = parser.format_instructions()
        assert "x" in instructions
        assert "y" in instructions


# ---------------------------------------------------------------------------
# StructuredGenerator
# ---------------------------------------------------------------------------

class TestStructuredGenerator:
    def test_requires_parser_or_model(self):
        service = MagicMock()
        with pytest.raises(ValueError, match="parser.*model"):
            StructuredGenerator(service)

    def test_generate_success_on_first_try(self):
        service = MagicMock()
        service.generate_completion.return_value = '{"key": "val"}'

        parser = JsonOutputParser()
        gen = StructuredGenerator(service, parser=parser, max_retries=2)
        result = gen.generate([{"role": "user", "content": "test"}])

        assert result == {"key": "val"}
        assert service.generate_completion.call_count == 1

    def test_generate_retries_on_failure(self):
        service = MagicMock()
        service.generate_completion.side_effect = [
            "not json",           # 1st attempt — will fail
            '{"key": "fixed"}',   # 2nd attempt — succeeds
        ]

        parser = JsonOutputParser()
        gen = StructuredGenerator(service, parser=parser, max_retries=1)
        result = gen.generate([{"role": "user", "content": "test"}])

        assert result == {"key": "fixed"}
        assert service.generate_completion.call_count == 2

    def test_generate_exhausts_retries(self):
        service = MagicMock()
        service.generate_completion.return_value = "bad data"

        parser = JsonOutputParser()
        gen = StructuredGenerator(service, parser=parser, max_retries=1)

        with pytest.raises(MaxRetriesExceededError) as exc_info:
            gen.generate([{"role": "user", "content": "test"}])

        assert exc_info.value.attempts == 2
        assert exc_info.value.raw == "bad data"

    @requires_pydantic
    def test_generate_with_pydantic_model(self):
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str
            price: float

        service = MagicMock()
        service.generate_completion.return_value = '{"name": "Widget", "price": 9.99}'

        gen = StructuredGenerator(service, model=Item)
        result = gen.generate([{"role": "user", "content": "test"}])

        assert isinstance(result, Item)
        assert result.name == "Widget"
        assert result.price == pytest.approx(9.99)

    def test_inject_instructions_appended_to_user_message(self):
        service = MagicMock()
        service.generate_completion.return_value = '{"a": 1}'

        parser = JsonOutputParser()
        gen = StructuredGenerator(service, parser=parser)
        gen.generate([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "original"},
        ])

        call_args = service.generate_completion.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "valid JSON" in user_msg["content"]

    def test_inject_false_skips_instructions(self):
        service = MagicMock()
        service.generate_completion.return_value = '{"a": 1}'

        parser = JsonOutputParser()
        gen = StructuredGenerator(service, parser=parser, inject_instructions=False)
        gen.generate([{"role": "user", "content": "plain"}])

        call_args = service.generate_completion.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "valid JSON" not in user_msg["content"]

    def test_response_format_set_to_json_for_json_parser(self):
        from nono.connector.connector_genai import ResponseFormat

        service = MagicMock()
        service.generate_completion.return_value = '{"a": 1}'

        parser = JsonOutputParser()
        gen = StructuredGenerator(service, parser=parser)
        gen.generate([{"role": "user", "content": "test"}])

        call_kwargs = service.generate_completion.call_args.kwargs
        assert call_kwargs["response_format"] == ResponseFormat.JSON

    def test_response_format_set_to_text_for_regex_parser(self):
        from nono.connector.connector_genai import ResponseFormat

        service = MagicMock()
        service.generate_completion.return_value = "Score: 42"

        parser = RegexOutputParser(r"Score:\s*(\d+)")
        gen = StructuredGenerator(service, parser=parser)
        gen.generate([{"role": "user", "content": "test"}])

        call_kwargs = service.generate_completion.call_args.kwargs
        assert call_kwargs["response_format"] == ResponseFormat.TEXT


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    def test_parse_json_simple(self):
        result = parse_json('{"x": 1}')
        assert result == {"x": 1}

    def test_parse_json_with_fence(self):
        result = parse_json('```json\n{"x": 1}\n```')
        assert result == {"x": 1}

    @requires_pydantic
    def test_parse_pydantic_simple(self):
        from pydantic import BaseModel

        class Coords(BaseModel):
            lat: float
            lon: float

        result = parse_pydantic('{"lat": 40.4, "lon": -3.7}', Coords)
        assert result.lat == pytest.approx(40.4)

    def test_parse_csv_simple(self):
        rows = parse_csv("a,b\n1,2\n3,4")
        assert len(rows) == 2
        assert rows[0]["a"] == "1"

    def test_parse_csv_with_expected_columns(self):
        rows = parse_csv("x,y\n1,2", expected_columns=["x", "y"])
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# LlmAgent integration
# ---------------------------------------------------------------------------

class TestLlmAgentStructuredOutput:
    """Test output_model / output_parser integration in LlmAgent."""

    @requires_pydantic
    def test_agent_with_output_model(self):
        from pydantic import BaseModel

        from nono.agent.base import Event, EventType, InvocationContext, Session
        from nono.agent.llm_agent import LlmAgent

        class Sentiment(BaseModel):
            label: str
            score: float

        agent = LlmAgent(
            name="test_agent",
            provider="google",
            instruction="Classify sentiment.",
            output_model=Sentiment,
        )

        # Mock the service
        mock_service = MagicMock()
        mock_service.generate_completion.return_value = (
            '{"label": "positive", "score": 0.95}'
        )
        agent._service = mock_service

        session = Session()
        ctx = InvocationContext(session=session, user_message="Great product!")
        response = agent.run(ctx)

        # Response should be the Pydantic model serialised as JSON
        parsed = json.loads(response)
        assert parsed["label"] == "positive"
        assert parsed["score"] == pytest.approx(0.95)

    def test_agent_with_output_parser(self):
        from nono.agent.base import InvocationContext, Session
        from nono.agent.llm_agent import LlmAgent

        parser = JsonOutputParser()
        agent = LlmAgent(
            name="test_agent",
            provider="google",
            instruction="Return JSON.",
            output_parser=parser,
        )

        mock_service = MagicMock()
        mock_service.generate_completion.return_value = '{"status": "ok"}'
        agent._service = mock_service

        session = Session()
        ctx = InvocationContext(session=session, user_message="test")
        response = agent.run(ctx)

        parsed = json.loads(response)
        assert parsed["status"] == "ok"

    @requires_pydantic
    def test_agent_retries_on_bad_output(self):
        from pydantic import BaseModel

        from nono.agent.base import InvocationContext, Session
        from nono.agent.llm_agent import LlmAgent

        class Item(BaseModel):
            name: str

        agent = LlmAgent(
            name="retry_agent",
            provider="google",
            instruction="Return item.",
            output_model=Item,
            output_retries=1,
        )

        mock_service = MagicMock()
        mock_service.generate_completion.side_effect = [
            "not valid json",                  # initial call fails
            '{"name": "Widget"}',              # retry succeeds
        ]
        agent._service = mock_service

        session = Session()
        ctx = InvocationContext(session=session, user_message="give me an item")
        response = agent.run(ctx)

        parsed = json.loads(response)
        assert parsed["name"] == "Widget"
        assert mock_service.generate_completion.call_count == 2

    def test_agent_without_output_model_unchanged(self):
        from nono.agent.base import InvocationContext, Session
        from nono.agent.llm_agent import LlmAgent

        agent = LlmAgent(
            name="plain_agent",
            provider="google",
            instruction="You are helpful.",
        )

        mock_service = MagicMock()
        mock_service.generate_completion.return_value = "Hello, world!"
        agent._service = mock_service

        session = Session()
        ctx = InvocationContext(session=session, user_message="Hi")
        response = agent.run(ctx)

        assert response == "Hello, world!"
