"""
LLM-powered agent with tool calling and sub-agent delegation.

Implements the Nono ``LlmAgent`` / ``Agent`` on top of Nono's unified
connector layer.  Supports any provider available in ``nono.connector``.

Usage:
    from nono.agent import Agent
    from nono.agent.tool import tool

    @tool(description="Get the weather for a city.")
    def get_weather(city: str) -> str:
        return f"Sunny, 22°C in {city}"

    agent = Agent(
        name="weather_assistant",
        model="gemini-3-flash-preview",
        provider="google",
        instruction="You are a helpful weather assistant.",
        tools=[get_weather],
    )

    from nono.agent.base import Session, InvocationContext
    session = Session()
    ctx = InvocationContext(session=session, user_message="What's the weather in Madrid?")
    response = agent.run(ctx)
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import threading
import time
from typing import Any, AsyncIterator, Callable, Generator, Iterator, Optional

from .base import (
    BaseAgent,
    Event,
    EventType,
    InvocationContext,
    MAX_TRANSFER_DEPTH,
    Session,
)
from .tool import FunctionTool, ToolContext, parse_tool_calls, validate_tools
from .tracing import LLMCall, TokenUsage, ToolRecord

# Fallback support
from nono.connector.fallback import FallbackHandler, load_fallback_config

# Late-bound type for skills to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .skill import BaseSkill

logger = logging.getLogger("Nono.Agent.LlmAgent")

# Provider → (ServiceClass, default_model) mapping
# NOTE: Keep in sync with the "Supported Providers" table in the docs
# and the connector classes in nono/connector/connector_genai.py.
_PROVIDER_MAP: dict[str, tuple[str, str]] = {
    "google": ("GeminiService", "gemini-3-flash-preview"),
    "gemini": ("GeminiService", "gemini-3-flash-preview"),
    "openai": ("OpenAIService", "gpt-4o-mini"),
    "perplexity": ("PerplexityService", "sonar"),
    "deepseek": ("DeepSeekService", "deepseek-chat"),
    "xai": ("XAIService", "grok-3"),
    "groq": ("GroqService", "llama-3.3-70b-versatile"),
    "cerebras": ("CerebrasService", "llama-3.3-70b"),
    "anthropic": ("AnthropicService", "claude-sonnet-4"),
    "claude": ("AnthropicService", "claude-sonnet-4"),
    "nvidia": ("NvidiaService", "meta/llama-3.3-70b-instruct"),
    "github": ("GitHubModelsService", "openai/gpt-5"),
    "openrouter": ("OpenRouterService", "openrouter/auto"),
    "azure": ("AzureAIService", "openai/gpt-4o"),
    "vercel": ("VercelAIService", "anthropic/claude-opus-4.5"),
    "ollama": ("OllamaService", "llama3"),
}


def _load_agent_int(key: str, default: int) -> int:
    """Load an integer from ``[agent]`` in config.toml, with fallback."""
    try:
        from nono.config import load_config as _lc
        val = _lc().get(f"agent.{key}")
        if val is not None:
            return int(val)
    except Exception:
        pass
    return default


def _load_agent_ints() -> dict[str, int]:
    """Load all agent integer settings in a single config parse."""
    defaults = {
        "max_tool_iterations": 10,
        "max_loop_messages": 40,
        "max_tool_result_chars": 20_000,
    }
    try:
        from nono.config import load_config as _lc
        cfg = _lc()
        for key in defaults:
            val = cfg.get(f"agent.{key}")
            if val is not None:
                defaults[key] = int(val)
    except Exception:
        pass
    return defaults


_AGENT_INTS = _load_agent_ints()

# Maximum tool-call loop iterations to prevent infinite loops
_MAX_TOOL_ITERATIONS: int = _AGENT_INTS["max_tool_iterations"]

# Maximum messages to keep in the tool-calling loop to avoid context overflow
_MAX_LOOP_MESSAGES: int = _AGENT_INTS["max_loop_messages"]

# Maximum characters from a single tool result before truncation
_MAX_TOOL_RESULT_CHARS: int = _AGENT_INTS["max_tool_result_chars"]


# Compaction strategies — import here to avoid circular deps at module load
from nono.agent.compaction import (
    CallableStrategy,
    CompactionResult,
    CompactionStrategy,
    SummarizationStrategy,
    TokenAwareStrategy,
)


def estimate_tokens(text: str) -> int:
    """Estimate the token count for a text string.

    Uses a simple heuristic of ~4 characters per token.  Replace with a
    proper tokenizer (e.g. ``tiktoken``) for accurate counts.

    Args:
        text: The text to estimate.

    Returns:
        Estimated token count.
    """
    return len(text) // 4


def _estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate total tokens across a list of messages without joining strings."""
    return sum(len(m.get("content", "")) for m in messages) // 4


def _inject_format_instructions(
    messages: list[dict[str, str]], instructions: str,
) -> list[dict[str, str]]:
    """Append format instructions to the last user message.

    Args:
        messages: Chat message list (will be shallow-copied).
        instructions: Text to append.

    Returns:
        New message list with instructions injected.
    """
    result = [m.copy() for m in messages]

    for msg in reversed(result):
        if msg.get("role") == "user":
            msg["content"] += f"\n\n{instructions}"
            return result

    # Fallback: inject as system message
    result.insert(0, {"role": "system", "content": instructions})
    return result


def _create_service(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    **service_kwargs: Any,
) -> Any:
    """Lazily import and instantiate a Nono connector service.

    Args:
        provider: Provider name (google, openai, etc.).
        model: Model name override.
        api_key: API key override.
        **service_kwargs: Extra kwargs forwarded to the service constructor.

    Returns:
        An instance of the appropriate ``GenerativeAIService`` subclass.

    Raises:
        ValueError: If the provider is unknown.
    """
    provider_lower = provider.lower()

    if provider_lower not in _PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Supported: {sorted(_PROVIDER_MAP.keys())}"
        )

    class_name, default_model = _PROVIDER_MAP[provider_lower]
    model = model or default_model

    # Dynamic import to avoid circular dependencies and heavy startup costs
    mod = importlib.import_module("nono.connector.connector_genai")
    service_cls = getattr(mod, class_name)

    kwargs: dict[str, Any] = {"model_name": model}
    if api_key:
        kwargs["api_key"] = api_key
    kwargs.update(service_kwargs)

    return service_cls(**kwargs)


_TRANSFER_TOOL_NAME = "transfer_to_agent"


class LlmAgent(BaseAgent):
    """An agent powered by a Large Language Model with tool calling.

    Core LLM agent in the Nono Agent Architecture (NAA).  Uses Nono's
    unified connector to talk to any supported provider.

    When ``sub_agents`` are configured, a ``transfer_to_agent`` tool is
    automatically registered so the LLM can delegate tasks via the standard
    function-calling mechanism instead of ad-hoc JSON.

    Args:
        name: Unique agent name.
        model: Model name (e.g. ``"gemini-3-flash-preview"``).
        provider: Provider name (e.g. ``"google"``, ``"openai"``).
        instruction: System prompt / instruction for the agent.
        description: Short description of the agent (used for delegation).
        tools: List of ``FunctionTool`` instances the agent can call.
        skills: List of ``BaseSkill`` instances — auto-converted to tools.
        api_key: API key override (default: read from config/env).
        temperature: LLM temperature.
        max_tokens: Maximum output tokens.
        output_format: Response format (``"text"`` or ``"json"``).
        sub_agents: Child agents for delegation.
        before_agent_callback: Lifecycle callback.
        after_agent_callback: Lifecycle callback.
        before_tool_callback: Lifecycle callback.
        after_tool_callback: Lifecycle callback.
        hook_manager: Optional ``HookManager`` for lifecycle hooks.
        service_kwargs: Extra kwargs for the connector service constructor.

    Example:
        >>> agent = Agent(
        ...     name="helper",
        ...     model="gemini-3-flash-preview",
        ...     provider="google",
        ...     instruction="You are helpful.",
        ...     skills=[SummarizeSkill(), ClassifySkill()],
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        model: str | None = None,
        provider: str = "google",
        instruction: str = "You are a helpful assistant.",
        description: str = "",
        tools: list[FunctionTool] | None = None,
        skills: list[BaseSkill] | None = None,
        api_key: str | None = None,
        temperature: float | str = 0.7,
        max_tokens: int | None = None,
        output_format: str = "text",
        output_model: type | None = None,
        output_parser: Any | None = None,
        output_retries: int = 2,
        compaction: CompactionStrategy | Callable | str | bool | None = None,
        sub_agents: list[BaseAgent] | None = None,
        before_agent_callback: Any = None,
        after_agent_callback: Any = None,
        before_tool_callback: Any = None,
        after_tool_callback: Any = None,
        hook_manager: Any = None,
        **service_kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            sub_agents=sub_agents,
            before_agent_callback=before_agent_callback,
            after_agent_callback=after_agent_callback,
            before_tool_callback=before_tool_callback,
            after_tool_callback=after_tool_callback,
            hook_manager=hook_manager,
        )
        self.instruction = instruction
        self.tools: list[FunctionTool] = tools or []
        self.skills: list[BaseSkill] = skills or []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.output_format = output_format
        self.output_model = output_model
        self.output_parser = output_parser
        self.output_retries = output_retries

        # Compaction strategy: True → SummarizationStrategy, instance → custom,
        # callable → CallableStrategy adapter, str → dynamic import,
        # False/None → _prune_messages
        if compaction is True:
            self._compaction: CompactionStrategy | None = SummarizationStrategy()
        elif isinstance(compaction, str):
            self._compaction = self._load_compaction_strategy(compaction)
        elif isinstance(compaction, CompactionStrategy):
            self._compaction = compaction
        elif callable(compaction):
            self._compaction = CallableStrategy(compaction)
        else:
            self._compaction = None

        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._service_kwargs = service_kwargs
        self._service: Any = None  # lazy init
        self._fallback_handler: FallbackHandler | None = None  # lazy init
        self._init_lock = threading.Lock()
        self._cached_all_tools: list[FunctionTool] | None = None
        self._cached_transfer_tool: FunctionTool | None = None

        # ACI quality: auto-validate user-supplied tool descriptions.
        # Emits warnings via logging — does not raise.
        if self.tools:
            validate_tools(self.tools, warn=True)

    @property
    def service(self) -> Any:
        """Lazily initialize and return the connector service."""
        if self._service is None:
            with self._init_lock:
                if self._service is None:
                    self._service = _create_service(
                        self._provider,
                        self._model,
                        self._api_key,
                        **self._service_kwargs,
                    )
        return self._service

    @property
    def fallback_handler(self) -> FallbackHandler:
        """Lazily initialize and return the fallback handler."""
        if self._fallback_handler is None:
            with self._init_lock:
                if self._fallback_handler is None:
                    self._fallback_handler = FallbackHandler(
                        primary_provider=self._provider,
                        primary_model=self._model,
                        api_key=self._api_key,
                        **self._service_kwargs,
                    )
        return self._fallback_handler

    def _call_llm(self, *, messages: list[dict], **kwargs: Any) -> str:
        """Call the LLM with automatic fallback on failure.

        When the fallback handler is enabled (``[fallback].enabled = true``
        in config.toml) and the primary provider fails, subsequent
        providers in the chain are tried transparently.

        Args:
            messages: Chat messages in OpenAI format.
            **kwargs: Extra arguments for ``generate_completion``.

        Returns:
            Generated text from the first successful provider.
        """
        if self.fallback_handler.enabled:
            # Ensure the handler reuses our (possibly injected) primary service
            cache_key = (self._provider, self._model or "")
            self.fallback_handler._services[cache_key] = self.service
            return self.fallback_handler.generate_completion(
                messages=messages, **kwargs,
            )
        return self.service.generate_completion(
            messages=messages, **kwargs,
        )

    def _call_llm_stream(self, *, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        """Stream tokens from the LLM.

        Falls back to the non-streaming path (yielding the full
        response as a single chunk) when the service's
        ``generate_completion_stream`` is the base-class default.

        Args:
            messages: Chat messages in OpenAI format.
            **kwargs: Extra arguments for ``generate_completion_stream``.

        Yields:
            Text chunks as they arrive from the provider.
        """
        return self.service.generate_completion_stream(
            messages=messages, **kwargs,
        )

    def _stream_llm_full(
        self, *, messages: list[dict], **kwargs: Any,
    ) -> Iterator:
        """Stream structured chunks (text + tool-call deltas) from the LLM.

        Delegates to the connector's ``generate_stream()`` which yields
        ``StreamChunk`` objects.

        Args:
            messages: Chat messages in OpenAI format.
            **kwargs: Extra arguments forwarded to ``generate_stream``.

        Yields:
            ``StreamChunk`` objects from the provider.
        """
        return self.service.generate_stream(
            messages=messages, **kwargs,
        )

    # ── structured output helpers ─────────────────────────────────────────

    def _build_structured_parser(self) -> Any:
        """Build a parser from ``output_model`` or ``output_parser``.

        Returns:
            An ``OutputParser`` instance or ``None``.
        """
        if self.output_parser is not None:
            return self.output_parser

        if self.output_model is not None:
            from nono.connector.structured_output import PydanticOutputParser
            return PydanticOutputParser(self.output_model)

        return None

    def _resolve_output_format(
        self, response_format_enum: type, parser: Any,
    ) -> tuple[Any, dict | None]:
        """Determine ResponseFormat and json_schema for the LLM call.

        Args:
            response_format_enum: The ``ResponseFormat`` enum class.
            parser: The structured output parser (or ``None``).

        Returns:
            Tuple of (ResponseFormat, json_schema dict or None).
        """
        if parser is not None:
            from nono.connector.structured_output import (
                PydanticOutputParser,
                JsonOutputParser,
            )

            schema = None

            if isinstance(parser, PydanticOutputParser):
                schema = parser.json_schema()
            elif isinstance(parser, JsonOutputParser) and parser.schema:
                schema = parser.schema

            return response_format_enum.JSON, schema

        if self.output_format == "json":
            return response_format_enum.JSON, None

        return response_format_enum.TEXT, None

    def _parse_structured_response(
        self,
        ctx: Any,
        messages: list[dict],
        raw: str,
        parser: Any,
        response_format: Any,
        json_schema: dict | None,
        tool_declarations: list | None,
    ) -> Generator[Event, None, str]:
        """Parse and validate structured output, retrying on failure.

        This is a generator that yields ``Event`` objects for tracing and
        returns the final serialised output string.

        Args:
            ctx: Invocation context.
            messages: Current message history.
            raw: Raw LLM response text.
            parser: The ``OutputParser`` to apply.
            response_format: The ``ResponseFormat`` enum value.
            json_schema: JSON schema dict (or ``None``).
            tool_declarations: Tool declarations for the LLM call.

        Yields:
            Error events on parse failure.

        Returns:
            Serialised parsed output as a string.
        """
        from nono.connector.structured_output import ParseError

        for attempt in range(1, self.output_retries + 2):
            try:
                parsed = parser.parse(raw)

                # Serialise Pydantic models to JSON string for the event
                if hasattr(parsed, "model_dump_json"):
                    return parsed.model_dump_json()
                elif hasattr(parsed, "json"):
                    return parsed.json()  # Pydantic v1

                return json.dumps(parsed) if not isinstance(parsed, str) else parsed
            except ParseError as exc:
                logger.warning(
                    "[%s] Structured output parse failed (attempt %d/%d): %s",
                    self.name, attempt, self.output_retries + 1, exc,
                )

                yield Event(
                    EventType.ERROR,
                    self.name,
                    f"Structured output parse error (attempt {attempt}): {exc}",
                )

                if attempt > self.output_retries:
                    # Retries exhausted — return raw response
                    logger.error(
                        "[%s] Structured output retries exhausted; "
                        "returning raw response.",
                        self.name,
                    )
                    return raw

                # Build repair prompt and retry the LLM
                repair = parser.repair_prompt(raw, exc)
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": repair},
                ]
                messages = self._compact_messages(messages)

                raw = self._call_llm(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format=response_format,
                    json_schema=json_schema,
                    tools=tool_declarations,
                )

        return raw  # pragma: no cover — loop always returns above

    # ── transfer_to_agent auto-tool ───────────────────────────────────────

    def _build_transfer_tool(self) -> FunctionTool:
        """Build the automatic ``transfer_to_agent`` tool (cached).

        The LLM calls this tool with ``agent_name`` and ``message`` to
        delegate a task to one of the configured ``sub_agents``.

        Returns:
            A ``FunctionTool`` describing the transfer function.
        """
        if self._cached_transfer_tool is not None:
            return self._cached_transfer_tool

        agent_names = [a.name for a in self.sub_agents]
        descriptions = "\n".join(
            f"  - {a.name}: {a.description or 'No description'}"
            for a in self.sub_agents
        )

        def transfer_to_agent(agent_name: str, message: str) -> str:
            """Delegate a task to a sub-agent by name."""
            ...  # placeholder — actual execution handled in _run_impl

        tool = FunctionTool(
            fn=transfer_to_agent,
            name=_TRANSFER_TOOL_NAME,
            description=(
                f"Transfer the conversation to a specialist sub-agent. "
                f"Available agents:\n{descriptions}"
            ),
        )
        self._cached_transfer_tool = tool
        return tool

    @property
    def _all_tools(self) -> list[FunctionTool]:
        """User tools + skill tools + auto-generated transfer tool (cached)."""
        if self._cached_all_tools is not None:
            return self._cached_all_tools
        tools = list(self.tools)
        for skill in self.skills:
            tools.append(skill.as_tool())
        if self.sub_agents:
            tools.append(self._build_transfer_tool())
        self._cached_all_tools = tools
        return tools

    # ── Core execution ────────────────────────────────────────────────────

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Execute LLM call with optional tool-calling loop.

        The loop:
        1. Build messages from session + system instruction + user message.
        2. Call the LLM (with tool declarations if tools are configured).
        3. If the LLM requests a tool call, execute it and feed the result back.
           Special case: ``transfer_to_agent`` runs a sub-agent.
        4. Repeat until the LLM produces a final text response or the max
           iteration limit is reached.

        Args:
            ctx: The invocation context.

        Yields:
            Event objects produced during execution.
        """
        from nono.connector.connector_genai import ResponseFormat

        # Build structured output parser if output_model or output_parser given
        _structured_parser = self._build_structured_parser()

        # Record user message as event
        if ctx.user_message:
            yield Event(EventType.USER_MESSAGE, "user", ctx.user_message)

        # Build message history
        messages = self._build_messages(ctx)

        # Inject format instructions into message history
        if _structured_parser is not None:
            _fmt_instr = _structured_parser.format_instructions()
            messages = _inject_format_instructions(messages, _fmt_instr)

        # Tool declarations for the LLM (user tools + transfer tool)
        all_tools = self._all_tools
        tool_declarations = (
            [t.to_function_declaration() for t in all_tools]
            if all_tools
            else None
        )

        response_format, json_schema = self._resolve_output_format(
            ResponseFormat, _structured_parser,
        )

        for iteration in range(_MAX_TOOL_ITERATIONS):
            logger.debug(
                "[%s] LLM call iteration %d, messages=%d",
                self.name, iteration, len(messages),
            )

            # Compact messages to avoid context window overflow
            messages = self._compact_messages(messages)

            _llm_start = time.perf_counter_ns()
            response = self._call_llm(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_format,
                json_schema=json_schema,
                tools=tool_declarations,
            )
            _llm_elapsed = (time.perf_counter_ns() - _llm_start) / 1_000_000

            # Record LLM call in trace
            if ctx.trace_collector is not None:
                _output_text = str(response)
                ctx.trace_collector.record_llm_call(LLMCall(
                    provider=self._provider,
                    model=self._model or "",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    token_usage=TokenUsage(
                        input_tokens=_estimate_messages_tokens(messages),
                        output_tokens=estimate_tokens(_output_text),
                    ),
                    duration_ms=_llm_elapsed,
                ))

            # Check if the response contains tool calls
            tool_calls = self._extract_tool_calls(response, all_tools)

            if not tool_calls:
                # Final text response — apply structured output parsing
                final = str(response)

                if _structured_parser is not None:
                    final = yield from self._parse_structured_response(
                        ctx, messages, final, _structured_parser,
                        response_format, json_schema, tool_declarations,
                    )

                yield Event(EventType.AGENT_MESSAGE, self.name, final)
                return

            # Process each tool call
            for call_info in tool_calls:
                tool_name = call_info["name"]
                tool_args = call_info["arguments"]
                tool_obj = call_info["tool"]

                # ── transfer_to_agent handling ────────────────────────────
                if tool_name == _TRANSFER_TOOL_NAME:
                    target_name = tool_args.get("agent_name", "")
                    sub_message = tool_args.get("message", ctx.user_message)
                    target = self.find_sub_agent(target_name)

                    yield Event(
                        EventType.AGENT_TRANSFER,
                        self.name,
                        f"Transferring to {target_name}",
                        data={"target_agent": target_name, "message": sub_message},
                    )

                    if target is None:
                        result = f"Error: agent '{target_name}' not found."
                        logger.warning(
                            "[%s] transfer_to_agent: unknown agent %r",
                            self.name, target_name,
                        )
                    elif ctx.transfer_depth >= MAX_TRANSFER_DEPTH:
                        result = (
                            f"Error: maximum transfer depth ({MAX_TRANSFER_DEPTH}) "
                            f"exceeded. Aborting delegation to '{target_name}'."
                        )
                        logger.error(
                            "[%s] transfer_to_agent: depth %d exceeds limit %d",
                            self.name, ctx.transfer_depth, MAX_TRANSFER_DEPTH,
                        )
                    else:
                        sub_ctx = InvocationContext(
                            session=ctx.session,
                            user_message=sub_message,
                            parent_agent=self,
                            trace_collector=ctx.trace_collector,
                            transfer_depth=ctx.transfer_depth + 1,
                        )
                        result = target.run(sub_ctx)

                    yield Event(
                        EventType.TOOL_RESULT,
                        self.name,
                        str(result),
                        data={"tool": _TRANSFER_TOOL_NAME, "result": result},
                    )

                    messages.append({
                        "role": "assistant",
                        "content": json.dumps({
                            "tool_call": _TRANSFER_TOOL_NAME,
                            "arguments": tool_args,
                        }),
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Agent '{target_name}' responded: {result}"
                        ),
                    })
                    continue

                # ── Regular tool handling ─────────────────────────────────
                # PreToolUse hook
                pre_tool_hook = self._fire_hook(
                    "PreToolUse", ctx,
                    tool_name=tool_name,
                    tool_input=tool_args,
                )
                if pre_tool_hook is not None and pre_tool_hook.should_block:
                    block_msg = pre_tool_hook.block_reason or f"Tool '{tool_name}' blocked by hook"
                    yield Event(
                        EventType.TOOL_RESULT, self.name, block_msg,
                        data={"tool": tool_name, "blocked": True},
                    )
                    messages.append({
                        "role": "user",
                        "content": f"Tool '{tool_name}' was blocked: {block_msg}",
                    })
                    continue

                # Apply updated input from hook
                if pre_tool_hook is not None and pre_tool_hook.updated_input:
                    tool_args = pre_tool_hook.updated_input

                # before_tool callback
                if self.before_tool_callback:
                    modified = self.before_tool_callback(self, tool_name, tool_args)
                    if modified is not None:
                        tool_args = modified

                yield Event(
                    EventType.TOOL_CALL,
                    self.name,
                    f"Calling {tool_name}",
                    data={"tool": tool_name, "arguments": tool_args},
                )

                _tool_start = time.perf_counter_ns()
                _tool_error: str | None = None
                try:
                    _tool_ctx = ToolContext(
                        state=ctx.session.state,
                        shared_content=ctx.session.shared_content,
                        local_content=self.local_content,
                        agent_name=self.name,
                        session_id=ctx.session.session_id,
                        _session=ctx.session,
                    )
                    result = tool_obj.invoke(tool_args, tool_context=_tool_ctx)
                except Exception as e:
                    logger.error("Tool %r failed: %s", tool_name, e)
                    result = f"Error: {e}"
                    _tool_error = str(e)
                finally:
                    _tool_elapsed = (time.perf_counter_ns() - _tool_start) / 1_000_000
                    if ctx.trace_collector is not None:
                        ctx.trace_collector.record_tool(ToolRecord(
                            tool_name=tool_name,
                            arguments=tool_args,
                            result=str(result),
                            duration_ms=_tool_elapsed,
                            error=_tool_error,
                        ))

                # after_tool callback
                if self.after_tool_callback:
                    modified = self.after_tool_callback(
                        self, tool_name, tool_args, result,
                    )
                    if modified is not None:
                        result = modified

                # PostToolUse hook
                post_tool_hook = self._fire_hook(
                    "PostToolUse", ctx,
                    tool_name=tool_name,
                    tool_input=tool_args,
                    tool_response=result,
                    tool_error=_tool_error,
                )
                if post_tool_hook is not None and post_tool_hook.additional_context:
                    result = f"{result}\n{post_tool_hook.additional_context}"

                yield Event(
                    EventType.TOOL_RESULT,
                    self.name,
                    str(result),
                    data={"tool": tool_name, "result": result},
                )

                # Append tool call + result to messages for the next LLM turn
                messages.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "tool_call": tool_name,
                        "arguments": tool_args,
                    }),
                })
                result_str = str(result)
                if len(result_str) > _MAX_TOOL_RESULT_CHARS:
                    result_str = result_str[:_MAX_TOOL_RESULT_CHARS] + "… (truncated)"
                messages.append({
                    "role": "user",
                    "content": f"Tool '{tool_name}' returned: {result_str}",
                })

        # Max iterations reached — ask LLM for a final answer
        messages.append({
            "role": "user",
            "content": "Please provide your final answer based on the tool results above.",
        })
        messages = self._compact_messages(messages)
        _llm_start = time.perf_counter_ns()
        response = self._call_llm(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=response_format,
        )
        _llm_elapsed = (time.perf_counter_ns() - _llm_start) / 1_000_000
        if ctx.trace_collector is not None:
            _output_text = str(response)
            ctx.trace_collector.record_llm_call(LLMCall(
                provider=self._provider,
                model=self._model or "",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                token_usage=TokenUsage(
                    input_tokens=_estimate_messages_tokens(messages),
                    output_tokens=estimate_tokens(_output_text),
                ),
                duration_ms=_llm_elapsed,
            ))
        yield Event(EventType.AGENT_MESSAGE, self.name, str(response))

    # ── Token-level streaming ─────────────────────────────────────────────

    def _run_stream_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Execute LLM call with full token-level streaming.

        All LLM calls — including those that return tool-call requests —
        are streamed.  The method yields:

        * ``TEXT_CHUNK`` for each text content delta.
        * ``TOOL_CALL_CHUNK`` for each tool-call argument fragment
          (incremental JSON).
        * ``TOOL_CALL`` once a tool call's arguments are fully assembled.
        * ``TOOL_RESULT`` after executing each tool.
        * ``AGENT_MESSAGE`` with the complete response text at the end.

        When the connector's ``generate_stream()`` falls back to the
        base-class default (no native streaming tool-call support), the
        method detects tool calls from the assembled text using
        ``parse_tool_calls`` — so the tool loop still works across
        all providers.

        Structured output parsing (``output_model`` / ``output_parser``)
        is applied to the fully-assembled text after streaming completes.

        Args:
            ctx: The invocation context.

        Yields:
            Event objects produced during streaming execution.
        """
        from nono.connector.connector_genai import ResponseFormat

        _structured_parser = self._build_structured_parser()

        if ctx.user_message:
            yield Event(EventType.USER_MESSAGE, "user", ctx.user_message)

        messages = self._build_messages(ctx)

        if _structured_parser is not None:
            _fmt_instr = _structured_parser.format_instructions()
            messages = _inject_format_instructions(messages, _fmt_instr)

        all_tools = self._all_tools
        tool_map = {t.name: t for t in all_tools} if all_tools else {}
        tool_declarations = (
            [t.to_function_declaration() for t in all_tools]
            if all_tools
            else None
        )
        response_format, json_schema = self._resolve_output_format(
            ResponseFormat, _structured_parser,
        )

        for iteration in range(_MAX_TOOL_ITERATIONS):
            messages = self._compact_messages(messages)

            # ── Stream from the LLM ──────────────────────────────────
            text_parts: list[str] = []
            tool_calls_acc: dict[int, dict[str, str]] = {}
            finish_reason = "stop"

            for chunk in self._stream_llm_full(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_format,
                json_schema=json_schema,
                tools=tool_declarations,
            ):
                if chunk.type == "text":
                    text_parts.append(chunk.content)
                    yield Event(
                        EventType.TEXT_CHUNK, self.name, chunk.content,
                    )

                elif chunk.type == "tool_call":
                    idx = chunk.tool_index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": chunk.tool_call_id,
                            "name": chunk.tool_name,
                            "arguments": "",
                        }
                    elif chunk.tool_name:
                        tool_calls_acc[idx]["name"] = chunk.tool_name
                    tool_calls_acc[idx]["arguments"] += chunk.content
                    yield Event(
                        EventType.TOOL_CALL_CHUNK, self.name,
                        chunk.content,
                        {
                            "tool_index": idx,
                            "tool_name": (
                                tool_calls_acc[idx]["name"]
                            ),
                            "arguments_delta": chunk.content,
                        },
                    )

                elif chunk.type == "finish":
                    finish_reason = chunk.finish_reason

            # ── Decide: tool calls or text response? ─────────────────
            assembled_text = "".join(text_parts)

            # Native streaming tool calls (from generate_stream)
            has_native_tool_calls = bool(tool_calls_acc)

            # Fallback: detect tool calls embedded in text
            # (for providers whose generate_stream defaults to base class)
            fallback_tool_calls: list[dict[str, Any]] = []
            if (
                not has_native_tool_calls
                and tool_declarations
                and assembled_text
            ):
                fallback_tool_calls = self._extract_tool_calls(
                    assembled_text, all_tools,
                )

            if has_native_tool_calls:
                # ── Process native streaming tool calls ──────────────
                for idx in sorted(tool_calls_acc):
                    tc = tool_calls_acc[idx]
                    tool_name = tc["name"]
                    try:
                        tool_args = json.loads(tc["arguments"])
                    except (json.JSONDecodeError, ValueError):
                        tool_args = {}

                    tool_obj = tool_map.get(tool_name)
                    if tool_obj is None:
                        messages.append({
                            "role": "assistant",
                            "content": f'{tool_name}({json.dumps(tool_args)})',
                        })
                        messages.append({
                            "role": "user",
                            "content": f"Error: Tool '{tool_name}' not found.",
                        })
                        continue

                    if tool_name == _TRANSFER_TOOL_NAME:
                        target_name = tool_args.get("agent_name", "")
                        target = self.find_sub_agent(target_name)
                        if target is None:
                            messages.append({
                                "role": "assistant",
                                "content": (
                                    f'transfer_to_agent({{"agent_name":'
                                    f' "{target_name}"}})'
                                ),
                            })
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"Error: Agent '{target_name}' not found."
                                ),
                            })
                            continue
                        yield Event(
                            EventType.AGENT_TRANSFER, self.name,
                            f"Transferring to {target_name}",
                            {"target_agent": target_name},
                        )
                        sub_ctx = InvocationContext(
                            session=ctx.session,
                            user_message=ctx.user_message,
                            trace_collector=ctx.trace_collector,
                        )
                        yield from target._run_impl_traced(sub_ctx)
                        return

                    yield Event(
                        EventType.TOOL_CALL, self.name,
                        f"Calling {tool_name}",
                        {"tool": tool_name, "arguments": tool_args},
                    )
                    _tool_ctx = ToolContext(
                        state=ctx.session.state,
                        shared_content=ctx.session.shared_content,
                        local_content=self.local_content,
                        agent_name=self.name,
                        session_id=ctx.session.session_id,
                        _session=ctx.session,
                    )
                    try:
                        result = tool_obj.invoke(tool_args, _tool_ctx)
                    except Exception as exc:
                        result = f"Error: {exc}"
                    result_str = str(result)[:_MAX_TOOL_RESULT_CHARS]
                    yield Event(
                        EventType.TOOL_RESULT, self.name,
                        result_str,
                        {"tool": tool_name, "result": result_str},
                    )
                    messages.append({
                        "role": "assistant",
                        "content": f'{tool_name}({json.dumps(tool_args)})',
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Tool '{tool_name}' returned: {result_str}"
                        ),
                    })
                continue  # next iteration

            if fallback_tool_calls:
                # ── Process text-based tool calls (fallback) ─────────
                for call_info in fallback_tool_calls:
                    tool_name = call_info["name"]
                    tool_args = call_info["arguments"]
                    tool_obj = call_info["tool"]

                    if tool_name == _TRANSFER_TOOL_NAME:
                        target_name = tool_args.get("agent_name", "")
                        target = self.find_sub_agent(target_name)
                        if target is None:
                            messages.append({
                                "role": "assistant",
                                "content": (
                                    f'transfer_to_agent({{"agent_name":'
                                    f' "{target_name}"}})'
                                ),
                            })
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"Error: Agent '{target_name}' not found."
                                ),
                            })
                            continue
                        yield Event(
                            EventType.AGENT_TRANSFER, self.name,
                            f"Transferring to {target_name}",
                            {"target_agent": target_name},
                        )
                        sub_ctx = InvocationContext(
                            session=ctx.session,
                            user_message=ctx.user_message,
                            trace_collector=ctx.trace_collector,
                        )
                        yield from target._run_impl_traced(sub_ctx)
                        return

                    yield Event(
                        EventType.TOOL_CALL, self.name,
                        f"Calling {tool_name}",
                        {"tool": tool_name, "arguments": tool_args},
                    )
                    _tool_ctx = ToolContext(
                        state=ctx.session.state,
                        shared_content=ctx.session.shared_content,
                        local_content=self.local_content,
                        agent_name=self.name,
                        session_id=ctx.session.session_id,
                        _session=ctx.session,
                    )
                    try:
                        result = tool_obj.invoke(tool_args, _tool_ctx)
                    except Exception as exc:
                        result = f"Error: {exc}"
                    result_str = str(result)[:_MAX_TOOL_RESULT_CHARS]
                    yield Event(
                        EventType.TOOL_RESULT, self.name,
                        result_str,
                        {"tool": tool_name, "result": result_str},
                    )
                    messages.append({
                        "role": "assistant",
                        "content": (
                            f'{tool_name}({json.dumps(tool_args)})'
                        ),
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Tool '{tool_name}' returned: {result_str}"
                        ),
                    })
                continue  # next iteration

            # ── Text response (no tool calls) ────────────────────────
            final = assembled_text
            if _structured_parser is not None:
                final = yield from self._parse_structured_response(
                    ctx, messages, final, _structured_parser,
                    response_format, json_schema, tool_declarations,
                )
            yield Event(EventType.AGENT_MESSAGE, self.name, final)
            return

        # Max iterations reached — stream a final answer
        messages.append({
            "role": "user",
            "content": (
                "Please provide your final answer based on the tool "
                "results above."
            ),
        })
        messages = self._compact_messages(messages)
        chunks: list[str] = []
        for delta in self._call_llm_stream(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=response_format,
            json_schema=json_schema,
        ):
            chunks.append(delta)
            yield Event(EventType.TEXT_CHUNK, self.name, delta)

        final = "".join(chunks)
        if _structured_parser is not None:
            final = yield from self._parse_structured_response(
                ctx, messages, final, _structured_parser,
                response_format, json_schema, tool_declarations,
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, final)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        """Async LLM call with tool-calling loop.

        Offloads each ``generate_completion`` call to a thread so the event
        loop is never blocked.  Follows the same logic as ``_run_impl``,
        including ``transfer_to_agent`` handling.

        Args:
            ctx: The invocation context.

        Yields:
            Event objects produced during execution.
        """
        from nono.connector.connector_genai import ResponseFormat

        if ctx.user_message:
            yield Event(EventType.USER_MESSAGE, "user", ctx.user_message)

        messages = self._build_messages(ctx)
        all_tools = self._all_tools
        tool_declarations = (
            [t.to_function_declaration() for t in all_tools]
            if all_tools
            else None
        )
        response_format = (
            ResponseFormat.JSON if self.output_format == "json"
            else ResponseFormat.TEXT
        )

        for iteration in range(_MAX_TOOL_ITERATIONS):
            logger.debug(
                "[%s] async LLM call iteration %d, messages=%d",
                self.name, iteration, len(messages),
            )

            # Compact messages to avoid context window overflow
            messages = self._compact_messages(messages)

            _llm_start = time.perf_counter_ns()
            response = await asyncio.to_thread(
                self._call_llm,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_format,
                tools=tool_declarations,
            )
            _llm_elapsed = (time.perf_counter_ns() - _llm_start) / 1_000_000

            if ctx.trace_collector is not None:
                _output_text = str(response)
                ctx.trace_collector.record_llm_call(LLMCall(
                    provider=self._provider,
                    model=self._model or "",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    token_usage=TokenUsage(
                        input_tokens=_estimate_messages_tokens(messages),
                        output_tokens=estimate_tokens(_output_text),
                    ),
                    duration_ms=_llm_elapsed,
                ))

            tool_calls = self._extract_tool_calls(response, all_tools)

            if not tool_calls:
                yield Event(EventType.AGENT_MESSAGE, self.name, str(response))
                return

            for call_info in tool_calls:
                tool_name = call_info["name"]
                tool_args = call_info["arguments"]
                tool_obj = call_info["tool"]

                # ── transfer_to_agent handling ────────────────────────────
                if tool_name == _TRANSFER_TOOL_NAME:
                    target_name = tool_args.get("agent_name", "")
                    sub_message = tool_args.get("message", ctx.user_message)
                    target = self.find_sub_agent(target_name)

                    yield Event(
                        EventType.AGENT_TRANSFER,
                        self.name,
                        f"Transferring to {target_name}",
                        data={"target_agent": target_name, "message": sub_message},
                    )

                    if target is None:
                        result = f"Error: agent '{target_name}' not found."
                        logger.warning(
                            "[%s] transfer_to_agent: unknown agent %r",
                            self.name, target_name,
                        )
                    elif ctx.transfer_depth >= MAX_TRANSFER_DEPTH:
                        result = (
                            f"Error: maximum transfer depth ({MAX_TRANSFER_DEPTH}) "
                            f"exceeded. Aborting delegation to '{target_name}'."
                        )
                        logger.error(
                            "[%s] transfer_to_agent: depth %d exceeds limit %d",
                            self.name, ctx.transfer_depth, MAX_TRANSFER_DEPTH,
                        )
                    else:
                        sub_ctx = InvocationContext(
                            session=ctx.session,
                            user_message=sub_message,
                            parent_agent=self,
                            trace_collector=ctx.trace_collector,
                            transfer_depth=ctx.transfer_depth + 1,
                        )
                        result = await target.run_async(sub_ctx)

                    yield Event(
                        EventType.TOOL_RESULT,
                        self.name,
                        str(result),
                        data={"tool": _TRANSFER_TOOL_NAME, "result": result},
                    )

                    messages.append({
                        "role": "assistant",
                        "content": json.dumps({
                            "tool_call": _TRANSFER_TOOL_NAME,
                            "arguments": tool_args,
                        }),
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Agent '{target_name}' responded: {result}"
                        ),
                    })
                    continue

                # ── Regular tool handling ─────────────────────────────────
                if self.before_tool_callback:
                    modified = self.before_tool_callback(self, tool_name, tool_args)
                    if modified is not None:
                        tool_args = modified

                yield Event(
                    EventType.TOOL_CALL,
                    self.name,
                    f"Calling {tool_name}",
                    data={"tool": tool_name, "arguments": tool_args},
                )

                _tool_start = time.perf_counter_ns()
                _tool_error: str | None = None
                try:
                    _tool_ctx = ToolContext(
                        state=ctx.session.state,
                        shared_content=ctx.session.shared_content,
                        local_content=self.local_content,
                        agent_name=self.name,
                        session_id=ctx.session.session_id,
                        _session=ctx.session,
                    )
                    result = await asyncio.to_thread(
                        tool_obj.invoke, tool_args, _tool_ctx,
                    )
                except Exception as e:
                    logger.error("Tool %r failed: %s", tool_name, e)
                    result = f"Error: {e}"
                    _tool_error = str(e)
                finally:
                    _tool_elapsed = (time.perf_counter_ns() - _tool_start) / 1_000_000
                    if ctx.trace_collector is not None:
                        ctx.trace_collector.record_tool(ToolRecord(
                            tool_name=tool_name,
                            arguments=tool_args,
                            result=str(result),
                            duration_ms=_tool_elapsed,
                            error=_tool_error,
                        ))

                if self.after_tool_callback:
                    modified = self.after_tool_callback(
                        self, tool_name, tool_args, result,
                    )
                    if modified is not None:
                        result = modified

                yield Event(
                    EventType.TOOL_RESULT,
                    self.name,
                    str(result),
                    data={"tool": tool_name, "result": result},
                )

                messages.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "tool_call": tool_name,
                        "arguments": tool_args,
                    }),
                })
                result_str = str(result)
                if len(result_str) > _MAX_TOOL_RESULT_CHARS:
                    result_str = result_str[:_MAX_TOOL_RESULT_CHARS] + "… (truncated)"
                messages.append({
                    "role": "user",
                    "content": f"Tool '{tool_name}' returned: {result_str}",
                })

        messages.append({
            "role": "user",
            "content": "Please provide your final answer based on the tool results above.",
        })
        messages = self._compact_messages(messages)
        _llm_start = time.perf_counter_ns()
        response = await asyncio.to_thread(
            self._call_llm,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=response_format,
        )
        _llm_elapsed = (time.perf_counter_ns() - _llm_start) / 1_000_000
        if ctx.trace_collector is not None:
            _output_text = str(response)
            ctx.trace_collector.record_llm_call(LLMCall(
                provider=self._provider,
                model=self._model or "",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                token_usage=TokenUsage(
                    input_tokens=_estimate_messages_tokens(messages),
                    output_tokens=estimate_tokens(_output_text),
                ),
                duration_ms=_llm_elapsed,
            ))
        yield Event(EventType.AGENT_MESSAGE, self.name, str(response))

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_messages(self, ctx: InvocationContext) -> list[dict[str, str]]:
        """Build the message list for the LLM call.

        Combines: system instruction → session history → current user message.
        When sub-agents exist, the system prompt includes a note about
        available agents — actual delegation goes through the
        ``transfer_to_agent`` tool.

        Args:
            ctx: The invocation context.

        Returns:
            List of message dicts.
        """
        messages: list[dict[str, str]] = []

        # System instruction
        system_content = self.instruction
        if self.sub_agents:
            agent_list = "\n".join(
                f"  - {a.name}: {a.description or 'No description'}"
                for a in self.sub_agents
            )
            system_content += (
                "\n\nYou have access to specialist sub-agents via the "
                f"'{_TRANSFER_TOOL_NAME}' tool:\n{agent_list}"
            )

        messages.append({"role": "system", "content": system_content})

        # Session history
        messages.extend(ctx.session.get_messages())

        # Current user message
        if ctx.user_message:
            messages.append({"role": "user", "content": ctx.user_message})

        return messages

    @staticmethod
    def _prune_messages(
        messages: list[dict[str, str]],
        max_messages: int = _MAX_LOOP_MESSAGES,
    ) -> list[dict[str, str]]:
        """Keep messages within a sliding window to avoid context overflow.

        Always preserves the first message (system prompt) and the most
        recent *max_messages - 1* non-system messages.

        Args:
            messages: Full message list.
            max_messages: Maximum number of messages to retain.

        Returns:
            Pruned message list.
        """
        if len(messages) <= max_messages:
            return messages

        # Keep system prompt + most recent messages
        system = [m for m in messages[:1] if m.get("role") == "system"]
        rest = messages[1:] if system else messages
        kept = rest[-(max_messages - len(system)):]

        logger.debug(
            "Pruned messages from %d to %d (max=%d)",
            len(messages), len(system) + len(kept), max_messages,
        )
        return system + kept

    def _compact_messages(
        self,
        messages: list[dict[str, str]],
        max_messages: int = _MAX_LOOP_MESSAGES,
    ) -> list[dict[str, str]]:
        """Compact messages using the configured compaction strategy.

        When no strategy is configured (``compaction=None``), falls back
        to the static ``_prune_messages`` sliding window directly.

        Injects the agent's service into ``SummarizationStrategy`` lazily
        so that compaction can use the same LLM provider as the agent.

        Args:
            messages: Full message list.
            max_messages: Maximum number of messages to retain.

        Returns:
            Compacted message list.
        """
        strategy = self._compaction

        # No strategy → legacy sliding-window prune
        if strategy is None:
            return self._prune_messages(messages, max_messages)

        # Inject our service lazily for LLM-based strategies
        if isinstance(strategy, SummarizationStrategy) and strategy.service is None:
            strategy.service = self.service

        if not strategy.should_compact(messages, max_messages):
            return messages

        compacted, result = strategy.compact(messages, max_messages)

        if result.summary:
            logger.info(
                "[%s] Auto-compaction: %d → %d msgs (saved ~%d tokens)",
                self.name, result.original_count,
                result.compacted_count, result.estimated_tokens_saved,
            )

        return compacted

    @staticmethod
    def _load_compaction_strategy(dotted_path: str) -> CompactionStrategy:
        """Import and instantiate a compaction strategy from a dotted path.

        The path must point to a class that inherits from
        ``CompactionStrategy``.  The class is instantiated with no
        arguments.

        Args:
            dotted_path: Fully qualified class name,
                e.g. ``"mypackage.compaction.MyStrategy"``.

        Returns:
            An instance of the loaded strategy.

        Raises:
            ImportError: If the module cannot be imported.
            AttributeError: If the class is not found in the module.
            TypeError: If the class is not a ``CompactionStrategy`` subclass.
        """
        module_path, _, class_name = dotted_path.rpartition(".")
        if not module_path:
            raise ImportError(
                f"Invalid compaction strategy path: {dotted_path!r}. "
                "Expected 'module.ClassName'."
            )
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        if not (isinstance(cls, type) and issubclass(cls, CompactionStrategy)):
            raise TypeError(
                f"{dotted_path!r} is not a CompactionStrategy subclass"
            )
        return cls()

    def _extract_tool_calls(
        self,
        response: str,
        tools: list[FunctionTool] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract tool call instructions from an LLM response.

        Args:
            response: Raw LLM response text.
            tools: Available tools to match against (defaults to ``_all_tools``).

        Returns:
            List of tool call dicts, or empty list for a final text response.
        """
        available = tools if tools is not None else self._all_tools
        if not available:
            return []

        return parse_tool_calls(str(response), available)


# Alias: ``Agent`` is the recommended short name for ``LlmAgent``
Agent = LlmAgent
