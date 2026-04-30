"""
Tool system for Nono agents.

Provides the ``Tool`` protocol, ``FunctionTool`` wrapper, and the ``@tool``
decorator.  Tools are callable objects with metadata (name, description,
parameter schema) that an LLM agent can invoke via function calling.

Part of the Nono Agent Architecture (NAA).

Usage:
    from nono.agent.tool import tool

    @tool(description="Get the current weather for a city.")
    def get_weather(city: str, unit: str = "celsius") -> str:
        return f"22°{unit[0].upper()} in {city}"

    # Or wrap an existing function manually:
    from nono.agent.tool import FunctionTool

    def search_db(query: str) -> list[dict]:
        ...

    search_tool = FunctionTool(search_db, description="Search the database.")
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_type_hints

logger = logging.getLogger("Nono.Agent.Tool")


class ToolContext:
    """Context injected automatically into tools that request it.

    When a tool function declares a parameter typed as ``ToolContext``, the
    agent framework will **exclude** it from the JSON Schema sent to the LLM
    and **inject** a populated instance at invocation time.

    Part of the Nono Agent Architecture (NAA).

    Two content scopes are available:
    - ``shared_content``: Session-level, visible to **all** agents.
    - ``local_content``: Agent-level, **private** to the invoking agent.

    Attributes:
        state: Mutable session state dict — changes persist across turns.
            **Warning:** Direct ``state[key] = val`` is *not* thread-safe.
            Use :meth:`state_set` / :meth:`state_get` for safe access.
        shared_content: Session-scoped content store (visible to all agents).
        local_content: Agent-scoped content store (private to this agent).
        agent_name: Name of the agent that invoked the tool.
        session_id: Active session identifier.

    Example:
        >>> from nono.agent import tool, ToolContext
        >>>
        >>> @tool(description="Save a report.")
        ... def save_report(text: str, tool_context: ToolContext) -> str:
        ...     tool_context.shared_content.save("report", text)   # visible to all agents
        ...     tool_context.local_content.save("draft", text)     # private to this agent
        ...     tool_context.state_set("has_report", True)
        ...     return "Report saved"
    """

    def __init__(
        self,
        *,
        state: dict[str, Any] | None = None,
        shared_content: Any = None,
        local_content: Any = None,
        agent_name: str = "",
        session_id: str = "",
        _session: Any = None,
    ) -> None:
        self._session = _session
        self.state: dict[str, Any] = state if state is not None else {}
        self.agent_name: str = agent_name
        self.session_id: str = session_id

        # Lazy import to avoid circular dependency
        from .base import SharedContent

        if shared_content is None:
            shared_content = SharedContent()
        self.shared_content = shared_content

        if local_content is None:
            local_content = SharedContent()
        self.local_content = local_content

    # ── Thread-safe state helpers ──────────────────────────────────────

    def state_set(self, key: str, value: Any) -> None:
        """Set a state key — thread-safe when ``_session`` is available.

        Args:
            key: State key.
            value: Value to store.
        """
        if self._session is not None:
            self._session.state_set(key, value)
        else:
            self.state[key] = value

    def state_get(self, key: str, default: Any = None) -> Any:
        """Get a state value — thread-safe when ``_session`` is available.

        Args:
            key: State key.
            default: Value returned when key is absent.

        Returns:
            The stored value, or *default*.
        """
        if self._session is not None:
            return self._session.state_get(key, default)
        return self.state.get(key, default)

    def state_update(self, mapping: dict[str, Any]) -> None:
        """Merge keys into state — thread-safe when ``_session`` is available.

        Args:
            mapping: Dict of key/value pairs to merge.
        """
        if self._session is not None:
            self._session.state_update(mapping)
        else:
            self.state.update(mapping)

    def save_content(
        self,
        name: str,
        data: Any,
        *,
        content_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
        scope: str = "shared",
    ) -> Any:
        """Save content to the shared or local store.

        Args:
            name: Key for the content.
            data: Payload to store.
            content_type: MIME type hint.
            metadata: Extra information.
            scope: ``"shared"`` (session-level, default) or ``"local"``
                (agent-private).

        Returns:
            The created ``ContentItem``.
        """
        store = self.local_content if scope == "local" else self.shared_content
        return store.save(
            name,
            data,
            content_type=content_type,
            metadata=metadata,
            created_by=self.agent_name,
        )

    def load_content(
        self,
        name: str,
        *,
        scope: str = "shared",
    ) -> Any:
        """Load content from the shared or local store.

        Args:
            name: Key to retrieve.
            scope: ``"shared"`` (session-level, default) or ``"local"``
                (agent-private).

        Returns:
            The ``ContentItem``, or ``None``.
        """
        store = self.local_content if scope == "local" else self.shared_content
        return store.load(name)

    def __repr__(self) -> str:
        return (
            f"ToolContext(agent={self.agent_name!r}, "
            f"session={self.session_id!r}, "
            f"state_keys={list(self.state.keys())}, "
            f"shared={self.shared_content!r}, "
            f"local={self.local_content!r})"
        )


# Python type → JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(py_type: Any) -> str:
    """Map a Python type annotation to a JSON Schema type string.

    Args:
        py_type: Python type or annotation.

    Returns:
        JSON Schema type name.
    """
    origin = getattr(py_type, "__origin__", None)

    if origin is list:
        return "array"
    if origin is dict:
        return "object"

    return _TYPE_MAP.get(py_type, "string")


def _build_parameters_schema(fn: Callable) -> dict[str, Any]:
    """Introspect a function to build an OpenAI-compatible parameters schema.

    Args:
        fn: The function to introspect.

    Returns:
        A JSON Schema ``object`` describing the function's parameters.
    """
    sig = inspect.signature(fn)
    hints = get_type_hints(fn) if hasattr(fn, "__annotations__") else {}
    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        py_type = hints.get(param_name, str)

        # Exclude ToolContext — it is injected at runtime, not by the LLM
        if py_type is ToolContext:
            continue
        schema_type = _python_type_to_json_schema(py_type)
        properties[param_name] = {"type": schema_type}

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }

    if required:
        schema["required"] = required

    return schema


@dataclass
class FunctionTool:
    """A callable tool with metadata for LLM function-calling.

    Args:
        fn: The underlying Python function.
        name: Tool name (defaults to ``fn.__name__``).
        description: Human-readable description for the LLM.
        parameters_schema: JSON Schema for the parameters.  Auto-generated
            from type hints if not provided.

    Example:
        >>> def add(a: int, b: int) -> int:
        ...     return a + b
        >>> t = FunctionTool(add, description="Add two numbers.")
        >>> t("a", 1, "b", 2)  # or t.invoke({"a": 1, "b": 2})
        3
    """
    fn: Callable
    name: str = ""
    description: str = ""
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    _needs_tool_context: bool = field(default=False, init=False, repr=False)
    _tool_context_param: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.fn.__name__
        if not self.description:
            self.description = (self.fn.__doc__ or "").strip().split("\n")[0]
        if not self.parameters_schema:
            self.parameters_schema = _build_parameters_schema(self.fn)

        # Detect if the function expects a ToolContext parameter
        hints = get_type_hints(self.fn) if hasattr(self.fn, "__annotations__") else {}
        for p_name, p_type in hints.items():
            if p_type is ToolContext:
                self._needs_tool_context = True
                self._tool_context_param = p_name
                break

    def invoke(
        self,
        args: dict[str, Any],
        tool_context: ToolContext | None = None,
    ) -> Any:
        """Invoke the tool with a dict of arguments.

        If the underlying function declares a ``ToolContext`` parameter, a
        populated instance is injected automatically.

        Args:
            args: Keyword arguments from the LLM.
            tool_context: Optional context to inject.  When ``None`` and the
                function requires one, an empty ``ToolContext`` is used.

        Returns:
            The function's return value.
        """
        logger.debug("Tool %r invoked with args: %s", self.name, args)
        call_args = dict(args)
        if self._needs_tool_context:
            call_args[self._tool_context_param] = tool_context or ToolContext()
        return self.fn(**call_args)

    def __call__(self, **kwargs: Any) -> Any:
        """Shorthand: call the tool with keyword arguments."""
        return self.invoke(kwargs)

    def to_function_declaration(self) -> dict[str, Any]:
        """Convert to an OpenAI-compatible function declaration.

        Returns:
            Dict ready for the ``tools`` parameter of a chat completion call.

        Example:
            >>> t.to_function_declaration()
            {"type": "function", "function": {"name": "add", ...}}
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def __repr__(self) -> str:
        return f"FunctionTool({self.name!r})"


def tool(
    fn: Callable | None = None,
    *,
    name: str = "",
    description: str = "",
) -> FunctionTool | Callable[[Callable], FunctionTool]:
    """Decorator to create a ``FunctionTool`` from a function.

    Can be used with or without arguments:

        @tool
        def my_tool(x: int) -> str: ...

        @tool(description="Does something special.")
        def my_tool(x: int) -> str: ...

    Args:
        fn: The function to wrap (when used without parentheses).
        name: Override tool name (default: function name).
        description: Override description (default: first docstring line).

    Returns:
        A ``FunctionTool`` instance.
    """
    def _wrap(func: Callable) -> FunctionTool:
        return FunctionTool(func, name=name, description=description)

    if fn is not None:
        return _wrap(fn)

    return _wrap


def parse_tool_calls(response_text: str, tools: list[FunctionTool]) -> list[dict[str, Any]]:
    """Parse tool call instructions from an LLM response.

    Handles responses that contain JSON function-call blocks in the OpenAI
    format: ``{"name": "tool_name", "arguments": {...}}``.

    Args:
        response_text: Raw LLM response text.
        tools: Available tools to match against.

    Returns:
        List of dicts with ``"name"``, ``"arguments"``, and ``"tool"`` keys.
    """
    tool_map = {t.name: t for t in tools}
    calls: list[dict[str, Any]] = []

    # Try to parse the whole response as JSON (single call)
    try:
        data = json.loads(response_text)
        if isinstance(data, dict) and "name" in data:
            data = [data]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "name" in item:
                    name = item["name"]
                    args = item.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    if name in tool_map:
                        calls.append({
                            "name": name,
                            "arguments": args,
                            "tool": tool_map[name],
                        })
    except (json.JSONDecodeError, TypeError):
        pass

    return calls


# ============================================================================
# ACI QUALITY VALIDATION
# ============================================================================

# Minimum description length to be considered meaningful.
_MIN_DESCRIPTION_LEN: int = 10


@dataclass
class ToolIssue:
    """A single quality issue found during tool validation.

    Attributes:
        tool_name: Name of the tool with the issue.
        severity: ``"error"`` (will likely break LLM usage) or
            ``"warning"`` (could degrade quality).
        message: Human-readable explanation of the issue.
    """

    tool_name: str
    severity: str  # "error" | "warning"
    message: str


def validate_tools(
    tools: list[FunctionTool],
    *,
    min_description_len: int = _MIN_DESCRIPTION_LEN,
    warn: bool = True,
) -> list[ToolIssue]:
    """Validate tool descriptions and schemas for ACI quality.

    Checks each tool for common issues that degrade LLM function-calling
    accuracy.  Follows the Anthropic principle: *"Make your tool
    descriptions extremely detailed — this is essentially the tool's
    ACI (Agent-Computer Interface)."*

    Checks performed:
    - Description is non-empty.
    - Description meets minimum length threshold.
    - Tool name is meaningful (>= 2 characters, not a single letter).
    - Parameters have types defined.
    - At least one parameter exists (tools with zero params are suspicious).

    Args:
        tools: List of ``FunctionTool`` instances to validate.
        min_description_len: Minimum acceptable description length
            (default: 10 characters).
        warn: If ``True``, emit ``logging.WARNING`` for each issue.

    Returns:
        List of ``ToolIssue`` objects.  Empty list means all tools pass.

    Example:
        >>> issues = validate_tools([my_tool, other_tool])
        >>> for issue in issues:
        ...     print(f"[{issue.severity}] {issue.tool_name}: {issue.message}")
    """
    issues: list[ToolIssue] = []

    for t in tools:
        # --- Description checks ---
        if not t.description or not t.description.strip():
            issues.append(ToolIssue(
                tool_name=t.name,
                severity="error",
                message=(
                    "Missing description. The LLM relies on the description "
                    "to decide when and how to call this tool. Add a clear, "
                    "detailed description via @tool(description=...) or a "
                    "docstring."
                ),
            ))
        elif len(t.description.strip()) < min_description_len:
            issues.append(ToolIssue(
                tool_name=t.name,
                severity="warning",
                message=(
                    f"Description is only {len(t.description.strip())} chars "
                    f"(minimum recommended: {min_description_len}). "
                    "Short descriptions reduce the LLM's ability to choose "
                    "the right tool. Explain what the tool does, when to use "
                    "it, and what it returns."
                ),
            ))

        # --- Name checks ---
        if len(t.name) < 2:
            issues.append(ToolIssue(
                tool_name=t.name,
                severity="warning",
                message=(
                    "Tool name is too short. Use a descriptive name "
                    "like 'search_database' instead of single letters."
                ),
            ))

        # --- Parameter checks ---
        props = t.parameters_schema.get("properties", {})

        if not props:
            issues.append(ToolIssue(
                tool_name=t.name,
                severity="warning",
                message=(
                    "Tool has no parameters. If the tool truly needs no "
                    "input, this is fine. Otherwise, add typed parameters "
                    "so the LLM knows what arguments to provide."
                ),
            ))

        for param_name, param_schema in props.items():
            if not param_schema.get("type"):
                issues.append(ToolIssue(
                    tool_name=t.name,
                    severity="warning",
                    message=(
                        f"Parameter '{param_name}' has no type annotation. "
                        "Add a type hint (str, int, float, bool, list, dict) "
                        "so the LLM generates correct argument values."
                    ),
                ))

    if warn:
        for issue in issues:
            log_fn = logger.error if issue.severity == "error" else logger.warning
            log_fn("Tool '%s': %s", issue.tool_name, issue.message)

    return issues
