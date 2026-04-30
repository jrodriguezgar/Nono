"""
Tasker-Agent integration — create agent tools from TaskExecutor.

Bridges ``nono.tasker`` and ``nono.agent``: wraps TaskExecutor capabilities
as ``FunctionTool`` instances that LLM agents can invoke via function calling.

Two factory functions are provided:

- ``tasker_tool``: Build a tool from explicit provider/model/schema parameters.
- ``json_task_tool``: Build a tool from a JSON task definition file.

Usage:
    from nono.agent import Agent, Runner
    from nono.agent.tasker_tool import tasker_tool, json_task_tool

    # Option 1 — inline configuration
    classify = tasker_tool(
        name="classify_names",
        description="Classify strings as person names or other entities.",
        provider="google",
        model="gemini-3-flash-preview",
        temperature="data_cleaning",
    )

    # Option 2 — from a JSON task file
    classify = json_task_tool("nono/tasker/prompts/name_classifier.json")

    agent = Agent(
        name="analyst",
        model="gemini-3-flash-preview",
        instruction="Use classify_names when the user provides names.",
        tools=[classify],
    )
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from .tool import FunctionTool

logger = logging.getLogger("Nono.Agent.TaskerTool")


def tasker_tool(
    *,
    name: str = "execute_task",
    description: str = "Execute an AI task using TaskExecutor.",
    provider: str = "google",
    model: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    temperature: Union[float, str] = 0.7,
    max_tokens: int = 2048,
    output_schema: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
) -> FunctionTool:
    """Create a FunctionTool that wraps ``TaskExecutor.execute()``.

    The returned tool accepts a single ``prompt`` parameter.  The LLM agent
    can invoke it to delegate a sub-task to a dedicated ``TaskExecutor``
    with its own provider/model/temperature settings.

    Args:
        name: Tool name visible to the agent.
        description: Tool description for the LLM.
        provider: AI provider for the TaskExecutor.
        model: Model name.
        api_key: Optional API key (auto-resolved if ``None``).
        temperature: Sampling temperature (float or preset name).
        max_tokens: Maximum response tokens.
        output_schema: Optional JSON schema for structured output.
        system_prompt: Optional system instruction prepended to each call.

    Returns:
        A ``FunctionTool`` wrapping ``TaskExecutor.execute()``.

    Example:
        >>> from nono.agent.tasker_tool import tasker_tool
        >>> tool = tasker_tool(
        ...     name="summarise",
        ...     description="Summarise a document.",
        ...     provider="google",
        ...     model="gemini-3-flash-preview",
        ...     system_prompt="You are a professional summariser.",
        ... )
        >>> tool.name
        'summarise'
    """

    def _execute_task(prompt: str) -> str:
        """Run a prompt through TaskExecutor.

        Args:
            prompt: The user prompt to process.

        Returns:
            Generated text from the AI provider.
        """
        from ..tasker.genai_tasker import TaskExecutor

        executor = TaskExecutor(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        input_data: Union[str, List[Dict[str, str]]]
        if system_prompt:
            input_data = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            input_data = prompt

        return executor.execute(input_data, output_schema=output_schema)

    return FunctionTool(
        fn=_execute_task,
        name=name,
        description=description,
    )


def json_task_tool(
    task_file: str,
    *,
    name: str = "",
    description: str = "",
    provider: str = "",
    model: str = "",
    api_key: Optional[str] = None,
) -> FunctionTool:
    """Create a FunctionTool from a JSON task definition file.

    Reads the task JSON to extract name, description, provider/model config,
    and output schema.  The returned tool accepts ``data`` (the input to
    process) and delegates to ``TaskExecutor.run_json_task()``.

    Args:
        task_file: Path to the JSON task file.
        name: Override tool name (default: task name from JSON).
        description: Override description (default: task description from JSON).
        provider: Override provider (default: from JSON ``genai.provider``).
        model: Override model (default: from JSON ``genai.model``).
        api_key: Optional API key override.

    Returns:
        A ``FunctionTool`` wrapping ``TaskExecutor.run_json_task()``.

    Raises:
        FileNotFoundError: If *task_file* does not exist.

    Example:
        >>> from nono.agent.tasker_tool import json_task_tool
        >>> tool = json_task_tool("nono/tasker/prompts/name_classifier.json")
        >>> tool.name
        'name_classifier'
    """
    if not os.path.exists(task_file):
        raise FileNotFoundError(f"Task file not found: {task_file}")

    with open(task_file, "r", encoding="utf-8") as f:
        task_def = json.load(f)

    task_meta = task_def.get("task", {})
    genai_config = task_def.get("genai", {})

    tool_name = name or task_meta.get(
        "name",
        os.path.splitext(os.path.basename(task_file))[0],
    )
    tool_desc = description or task_meta.get(
        "description",
        f"Execute task from {task_file}",
    )
    task_provider = provider or genai_config.get("provider", "google")
    task_model = model or genai_config.get("model", "gemini-3-flash-preview")

    _task_file = task_file  # capture in closure

    def _run_json_task(data: str) -> str:
        """Execute the JSON-defined task with the given data.

        Args:
            data: Input data string for the task.

        Returns:
            Task execution result.
        """
        from ..tasker.genai_tasker import TaskExecutor

        executor = TaskExecutor(
            provider=task_provider,
            model=task_model,
            api_key=api_key,
        )
        return executor.run_json_task(_task_file, data)

    return FunctionTool(
        fn=_run_json_task,
        name=tool_name,
        description=tool_desc,
    )
