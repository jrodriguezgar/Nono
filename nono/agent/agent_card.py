"""
Agent Card — A2A Protocol-compliant agent discovery metadata.

Implements the **Agent Card** specification from the
`Agent-to-Agent (A2A) Protocol <https://a2a-protocol.org/specification/>`_
(Section 4.4 — Agent Discovery Objects, Section 8 — Agent Discovery).

An Agent Card is a JSON metadata document published by an A2A Server that
describes its identity, capabilities, skills, service endpoint, and
authentication requirements.  Clients discover agents by fetching the
well-known URI ``/.well-known/agent-card.json``.

Nono can generate Agent Cards automatically from:
    - ``LlmAgent`` / ``Agent`` instances
    - ``Workflow`` pipelines
    - ``BaseAgent`` orchestration hierarchies

Usage::

    from nono.agent import Agent
    from nono.agent.agent_card import AgentCard, to_agent_card

    agent = Agent(
        name="weather_assistant",
        model="gemini-3-flash-preview",
        provider="google",
        instruction="You are a helpful weather assistant.",
        description="Provides weather forecasts and analysis.",
    )

    card = to_agent_card(agent, url="https://my-agent.example.com")
    print(card.to_json(indent=2))

    # Serve the card via HTTP
    from nono.agent.agent_card import serve_agent_card
    serve_agent_card(card, host="0.0.0.0", port=8080)

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("Nono.Agent.AgentCard")


# ── A2A Protocol Version ──────────────────────────────────────────────────────

A2A_PROTOCOL_VERSION: str = "0.3"
"""Current A2A protocol version supported by Nono."""


# ── Data Model (Section 4.4 — Agent Discovery Objects) ────────────────────────

@dataclass
class AgentSkill:
    """A specific skill/capability offered by an agent (A2A §4.4.5).

    Args:
        id: Unique identifier for the skill.
        name: Human-readable skill name.
        description: Description of what the skill does.
        tags: Categorization tags for discovery.
        examples: Example prompts or inputs.
        input_modes: Accepted MIME types for input.
        output_modes: Produced MIME types for output.
    """
    id: str
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    input_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    output_modes: list[str] = field(default_factory=lambda: ["text/plain"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to A2A-compliant dict (camelCase)."""
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
        }
        if self.tags:
            d["tags"] = self.tags
        if self.examples:
            d["examples"] = self.examples
        if self.input_modes:
            d["inputModes"] = self.input_modes
        if self.output_modes:
            d["outputModes"] = self.output_modes
        return d


@dataclass
class AgentProvider:
    """Organization that provides the agent (A2A §4.4.2).

    Args:
        organization: Name of the organization.
        url: Organization website URL.
    """
    organization: str
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to A2A-compliant dict."""
        d: dict[str, Any] = {"organization": self.organization}
        if self.url:
            d["url"] = self.url
        return d


@dataclass
class AgentCapabilities:
    """Optional capabilities declared by the agent (A2A §4.4.3).

    Args:
        streaming: Whether the agent supports streaming responses.
        push_notifications: Whether the agent supports push notifications.
        state_transition_history: Whether task state history is available.
        extended_agent_card: Whether an extended card is available.
    """
    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = False
    extended_agent_card: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to A2A-compliant dict (camelCase)."""
        d: dict[str, Any] = {}
        if self.streaming:
            d["streaming"] = True
        if self.push_notifications:
            d["pushNotifications"] = True
        if self.state_transition_history:
            d["stateTransitionHistory"] = True
        if self.extended_agent_card:
            d["extendedAgentCard"] = True
        return d


@dataclass
class AgentInterface:
    """A supported protocol interface (A2A §4.4.6).

    Args:
        url: Endpoint URL for this interface.
        protocol_binding: Protocol identifier (``"JSONRPC"``, ``"HTTP+JSON"``, etc.).
        protocol_version: A2A protocol version supported.
    """
    url: str
    protocol_binding: str = "HTTP+JSON"
    protocol_version: str = A2A_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize to A2A-compliant dict (camelCase)."""
        return {
            "url": self.url,
            "protocolBinding": self.protocol_binding,
            "protocolVersion": self.protocol_version,
        }


@dataclass
class AgentCard:
    """A2A Agent Card — agent discovery metadata (A2A §4.4.1).

    This is the root document served at ``/.well-known/agent-card.json``.

    Args:
        name: Human-readable agent name.
        description: Description of the agent's purpose.
        supported_interfaces: Protocol endpoints the agent supports.
        skills: List of skills the agent offers.
        provider: Organization that provides the agent.
        capabilities: Optional capability flags.
        version: Agent version string.
        icon_url: URL to the agent's icon.
        documentation_url: URL to the agent's documentation.
        default_input_modes: Default accepted input MIME types.
        default_output_modes: Default produced output MIME types.
    """
    name: str
    description: str = ""
    supported_interfaces: list[AgentInterface] = field(default_factory=list)
    skills: list[AgentSkill] = field(default_factory=list)
    provider: AgentProvider | None = None
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)
    version: str = ""
    icon_url: str = ""
    documentation_url: str = ""
    default_input_modes: list[str] = field(
        default_factory=lambda: ["text/plain"],
    )
    default_output_modes: list[str] = field(
        default_factory=lambda: ["text/plain"],
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to A2A-compliant dict (camelCase keys).

        Returns:
            Dict ready for JSON serialization following A2A §5.5.
        """
        d: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }

        if self.supported_interfaces:
            d["supportedInterfaces"] = [
                iface.to_dict() for iface in self.supported_interfaces
            ]

        if self.provider:
            d["provider"] = self.provider.to_dict()

        if self.icon_url:
            d["iconUrl"] = self.icon_url

        if self.version:
            d["version"] = self.version

        if self.documentation_url:
            d["documentationUrl"] = self.documentation_url

        caps = self.capabilities.to_dict()
        if caps:
            d["capabilities"] = caps

        if self.default_input_modes:
            d["defaultInputModes"] = self.default_input_modes

        if self.default_output_modes:
            d["defaultOutputModes"] = self.default_output_modes

        if self.skills:
            d["skills"] = [s.to_dict() for s in self.skills]

        return d

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to A2A-compliant JSON string.

        Args:
            indent: JSON indentation (``None`` for compact).

        Returns:
            JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCard:
        """Deserialize from an A2A-compliant dict.

        Args:
            data: Dict with camelCase keys per A2A spec.

        Returns:
            Populated ``AgentCard`` instance.
        """
        interfaces = [
            AgentInterface(
                url=iface["url"],
                protocol_binding=iface.get("protocolBinding", "HTTP+JSON"),
                protocol_version=iface.get("protocolVersion", A2A_PROTOCOL_VERSION),
            )
            for iface in data.get("supportedInterfaces", [])
        ]

        skills = [
            AgentSkill(
                id=s["id"],
                name=s["name"],
                description=s.get("description", ""),
                tags=s.get("tags", []),
                examples=s.get("examples", []),
                input_modes=s.get("inputModes", ["text/plain"]),
                output_modes=s.get("outputModes", ["text/plain"]),
            )
            for s in data.get("skills", [])
        ]

        provider_data = data.get("provider")
        provider = (
            AgentProvider(
                organization=provider_data["organization"],
                url=provider_data.get("url", ""),
            )
            if provider_data
            else None
        )

        caps_data = data.get("capabilities", {})
        capabilities = AgentCapabilities(
            streaming=caps_data.get("streaming", False),
            push_notifications=caps_data.get("pushNotifications", False),
            state_transition_history=caps_data.get("stateTransitionHistory", False),
            extended_agent_card=caps_data.get("extendedAgentCard", False),
        )

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            supported_interfaces=interfaces,
            skills=skills,
            provider=provider,
            capabilities=capabilities,
            version=data.get("version", ""),
            icon_url=data.get("iconUrl", ""),
            documentation_url=data.get("documentationUrl", ""),
            default_input_modes=data.get("defaultInputModes", ["text/plain"]),
            default_output_modes=data.get("defaultOutputModes", ["text/plain"]),
        )

    @classmethod
    def from_json(cls, json_str: str) -> AgentCard:
        """Deserialize from a JSON string.

        Args:
            json_str: A2A-compliant JSON string.

        Returns:
            Populated ``AgentCard`` instance.
        """
        return cls.from_dict(json.loads(json_str))


# ── Conversion helpers ────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """Convert a name to a URL-safe slug.

    Args:
        name: Agent or skill name.

    Returns:
        Lowercase, hyphenated slug.
    """
    return name.lower().replace(" ", "-").replace("_", "-")


def _extract_skills_from_agent(agent: Any) -> list[AgentSkill]:
    """Extract AgentSkill entries from an agent's tools and skills.

    Args:
        agent: A ``BaseAgent`` (typically ``LlmAgent``) instance.

    Returns:
        List of ``AgentSkill`` objects.
    """
    skills: list[AgentSkill] = []

    # From skill objects (nono.agent.skill.BaseSkill)
    if hasattr(agent, "skills") and agent.skills:
        for sk in agent.skills:
            desc = getattr(sk, "descriptor", None)
            if desc is not None:
                skills.append(AgentSkill(
                    id=_slugify(desc.name),
                    name=desc.name,
                    description=desc.description or "",
                    tags=list(desc.tags) if hasattr(desc, "tags") and desc.tags else [],
                ))
            else:
                sk_name = getattr(sk, "name", type(sk).__name__)
                skills.append(AgentSkill(
                    id=_slugify(sk_name),
                    name=sk_name,
                    description=getattr(sk, "description", ""),
                ))

    # From function tools (nono.agent.tool.FunctionTool)
    if hasattr(agent, "tools") and agent.tools:
        for t in agent.tools:
            t_name = getattr(t, "name", "")
            if t_name and t_name != "transfer_to_agent":
                skills.append(AgentSkill(
                    id=_slugify(t_name),
                    name=t_name,
                    description=getattr(t, "description", ""),
                    tags=["tool"],
                ))

    # From sub-agents (delegation capabilities)
    if hasattr(agent, "sub_agents") and agent.sub_agents:
        for sub in agent.sub_agents:
            skills.append(AgentSkill(
                id=_slugify(sub.name),
                name=sub.name,
                description=sub.description or f"Delegate to {sub.name}",
                tags=["sub-agent"],
            ))

    return skills


def _extract_skills_from_workflow(workflow: Any) -> list[AgentSkill]:
    """Extract AgentSkill entries from a Workflow's steps.

    Args:
        workflow: A ``Workflow`` instance.

    Returns:
        List of ``AgentSkill`` objects — one per workflow step.
    """
    skills: list[AgentSkill] = []

    step_order: list[str] = getattr(workflow, "_step_order", [])
    step_fns: dict = getattr(workflow, "_step_fns", {})

    for step_name in step_order:
        fn = step_fns.get(step_name)
        doc = (fn.__doc__ or "").strip().split("\n")[0] if fn and fn.__doc__ else ""
        skills.append(AgentSkill(
            id=_slugify(step_name),
            name=step_name,
            description=doc or f"Workflow step: {step_name}",
            tags=["workflow-step"],
        ))

    return skills


def to_agent_card(
    source: Any,
    *,
    url: str = "http://localhost:8080",
    protocol_binding: str = "HTTP+JSON",
    version: str = "",
    provider: AgentProvider | None = None,
    capabilities: AgentCapabilities | None = None,
    icon_url: str = "",
    documentation_url: str = "",
    extra_skills: list[AgentSkill] | None = None,
) -> AgentCard:
    """Generate an A2A Agent Card from a Nono agent or workflow.

    Automatically extracts metadata (name, description, skills, tools,
    sub-agents) from the source object and maps them to the A2A schema.

    Supported source types:
        - ``LlmAgent`` / ``Agent``
        - ``BaseAgent`` subclasses (orchestrators)
        - ``Workflow``

    Args:
        source: The agent or workflow to describe.
        url: Base URL where the agent's A2A endpoint is hosted.
        protocol_binding: Protocol binding identifier.
        version: Agent version string.
        provider: Organization providing the agent.
        capabilities: Capability flags override.
        icon_url: URL to the agent's icon.
        documentation_url: URL to agent documentation.
        extra_skills: Additional skills to append.

    Returns:
        A populated ``AgentCard``.

    Raises:
        TypeError: If *source* is not a recognized type.

    Example::

        card = to_agent_card(my_agent, url="https://api.example.com")
        print(card.to_json())
    """
    from .base import BaseAgent

    name: str = ""
    description: str = ""
    skills: list[AgentSkill] = []

    # --- Agent ---
    if isinstance(source, BaseAgent):
        name = source.name
        description = source.description or ""

        # Enrich description with instruction if available
        instruction = getattr(source, "instruction", "")
        if not description and instruction:
            description = instruction[:200]

        skills = _extract_skills_from_agent(source)

        # Auto-detect streaming capability
        if capabilities is None:
            has_streaming = hasattr(source, "run_stream") or hasattr(source, "run_async_stream")
            capabilities = AgentCapabilities(streaming=has_streaming)

    # --- Workflow ---
    elif _is_workflow(source):
        wf_name = getattr(source, "_name", "workflow")
        name = wf_name
        description = (source.__doc__ or "").strip().split("\n")[0] if source.__doc__ else f"Workflow: {wf_name}"
        skills = _extract_skills_from_workflow(source)

        if capabilities is None:
            capabilities = AgentCapabilities(streaming=True)

    else:
        raise TypeError(
            f"Cannot generate Agent Card from {type(source).__name__}. "
            "Expected BaseAgent or Workflow."
        )

    # Append extra skills
    if extra_skills:
        skills.extend(extra_skills)

    # Version fallback
    if not version:
        try:
            from nono import __version__
            version = __version__
        except ImportError:
            version = "0.0.0"

    interface = AgentInterface(
        url=url,
        protocol_binding=protocol_binding,
        protocol_version=A2A_PROTOCOL_VERSION,
    )

    # Detect output modes from agent's output_format
    output_modes = ["text/plain"]
    output_format = getattr(source, "output_format", "text")
    if output_format == "json":
        output_modes = ["application/json", "text/plain"]

    card = AgentCard(
        name=name,
        description=description,
        supported_interfaces=[interface],
        skills=skills,
        provider=provider,
        capabilities=capabilities or AgentCapabilities(),
        version=version,
        icon_url=icon_url,
        documentation_url=documentation_url,
        default_input_modes=["text/plain"],
        default_output_modes=output_modes,
    )

    logger.info("Generated Agent Card for %r (%d skills)", name, len(skills))
    return card


def _is_workflow(obj: Any) -> bool:
    """Check if an object is a Workflow instance without importing."""
    cls_name = type(obj).__name__
    return cls_name == "Workflow" and hasattr(obj, "_step_order")


# ── HTTP Server for Agent Card ────────────────────────────────────────────────

def serve_agent_card(
    card: AgentCard,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
    path: str = "/.well-known/agent-card.json",
) -> None:
    """Start an HTTP server that serves the Agent Card.

    Serves the card at the well-known URI ``/.well-known/agent-card.json``
    as specified by A2A §8.2.  Uses Python's built-in ``http.server`` —
    suitable for development and lightweight deployments.

    Args:
        card: The ``AgentCard`` to serve.
        host: Bind address.
        port: Bind port.
        path: URI path for the card endpoint.

    Example::

        serve_agent_card(card, host="0.0.0.0", port=8080)
        # Card available at http://localhost:8080/.well-known/agent-card.json
    """
    import http.server

    card_json = card.to_json()
    card_bytes = card_json.encode("utf-8")

    class AgentCardHandler(http.server.BaseHTTPRequestHandler):
        """HTTP handler that serves the Agent Card."""

        def do_GET(self) -> None:
            """Handle GET requests."""
            if self.path == path or self.path == path.rstrip("/"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "public, max-age=3600")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(card_bytes)))
                self.end_headers()
                self.wfile.write(card_bytes)
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "Not Found"}')

        def log_message(self, format: str, *args: Any) -> None:
            """Route logs through Nono's logger."""
            logger.debug(format, *args)

    server = http.server.HTTPServer((host, port), AgentCardHandler)
    logger.info(
        "Serving Agent Card at http://%s:%d%s",
        host if host != "0.0.0.0" else "localhost",
        port,
        path,
    )
    print(f"Agent Card server running at http://{host}:{port}{path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Agent Card server stopped.")
        server.server_close()


def save_agent_card(card: AgentCard, path: str) -> None:
    """Write an Agent Card to a JSON file.

    Args:
        card: The ``AgentCard`` to persist.
        path: File path for the output JSON.
    """
    from pathlib import Path
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(card.to_json(), encoding="utf-8")
    logger.info("Agent Card saved to %s", path)


def load_agent_card(path: str) -> AgentCard:
    """Load an Agent Card from a JSON file.

    Args:
        path: File path to the agent card JSON.

    Returns:
        Populated ``AgentCard`` instance.
    """
    from pathlib import Path
    content = Path(path).read_text(encoding="utf-8")
    return AgentCard.from_json(content)
