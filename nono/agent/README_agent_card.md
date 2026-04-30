# Agent Card — A2A Protocol Discovery

Nono implements the **Agent Card** protocol from the [Agent-to-Agent (A2A) specification](https://a2a-protocol.org/specification/) (v0.3+), enabling automated discovery of agents and workflows by external systems.

An **Agent Card** is a JSON metadata document that describes an agent's identity, capabilities, skills, service endpoint, and authentication requirements. It is served at the well-known URI `/.well-known/agent-card.json`.

---

## Quick Start

### Generate from an Agent

```python
from nono.agent import Agent, to_agent_card

agent = Agent(
    name="weather_assistant",
    model="gemini-3-flash-preview",
    provider="google",
    instruction="You are a helpful weather assistant.",
    description="Provides weather forecasts and climate analysis.",
)

# Generate the card
card = to_agent_card(agent, url="https://my-agent.example.com")
print(card.to_json(indent=2))
```

### Generate from any Agent (convenience method)

Every `BaseAgent` subclass has a built-in `.agent_card()` method:

```python
card = agent.agent_card(url="https://api.example.com")
print(card.to_json())
```

### Generate from a Workflow

```python
from nono.workflows import Workflow
from nono.agent.agent_card import to_agent_card

flow = Workflow("research_pipeline")
flow.step("research", research_fn)
flow.step("analyze", analyze_fn)
flow.step("write", write_fn)
flow.connect("research", "analyze")
flow.connect("analyze", "write")

card = to_agent_card(flow, url="https://pipeline.example.com")
```

---

## Serving the Agent Card

### Built-in HTTP Server

```python
from nono.agent.agent_card import serve_agent_card

# Serves at http://localhost:8080/.well-known/agent-card.json
serve_agent_card(card, host="0.0.0.0", port=8080)
```

### Save to File (for static hosting)

```python
from nono.agent.agent_card import save_agent_card

save_agent_card(card, ".well-known/agent-card.json")
```

### Load from File

```python
from nono.agent.agent_card import load_agent_card

card = load_agent_card(".well-known/agent-card.json")
```

---

## Output Format

The generated JSON follows the A2A specification (§4.4, §8) with camelCase field names:

```json
{
  "name": "weather_assistant",
  "description": "Provides weather forecasts and climate analysis.",
  "supportedInterfaces": [
    {
      "url": "https://my-agent.example.com",
      "protocolBinding": "HTTP+JSON",
      "protocolVersion": "0.3"
    }
  ],
  "version": "1.1.0",
  "capabilities": {
    "streaming": true
  },
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/plain"],
  "skills": [
    {
      "id": "get-weather",
      "name": "get_weather",
      "description": "Get the current weather for a city.",
      "tags": ["tool"]
    }
  ]
}
```

---

## Data Model

### `AgentCard`

Root document served at `/.well-known/agent-card.json`.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Human-readable agent name |
| `description` | `str` | Agent purpose description |
| `supported_interfaces` | `list[AgentInterface]` | Protocol endpoints |
| `skills` | `list[AgentSkill]` | Skills/capabilities offered |
| `provider` | `AgentProvider \| None` | Organization info |
| `capabilities` | `AgentCapabilities` | Capability flags |
| `version` | `str` | Agent version |
| `icon_url` | `str` | Icon URL |
| `documentation_url` | `str` | Documentation URL |
| `default_input_modes` | `list[str]` | Default input MIME types |
| `default_output_modes` | `list[str]` | Default output MIME types |

### `AgentSkill`

A specific skill offered by the agent (A2A §4.4.5).

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique identifier |
| `name` | `str` | Human-readable name |
| `description` | `str` | What the skill does |
| `tags` | `list[str]` | Discovery tags |
| `examples` | `list[str]` | Example inputs |
| `input_modes` | `list[str]` | Accepted MIME types |
| `output_modes` | `list[str]` | Produced MIME types |

### `AgentCapabilities`

Optional capability flags (A2A §4.4.3).

| Field | Type | Default | Description |
|---|---|---|---|
| `streaming` | `bool` | `False` | Supports streaming responses |
| `push_notifications` | `bool` | `False` | Supports push notifications |
| `state_transition_history` | `bool` | `False` | Task history available |
| `extended_agent_card` | `bool` | `False` | Extended card available |

### `AgentProvider`

Organization providing the agent (A2A §4.4.2).

| Field | Type | Description |
|---|---|---|
| `organization` | `str` | Organization name |
| `url` | `str` | Organization URL |

### `AgentInterface`

Protocol endpoint (A2A §4.4.6).

| Field | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | — | Endpoint URL |
| `protocol_binding` | `str` | `"HTTP+JSON"` | Protocol identifier |
| `protocol_version` | `str` | `"0.3"` | A2A version |

---

## Automatic Skill Extraction

`to_agent_card()` automatically extracts skills from:

| Source | Extracted As |
|---|---|
| `agent.skills` (BaseSkill) | Skill with metadata from `SkillDescriptor` |
| `agent.tools` (FunctionTool) | Skill tagged `["tool"]` |
| `agent.sub_agents` | Skill tagged `["sub-agent"]` |
| `workflow._step_order` | Skill tagged `["workflow-step"]` |

---

## Advanced Usage

### Custom Provider and Capabilities

```python
from nono.agent.agent_card import (
    to_agent_card,
    AgentProvider,
    AgentCapabilities,
    AgentSkill,
)

card = to_agent_card(
    agent,
    url="https://api.mycompany.com/agents/v1",
    provider=AgentProvider(
        organization="MyCompany Inc.",
        url="https://mycompany.com",
    ),
    capabilities=AgentCapabilities(
        streaming=True,
        push_notifications=True,
    ),
    icon_url="https://mycompany.com/agent-icon.png",
    documentation_url="https://docs.mycompany.com/agents/weather",
    extra_skills=[
        AgentSkill(
            id="custom-analysis",
            name="Custom Analysis",
            description="Run a custom analysis pipeline.",
            tags=["analysis", "custom"],
            examples=["Analyze Q4 sales trends"],
        ),
    ],
)
```

### Build Card Manually

```python
from nono.agent.agent_card import (
    AgentCard,
    AgentInterface,
    AgentSkill,
    AgentCapabilities,
)

card = AgentCard(
    name="My Custom Agent",
    description="Does amazing things.",
    supported_interfaces=[
        AgentInterface(url="https://api.example.com"),
    ],
    capabilities=AgentCapabilities(streaming=True),
    skills=[
        AgentSkill(id="task-1", name="Task One", description="Does task one."),
        AgentSkill(id="task-2", name="Task Two", description="Does task two."),
    ],
    version="2.0.0",
)
```

### Orchestration Agents

Works with any `BaseAgent` subclass — orchestrators expose their sub-agents as skills:

```python
from nono.agent import SequentialAgent, Agent

pipeline = SequentialAgent(
    name="research_pipeline",
    description="Research, analyze, and write reports.",
    sub_agents=[
        Agent(name="researcher", instruction="Research the topic."),
        Agent(name="analyzer", instruction="Analyze findings."),
        Agent(name="writer", instruction="Write the report."),
    ],
)

card = pipeline.agent_card(url="https://pipeline.example.com")
# Skills: researcher, analyzer, writer (tagged "sub-agent")
```

---

## A2A Discovery

Per the A2A specification (§8.2), clients discover agents via:

1. **Well-Known URI**: `GET https://{domain}/.well-known/agent-card.json`
2. **Registries/Catalogs**: Querying curated agent directories
3. **Direct Configuration**: Pre-configured Agent Card URLs

Nono's `serve_agent_card()` handles option 1. For options 2 and 3, use `save_agent_card()` or `card.to_json()` to export the card.

---

## API Reference

| Function | Description |
|---|---|
| `to_agent_card(source, *, url, ...)` | Generate card from agent/workflow |
| `serve_agent_card(card, *, host, port)` | Start HTTP server for the card |
| `save_agent_card(card, path)` | Save card to JSON file |
| `load_agent_card(path)` | Load card from JSON file |
| `agent.agent_card(*, url, ...)` | Convenience method on any `BaseAgent` |
| `AgentCard.to_json()` | Serialize to JSON string |
| `AgentCard.to_dict()` | Serialize to dict |
| `AgentCard.from_json(str)` | Deserialize from JSON |
| `AgentCard.from_dict(dict)` | Deserialize from dict |
