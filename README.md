# Nono — No Overhead, Neural Operations

> Unified AI framework for tasks, agents, workflows, and code execution across 15 LLM providers.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-1.1.0-orange)

---

## What is Nono?

**Nono** is a modular Python framework that replaces complex multi-library setups with a single, batteries-included package for Generative AI. From a one-line prompt to a multi-agent pipeline with 100 orchestration patterns and 29 ready-to-use pipelines — everything works with one import and minimal dependencies.

```python
from nono.agent import Agent, Runner

agent = Agent(name="assistant", model="gemini-3-flash-preview")
response = Runner.run(agent, "Explain quantum computing in simple terms.")
print(response)
```

---

## Key Features

| Feature | Summary |
|---------|---------|
| **16 native providers** | Gemini, OpenAI, Anthropic Claude, DeepSeek, Groq, Cerebras, NVIDIA, Hugging Face, Azure AI, Ollama… — switch with one config change |
| **100 orchestration patterns + 29 pipelines** | Tree of Thoughts, Monte Carlo, Genetic Algorithm, Circuit Breaker, Saga, and 29 domain pipelines (bug fix, security audit, RAG, ETL, prompt engineering…) |
| **Agent framework (NAA)** | `LlmAgent` with tools, skills, hooks, content scopes, and transfer-to-agent delegation |
| **DAG workflow engine** | Checkpointing, time-travel debugging, parallel execution, loops, YAML/JSON definitions |
| **Task-based execution** | JSON prompt definitions + Jinja2 templates — prompts-as-data, not code strings |
| **Code execution** | Sandboxed local executor + 7 cloud sandbox providers (E2B, Modal, Daytona, …) |
| **5 built-in tool collections** | DateTime, Text, Web, Python, ShortFx (3,000+ formulas) — ready to use |
| **MCP client** | Connect to any Model Context Protocol server |
| **20 agent skills** | Triple-mode: standalone, as tool, or as pipeline component |
| **6 hook types × 15 events** | Functions, commands, prompts, tasks, skills, and tools as lifecycle hooks |
| **Dynamic Agent Factory** | Describe what you need in plain English → production agent with security controls |
| **Decision Wizard** | Recommends the optimal orchestration pattern for your task |
| **Built-in observability** | Hierarchical `TraceCollector` — zero external dependencies |
| **Enterprise-ready** | SSL modes, rate limiting, provider fallback chains, HITL protocols |

Full feature breakdown and framework comparisons → [FEATURES.md](FEATURES.md)

---

## Supported Providers

| Provider | Recommended Model | Context Window |
|----------|-------------------|----------------|
| **Google Gemini** | `gemini-3-flash-preview` (default) | Up to 4M chars |
| **OpenAI** | `gpt-4o-mini` | Up to 500K chars |
| **Anthropic Claude** | `claude-sonnet-4` | Up to 200K chars |
| **Perplexity** | `sonar` | Up to 120K chars |
| **DeepSeek** | `deepseek-chat` | Up to 120K chars |
| **xAI** | `grok-3` | Up to 100K chars |
| **Groq** | `llama-3.3-70b-versatile` | Varies |
| **Cerebras** | `llama-3.3-70b` | Varies |
| **NVIDIA** | `meta/llama-3.3-70b-instruct` | Varies |
| **Hugging Face** | `meta-llama/Llama-3.3-70B-Instruct` | Varies |
| **GitHub Models** | `openai/gpt-5` | Varies |
| **OpenRouter** | `openrouter/auto` | Varies |
| **Azure AI** | `openai/gpt-4o` | Varies |
| **Vercel** | `anthropic/claude-opus-4.5` | Varies |
| **Ollama** | Any local model | Varies |

---

## Installation

```bash
# 1. Clone
git clone https://github.com/DatamanEdge/Nono.git
cd Nono

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# 3. Install
pip install -e .
```

Configure your API key:

```python
import keyring
keyring.set_password("gemini", "api_key", "your-api-key")  # recommended
```

Or use a key file: `echo "your-api-key" > nono/apikey.txt`

Full API key and SSL setup → [API Manager](nono/connector/README_api_manager.md) | [SSL Guide](nono/connector/README_connector_genai_ssl.md)

---

## Quick Start

### Agent (stateful, multi-turn)

```python
from nono.agent import Agent, Runner

agent = Agent(
    name="researcher",
    model="gemini-3-flash-preview",
    instruction="You are a helpful research assistant.",
)
response = Runner.run(agent, "What are the latest trends in AI agents?")
```

### Tasker (stateless, single-turn)

```python
from nono.tasker import TaskExecutor

executor = TaskExecutor()
result = executor.execute_task(
    task_name="product_categorizer",
    input_data={"product": "iPhone 15 Pro Max 256GB"}
)
```

### Code Execution

```python
from nono.executer import CodeExecuter

executer = CodeExecuter()
result = executer.run("Calculate the factorial of 10")
print(result.output)  # 3628800
```

### CLI

```bash
python -m nono.cli --provider gemini --prompt "Explain quantum computing"
python -m nono.cli --provider openai --task summarize -i document.txt -o summary.json
python -m nono.cli --dry-run --provider gemini --prompt "Test" -v
```

---

## Architecture

```
nono/
├── agent/            # AI agent framework (NAA): LlmAgent, orchestration, runner
│   ├── tools/        # Built-in tool collections (DateTime, Text, Web, Python, ShortFx)
│   ├── skills/       # 20 reusable AI skills (class-based + SKILL.md)
│   └── templates/    # Pre-configured agent templates & multi-agent compositions
├── cli/              # Command-line interface module
├── config/           # Configuration management
├── connector/        # Low-level AI provider connectors + MCP client
├── executer/         # Code generation & sandboxed execution
├── sandbox/          # Remote sandbox execution (7 cloud providers)
├── tasker/           # Task execution framework
│   ├── prompts/      # JSON task definitions
│   └── templates/    # Jinja2 prompt templates
├── visualize/        # ASCII rendering (agent trees, workflows)
└── workflows/        # DAG workflow engine with checkpointing
```

---

## Configuration

Multi-source priority resolution (highest to lowest):

| Priority | Source | Example |
|----------|--------|---------|
| 1st | CLI arguments | `--provider gemini --model gpt-4o` |
| 2nd | Environment variables | `NONO_GOOGLE__DEFAULT_MODEL` |
| 3rd | Config file | `config.toml` |
| 4th | Defaults | Programmatic defaults |

```python
from nono.config import load_config

config = load_config(filepath='config.toml', env_prefix='NONO_')
model = config['google.default_model']
```

Full configuration reference → [Configuration Guide](nono/README_config.md)

---

## Documentation

All detailed documentation is organized by module. Each document expands on a specific area of the framework.

### Getting Started

| Document | Description |
|----------|-------------|
| [Step-by-Step Guide](nono/README_guide.md) | Beginner tutorial — from first prompt to multi-agent pipelines |
| [Progressive Disclosure](nono/README_progressive_disclosure.md) | Complexity levels (L0–L5) and how to grow incrementally |
| [Tasker vs Agent](nono/README_tasker_vs_agent.md) | When to use stateless `TaskExecutor` vs stateful `Agent` |

### Agent Framework

| Document | Description |
|----------|-------------|
| [Agent](nono/agent/README_agent.md) | `LlmAgent`, `Runner`, sessions, tools, and state management |
| [Orchestration](nono/agent/README_orchestration.md) | Sequential, Parallel, Loop, and 100 orchestration patterns + 29 domain pipelines |
| [Orchestration Guide](nono/workflows/README_orchestration_guide.md) | Deterministic, LLM-routing, and hybrid orchestration strategies |
| [Agent Templates](nono/agent/templates/README_agent_templates.md) | 9 single-agent templates and 29 multi-agent pipelines across 7 domains |
| [Dynamic Agent Factory](nono/agent/README_agent_factory.md) | Generate agents from natural language descriptions |
| [Events and Tracing](nono/agent/README_events_tracking.md) | 9 event types, `TraceCollector`, and observability |
| [State Introspection](nono/README_state_introspection.md) | Query states, transitions, events, and traces during and after execution |
| [Skills](nono/README_skills.md) | Triple-mode skills: standalone, as tool, or as pipeline component |
| [Skill Templates](nono/agent/skills/README_skill_templates.md) | Built-in reusable skill catalog |
| [Human-in-the-Loop](nono/README_human_in_the_loop.md) | `HumanInputAgent`, approval workflows, and HITL protocols |

### Tools and Hooks

| Document | Description |
|----------|-------------|
| [Tools](nono/README_tools.md) | `@tool` decorator, `ToolContext`, MCP, and Tasker tools |
| [ShortFx](nono/README_shortfx.md) | 3,000+ deterministic financial/math formulas for agents |
| [MCP Client](nono/README_mcp.md) | Connect to Model Context Protocol servers |
| [Hooks](nono/README_hooks.md) | 6 hook types × 15 lifecycle events |

### Task Execution

| Document | Description |
|----------|-------------|
| [Tasker](nono/tasker/README.md) | Task-based execution framework and templates |
| [Task Configuration](nono/tasker/README_task_configuration.md) | JSON prompt definition guide |
| [Technical Reference](nono/tasker/README_technical.md) | Tasker architecture and internals |
| [Batch Processing](nono/connector/README_genai_batch_processing.md) | Token-efficient batch operations with Gemini/OpenAI batch APIs |

### Connectors and Infrastructure

| Document | Description |
|----------|-------------|
| [Connector](nono/connector/README_connector_genai.md) | Low-level AI provider interface |
| [Provider Fallback](nono/connector/README_fallback.md) | Automatic failover between providers |
| [API Manager](nono/connector/README_api_manager.md) | API key management with keyring |
| [SSL Configuration](nono/connector/README_connector_genai_ssl.md) | SSL verification modes for corporate environments |

### Workflows and Sandbox

| Document | Description |
|----------|-------------|
| [Workflows](nono/workflows/README_workflow.md) | DAG engine with checkpointing, time-travel, and parallel execution |
| [Sandbox](nono/sandbox/README_sandbox.md) | External sandbox execution (E2B, Modal, Daytona, etc.) |
| [Sandbox Guide](nono/README_sandbox_guide.md) | Multi-provider sandboxing tutorial with manifests and patterns |
| [Workspace & Manifest](nono/README_workspace.md) | Declarative agent I/O — entry types, cloud storage, serialisation, and Manifest bridge |

### CLI, Config, and Deployment

| Document | Description |
|----------|-------------|
| [CLI Guide](nono/README_cli.md) | Command-line interface usage |
| [CLI Module](nono/cli/README_cli.md) | CLI module internals |
| [Configuration Guide](nono/README_config.md) | Unified configuration management |
| [Config Module](nono/config/README_config.md) | Configuration module internals |
| [Projects](nono/README_projects.md) | Git-style project discovery and isolated manifests |
| [Docker Deployment](README_docker.md) | Docker and docker-compose setup |
| [Executer](nono/executer/README.md) | Code generation and sandboxed execution |

### Community

| Document | Description |
|----------|-------------|
| [Features](FEATURES.md) | Full feature breakdown and framework comparisons |
| [Changelog](CHANGELOG.md) | Version history and release notes |
| [Contributing](CONTRIBUTING.md) | Contribution guidelines |
| [Code of Conduct](CODE_OF_CONDUCT.md) | Community code of conduct |
| [Security](SECURITY.md) | Security policy and vulnerability reporting |

---

## Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| `google-genai` | >= 1.0.0 | Google Gemini SDK |
| `openai` | >= 1.0.0 | OpenAI SDK (also used by Groq, NVIDIA, xAI, DeepSeek, Perplexity, Hugging Face, etc.) |
| `httpx` | >= 0.24.0 | Async HTTP client |
| `certifi` | >= 2023.0.0 | SSL certificates |
| `jsonschema` | >= 4.0.0 | JSON schema validation |
| `jinja2` | >= 3.0.0 | Template engine for prompts |

---

## Contact

- **Author**: [DatamanEdge](https://github.com/DatamanEdge)
- **Email**: [jrodriguezga@outlook.com](mailto:jrodriguezga@outlook.com)
- **LinkedIn**: [Javier Rodríguez](https://es.linkedin.com/in/javier-rodriguez-ga)

---

## License

MIT © 2026 DatamanEdge. See [LICENSE](LICENSE).
