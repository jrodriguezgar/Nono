# Connector Gen AI - Unified Generative AI Services Connector

> Provides a unified interface for connecting to multiple generative AI services including OpenAI, Google Gemini, Perplexity, DeepSeek, Grok (xAI), Groq, and Ollama. Features built-in rate limiting (Token Bucket), SSL configuration management, batch processing for large-scale operations, and auto-installation of required dependencies.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-1.2.0-orange)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Batch Processing](#batch-processing)
- [Configuration](#configuration)
- [Rate Limit Status](#rate-limit-status)
- [OpenRouter Service](#openrouter-service)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Contributing](#contributing)
- [Contact](#contact)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

### Prerequisites

- Python >= 3.10
- pip (package manager)

### Steps

1. Install dependencies (auto-installed on first use):

   ```bash
   pip install requests urllib3 certifi google-genai
   ```
2. Import the module:

   ```python
   from connector_genai import GeminiService, ResponseFormat
   from genai_batch_processing import GeminiBatchService
   ```

## Usage

### Quick Start

```python
from connector_genai import GeminiService, ResponseFormat

client = GeminiService(
    model_name="gemini-3-flash-preview",
    api_key="your-api-key"
)

messages = [{"role": "user", "content": "What is machine learning?"}]

response = client.generate_completion(
    messages=messages,
    temperature=0.7,
    response_format=ResponseFormat.TEXT
)

print(response)
```

## Batch Processing

For large-scale, non-urgent tasks, use `GeminiBatchService` from the `genai_batch_processing` module to process requests asynchronously at **50% of the standard cost**.

```python
from genai_batch_processing import GeminiBatchService

batch_service = GeminiBatchService(api_key="your-api-key")

# Build requests from prompts
requests = GeminiBatchService.build_requests_from_prompts(
    prompts=["Question 1?", "Question 2?", "Question 3?"],
    system_instruction="Answer briefly.",
    temperature=0.7
)

# Create batch job
job = batch_service.create_inline_batch(requests, display_name="my-batch")

# Wait for completion
completed = batch_service.wait_for_completion(job.name, poll_interval=30)

# Get results
if completed.state == GeminiBatchService.STATE_SUCCEEDED:
    results = batch_service.get_inline_results(job.name)
    for result in results:
        print(result)
```

📖 **Full documentation**: See [README_genai_batch_processing.md](README_genai_batch_processing.md) for detailed batch processing guide.

### Message Roles & Adaptation

The connector provides a **unified message interface**. You can use standard roles (`system`, `user`, `assistant`) for **all** providers. The connector automatically adapts them to the specific API requirements:

- **OpenAI / DeepSeek / Perplexity**: Passed as standard `system`, `user`, `assistant` roles.
- **Google Gemini**:
  - `system` messages are extracted and passed via the native `system_instruction` parameter (or merged into the first user prompt for older SDK versions).
  - `assistant` roles are mapped to `model`.
- **Ollama**: Passed as standard roles.

**Default Behavior:** If no `system` message is provided, the connector automatically injects a default system instruction: *"You are a helpful assistant."* to ensure consistent model behavior across providers.

### Supported Providers

| Service                 | Class                   |
| :---------------------- | :---------------------- |
| **OpenAI**        | `OpenAIService`       |
| **Google Gemini** | `GeminiService`       |
| **Perplexity**    | `PerplexityService`   |
| **DeepSeek**      | `DeepSeekService`     |
| **Grok (xAI)**    | `GrokService`         |
| **Groq**          | `GroqService`         |
| **OpenRouter**    | `OpenRouterService`   |
| **Ollama**        | `OllamaService`       |

---

### Provider Examples

#### OpenAI

```python
from connector_genai import OpenAIService, ResponseFormat, RateLimitConfig

client = OpenAIService(
    model_name="gpt-4o-mini",
    api_key="sk-your-api-key",
    rate_limit_config=RateLimitConfig(rps=1.0)  # Optional: defaults to 1 RPS
)

# Inspect service configuration
print(client.config)  # Shows provider, model, rate_limit, etc.
print(client.rate_limit['summary'])  # e.g., "RPS=1.0 (60.0 req/min)"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]

response = client.generate_completion(
    messages=messages,
    temperature=0.7,
    response_format=ResponseFormat.TEXT
)
```

#### Google Gemini

```python
from connector_genai import GeminiService, ResponseFormat

client = GeminiService(
    model_name="gemini-3-flash-preview",  # Default model for this project
    api_key="AIzaSy..."
    # rate_limit_config defaults to 15 RPM for Gemini
)

messages = [
    {"role": "system", "content": "You are a technical expert."},
    {"role": "user", "content": "What are the benefits of microservices?"}
]

response = client.generate_completion(
    messages=messages,
    temperature=0.5,
    response_format=ResponseFormat.TEXT
)
```

#### Perplexity

```python
from connector_genai import PerplexityService, ResponseFormat

client = PerplexityService(
    model_name="sonar",  # or "llama-3-sonar-large-32k-online", "mixtral-8x7b-instruct"
    api_key="pplx-your-api-key"
    # rate_limit_config defaults to 1 RPS
)

messages = [
    {"role": "user", "content": "What are the latest developments in AI?"}
]

response = client.generate_completion(
    messages=messages,
    temperature=0.7,
    response_format=ResponseFormat.TEXT
)
```

#### DeepSeek

```python
from connector_genai import DeepSeekService, ResponseFormat

client = DeepSeekService(
    model_name="deepseek-chat",  # or "deepseek-coder"
    api_key="your-deepseek-api-key"
    # rate_limit_config defaults to 1 RPS
)

messages = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "Write a Python function to calculate Fibonacci."}
]

response = client.generate_completion(
    messages=messages,
    temperature=0.0,  # Low temperature for code generation
    response_format=ResponseFormat.TEXT
)
```

#### Grok (xAI)

```python
from connector_genai import GrokService, ResponseFormat

client = GrokService(
    model_name="grok-1",
    api_key="your-xai-api-key"
    # rate_limit_config defaults to 1 RPS
)

messages = [
    {"role": "user", "content": "Explain the theory of relativity."}
]

response = client.generate_completion(
    messages=messages,
    temperature=0.7,
    response_format=ResponseFormat.TEXT
)
```

#### Groq (Ultra-fast Inference)

```python
from connector_genai import GroqService, ResponseFormat

# Groq provides ultra-fast inference using LPU technology
# Supports LLaMA, Qwen, Kimi, and other open models
# Rate limits are automatically loaded from model_rate_limits.csv
client = GroqService(
    model_name="llama-3.3-70b-versatile",  # Default model
    api_key="gsk_your-groq-api-key"
    # Rate limit auto-loaded from CSV (30 RPM for this model)
)

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to calculate prime numbers."}
]

response = client.generate_completion(
    messages=messages,
    temperature=0.0,  # Low temperature for code
    response_format=ResponseFormat.TEXT
)
```

#### Ollama (Local)

```python
from connector_genai import OllamaService, ResponseFormat

client = OllamaService(
    model_name="llama3",  # Any model installed locally
    host="http://localhost:11434"
    # rate_limit_config defaults to 1 RPS
)

messages = [
    {"role": "system", "content": "You are a creative writer."},
    {"role": "user", "content": "Write a haiku about programming."}
]

response = client.generate_completion(
    messages=messages,
    temperature=1.5,  # Higher temperature for creativity
    response_format=ResponseFormat.TEXT
)
```

---

### Advanced Examples

#### JSON Response with Schema

```python
from connector_genai import GeminiService, ResponseFormat

client = GeminiService(api_key="...")

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"}
    },
    "required": ["name", "age"]
}

messages = [{"role": "user", "content": "Generate data for a fictional person."}]

response = client.generate_completion(
    messages=messages,
    response_format=ResponseFormat.JSON,
    json_schema=schema
)
```

#### Multi-turn Conversation

```python
from connector_genai import OpenAIService, ResponseFormat

client = OpenAIService(model_name="gpt-4o-mini", api_key="...")

conversation = [
    {"role": "system", "content": "You are a history expert."},
    {"role": "user", "content": "Who was Napoleon?"},
    {"role": "assistant", "content": "Napoleon Bonaparte was a French military leader..."},
    {"role": "user", "content": "When was he born?"}
]

response = client.generate_completion(
    messages=conversation,
    temperature=0.7,
    response_format=ResponseFormat.TEXT
)
```

#### Structured Data Extraction

```python
from connector_genai import PerplexityService, ResponseFormat

client = PerplexityService(api_key="...")

schema = {
    "type": "object",
    "properties": {
        "companies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "sector": {"type": "string"},
                    "valuation": {"type": "number"}
                }
            }
        }
    },
    "required": ["companies"]
}

messages = [{"role": "user", "content": "List the top 3 tech companies by market cap."}]

response = client.generate_completion(
    messages=messages,
    response_format=ResponseFormat.JSON,
    json_schema=schema
)
```

## Configuration

### SSL Verification Modes

| Variable                         | Description                                       | Required   | Default |
| -------------------------------- | ------------------------------------------------- | ---------- | ------- |
| `SSLVerificationMode.INSECURE` | Disables SSL verification (development only)      | No         | ✓      |
| `SSLVerificationMode.CERTIFI`  | Uses certifi package (recommended for production) | No         | —      |
| `SSLVerificationMode.CUSTOM`   | Uses custom certificate file                      | Yes (path) | —      |

```python
from connector_genai import configure_ssl_verification, SSLVerificationMode

# Development (default)
configure_ssl_verification(SSLVerificationMode.INSECURE)

# Production (recommended)
configure_ssl_verification(SSLVerificationMode.CERTIFI)

# Corporate certificate
configure_ssl_verification(
    SSLVerificationMode.CUSTOM,
    custom_cert_path=r"C:\certs\corporate-ca.crt"
)
```

> ⚠️ **Important:** Configure SSL **BEFORE** creating service instances.

📖 **Full SSL documentation**: See [connector_genai_ssl.md](connector_genai_ssl.md) for detailed SSL configuration guide, including corporate proxy setup and troubleshooting.

### Response Formats

| Format | Enum                     | Description             |
| :----- | :----------------------- | :---------------------- |
| Text   | `ResponseFormat.TEXT`  | Plain text response     |
| JSON   | `ResponseFormat.JSON`  | JSON formatted response |
| XML    | `ResponseFormat.XML`   | XML formatted response  |
| Table  | `ResponseFormat.TABLE` | Markdown table response |

### Rate Limiting

Rate limiting is managed by the `api_manager` module using Token Bucket algorithm by default. Each service has configurable rate limits loaded from `model_rate_limits.csv`.

📖 **Full Rate Limiting documentation**: See [README_api_manager.md](README_api_manager.md) for detailed configuration options, algorithms, and advanced features.

#### Simple Rate Limit Configuration

```python
from connector_genai import OpenAIService, RateLimitConfig

# Custom rate limit: 2 requests per second
rate_config = RateLimitConfig(rps=2.0)

client = OpenAIService(
    model_name="gpt-4o-mini",
    api_key="...",
    rate_limit_config=rate_config
)
```

#### Advanced Rate Limit with RateLimitConfig

```python
from connector_genai import GeminiService, RateLimitConfig

# Advanced configuration with multiple limits
rate_config = RateLimitConfig(
    rps=0.5,              # 0.5 requests per second
    rpm=30,               # 30 requests per minute
    tpm=100000,           # 100K tokens per minute
    concurrent_limit=5,   # Max 5 concurrent requests
    burst_size=10         # Allow burst of 10 requests
)

client = GeminiService(
    model_name="gemini-3-flash-preview",
    api_key="...",
    rate_limit_config=rate_config
)
```

#### Inspecting Rate Limit Configuration

```python
# Get rate limit summary
print(client.rate_limit['summary'])  
# Output: "RPS=0.5 (30.0 req/min), RPM=30, TPM=100000, Concurrent=5"

# Get full configuration
print(client.rate_limit['config'])
# Output: {'algorithm': 'token_bucket', 'action': 'wait', 'rps': 0.5, 'rpm': 30, ...}

# Get limiter statistics
print(client.rate_limit['stats'])

# Access the underlying RateLimitConfig object
config = client.rate_limit_config
print(config.rps, config.rpm, config.algorithm)
```

### Recommended Temperatures

Temperature can be specified as a **float value** (0.0 - 2.0) or as a **use case name** (string):

```python
from connector_genai import GenerativeAIService, GeminiService

# Get recommended temperature for a use case
temp = GenerativeAIService.get_recommended_temperature("coding")  # Returns 0.0

# Or use the string directly in generate_completion
client = GeminiService(model_name="gemini-3-flash-preview", api_key="...")

# These are equivalent:
response = client.generate_completion(messages=msgs, temperature=0.0)
response = client.generate_completion(messages=msgs, temperature="coding")
```

| Use Case          | Temperature | Description                              |
| :---------------- | :---------- | :--------------------------------------- |
| `coding`        | 0.0         | Maximum precision, deterministic code    |
| `math`          | 0.0         | Exact mathematical answers               |
| `data_cleaning` | 0.1         | High precision for data transformations  |
| `data_analysis` | 0.3         | Consistency in analysis, some flexibility|
| `translation`   | 0.3         | Precise and faithful translations        |
| `conversation`  | 0.7         | Balance between coherence and naturalness|
| `creative`      | 1.0         | Higher variability for creative content  |
| `poetry`        | 1.2         | High creativity for artistic expression  |
| `default`       | 0.7         | Balanced default for general use         |

---

## Rate Limit Status

The `RateLimitStatus` dataclass provides a unified way to inspect and monitor rate limits across all AI services. It combines configured limits with real-time usage information.

### RateLimitStatus Attributes

| Attribute | Type | Description |
| :-------- | :--- | :---------- |
| `configured_rpm` | `int \| None` | Configured requests per minute limit |
| `configured_rpd` | `int \| None` | Configured requests per day limit |
| `configured_tpm` | `int \| None` | Configured tokens per minute limit |
| `configured_tpd` | `int \| None` | Configured tokens per day limit |
| `remaining_requests` | `int \| None` | Remaining requests in current window |
| `remaining_tokens` | `int \| None` | Remaining tokens in current window |
| `reset_time` | `float \| None` | Seconds until limits reset |
| `credits_remaining` | `float \| None` | Remaining credits (paid APIs) |
| `credits_used` | `float \| None` | Credits used in current period |
| `is_free_tier` | `bool` | Whether using free tier |
| `rate_limit_tier` | `str \| None` | Rate limit tier name |
| `is_rate_limited` | `bool` | Property: True if currently rate limited |
| `has_credits` | `bool` | Property: True if credits remain |

### Basic Usage

```python
from connector_genai import GeminiService

client = GeminiService(api_key="...")

# Get current rate limit status
status = client.get_rate_limit_status()

# Check if rate limited
if status.is_rate_limited:
    print(f"Rate limited! Wait {status.reset_time:.1f}s")
else:
    print(f"Remaining requests: {status.remaining_requests}")

# Print summary
print(status)
# Output: RateLimitStatus(RPM=15 (12 left), Tier=Gemini)

# Convert to dictionary
print(status.to_dict())
```

### OpenRouter Rate Limit Status

OpenRouterService provides enhanced rate limit status by querying the OpenRouter API directly:

```python
from connector_genai import OpenRouterService

service = OpenRouterService(api_key="sk-or-...")

# Fetch real-time status from OpenRouter API
status = service.get_rate_limit_status()

print(f"Tier: {status.rate_limit_tier}")      # e.g., "explorer"
print(f"Credits: ${status.credits_remaining:.2f}")  # e.g., "$98.50"
print(f"Free tier: {status.is_free_tier}")    # True/False
print(f"Rate limited: {status.is_rate_limited}")

# Access raw API response for debugging
if status.raw_response:
    print(status.raw_response)
```

---

## OpenRouter Service

`OpenRouterService` provides access to 300+ AI models from multiple providers (OpenAI, Anthropic, Google, Meta, Mistral, DeepSeek, and more) through a single unified API.

### Features

- **Unified API**: Access all major AI providers with one API key
- **Model routing**: Automatic fallback to alternative models
- **Provider preferences**: Select specific providers for a model
- **Plugins**: Web search, PDF parsing, response healing
- **Structured outputs**: JSON schema support
- **Real-time rate limits**: Query credits and limits via API

### Basic Usage

```python
from connector_genai import OpenRouterService, ResponseFormat

service = OpenRouterService(
    api_key="sk-or-v1-...",
    app_name="My Application",      # Shown in OpenRouter dashboard
    app_url="https://myapp.com"     # Used for rankings
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing."}
]

response = service.generate_completion(
    messages=messages,
    temperature=0.7,
    response_format=ResponseFormat.TEXT
)
print(response)
```

### OpenRouter-Specific Parameters

```python
response = service.generate_completion(
    messages=messages,
    
    # Standard parameters
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    
    # OpenRouter-specific
    top_k=50,                    # Not available for OpenAI models
    repetition_penalty=1.1,      # Range: (0, 2]
    min_p=0.05,                  # Min-p sampling
    seed=42,                     # For reproducibility
    
    # Provider preferences
    provider_preferences={
        "order": ["anthropic", "openai"],
        "allow_fallbacks": True
    },
    
    # Plugins
    plugins=[
        {"id": "web"},           # Enable web search
        {"id": "response-healing"}  # Auto-fix malformed JSON
    ],
    
    # Model routing (fallback)
    models=["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
    route="fallback"
)
```

### Structured Outputs

```python
# JSON mode (basic)
response = service.generate_completion(
    messages=[{"role": "user", "content": "List 3 colors as JSON"}],
    response_format=ResponseFormat.JSON
)

# JSON schema mode (strict)
schema = {
    "title": "Person",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

response = service.generate_completion(
    messages=[{"role": "user", "content": "Create a person"}],
    response_format=ResponseFormat.JSON,
    json_schema=schema
)
```

### Using OpenAI SDK

For advanced features like streaming, you can use the OpenAI SDK directly:

```python
service = OpenRouterService(
    api_key="sk-or-...",
    app_name="My App",
    app_url="https://myapp.com"
)

# Get configured OpenAI client
client = service.get_openai_client()

# Use with extra_headers for app identification
completion = client.chat.completions.create(
    extra_headers=service.extra_headers,
    model="anthropic/claude-3.5-sonnet",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True  # Streaming supported
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
```

### Checking Credits and Limits

```python
service = OpenRouterService(api_key="sk-or-...")

# Fetch raw API key info
info = service.fetch_api_key_info()
print(info)
# {
#     "data": {
#         "label": "My API Key",
#         "usage": 1.50,
#         "limit": 100.0,
#         "is_free_tier": False,
#         "rate_limit": {"requests": 200, "interval": "10s"}
#     }
# }

# Get structured status
status = service.get_rate_limit_status()
print(f"Credits remaining: ${status.credits_remaining:.2f}")
print(f"Tier: {status.rate_limit_tier}")
print(f"Is free tier: {status.is_free_tier}")
```

---

## API Reference

### Classes

#### Base Class: `GenerativeAIService`

```python
class GenerativeAIService(ABC):
    """Abstract base class for generative AI services."""
  
    def __init__(
        self,
        model_name: str,
        max_input_chars: int,
        api_key: Optional[str] = None,
        rate_limit_config: Optional[RateLimitConfig] = None
    )
  
    # --- Properties for configuration inspection ---
    @property
    def model_name(self) -> str: ...           # Model name in use
  
    @property
    def api_key_masked(self) -> str: ...       # Masked API key (e.g., "abc1...xyz9")
  
    @property
    def max_input_chars(self) -> int: ...      # Max input characters
  
    @property
    def provider(self) -> str: ...             # Provider name (e.g., "Gemini")
  
    @property
    def rate_limit(self) -> dict: ...          # Rate limit config, stats, and summary
  
    @property
    def rate_limit_config(self) -> RateLimitConfig: ...  # Underlying config object
  
    @property
    def config(self) -> dict: ...              # Full configuration dictionary
  
    # --- Methods ---
    def get_config(
        self, 
        include_api_key: bool = False,
        include_rate_stats: bool = False
    ) -> dict
  
    @abstractmethod
    def generate_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | str = 0.7,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: list[str] | None = None,
        response_format: ResponseFormat = ResponseFormat.JSON,
        json_schema: dict | None = None,
        use_case: str | None = None,
        **kwargs
    ) -> str
  
    @classmethod
    def get_recommended_temperature(cls, use_case: str) -> float

    @classmethod
    def _resolve_temperature(cls, temperature: float | str) -> float
```

#### Configuration Properties

| Property              | Type                | Description                                                          |
| :-------------------- | :------------------ | :------------------------------------------------------------------- |
| `model_name`        | `str`             | Name of the model being used                                         |
| `api_key_masked`    | `str`             | API key masked for security (shows first/last 4 chars)               |
| `max_input_chars`   | `int`             | Maximum input characters allowed                                     |
| `provider`          | `str`             | AI provider name (derived from class name)                           |
| `rate_limit`        | `dict`            | Rate limit configuration with `config`, `stats`, and `summary` |
| `rate_limit_config` | `RateLimitConfig` | Underlying rate limit configuration object                           |
| `config`            | `dict`            | Full configuration dictionary                                        |

#### Example: Inspecting Service Configuration

```python
from connector_genai import GeminiService

client = GeminiService(
    model_name="gemini-3-flash-preview",
    api_key="AIzaSyABC123..."
)

# Quick inspection
print(client)  
# Output: "Gemini Service using model 'gemini-3-flash-preview' (max 4,000,000 chars, RPS=0.25 (15.0 req/min))"

print(repr(client))
# Output: "GeminiService(model='gemini-3-flash-preview', max_chars=4,000,000, rate_limit='RPS=0.25 (15.0 req/min)')"

# Full configuration
print(client.config)
# Output: {
#     'provider': 'Gemini',
#     'model_name': 'gemini-3-flash-preview',
#     'max_input_chars': 4000000,
#     'rate_limit': 'RPS=0.25 (15.0 req/min)',
#     'api_key': 'AIza...123...'
# }

# Detailed rate limit info
print(client.rate_limit)
# Output: {
#     'config': {'algorithm': 'token_bucket', 'action': 'wait', 'rps': 0.25, ...},
#     'stats': {'limiters': {...}},
#     'summary': 'RPS=0.25 (15.0 req/min)'
# }
```

### Supported Generation Parameters

The method `generate_completion` supports standard parameters which are automatically mapped to the specific provider's API.

> **Note on Defaults:** If optional parameters (like `max_tokens`, `top_p`, etc.) are not provided (left as `None`), the connector does not enforce any local defaults. Instead, the request allows the **provider's API to use its own internal default values** for that specific model.

| Parameter             | Type              | Description                         | Supported Providers                                         |
| :-------------------- | :---------------- | :---------------------------------- | :---------------------------------------------------------- |
| `temperature`       | `float \| str`  | Controls randomness (0.0-2.0) or use case name (e.g., "coding"). | All                                                         |
| `max_tokens`        | `int`       | Maximum tokens to generate.         | All (Ollama:`num_predict`, Gemini: `max_output_tokens`) |
| `top_p`             | `float`     | Nucleus sampling probability.       | All                                                         |
| `frequency_penalty` | `float`     | Penalizes token repetition.         | OpenAI, Perplexity, DeepSeek, Ollama (`repeat_penalty`)   |
| `presence_penalty`  | `float`     | Penalizes token presence.           | OpenAI, Perplexity, DeepSeek                                |
| `stop`              | `list[str]` | Stop sequences.                     | All                                                         |
| `**kwargs`          | `key=val`   | Provider-specific extra parameters. | See below                                                   |

#### Provider-Specific Parameters via **kwargs

**Google Gemini**

- `top_k`: (int) Top K sampling.
- `candidate_count`: (int) Number of responses.

**Ollama**

- `top_k`: (int) Top K sampling.
- `seed`: (int) Random seed.
- `num_ctx`: (int) Context window size.

**Perplexity**

- `return_citations`: (bool)
- `search_domain_filter`: (list[str])

**OpenRouter**

- `top_k`: (int) Top K sampling (not available for OpenAI models).
- `repetition_penalty`: (float) Repetition penalty (0, 2].
- `min_p`: (float) Min-p sampling [0, 1].
- `top_a`: (float) Top-a sampling [0, 1].
- `seed`: (int) Random seed for reproducibility.
- `logit_bias`: (dict) Token logit bias.
- `tools`: (list) Tool definitions for function calling.
- `tool_choice`: (str|dict) Tool selection strategy.
- `plugins`: (list) OpenRouter plugins (web, file-parser, response-healing).
- `provider_preferences`: (dict) Provider routing preferences.
- `transforms`: (list) Prompt transforms.
- `route`: (str) Routing strategy ("fallback").
- `models`: (list) Fallback model list.

#### Service Classes

| Class                 | Constructor Parameters                                                                   |
| :-------------------- | :--------------------------------------------------------------------------------------- |
| `OpenAIService`     | `model_name`, `api_key`, `rate_limit_config`                                       |
| `GeminiService`     | `model_name`, `api_key`, `rate_limit_config`                                       |
| `PerplexityService` | `model_name`, `api_key`, `rate_limit_config`                                       |
| `DeepSeekService`   | `model_name`, `api_key`, `rate_limit_config`                                       |
| `GrokService`       | `model_name`, `api_key`, `rate_limit_config`                                       |
| `GroqService`       | `model_name`, `api_key`, `rate_limit_config` (auto-calculated from model RPM)      |
| `OpenRouterService` | `model_name`, `api_key`, `app_name`, `app_url`, `rate_limit_config`          |
| `OllamaService`     | `model_name`, `host`, `rate_limit_config`                                          |

> **Notes:**
> - All services accept `rate_limit_config: RateLimitConfig` for advanced rate limiting.
> - All services provide `get_rate_limit_status() -> RateLimitStatus` method.
> - `OpenAICompatibleService` classes expose a `base_url` property.
> - `OllamaService` exposes a `host` property.
> - `GroqService` and `OpenRouterService` load rate limits from `model_rate_limits.csv`.
> - `OpenRouterService` provides `get_openai_client()`, `extra_headers`, `fetch_api_key_info()`, and enhanced `get_rate_limit_status()` with real-time API data.

### Enums

#### `ResponseFormat`

| Value     | Description             |
| :-------- | :---------------------- |
| `TEXT`  | Plain text response     |
| `JSON`  | JSON formatted response |
| `XML`   | XML formatted response  |
| `TABLE` | Markdown table response |

#### `SSLVerificationMode`

| Value        | Description                                  |
| :----------- | :------------------------------------------- |
| `INSECURE` | Disables SSL verification (development only) |
| `CERTIFI`  | Uses certifi package (recommended)           |
| `CUSTOM`   | Uses custom certificate file                 |

### Helper Functions

| Function                                               | Description                                        |
| :----------------------------------------------------- | :------------------------------------------------- |
| `configure_ssl_verification(mode, custom_cert_path)` | Configures SSL verification mode                   |
| `install_library(library_name, import_name)`         | Installs a library if not available                |
| `convert_json_schema(input_schema, output_title)`    | Converts simplified JSON schema to detailed format |
| `get_prompt_size(provider, model, default)`          | Gets max prompt size from model_features.csv       |
| `get_rate_limit_config(provider, model)`             | Gets RateLimitConfig from model_rate_limits.csv    |
| `get_service_for_provider(provider, model, apikey)`  | Factory function that returns appropriate service  |
| `resolve_api_key_for_provider(provider)`             | Resolves API key from keyring or CSV file          |

### Error Handling

```python
from connector_genai import GeminiService

client = GeminiService(api_key="...")

try:
    response = client.generate_completion(messages=very_long_messages)
except ValueError as e:
    print(f"Message too long: {e}")
except requests.exceptions.RequestException as e:
    print(f"Connection error: {e}")
except ImportError as e:
    print(f"Missing dependency: {e}")
```

---

## Dependencies

| Package          | Version     | Description                                                    |
| ---------------- | ----------- | -------------------------------------------------------------- |
| `requests`     | >= 2.28.0   | HTTP library for API calls                                     |
| `google-genai` | >= 1.0.0    | Google Gemini SDK ([docs](https://ai.google.dev/gemini-api/docs)) |
| `certifi`      | >= 2023.0.0 | SSL certificates for secure connections                        |
| `jsonschema`   | >= 4.0.0    | JSON schema validation (optional)                              |

---

## Contact

- **Author**: [DatamanEdge](https://github.com/DatamanEdge)
- **Email**: [jrodriguezga@outlook.com](mailto:jrodriguezga@outlook.com)
- **LinkedIn**: [Javier Rodríguez](https://es.linkedin.com/in/javier-rodriguez-ga)

---

## License

MIT © 2026 DatamanEdge. See [LICENSE](../../LICENSE).