# Connector Gen AI - Unified Generative AI Services Connector

> Provides a unified interface for connecting to multiple generative AI services including OpenAI, Google Gemini, Perplexity, DeepSeek, Grok (xAI), and Ollama. Features built-in rate limiting (Token Bucket), SSL configuration management, and auto-installation of required dependencies.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-1.0.0-orange)


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Contributing](#contributing)
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
   ```

## Usage

### Quick Start

```python
from connector_genai import GeminiService, ResponseFormat

client = GeminiService(
    model_name="gemini-1.5-flash",
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

### Message Roles & Adaptation

The connector provides a **unified message interface**. You can use standard roles (`system`, `user`, `assistant`) for **all** providers. The connector automatically adapts them to the specific API requirements:

- **OpenAI / DeepSeek / Perplexity**: Passed as standard `system`, `user`, `assistant` roles.
- **Google Gemini**:
  - `system` messages are extracted and passed via the native `system_instruction` parameter (or merged into the first user prompt for older SDK versions).
  - `assistant` roles are mapped to `model`.
- **Ollama**: Passed as standard roles.

**Default Behavior:** If no `system` message is provided, the connector automatically injects a default system instruction: *"You are a helpful assistant."* to ensure consistent model behavior across providers.

### Supported Providers

| Service | Class | Max Characters |
|:---|:---|:---|
| **OpenAI** | `OpenAIService` | 60K - 500K |
| **Google Gemini** | `GeminiService` | 120K - 4M |
| **Perplexity** | `PerplexityService` | 30K - 120K |
| **DeepSeek** | `DeepSeekService` | 500K |
| **Grok (xAI)** | `GrokService` | 30K |
| **Ollama** | `OllamaService` | 200K |

---

### Provider Examples

#### OpenAI

```python
from connector_genai import OpenAIService, ResponseFormat

client = OpenAIService(
    model_name="gpt-4o-mini",
    api_key="sk-your-api-key",
    requests_per_second=1.0
)

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
    model_name="gemini-1.5-flash",  # or "gemini-1.5-pro-latest", "gemini-pro"
    api_key="AIzaSy...",
    requests_per_second=0.25  # 15 req/min default
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
    api_key="pplx-your-api-key",
    requests_per_second=1.0
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
    api_key="your-deepseek-api-key",
    requests_per_second=1.0
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
    api_key="your-xai-api-key",
    requests_per_second=1.0
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

#### Ollama (Local)

```python
from connector_genai import OllamaService, ResponseFormat

client = OllamaService(
    model_name="llama3",  # Any model installed locally
    host="http://localhost:11434",
    requests_per_second=1.0
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

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `SSLVerificationMode.INSECURE` | Disables SSL verification (development only) | No | ✓ |
| `SSLVerificationMode.CERTIFI` | Uses certifi package (recommended for production) | No | — |
| `SSLVerificationMode.CUSTOM` | Uses custom certificate file | Yes (path) | — |

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

### Response Formats

| Format | Enum | Description |
|:---|:---|:---|
| Text | `ResponseFormat.TEXT` | Plain text response |
| JSON | `ResponseFormat.JSON` | JSON formatted response |
| XML | `ResponseFormat.XML` | XML formatted response |
| Table | `ResponseFormat.TABLE` | Markdown table response |

### Rate Limiting

Token Bucket rate limiter controls request frequency per service:

| Service | Default Rate Limit |
|:---|:---|
| OpenAI | 1 req/sec |
| Gemini | 0.25 req/sec (15 req/min) |
| Perplexity | 1 req/sec |
| DeepSeek | 1 req/sec |
| Ollama | 1 req/sec |

```python
# Custom rate limit: 2 requests per second
client = OpenAIService(
    model_name="gpt-4o-mini",
    api_key="...",
    requests_per_second=2.0
)
```

### Recommended Temperatures

```python
from connector_genai import GenerativeAIService

temp = GenerativeAIService.get_recommended_temperature("coding")  # Returns 0.0
```

| Use Case | Temperature | Description |
|:---|:---|:---|
| `coding` | 0.0 | Precise code generation |
| `math` | 0.0 | Mathematical calculations |
| `data_cleaning` | 1.0 | Data cleaning tasks |
| `data_analysis` | 1.0 | Data analysis |
| `conversation` | 1.3 | General conversation |
| `translation` | 1.3 | Text translation |
| `creative` | 1.5 | Creative writing |
| `poetry` | 1.5 | Poetry and artistic prose |

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
        requests_per_second: float = 1.0,
        api_key: Optional[str] = None
    )
    
    @abstractmethod
    def generate_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
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
```

### Supported Generation Parameters

The method `generate_completion` supports standard parameters which are automatically mapped to the specific provider's API.

> **Note on Defaults:** If optional parameters (like `max_tokens`, `top_p`, etc.) are not provided (left as `None`), the connector does not enforce any local defaults. Instead, the request allows the **provider's API to use its own internal default values** for that specific model.

| Parameter | Type | Description | Supported Providers |
|:---|:---|:---|:---|
| `temperature` | `float` | Controls randomness (0.0 - 2.0). | All |
| `max_tokens` | `int` | Maximum tokens to generate. | All (Ollama: `num_predict`, Gemini: `max_output_tokens`) |
| `top_p` | `float` | Nucleus sampling probability. | All |
| `frequency_penalty` | `float` | Penalizes token repetition. | OpenAI, Perplexity, DeepSeek, Ollama (`repeat_penalty`) |
| `presence_penalty` | `float` | Penalizes token presence. | OpenAI, Perplexity, DeepSeek |
| `stop` | `list[str]` | Stop sequences. | All |
| `**kwargs` | `key=val` | Provider-specific extra parameters. | See below |

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

#### Service Classes

| Class | Constructor Parameters |
|:---|:---|
| `OpenAIService` | `model_name`, `api_key`, `requests_per_second` |
| `GeminiService` | `model_name`, `api_key`, `requests_per_second` |
| `PerplexityService` | `model_name`, `api_key`, `requests_per_second` |
| `DeepSeekService` | `model_name`, `api_key`, `requests_per_second` |
| `GrokService` | `model_name`, `api_key`, `requests_per_second` |
| `OllamaService` | `model_name`, `host`, `requests_per_second` |

### Enums

#### `ResponseFormat`

| Value | Description |
|:---|:---|
| `TEXT` | Plain text response |
| `JSON` | JSON formatted response |
| `XML` | XML formatted response |
| `TABLE` | Markdown table response |

#### `SSLVerificationMode`

| Value | Description |
|:---|:---|
| `INSECURE` | Disables SSL verification (development only) |
| `CERTIFI` | Uses certifi package (recommended) |
| `CUSTOM` | Uses custom certificate file |

### Helper Functions

| Function | Description |
|:---|:---|
| `configure_ssl_verification(mode, custom_cert_path)` | Configures SSL verification mode |
| `install_library(library_name, import_name)` | Installs a library if not available |
| `convert_json_schema(input_schema, output_title)` | Converts simplified JSON schema to detailed format |

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

## Testing

```bash
python -m pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Credits

**Author:** DatamanEdge  
**License:** MIT

## Related Files

| File | Description |
|:---|:---|
| [connector_genai.py](connector_genai.py) | Main module |
| [connector_genai_ssl.md](connector_genai_ssl.md) | SSL configuration guide |
