# API Manager - Documentation

Professional module for comprehensive API management with rate limiting, circuit breaker, retries, and metrics.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Migration from api_rate_limiter](#migration-from-api_rate_limiter)
- [Quick Start](#quick-start)
- [API Manager](#api-manager)
- [Rate Limiting](#rate-limiting)
- [Circuit Breaker](#circuit-breaker)
- [Retry Policies](#retry-policies)
- [Metrics](#metrics)
- [Decorators](#decorators)
- [Provider Presets](#provider-presets)
- [API Reference](#api-reference)

---

## ✨ Features

### Rate Limiting
- **Multiple simultaneous limits**: RPM, RPD, TPM, TPD, RPS and concurrent limits
- **Configurable algorithms**: Token Bucket, Sliding Window, Fixed Window
- **AI API presets**: OpenAI, Gemini, Claude, Perplexity, DeepSeek, Ollama

### Circuit Breaker
- **Circuit Breaker pattern**: Prevents cascading failures
- **States**: Closed, Open, Half-Open
- **Auto-recovery**: Automatic recovery testing

### Retries
- **Multiple strategies**: Fixed, Linear, Exponential, Fibonacci
- **Configurable jitter**: Avoids thundering herd
- **Configurable exceptions**: Define which errors to retry

### API Management
- **Centralized registration**: Manage multiple APIs from a single point
- **Complete metrics**: Usage statistics, latency, errors
- **Configuration presets**: Ready configurations for known providers

---

## 📦 Installation

The module is in the `connector/` directory. Import directly:

```python
from api_manager import (
    # API Manager
    APIManager,
    APIConfig,
    APIConfigPresets,
    
    # Rate Limiting (backward compatible)
    APIRateLimiter,
    RateLimitConfig,
    AIProviderPresets,
    create_limiter_for_provider,
    
    # Circuit Breaker
    CircuitBreaker,
    CircuitBreakerConfig,
    
    # Retry
    RetryConfig,
    RetryStrategy,
    
    # Decorators
    with_rate_limit,
    with_retry,
    with_circuit_breaker,
    with_managed_api,
)
```

---

## 🔄 Migration from api_rate_limiter

The new `api_manager` module is **100% backward compatible** with `api_rate_limiter`. 
Simply change the import:

```python
# Before
from api_rate_limiter import APIRateLimiter, RateLimitConfig

# Now
from api_manager import APIRateLimiter, RateLimitConfig
```

All existing code will continue to work without changes.

---

## 🚀 Quick Start

### Option 1: API Manager (Recommended)

```python
from api_manager import APIManager, APIConfigPresets

# Create manager
manager = APIManager()

# Register APIs with presets
manager.register_provider("openai", api_key="sk-...")
manager.register_provider("gemini", api_key="AI...")

# Use with context manager
with manager.acquire("openai", tokens=1000):
    response = call_openai_api()

# Or async version
async with manager.async_acquire("gemini", tokens=500):
    response = await call_gemini_api()
```

### Option 2: Standalone Rate Limiter

```python
from api_manager import APIRateLimiter, RateLimitConfig

# Configure limits
config = RateLimitConfig(
    rpm=60,
    tpm=100000,
    concurrent_limit=5
)

limiter = APIRateLimiter(config)

# Use before each call
with limiter.acquire_context(tokens=500):
    response = make_api_call()
```

### Option 3: Use Presets

```python
from api_manager import create_limiter_for_provider

# Create preconfigured limiter
limiter = create_limiter_for_provider("gemini")
```

---

## 🏢 API Manager

The `APIManager` is the central point for managing multiple APIs.

### API Registration

```python
from api_manager import APIManager, APIConfig, RateLimitConfig

manager = APIManager()

# Register with custom configuration
config = APIConfig(
    base_url="https://api.example.com",
    api_key="your-key",
    rate_limit=RateLimitConfig(rpm=100, tpm=50000),
    timeout=30.0
)
manager.register_api("my-api", config)

# Or use presets for known providers
manager.register_provider("openai", api_key="sk-...")
manager.register_provider("gemini", api_key="AI...")
```

### Usage with Context Manager

```python
# Context manager automatically handles:
# - Rate limiting
# - Circuit breaker
# - Metrics
# - Resource release

with manager.acquire("openai", tokens=1000):
    response = call_api()
    # If error occurs, it's automatically logged

# Async version
async with manager.async_acquire("openai", tokens=1000):
    response = await async_call_api()
```

### Get Statistics

```python
# Statistics for one API
api = manager.get_api("openai")
stats = api.get_stats()
print(stats)

# Statistics for all APIs
all_stats = manager.get_all_stats()
```

---

## ⚡ Rate Limiting

### Limits Configuration

```python
from api_manager import RateLimitConfig, RateLimitAlgorithm

config = RateLimitConfig(
    # Request limits
    rpm=60,           # 60 requests per minute
    rpd=10000,        # 10,000 requests per day
    rps=1.0,          # 1 request per second
    
    # Token limits (for AI APIs)
    tpm=100000,       # 100,000 tokens per minute
    tpd=1000000,      # 1,000,000 tokens per day
    
    # Concurrency
    concurrent_limit=10,  # 10 simultaneous requests
    
    # Algorithm
    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
    burst_size=15,    # Maximum burst
    
    # Behavior
    max_wait_time=300.0,  # Wait maximum 5 minutes
)
```

### Limits Glossary

| Acronym | Name | Description |
|----------|--------|-------------|
| **RPM** | Requests Per Minute | Requests per minute |
| **RPD** | Requests Per Day | Requests per day |
| **RPS** | Requests Per Second | Requests per second |
| **TPM** | Tokens Per Minute | Tokens per minute (AI APIs) |
| **TPD** | Tokens Per Day | Tokens per day |

### Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `TOKEN_BUCKET` | Controlled bursts | General use, AI APIs |
| `SLIDING_WINDOW` | Precise counting | Strict limits |
| `FIXED_WINDOW` | Simple counter | Simplicity, compatibility |

---

## 🔌 Circuit Breaker

Previene cascada de fallos cuando un servicio está degradado.

### Configuración

```python
from api_manager import CircuitBreakerConfig, CircuitBreaker

config = CircuitBreakerConfig(
    failure_threshold=5,      # Abrir después de 5 fallos
    success_threshold=3,      # Cerrar después de 3 éxitos
    timeout=60.0,             # Intentar recovery después de 60s
    half_open_max_calls=3,    # Máx llamadas en half-open
)

breaker = CircuitBreaker(config)
```

### Estados

```
CLOSED ──(failures >= threshold)──> OPEN
   ^                                   │
   │                              (timeout)
   │                                   ▼
   └──(successes >= threshold)── HALF_OPEN
```

- **CLOSED**: Operación normal
- **OPEN**: Rechaza todas las solicitudes
- **HALF_OPEN**: Permite pruebas limitadas

### Uso Manual

```python
if breaker.can_execute():
    try:
        result = call_api()
        breaker.record_success()
    except Exception as e:
        breaker.record_failure(e)
        raise
else:
    raise CircuitBreakerOpenError("Circuit is open")
```

---

## 🔁 Políticas de Reintentos

### Configuración

```python
from api_manager import RetryConfig, RetryStrategy

config = RetryConfig(
    strategy=RetryStrategy.EXPONENTIAL,
    max_retries=3,
    base_delay=1.0,       # 1 segundo base
    max_delay=60.0,       # Máximo 60 segundos
    jitter=0.1,           # 10% de variación aleatoria
    retryable_exceptions=(ConnectionError, TimeoutError),
    retryable_status_codes=(429, 500, 502, 503, 504),
)
```

### Estrategias

| Estrategia | Delays (base=1s) | Descripción |
|------------|------------------|-------------|
| `NONE` | - | Sin reintentos |
| `FIXED` | 1, 1, 1, ... | Delay constante |
| `LINEAR` | 1, 2, 3, ... | Incremento lineal |
| `EXPONENTIAL` | 1, 2, 4, 8, ... | Incremento exponencial |
| `FIBONACCI` | 1, 1, 2, 3, 5, ... | Secuencia Fibonacci |

---

## 📊 Métricas

### Métricas de API

```python
api = manager.get_api("openai")
stats = api.get_stats()

print(f"Total requests: {stats['metrics']['total_requests']}")
print(f"Success rate: {stats['metrics']['success_rate']:.2%}")
print(f"Avg latency: {stats['metrics']['average_latency_ms']:.2f}ms")
print(f"Total tokens: {stats['metrics']['total_tokens_used']}")
```

### Métricas de Rate Limiter

```python
limiter_stats = api.rate_limiter.get_stats()

for name, data in limiter_stats["limiters"].items():
    print(f"{name.upper()}:")
    print(f"  Acquired: {data.get('total_acquired', 0)}")
    print(f"  Denied: {data.get('total_denied', 0)}")
```

### Métricas de Circuit Breaker

```python
cb_stats = api.circuit_breaker.get_stats()

print(f"State: {cb_stats['state']}")
print(f"Failures: {cb_stats['failure_count']}")
print(f"Total calls: {cb_stats['total_calls']}")
```

---

## 🎯 Decoradores

### @with_rate_limit

```python
from api_manager import with_rate_limit, set_default_limiter

limiter = create_limiter_for_provider("openai")
set_default_limiter(limiter)

@with_rate_limit(tokens=500)
def call_openai(prompt: str) -> str:
    return openai_client.generate(prompt)
```

### @with_retry

```python
from api_manager import with_retry, RetryConfig

@with_retry(RetryConfig(max_retries=3))
def unreliable_api_call():
    return requests.get("https://api.example.com/data")
```

### @with_circuit_breaker

```python
from api_manager import with_circuit_breaker, CircuitBreaker, CircuitBreakerConfig

breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))

@with_circuit_breaker(breaker)
def protected_call():
    return external_service.call()
```

### @with_managed_api

```python
from api_manager import with_managed_api, get_default_manager

manager = get_default_manager()
manager.register_provider("openai", api_key="sk-...")

@with_managed_api(api_name="openai", tokens=500)
def call_openai(prompt: str) -> str:
    return openai_client.generate(prompt)
```

---

## 🤖 Presets de Proveedores

### Rate Limit Presets

```python
from api_manager import AIProviderPresets

# Obtener configuración preconfigurada
config = AIProviderPresets.openai_gpt4()
config = AIProviderPresets.google_gemini()
config = AIProviderPresets.anthropic_claude()
config = AIProviderPresets.perplexity()
config = AIProviderPresets.deepseek()
config = AIProviderPresets.ollama_local()
```

### API Config Presets

```python
from api_manager import APIConfigPresets

# Configuración completa con rate limit, circuit breaker, retry
config = APIConfigPresets.openai(api_key="sk-...")
config = APIConfigPresets.gemini(api_key="AI...")
config = APIConfigPresets.anthropic(api_key="sk-ant-...")
config = APIConfigPresets.perplexity(api_key="pplx-...")
config = APIConfigPresets.deepseek(api_key="sk-...")
config = APIConfigPresets.ollama(base_url="http://localhost:11434")
```

### Tabla de Límites por Proveedor

| Proveedor | RPM | TPM | Concurrent |
|-----------|-----|-----|------------|
| OpenAI GPT-4 | 500 | 10,000 | 100 |
| OpenAI GPT-4 Turbo | 5,000 | 450,000 | 100 |
| OpenAI GPT-3.5 | 10,000 | 2,000,000 | 200 |
| Gemini Free | 60 | 1,000,000 | 10 |
| Gemini Pro | 1,000 | 4,000,000 | 100 |
| Claude | 50 | 100,000 | 10 |
| Perplexity | 60 | 100,000 | - |
| DeepSeek | 60 | 1,000,000 | - |
| Ollama | - | - | 2 |

---

## 📚 API Reference

### Clases Principales

| Clase | Descripción |
|-------|-------------|
| `APIManager` | Gestor central de APIs |
| `APIRateLimiter` | Rate limiter configurable |
| `CircuitBreaker` | Implementación circuit breaker |
| `ManagedAPI` | API gestionada individual |
| `APIMetrics` | Métricas de una API |

### Configuraciones

| Clase | Descripción |
|-------|-------------|
| `APIConfig` | Configuración completa de API |
| `RateLimitConfig` | Configuración de rate limiting |
| `CircuitBreakerConfig` | Configuración de circuit breaker |
| `RetryConfig` | Configuración de reintentos |

### Excepciones

| Excepción | Descripción |
|-----------|-------------|
| `APIManagerError` | Base para errores del manager |
| `RateLimitExceededError` | Rate limit excedido |
| `CircuitBreakerOpenError` | Circuit breaker abierto |
| `RetryExhaustedError` | Reintentos agotados |
| `APINotRegisteredError` | API no registrada |

### Enumeraciones

| Enum | Valores |
|------|---------|
| `RateLimitAlgorithm` | TOKEN_BUCKET, SLIDING_WINDOW, FIXED_WINDOW |
| `CircuitBreakerState` | CLOSED, OPEN, HALF_OPEN |
| `RetryStrategy` | NONE, FIXED, LINEAR, EXPONENTIAL, FIBONACCI |
| `APIStatus` | HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN |

---

## 📝 Ejemplos Completos

### Ejemplo: Cliente de IA Robusto

```python
from api_manager import (
    APIManager,
    APIConfig,
    RateLimitConfig,
    RetryConfig,
    CircuitBreakerConfig,
    RetryStrategy
)

# Configuración robusta
config = APIConfig(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    rate_limit=RateLimitConfig(
        rpm=100,
        tpm=150000,
        concurrent_limit=10
    ),
    retry=RetryConfig(
        strategy=RetryStrategy.EXPONENTIAL,
        max_retries=3,
        base_delay=1.0
    ),
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=5,
        timeout=60.0
    ),
    timeout=60.0
)

# Crear manager y registrar
manager = APIManager()
manager.register_api("openai", config)

# Usar
async def generate_text(prompt: str) -> str:
    async with manager.async_acquire("openai", tokens=len(prompt)//4 + 500):
        response = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# Ver estadísticas
print(manager.get_all_stats())
```

### Ejemplo: Procesamiento por Lotes

```python
from api_manager import APIManager
import asyncio

manager = APIManager()
manager.register_provider("gemini", api_key="AI...")

async def process_batch(items: list) -> list:
    results = []
    
    for item in items:
        tokens = len(item) // 4 + 200
        
        async with manager.async_acquire("gemini", tokens=tokens):
            result = await process_item(item)
            results.append(result)
    
    # Ver métricas después del procesamiento
    stats = manager.get_api("gemini").get_stats()
    print(f"Processed: {stats['metrics']['total_requests']}")
    print(f"Success rate: {stats['metrics']['success_rate']:.2%}")
    
    return results
```

---

## Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| `requests` | >= 2.28.0 | HTTP library for API calls |
| `asyncio` | built-in | Async support for Python |

---

## Contact

- **Author**: [DatamanEdge](https://github.com/DatamanEdge)
- **Email**: [jrodriguezga@outlook.com](mailto:jrodriguezga@outlook.com)
- **LinkedIn**: [Javier Rodríguez](https://es.linkedin.com/in/javier-rodriguez-ga)

---

## License

MIT © 2026 DatamanEdge. See [LICENSE](../../LICENSE).
