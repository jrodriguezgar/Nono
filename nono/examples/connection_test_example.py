"""
Example: Connection Test - Verify connectivity with all available AI providers.
Usage: python connection_test_example.py

This example demonstrates:
1. Testing each provider connector individually
2. Verifying API keys and endpoints are configured correctly
3. A simple "ping" prompt to confirm round-trip communication
"""

import sys
import os
import time
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nono.connector.connector_genai import (
    GeminiService,
    OpenAIService,
    PerplexityService,
    DeepSeekService,
    XAIService,
    GroqService,
    CerebrasService,
    NvidiaService,
    GitHubModelsService,
    OpenRouterService,
    VercelAIService,
    OllamaService,
    ResponseFormat,
)

# Simple test prompt — short response to minimize token usage
TEST_MESSAGES = [{"role": "user", "content": "Respond with exactly one word: OK"}]
TIMEOUT_SECONDS = 60


def _run_provider(service_factory, **kwargs):
    """Run a single provider call (used inside a thread for timeout)."""
    service = service_factory(**kwargs)
    return service.generate_completion(
        messages=TEST_MESSAGES,
        temperature=0.0,
        max_tokens=10,
        response_format=ResponseFormat.TEXT,
    )


def test_provider(name: str, service_factory, **kwargs) -> dict:
    """Test a single provider and return the result.
    
    Args:
        name: Display name for the provider.
        service_factory: Callable that creates the service instance.
        **kwargs: Extra keyword arguments for the factory.

    Returns:
        dict with keys: provider, status, response, elapsed, error
    """
    result = {"provider": name, "status": "⏳", "response": "", "elapsed": 0.0, "error": ""}
    start = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_provider, service_factory, **kwargs)
            response = future.result(timeout=TIMEOUT_SECONDS)
        elapsed = time.perf_counter() - start
        result["status"] = "✅"
        result["response"] = (response or "").strip()[:80]
        result["elapsed"] = elapsed
    except concurrent.futures.TimeoutError:
        elapsed = time.perf_counter() - start
        result["status"] = "⏱️"
        result["error"] = f"Timeout after {TIMEOUT_SECONDS}s"
        result["elapsed"] = elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start
        result["status"] = "❌"
        result["error"] = str(e)[:120]
        result["elapsed"] = elapsed
    return result


# ---------------------------------------------------------------------------
# Provider definitions: (display_name, factory_callable, kwargs)
# Keys are resolved automatically from apikeys.csv by each service.
# Providers without a valid key will fail gracefully and be reported.
# ---------------------------------------------------------------------------
PROVIDERS = [
    ("Google Gemini", lambda **kw: GeminiService(model_name="gemini-3-flash-preview", **kw)),
    ("OpenAI", lambda **kw: OpenAIService(model_name="gpt-4o-mini", **kw)),
    ("Groq", lambda **kw: GroqService(model_name="llama-3.3-70b-versatile", **kw)),
    ("Cerebras", lambda **kw: CerebrasService(model_name="gpt-oss-120b", **kw)),
    ("NVIDIA", lambda **kw: NvidiaService(model_name="meta/llama-3.3-70b-instruct", **kw)),
    ("GitHub Models", lambda **kw: GitHubModelsService(model_name="openai/gpt-4o-mini", **kw)),
    ("OpenRouter", lambda **kw: OpenRouterService(model_name="openrouter/auto", **kw)),
    ("Perplexity", lambda **kw: PerplexityService(model_name="sonar", **kw)),
    ("DeepSeek", lambda **kw: DeepSeekService(model_name="deepseek-chat", **kw)),
    ("xAI", lambda **kw: XAIService(model_name="grok-3", **kw)),
    ("Vercel AI Gateway", lambda **kw: VercelAIService(model_name="google/gemini-3-flash", **kw)),
    ("Ollama (local)", lambda **kw: OllamaService(model_name="llama3", **kw)),
]


def run_all_tests():
    """Execute connection tests for every configured provider."""
    print(f"\n{'='*72}")
    print("  Nono — Connection Test for All Providers")
    print(f"{'='*72}\n")

    results: list[dict] = []

    for i, (name, factory) in enumerate(PROVIDERS, 1):
        tag = f"[{i:>2}/{len(PROVIDERS)}]"
        print(f"  {tag} Testing {name:<20s} ... ", end="", flush=True)
        result = test_provider(name, factory)
        results.append(result)

        if result["status"] == "✅":
            print(f"{result['status']}  {result['elapsed']:.2f}s  → {result['response']}")
        else:
            print(f"{result['status']}  {result['elapsed']:.2f}s  → {result['error']}")

    # Summary table
    ok = sum(1 for r in results if r["status"] == "✅")
    fail = len(results) - ok

    print(f"\n{'─'*72}")
    print(f"  Results: {ok} passed, {fail} failed, {len(results)} total")
    print(f"{'─'*72}")

    print(f"\n  {'Provider':<20s} {'Status':<6s} {'Time':>7s}  {'Response / Error'}")
    print(f"  {'─'*20} {'─'*6} {'─'*7}  {'─'*35}")
    for r in results:
        time_str = f"{r['elapsed']:.2f}s"
        detail = r["response"] if r["status"] == "✅" else r["error"]
        print(f"  {r['provider']:<20s} {r['status']:<6s} {time_str:>7s}  {detail}")

    print()


if __name__ == "__main__":
    run_all_tests()
