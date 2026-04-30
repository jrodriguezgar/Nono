"""
Provider Fallback Handler — automatic failover across LLM providers.

When a ``generate_completion`` call fails (network error, timeout, API error,
rate-limit), the handler transparently retries with the next provider/model
in the configured fallback chain.

Configuration lives in ``config.toml`` under the ``[fallback]`` section.

Usage (standalone)::

    from nono.connector.fallback import FallbackHandler

    handler = FallbackHandler(primary_provider="google", primary_model="gemini-3-flash-preview")
    response = handler.generate_completion(messages=[{"role": "user", "content": "Hello"}])

Usage (agent integration — automatic when fallback is enabled in config)::

    agent = Agent(name="demo", provider="google", model="gemini-3-flash-preview")
    # If Google fails, the agent automatically falls back to the next provider.

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_logger = logging.getLogger("Nono.Connector.Fallback")


@dataclass
class FallbackEntry:
    """A single provider/model in the fallback chain.

    Attributes:
        provider: Provider identifier (e.g. ``"google"``, ``"openai"``).
        model: Model name override.  Empty string means *use the provider
               default from config.toml*.
    """

    provider: str
    model: str = ""


@dataclass
class FallbackConfig:
    """Parsed ``[fallback]`` configuration.

    Attributes:
        enabled: Whether fallback is active.
        max_retries: Retries per provider before moving to the next one.
        timeout: Per-request timeout in seconds (0 = no override).
        chain: Ordered list of fallback entries.
    """

    enabled: bool = True
    max_retries: int = 1
    timeout: int = 30
    chain: list[FallbackEntry] = field(default_factory=list)


def load_fallback_config(config_path: Optional[str | Path] = None) -> FallbackConfig:
    """Load the ``[fallback]`` section from *config.toml*.

    Args:
        config_path: Explicit path to config.toml.  When *None*, the
                     default ``nono/config/config.toml`` is used.

    Returns:
        A populated ``FallbackConfig``.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config.toml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        _logger.debug("Config file not found at %s — fallback disabled.", config_path)
        return FallbackConfig(enabled=False)

    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(config_path, "rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:
        _logger.warning("Failed to read fallback config: %s", exc)
        return FallbackConfig(enabled=False)

    fb = data.get("fallback", {})

    if not fb:
        return FallbackConfig(enabled=False)

    chain: list[FallbackEntry] = []
    for entry in fb.get("chain", []):
        chain.append(FallbackEntry(
            provider=entry.get("provider", ""),
            model=entry.get("model", ""),
        ))

    return FallbackConfig(
        enabled=fb.get("enabled", True),
        max_retries=fb.get("max_retries", 1),
        timeout=fb.get("timeout", 30),
        chain=chain,
    )


class FallbackHandler:
    """Wraps LLM calls with automatic provider failover.

    The handler keeps a *primary* service and lazily creates backup services
    from the configured fallback chain when the primary fails.

    Args:
        primary_provider: Primary provider name.
        primary_model: Primary model name (``None`` = provider default).
        api_key: API key override for the primary provider.
        config: Pre-loaded ``FallbackConfig`` (loaded automatically if *None*).
        service_kwargs: Extra kwargs forwarded to every service constructor.

    Example:
        >>> handler = FallbackHandler("google", "gemini-3-flash-preview")
        >>> text = handler.generate_completion(
        ...     messages=[{"role": "user", "content": "Hi"}],
        ...     temperature=0.7,
        ... )
    """

    def __init__(
        self,
        primary_provider: str,
        primary_model: str | None = None,
        api_key: str | None = None,
        config: FallbackConfig | None = None,
        **service_kwargs: Any,
    ) -> None:
        self._primary_provider = primary_provider
        self._primary_model = primary_model
        self._api_key = api_key
        self._service_kwargs = service_kwargs
        self._config = config if config is not None else load_fallback_config()

        # LRU cache of instantiated services keyed by (provider, model).
        # Bounded to prevent unbounded growth when many provider/model
        # combinations are used.
        self._services: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._max_cached_services: int = 20

    @property
    def enabled(self) -> bool:
        """Whether fallback is enabled and there are alternative providers."""
        return self._config.enabled and len(self._config.chain) > 0

    def _get_service(self, provider: str, model: str | None = None) -> Any:
        """Return a cached or newly created service instance.

        Args:
            provider: Provider name.
            model: Model name override.

        Returns:
            A ``GenerativeAIService`` subclass instance.
        """
        cache_key = (provider, model or "")

        if cache_key not in self._services:
            from nono.agent.llm_agent import _create_service
            self._services[cache_key] = _create_service(
                provider, model, self._api_key, **self._service_kwargs,
            )
            # Evict oldest if over capacity
            while len(self._services) > self._max_cached_services:
                self._services.popitem(last=False)
        else:
            # Move to end (most recently used)
            self._services.move_to_end(cache_key)

        return self._services[cache_key]

    def _build_chain(self) -> list[tuple[str, str | None]]:
        """Build the ordered list of (provider, model) to try.

        The primary provider is always first.  Remaining entries come from
        the config chain, skipping the primary to avoid duplication.

        Returns:
            Ordered list of (provider, model) tuples.
        """
        chain: list[tuple[str, str | None]] = [
            (self._primary_provider, self._primary_model),
        ]

        for entry in self._config.chain:
            key = entry.provider.lower()

            if key == self._primary_provider.lower() and (
                not entry.model or entry.model == (self._primary_model or "")
            ):
                continue  # skip duplicate of primary

            chain.append((entry.provider, entry.model or None))

        return chain

    def generate_completion(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Call ``generate_completion`` with automatic fallback.

        Tries the primary provider first, then walks through the fallback
        chain.  Each provider is retried up to ``max_retries`` times.

        Args:
            messages: Chat messages in OpenAI format.
            **kwargs: Extra arguments forwarded to ``generate_completion``
                      (temperature, max_tokens, response_format, tools, …).

        Returns:
            The generated text from the first successful provider.

        Raises:
            RuntimeError: When all providers in the chain have been exhausted.
        """
        if not self.enabled:
            # Fallback disabled — single call, no retry
            service = self._get_service(self._primary_provider, self._primary_model)
            return service.generate_completion(messages=messages, **kwargs)

        chain = self._build_chain()
        last_error: Exception | None = None

        for idx, (provider, model) in enumerate(chain):
            for attempt in range(1, self._config.max_retries + 1):
                try:
                    service = self._get_service(provider, model)
                    _logger.debug(
                        "Fallback attempt provider=%s model=%s attempt=%d/%d",
                        provider, model or "(default)", attempt, self._config.max_retries,
                    )

                    _start = time.perf_counter()
                    result = service.generate_completion(messages=messages, **kwargs)
                    _elapsed = time.perf_counter() - _start

                    if idx > 0:
                        _logger.info(
                            "Fallback succeeded → provider=%s model=%s (%.1fs)",
                            provider, model or "(default)", _elapsed,
                        )

                    return result

                except Exception as exc:
                    last_error = exc
                    _logger.warning(
                        "Provider %s (model=%s) failed [attempt %d/%d]: %s",
                        provider, model or "(default)",
                        attempt, self._config.max_retries, exc,
                    )

        raise RuntimeError(
            f"All fallback providers exhausted. Last error: {last_error}"
        ) from last_error
