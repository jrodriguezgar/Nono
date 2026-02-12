# -*- coding: utf-8 -*-
"""
API Manager - Comprehensive API Management Module

Provides a complete API management solution including:
- Rate Limiting (Token Bucket, Sliding Window, Fixed Window algorithms)
- Circuit Breaker pattern for fault tolerance
- Retry policies with exponential backoff
- API health monitoring
- API registry and configuration management
- Comprehensive metrics and statistics

Supported Limit Types:
    - RPM (Requests Per Minute)
    - RPD (Requests Per Day)
    - TPM (Tokens Per Minute - for AI APIs)
    - TPD (Tokens Per Day - for AI APIs)
    - RPS (Requests Per Second)
    - Concurrent Requests Limit
    - Custom time windows

Example:
    >>> from api_manager import APIManager, APIConfig
    >>> manager = APIManager()
    >>> manager.register_api("openai", APIConfig(
    ...     base_url="https://api.openai.com/v1",
    ...     rate_limit=RateLimitConfig(rpm=60, tpm=100000)
    ... ))
    >>> async with manager.acquire("openai", tokens=1500):
    ...     response = await make_api_call()

Author: DatamanEdge
License: MIT
Date: 2026-02-05
Version: 2.0.0
"""

import asyncio
import threading
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, Callable, List, TypeVar, Generic
from collections import deque
from functools import wraps
import logging

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class RateLimitAlgorithm(Enum):
    """
    Rate limiting algorithm types.
    
    TOKEN_BUCKET: Classic token bucket algorithm - smooth rate limiting
    SLIDING_WINDOW: Sliding window log algorithm - precise counting
    FIXED_WINDOW: Fixed window counter - simple but can have burst at boundaries
    LEAKY_BUCKET: Leaky bucket algorithm - constant output rate
    """
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitScope(Enum):
    """
    Scope for rate limit application.
    
    GLOBAL: Single limit for all operations
    PER_ENDPOINT: Separate limits per API endpoint
    PER_MODEL: Separate limits per AI model (useful for AI APIs)
    PER_USER: Separate limits per user/client
    """
    GLOBAL = "global"
    PER_ENDPOINT = "per_endpoint"
    PER_MODEL = "per_model"
    PER_USER = "per_user"


class TimeUnit(Enum):
    """Time units for rate limit windows."""
    SECOND = 1
    MINUTE = 60
    HOUR = 3600
    DAY = 86400


class RateLimitExceededAction(Enum):
    """
    Action to take when rate limit is exceeded.
    
    WAIT: Block and wait for availability
    RAISE: Raise an exception immediately
    QUEUE: Add to queue for later processing
    DROP: Silently drop the request
    """
    WAIT = "wait"
    RAISE = "raise"
    QUEUE = "queue"
    DROP = "drop"


class CircuitBreakerState(Enum):
    """
    States for the Circuit Breaker pattern.
    
    CLOSED: Normal operation, requests pass through
    OPEN: Circuit is open, requests are rejected
    HALF_OPEN: Testing if service has recovered
    """
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RetryStrategy(Enum):
    """
    Retry strategies for failed requests.
    
    NONE: No retries
    FIXED: Fixed delay between retries
    LINEAR: Linearly increasing delay
    EXPONENTIAL: Exponentially increasing delay with jitter
    FIBONACCI: Fibonacci sequence delay
    """
    NONE = "none"
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"


class APIStatus(Enum):
    """API health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# =============================================================================
# EXCEPTIONS
# =============================================================================

class APIManagerError(Exception):
    """Base exception for API Manager errors."""
    pass


class RateLimitError(APIManagerError):
    """Base exception for rate limit errors."""
    pass


class RateLimitExceededError(RateLimitError):
    """Raised when rate limit is exceeded and action is RAISE."""
    def __init__(self, message: str, retry_after: Optional[float] = None,
                 limit_type: Optional[str] = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.limit_type = limit_type


class RateLimitConfigError(RateLimitError):
    """Raised when rate limit configuration is invalid."""
    pass


class CircuitBreakerOpenError(APIManagerError):
    """Raised when circuit breaker is open."""
    def __init__(self, message: str, api_name: str, reset_time: Optional[float] = None):
        super().__init__(message)
        self.api_name = api_name
        self.reset_time = reset_time


class RetryExhaustedError(APIManagerError):
    """Raised when all retries are exhausted."""
    def __init__(self, message: str, attempts: int, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


class APINotRegisteredError(APIManagerError):
    """Raised when accessing an unregistered API."""
    pass


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class RateLimitConfig:
    """
    Configuration for API rate limiting.
    
    Supports multiple simultaneous limits that work together.
    All limits are optional - only configure what you need.
    
    Attributes:
        rpm: Requests per minute limit
        rpd: Requests per day limit
        rps: Requests per second limit
        tpm: Tokens per minute limit (for AI APIs)
        tpd: Tokens per day limit (for AI APIs)
        concurrent_limit: Maximum concurrent requests
        burst_size: Maximum burst size for token bucket (default: rpm/10 or 10)
        algorithm: Rate limiting algorithm to use
        action: Action when limit is exceeded
        max_wait_time: Maximum time to wait for token (seconds)
        retry_after_header: Whether to respect Retry-After headers
        scope: Scope for rate limit application
        custom_limits: Dictionary of custom limits {name: (count, window_seconds)}
    
    Example:
        # OpenAI GPT-4 typical limits
        config = RateLimitConfig(
            rpm=10000,
            tpm=300000,
            concurrent_limit=100
        )
    """
    # Request limits
    rpm: Optional[int] = None  # Requests per minute
    rpd: Optional[int] = None  # Requests per day
    rps: Optional[float] = None  # Requests per second
    
    # Token limits (for AI APIs)
    tpm: Optional[int] = None  # Tokens per minute
    tpd: Optional[int] = None  # Tokens per day
    
    # Concurrency
    concurrent_limit: Optional[int] = None
    
    # Algorithm configuration
    burst_size: Optional[int] = None
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    
    # Behavior configuration
    action: RateLimitExceededAction = RateLimitExceededAction.WAIT
    max_wait_time: float = 300.0  # 5 minutes default
    retry_after_header: bool = True
    
    # Scope
    scope: RateLimitScope = RateLimitScope.GLOBAL
    
    # Custom limits: {name: (count, window_seconds)}
    custom_limits: Dict[str, tuple] = field(default_factory=dict)
    
    # Callbacks
    on_limit_exceeded: Optional[Callable[[str, float], None]] = None
    on_limit_reset: Optional[Callable[[str], None]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.burst_size is None:
            if self.rpm:
                self.burst_size = max(self.rpm // 10, 10)
            elif self.rps:
                self.burst_size = max(int(self.rps * 10), 10)
            else:
                self.burst_size = 10
        
        # Validate that at least one limit is set
        has_limit = any([
            self.rpm, self.rpd, self.rps,
            self.tpm, self.tpd, self.concurrent_limit,
            self.custom_limits
        ])
        
        if not has_limit:
            logger.warning("RateLimitConfig created without any limits configured")


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.
    
    Attributes:
        strategy: Retry strategy to use
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        jitter: Add random jitter to delays (0.0 to 1.0)
        retryable_exceptions: Exception types that should trigger retry
        retryable_status_codes: HTTP status codes that should trigger retry
    """
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: float = 0.1
    retryable_exceptions: tuple = (ConnectionError, TimeoutError)
    retryable_status_codes: tuple = (429, 500, 502, 503, 504)
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        if self.strategy == RetryStrategy.NONE:
            return 0.0
        elif self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * attempt
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** (attempt - 1))
        elif self.strategy == RetryStrategy.FIBONACCI:
            a, b = 1, 1
            for _ in range(attempt - 1):
                a, b = b, a + b
            delay = self.base_delay * a
        else:
            delay = self.base_delay
        
        # Apply max_delay cap
        delay = min(delay, self.max_delay)
        
        # Apply jitter
        if self.jitter > 0:
            jitter_amount = delay * self.jitter * random.random()
            delay += jitter_amount
        
        return delay


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for the Circuit Breaker pattern.
    
    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes in half-open to close circuit
        timeout: Time in seconds before attempting recovery (half-open)
        half_open_max_calls: Maximum calls allowed in half-open state
        exclude_exceptions: Exceptions that should not count as failures
    """
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: float = 60.0
    half_open_max_calls: int = 3
    exclude_exceptions: tuple = ()


@dataclass
class HealthCheckConfig:
    """
    Configuration for API health checking.
    
    Attributes:
        enabled: Whether health checking is enabled
        interval: Seconds between health checks
        timeout: Timeout for health check requests
        healthy_threshold: Consecutive successes to mark healthy
        unhealthy_threshold: Consecutive failures to mark unhealthy
        endpoint: Optional health check endpoint path
    """
    enabled: bool = True
    interval: float = 60.0
    timeout: float = 10.0
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    endpoint: Optional[str] = None


@dataclass
class APIConfig:
    """
    Complete configuration for a managed API.
    
    Attributes:
        base_url: Base URL for the API
        api_key: API key for authentication
        rate_limit: Rate limiting configuration
        retry: Retry configuration
        circuit_breaker: Circuit breaker configuration
        health_check: Health check configuration
        timeout: Default request timeout
        headers: Default headers to include
        metadata: Additional metadata for the API
    """
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    rate_limit: Optional[RateLimitConfig] = None
    retry: Optional[RetryConfig] = None
    circuit_breaker: Optional[CircuitBreakerConfig] = None
    health_check: Optional[HealthCheckConfig] = None
    timeout: float = 30.0
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# PRESET CONFIGURATIONS FOR POPULAR AI APIs
# =============================================================================

class AIProviderPresets:
    """
    Preset configurations for popular AI API providers.
    
    These are typical limits - actual limits may vary by tier/plan.
    Always verify with the provider's documentation.
    """
    
    @staticmethod
    def openai_gpt4() -> RateLimitConfig:
        """OpenAI GPT-4 typical limits (Tier 1)."""
        return RateLimitConfig(
            rpm=500,
            rpd=10000,
            tpm=10000,
            concurrent_limit=100,
            action=RateLimitExceededAction.WAIT
        )
    
    @staticmethod
    def openai_gpt4_turbo() -> RateLimitConfig:
        """OpenAI GPT-4 Turbo typical limits."""
        return RateLimitConfig(
            rpm=5000,
            tpm=450000,
            concurrent_limit=100,
            action=RateLimitExceededAction.WAIT
        )
    
    @staticmethod
    def openai_gpt35_turbo() -> RateLimitConfig:
        """OpenAI GPT-3.5 Turbo typical limits."""
        return RateLimitConfig(
            rpm=10000,
            tpm=2000000,
            concurrent_limit=200,
            action=RateLimitExceededAction.WAIT
        )
    
    @staticmethod
    def google_gemini() -> RateLimitConfig:
        """Google Gemini API typical limits (free tier)."""
        return RateLimitConfig(
            rpm=60,
            rpd=1500,
            tpm=1000000,
            concurrent_limit=10,
            action=RateLimitExceededAction.WAIT
        )
    
    @staticmethod
    def google_gemini_pro() -> RateLimitConfig:
        """Google Gemini Pro API typical limits (paid tier)."""
        return RateLimitConfig(
            rpm=1000,
            tpm=4000000,
            concurrent_limit=100,
            action=RateLimitExceededAction.WAIT
        )
    
    @staticmethod
    def anthropic_claude() -> RateLimitConfig:
        """Anthropic Claude API typical limits."""
        return RateLimitConfig(
            rpm=50,
            tpm=100000,
            concurrent_limit=10,
            action=RateLimitExceededAction.WAIT
        )
    
    @staticmethod
    def perplexity() -> RateLimitConfig:
        """Perplexity API typical limits."""
        return RateLimitConfig(
            rpm=60,
            tpm=100000,
            action=RateLimitExceededAction.WAIT
        )
    
    @staticmethod
    def deepseek() -> RateLimitConfig:
        """DeepSeek API typical limits."""
        return RateLimitConfig(
            rpm=60,
            tpm=1000000,
            action=RateLimitExceededAction.WAIT
        )
    
    @staticmethod
    def cerebras() -> RateLimitConfig:
        """Cerebras API typical limits (free tier - 1M tokens/day)."""
        return RateLimitConfig(
            rpm=30,
            tpd=1000000,
            action=RateLimitExceededAction.WAIT
        )
    
    @staticmethod
    def nvidia() -> RateLimitConfig:
        """NVIDIA NIM API typical limits."""
        return RateLimitConfig(
            rpm=60,
            tpm=100000,
            action=RateLimitExceededAction.WAIT
        )
    
    @staticmethod
    def foundry() -> RateLimitConfig:
        """Microsoft Foundry (GitHub Models) API typical limits (Copilot Free tier)."""
        return RateLimitConfig(
            rpm=15,
            rpd=150,
            tpm=120000,
            action=RateLimitExceededAction.WAIT
        )
    
    @staticmethod
    def ollama_local() -> RateLimitConfig:
        """Ollama local - no real limits, but prevent overload."""
        return RateLimitConfig(
            concurrent_limit=2,  # Limit concurrent to prevent resource issues
            action=RateLimitExceededAction.WAIT
        )
    
    @staticmethod
    def generic_conservative() -> RateLimitConfig:
        """Generic conservative settings for unknown APIs."""
        return RateLimitConfig(
            rpm=30,
            tpm=50000,
            concurrent_limit=5,
            max_wait_time=60.0,
            action=RateLimitExceededAction.WAIT
        )
    
    @staticmethod
    def generic_aggressive() -> RateLimitConfig:
        """Generic aggressive settings for high-throughput needs."""
        return RateLimitConfig(
            rpm=1000,
            tpm=500000,
            concurrent_limit=50,
            action=RateLimitExceededAction.WAIT
        )


class APIConfigPresets:
    """
    Complete API configuration presets for popular providers.
    """
    
    @staticmethod
    def openai(api_key: Optional[str] = None) -> APIConfig:
        """OpenAI complete configuration."""
        return APIConfig(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            rate_limit=AIProviderPresets.openai_gpt4_turbo(),
            retry=RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL,
                max_retries=3,
                retryable_status_codes=(429, 500, 502, 503, 504)
            ),
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=5,
                timeout=30.0
            ),
            timeout=60.0
        )
    
    @staticmethod
    def gemini(api_key: Optional[str] = None) -> APIConfig:
        """Google Gemini complete configuration."""
        return APIConfig(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key=api_key,
            rate_limit=AIProviderPresets.google_gemini(),
            retry=RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL,
                max_retries=3
            ),
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=5,
                timeout=30.0
            ),
            timeout=60.0
        )
    
    @staticmethod
    def anthropic(api_key: Optional[str] = None) -> APIConfig:
        """Anthropic Claude complete configuration."""
        return APIConfig(
            base_url="https://api.anthropic.com/v1",
            api_key=api_key,
            rate_limit=AIProviderPresets.anthropic_claude(),
            retry=RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL,
                max_retries=3
            ),
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=5,
                timeout=30.0
            ),
            timeout=120.0
        )
    
    @staticmethod
    def perplexity(api_key: Optional[str] = None) -> APIConfig:
        """Perplexity complete configuration."""
        return APIConfig(
            base_url="https://api.perplexity.ai",
            api_key=api_key,
            rate_limit=AIProviderPresets.perplexity(),
            retry=RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL,
                max_retries=3
            ),
            timeout=60.0
        )
    
    @staticmethod
    def deepseek(api_key: Optional[str] = None) -> APIConfig:
        """DeepSeek complete configuration."""
        return APIConfig(
            base_url="https://api.deepseek.com",
            api_key=api_key,
            rate_limit=AIProviderPresets.deepseek(),
            retry=RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL,
                max_retries=3
            ),
            timeout=60.0
        )
    
    @staticmethod
    def ollama(base_url: str = "http://localhost:11434") -> APIConfig:
        """Ollama local complete configuration."""
        return APIConfig(
            base_url=base_url,
            rate_limit=AIProviderPresets.ollama_local(),
            timeout=120.0
        )


# =============================================================================
# BASE LIMITER INTERFACE
# =============================================================================

class BaseLimiter(ABC):
    """Abstract base class for rate limiters."""
    
    @abstractmethod
    def acquire(self, count: int = 1) -> bool:
        """Attempt to acquire permission for a request."""
        pass
    
    @abstractmethod
    def wait_for_permit(self, count: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait until permission is available."""
        pass
    
    @abstractmethod
    def get_wait_time(self, count: int = 1) -> float:
        """Get estimated wait time until permission is available."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the limiter to initial state."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get current limiter statistics."""
        pass


# =============================================================================
# LIMITER IMPLEMENTATIONS
# =============================================================================

class TokenBucketLimiter(BaseLimiter):
    """
    Token bucket rate limiter implementation.
    
    Tokens are added at a constant rate up to a maximum (burst) capacity.
    Requests consume tokens, and if none available, must wait.
    """
    
    def __init__(self, rate: float, window: TimeUnit, burst_size: int):
        self.rate = rate
        self.window_seconds = window.value
        self.tokens_per_second = rate / self.window_seconds
        self.burst_size = burst_size
        
        self.tokens = float(burst_size)
        self.last_refill = time.monotonic()
        self.lock = threading.RLock()
        
        self._total_acquired = 0
        self._total_waited = 0
        self._total_denied = 0
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.tokens_per_second
        self.tokens = min(self.burst_size, self.tokens + new_tokens)
        self.last_refill = now
    
    def acquire(self, count: int = 1) -> bool:
        """Attempt to acquire tokens without waiting."""
        with self.lock:
            self._refill()
            if self.tokens >= count:
                self.tokens -= count
                self._total_acquired += count
                return True
            self._total_denied += count
            return False
    
    def wait_for_permit(self, count: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait until tokens are available."""
        start_time = time.monotonic()
        
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= count:
                    self.tokens -= count
                    self._total_acquired += count
                    wait_time = time.monotonic() - start_time
                    if wait_time > 0.01:
                        self._total_waited += 1
                    return True
                
                needed = count - self.tokens
                wait = needed / self.tokens_per_second
            
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed + wait > timeout:
                    self._total_denied += count
                    return False
                wait = min(wait, timeout - elapsed)
            
            time.sleep(min(wait, 0.1))
    
    def get_wait_time(self, count: int = 1) -> float:
        """Get estimated wait time for acquiring tokens."""
        with self.lock:
            self._refill()
            if self.tokens >= count:
                return 0.0
            needed = count - self.tokens
            return needed / self.tokens_per_second
    
    def reset(self) -> None:
        """Reset the bucket to full capacity."""
        with self.lock:
            self.tokens = float(self.burst_size)
            self.last_refill = time.monotonic()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bucket statistics."""
        with self.lock:
            self._refill()
            return {
                "type": "token_bucket",
                "current_tokens": self.tokens,
                "burst_size": self.burst_size,
                "rate_per_second": self.tokens_per_second,
                "total_acquired": self._total_acquired,
                "total_waited": self._total_waited,
                "total_denied": self._total_denied
            }


class SlidingWindowLimiter(BaseLimiter):
    """
    Sliding window log rate limiter implementation.
    
    Maintains a log of timestamps and removes old entries outside the window.
    """
    
    def __init__(self, limit: int, window: TimeUnit):
        self.limit = limit
        self.window_seconds = window.value
        self.requests: deque = deque()
        self.lock = threading.RLock()
        
        self._total_acquired = 0
        self._total_denied = 0
    
    def _cleanup(self) -> None:
        """Remove expired entries from the window."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
    
    def acquire(self, count: int = 1) -> bool:
        """Attempt to acquire without waiting."""
        with self.lock:
            self._cleanup()
            if len(self.requests) + count <= self.limit:
                now = time.monotonic()
                for _ in range(count):
                    self.requests.append(now)
                self._total_acquired += count
                return True
            self._total_denied += count
            return False
    
    def wait_for_permit(self, count: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait until slot is available in the window."""
        start_time = time.monotonic()
        
        while True:
            with self.lock:
                self._cleanup()
                if len(self.requests) + count <= self.limit:
                    now = time.monotonic()
                    for _ in range(count):
                        self.requests.append(now)
                    self._total_acquired += count
                    return True
                
                if self.requests:
                    oldest = self.requests[0]
                    wait = (oldest + self.window_seconds) - time.monotonic()
                    wait = max(wait, 0.01)
                else:
                    wait = 0.01
            
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    self._total_denied += count
                    return False
                wait = min(wait, timeout - elapsed)
            
            time.sleep(min(wait, 0.1))
    
    def get_wait_time(self, count: int = 1) -> float:
        """Get estimated wait time."""
        with self.lock:
            self._cleanup()
            if len(self.requests) + count <= self.limit:
                return 0.0
            if self.requests:
                oldest = self.requests[0]
                return max((oldest + self.window_seconds) - time.monotonic(), 0.0)
            return 0.0
    
    def reset(self) -> None:
        """Clear all entries."""
        with self.lock:
            self.requests.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get window statistics."""
        with self.lock:
            self._cleanup()
            return {
                "type": "sliding_window",
                "current_count": len(self.requests),
                "limit": self.limit,
                "window_seconds": self.window_seconds,
                "total_acquired": self._total_acquired,
                "total_denied": self._total_denied
            }


class FixedWindowLimiter(BaseLimiter):
    """
    Fixed window counter rate limiter implementation.
    
    Simple counter that resets at fixed intervals.
    """
    
    def __init__(self, limit: int, window: TimeUnit):
        self.limit = limit
        self.window_seconds = window.value
        self.count = 0
        self.window_start = time.monotonic()
        self.lock = threading.RLock()
        
        self._total_acquired = 0
        self._total_denied = 0
        self._windows_completed = 0
    
    def _check_window(self) -> None:
        """Check if current window has expired and reset if needed."""
        now = time.monotonic()
        if now - self.window_start >= self.window_seconds:
            self.count = 0
            self.window_start = now
            self._windows_completed += 1
    
    def acquire(self, count: int = 1) -> bool:
        """Attempt to acquire without waiting."""
        with self.lock:
            self._check_window()
            if self.count + count <= self.limit:
                self.count += count
                self._total_acquired += count
                return True
            self._total_denied += count
            return False
    
    def wait_for_permit(self, count: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait until next window if needed."""
        start_time = time.monotonic()
        
        while True:
            with self.lock:
                self._check_window()
                if self.count + count <= self.limit:
                    self.count += count
                    self._total_acquired += count
                    return True
                
                wait = (self.window_start + self.window_seconds) - time.monotonic()
                wait = max(wait, 0.01)
            
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    self._total_denied += count
                    return False
                wait = min(wait, timeout - elapsed)
            
            time.sleep(min(wait, 0.1))
    
    def get_wait_time(self, count: int = 1) -> float:
        """Get estimated wait time."""
        with self.lock:
            self._check_window()
            if self.count + count <= self.limit:
                return 0.0
            return max((self.window_start + self.window_seconds) - time.monotonic(), 0.0)
    
    def reset(self) -> None:
        """Reset to start of new window."""
        with self.lock:
            self.count = 0
            self.window_start = time.monotonic()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get window statistics."""
        with self.lock:
            self._check_window()
            time_remaining = max(
                (self.window_start + self.window_seconds) - time.monotonic(), 
                0.0
            )
            return {
                "type": "fixed_window",
                "current_count": self.count,
                "limit": self.limit,
                "window_seconds": self.window_seconds,
                "time_remaining": time_remaining,
                "total_acquired": self._total_acquired,
                "total_denied": self._total_denied,
                "windows_completed": self._windows_completed
            }


class ConcurrentLimiter:
    """Limits the number of concurrent operations."""
    
    def __init__(self, max_concurrent: int):
        self.max_concurrent = max_concurrent
        self.semaphore = threading.Semaphore(max_concurrent)
        self.lock = threading.RLock()
        
        self._current_active = 0
        self._peak_active = 0
        self._total_acquired = 0
        self._total_denied = 0
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire a concurrent slot."""
        if timeout is None:
            acquired = self.semaphore.acquire(blocking=False)
        else:
            acquired = self.semaphore.acquire(blocking=True, timeout=timeout)
        
        if acquired:
            with self.lock:
                self._current_active += 1
                self._peak_active = max(self._peak_active, self._current_active)
                self._total_acquired += 1
            return True
        
        with self.lock:
            self._total_denied += 1
        return False
    
    def release(self) -> None:
        """Release a concurrent slot."""
        self.semaphore.release()
        with self.lock:
            self._current_active = max(0, self._current_active - 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get concurrent limiter statistics."""
        with self.lock:
            return {
                "type": "concurrent",
                "current_active": self._current_active,
                "max_concurrent": self.max_concurrent,
                "available": self.max_concurrent - self._current_active,
                "peak_active": self._peak_active,
                "total_acquired": self._total_acquired,
                "total_denied": self._total_denied
            }


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Prevents cascading failures by stopping requests to failing services.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self.lock = threading.RLock()
        
        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._state_changes: List[tuple] = []
    
    def _change_state(self, new_state: CircuitBreakerState) -> None:
        """Change circuit breaker state."""
        old_state = self.state
        self.state = new_state
        self._state_changes.append((time.time(), old_state.value, new_state.value))
        logger.info(f"Circuit breaker state changed: {old_state.value} -> {new_state.value}")
    
    def can_execute(self) -> bool:
        """Check if a request can be executed."""
        with self.lock:
            self._total_calls += 1
            
            if self.state == CircuitBreakerState.CLOSED:
                return True
            
            if self.state == CircuitBreakerState.OPEN:
                if self.last_failure_time is None:
                    return False
                
                time_since_failure = time.monotonic() - self.last_failure_time
                if time_since_failure >= self.config.timeout:
                    self._change_state(CircuitBreakerState.HALF_OPEN)
                    self.half_open_calls = 0
                    return True
                return False
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls < self.config.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False
            
            return False
    
    def record_success(self) -> None:
        """Record a successful request."""
        with self.lock:
            self._total_successes += 1
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._change_state(CircuitBreakerState.CLOSED)
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0
    
    def record_failure(self, exception: Optional[Exception] = None) -> None:
        """Record a failed request."""
        # Check if this exception should be excluded
        if exception and isinstance(exception, self.config.exclude_exceptions):
            return
        
        with self.lock:
            self._total_failures += 1
            self.last_failure_time = time.monotonic()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self._change_state(CircuitBreakerState.OPEN)
                self.success_count = 0
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.config.failure_threshold:
                    self._change_state(CircuitBreakerState.OPEN)
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self.lock:
            self._change_state(CircuitBreakerState.CLOSED)
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
            self.last_failure_time = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self.lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "total_calls": self._total_calls,
                "total_successes": self._total_successes,
                "total_failures": self._total_failures,
                "last_failure_time": self.last_failure_time,
                "recent_state_changes": self._state_changes[-10:]
            }


# =============================================================================
# API RATE LIMITER (Compatible with original api_rate_limiter.py)
# =============================================================================

class APIRateLimiter:
    """
    Main API rate limiter that combines multiple limiting strategies.
    
    This class maintains backward compatibility with the original api_rate_limiter module.
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._limiters: Dict[str, BaseLimiter] = {}
        self._concurrent_limiter: Optional[ConcurrentLimiter] = None
        
        self._setup_limiters()
        
        logger.info(f"APIRateLimiter initialized with config: {self._get_config_summary()}")
    
    def _get_config_summary(self) -> str:
        """Get a summary of the configuration."""
        parts = []
        if self.config.rpm:
            parts.append(f"RPM={self.config.rpm}")
        if self.config.rpd:
            parts.append(f"RPD={self.config.rpd}")
        if self.config.rps:
            parts.append(f"RPS={self.config.rps}")
        if self.config.tpm:
            parts.append(f"TPM={self.config.tpm}")
        if self.config.tpd:
            parts.append(f"TPD={self.config.tpd}")
        if self.config.concurrent_limit:
            parts.append(f"Concurrent={self.config.concurrent_limit}")
        return ", ".join(parts) or "No limits configured"
    
    def _create_limiter(self, limit: int, window: TimeUnit) -> BaseLimiter:
        """Create a limiter based on configured algorithm."""
        burst_size = self.config.burst_size if self.config.burst_size else 10
        
        if self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return TokenBucketLimiter(limit, window, burst_size)
        elif self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return SlidingWindowLimiter(limit, window)
        elif self.config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return FixedWindowLimiter(limit, window)
        else:
            return TokenBucketLimiter(limit, window, burst_size)
    
    def _setup_limiters(self) -> None:
        """Set up all configured limiters."""
        if self.config.rpm:
            self._limiters["rpm"] = self._create_limiter(self.config.rpm, TimeUnit.MINUTE)
        
        if self.config.rpd:
            self._limiters["rpd"] = self._create_limiter(self.config.rpd, TimeUnit.DAY)
        
        if self.config.rps:
            self._limiters["rps"] = self._create_limiter(int(self.config.rps), TimeUnit.SECOND)
        
        if self.config.tpm:
            self._limiters["tpm"] = self._create_limiter(self.config.tpm, TimeUnit.MINUTE)
        
        if self.config.tpd:
            self._limiters["tpd"] = self._create_limiter(self.config.tpd, TimeUnit.DAY)
        
        for name, (limit, window_seconds) in self.config.custom_limits.items():
            if window_seconds <= 1:
                window = TimeUnit.SECOND
            elif window_seconds <= 60:
                window = TimeUnit.MINUTE
            elif window_seconds <= 3600:
                window = TimeUnit.HOUR
            else:
                window = TimeUnit.DAY
            self._limiters[name] = self._create_limiter(limit, window)
        
        if self.config.concurrent_limit:
            self._concurrent_limiter = ConcurrentLimiter(self.config.concurrent_limit)
    
    def try_acquire(self, tokens: int = 0) -> bool:
        """Try to acquire permission without waiting."""
        for name in ["rpm", "rpd", "rps"]:
            if name in self._limiters:
                if not self._limiters[name].acquire(1):
                    return False
        
        if tokens > 0:
            for name in ["tpm", "tpd"]:
                if name in self._limiters:
                    if not self._limiters[name].acquire(tokens):
                        for rname in ["rpm", "rpd", "rps"]:
                            if rname in self._limiters:
                                self._limiters[rname].reset()
                        return False
        
        if self._concurrent_limiter:
            if not self._concurrent_limiter.acquire(timeout=0):
                return False
        
        return True
    
    def wait_for_permit(self, tokens: int = 0, timeout: Optional[float] = None) -> bool:
        """Wait for permission from all limiters."""
        if timeout is None:
            timeout = self.config.max_wait_time
        
        start_time = time.monotonic()
        
        for name in ["rpm", "rpd", "rps"]:
            if name in self._limiters:
                remaining = timeout - (time.monotonic() - start_time) if timeout else None
                if not self._limiters[name].wait_for_permit(1, remaining):
                    if self.config.action == RateLimitExceededAction.RAISE:
                        wait_time = self._limiters[name].get_wait_time(1)
                        raise RateLimitExceededError(
                            f"Rate limit exceeded for {name.upper()}",
                            retry_after=wait_time,
                            limit_type=name
                        )
                    if self.config.on_limit_exceeded:
                        self.config.on_limit_exceeded(name, 0)
                    return False
        
        if tokens > 0:
            for name in ["tpm", "tpd"]:
                if name in self._limiters:
                    remaining = timeout - (time.monotonic() - start_time) if timeout else None
                    if not self._limiters[name].wait_for_permit(tokens, remaining):
                        if self.config.action == RateLimitExceededAction.RAISE:
                            wait_time = self._limiters[name].get_wait_time(tokens)
                            raise RateLimitExceededError(
                                f"Token limit exceeded for {name.upper()}",
                                retry_after=wait_time,
                                limit_type=name
                            )
                        if self.config.on_limit_exceeded:
                            self.config.on_limit_exceeded(name, tokens)
                        return False
        
        if self._concurrent_limiter:
            remaining = timeout - (time.monotonic() - start_time) if timeout else None
            if not self._concurrent_limiter.acquire(remaining):
                if self.config.action == RateLimitExceededAction.RAISE:
                    raise RateLimitExceededError(
                        "Concurrent request limit exceeded",
                        limit_type="concurrent"
                    )
                return False
        
        return True
    
    def release_concurrent(self) -> None:
        """Release a concurrent slot after request completes."""
        if self._concurrent_limiter:
            self._concurrent_limiter.release()
    
    def get_wait_time(self, tokens: int = 0) -> float:
        """Get estimated wait time across all limiters."""
        max_wait = 0.0
        
        for name in ["rpm", "rpd", "rps"]:
            if name in self._limiters:
                wait = self._limiters[name].get_wait_time(1)
                max_wait = max(max_wait, wait)
        
        if tokens > 0:
            for name in ["tpm", "tpd"]:
                if name in self._limiters:
                    wait = self._limiters[name].get_wait_time(tokens)
                    max_wait = max(max_wait, wait)
        
        return max_wait
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all limiters."""
        stats = {
            "limiters": {},
            "config": {
                "algorithm": self.config.algorithm.value,
                "action": self.config.action.value,
                "max_wait_time": self.config.max_wait_time
            }
        }
        
        for name, limiter in self._limiters.items():
            stats["limiters"][name] = limiter.get_stats()
        
        if self._concurrent_limiter:
            stats["limiters"]["concurrent"] = self._concurrent_limiter.get_stats()
        
        return stats
    
    def reset_all(self) -> None:
        """Reset all limiters to initial state."""
        for limiter in self._limiters.values():
            limiter.reset()
        logger.info("All rate limiters reset")
    
    class _AcquireContext:
        """Context manager for automatic acquire/release."""
        
        def __init__(self, limiter: 'APIRateLimiter', tokens: int, timeout: Optional[float]):
            self.limiter = limiter
            self.tokens = tokens
            self.timeout = timeout
            self.acquired = False
        
        def __enter__(self) -> 'APIRateLimiter._AcquireContext':
            self.acquired = self.limiter.wait_for_permit(self.tokens, self.timeout)
            if not self.acquired and self.limiter.config.action == RateLimitExceededAction.RAISE:
                raise RateLimitExceededError("Could not acquire rate limit permit")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            if self.acquired:
                self.limiter.release_concurrent()
            return False
    
    def acquire_context(self, tokens: int = 0, timeout: Optional[float] = None):
        """Get a context manager for automatic acquire/release."""
        return self._AcquireContext(self, tokens, timeout)
    
    async def async_wait_for_permit(self, tokens: int = 0, timeout: Optional[float] = None) -> bool:
        """Async version of wait_for_permit."""
        if timeout is None:
            timeout = self.config.max_wait_time
        
        start_time = time.monotonic()
        
        while True:
            can_proceed = True
            max_wait = 0.0
            
            for name in ["rpm", "rpd", "rps"]:
                if name in self._limiters:
                    wait = self._limiters[name].get_wait_time(1)
                    if wait > 0:
                        can_proceed = False
                        max_wait = max(max_wait, wait)
            
            if tokens > 0:
                for name in ["tpm", "tpd"]:
                    if name in self._limiters:
                        wait = self._limiters[name].get_wait_time(tokens)
                        if wait > 0:
                            can_proceed = False
                            max_wait = max(max_wait, wait)
            
            if can_proceed:
                if self.try_acquire(tokens):
                    return True
            
            elapsed = time.monotonic() - start_time
            if timeout and elapsed >= timeout:
                if self.config.action == RateLimitExceededAction.RAISE:
                    raise RateLimitExceededError(
                        "Rate limit timeout exceeded",
                        retry_after=max_wait
                    )
                return False
            
            sleep_time = min(max_wait, 0.1, timeout - elapsed if timeout else 0.1)
            await asyncio.sleep(max(sleep_time, 0.01))
    
    class _AsyncAcquireContext:
        """Async context manager for automatic acquire/release."""
        
        def __init__(self, limiter: 'APIRateLimiter', tokens: int, timeout: Optional[float]):
            self.limiter = limiter
            self.tokens = tokens
            self.timeout = timeout
            self.acquired = False
        
        async def __aenter__(self) -> 'APIRateLimiter._AsyncAcquireContext':
            self.acquired = await self.limiter.async_wait_for_permit(self.tokens, self.timeout)
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
            if self.acquired:
                self.limiter.release_concurrent()
    
    def async_acquire_context(self, tokens: int = 0, timeout: Optional[float] = None):
        """Get an async context manager for automatic acquire/release."""
        return self._AsyncAcquireContext(self, tokens, timeout)


# =============================================================================
# API METRICS
# =============================================================================

@dataclass
class APIMetrics:
    """Metrics for a managed API."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    total_latency_ms: float = 0.0
    requests_by_endpoint: Dict[str, int] = field(default_factory=dict)
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    hourly_requests: deque = field(default_factory=lambda: deque(maxlen=24))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests
    
    def record_request(self, success: bool, latency_ms: float, 
                       tokens: int = 0, endpoint: str = "", 
                       error_type: Optional[str] = None) -> None:
        """Record a request."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.total_latency_ms += latency_ms
        else:
            self.failed_requests += 1
            if error_type:
                self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1
        
        if tokens > 0:
            self.total_tokens_used += tokens
        
        if endpoint:
            self.requests_by_endpoint[endpoint] = self.requests_by_endpoint.get(endpoint, 0) + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "total_tokens_used": self.total_tokens_used,
            "average_latency_ms": self.average_latency_ms,
            "requests_by_endpoint": dict(self.requests_by_endpoint),
            "errors_by_type": dict(self.errors_by_type)
        }


# =============================================================================
# MANAGED API INSTANCE
# =============================================================================

class ManagedAPI:
    """
    A fully managed API instance with rate limiting, circuit breaker, and metrics.
    """
    
    def __init__(self, name: str, config: APIConfig):
        self.name = name
        self.config = config
        
        # Initialize components
        self.rate_limiter: Optional[APIRateLimiter] = None
        if config.rate_limit:
            self.rate_limiter = APIRateLimiter(config.rate_limit)
        
        self.circuit_breaker: Optional[CircuitBreaker] = None
        if config.circuit_breaker:
            self.circuit_breaker = CircuitBreaker(config.circuit_breaker)
        
        self.retry_config = config.retry or RetryConfig(strategy=RetryStrategy.NONE)
        self.metrics = APIMetrics()
        self.status = APIStatus.UNKNOWN
        self.last_health_check: Optional[float] = None
        
        self.lock = threading.RLock()
    
    def can_execute(self) -> bool:
        """Check if a request can be executed."""
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            return False
        return True
    
    def record_success(self, latency_ms: float, tokens: int = 0, endpoint: str = "") -> None:
        """Record a successful request."""
        self.metrics.record_request(True, latency_ms, tokens, endpoint)
        if self.circuit_breaker:
            self.circuit_breaker.record_success()
    
    def record_failure(self, exception: Optional[Exception] = None, 
                       endpoint: str = "") -> None:
        """Record a failed request."""
        error_type = type(exception).__name__ if exception else "Unknown"
        self.metrics.record_request(False, 0, 0, endpoint, error_type)
        if self.circuit_breaker:
            self.circuit_breaker.record_failure(exception)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "name": self.name,
            "status": self.status.value,
            "metrics": self.metrics.to_dict(),
            "config": {
                "base_url": self.config.base_url,
                "timeout": self.config.timeout
            }
        }
        
        if self.rate_limiter:
            stats["rate_limiter"] = self.rate_limiter.get_stats()
        
        if self.circuit_breaker:
            stats["circuit_breaker"] = self.circuit_breaker.get_stats()
        
        return stats


# =============================================================================
# API MANAGER
# =============================================================================

class APIManager:
    """
    Central API management hub.
    
    Manages multiple APIs with:
    - Rate limiting
    - Circuit breakers
    - Retry policies
    - Health monitoring
    - Metrics collection
    
    Example:
        >>> manager = APIManager()
        >>> manager.register_api("openai", APIConfigPresets.openai(api_key="sk-..."))
        >>> 
        >>> # Acquire and execute
        >>> async with manager.acquire("openai", tokens=1000):
        ...     response = await call_openai_api()
    """
    
    def __init__(self):
        self._apis: Dict[str, ManagedAPI] = {}
        self._lock = threading.RLock()
        
        logger.info("APIManager initialized")
    
    def register_api(self, name: str, config: APIConfig) -> ManagedAPI:
        """
        Register a new API for management.
        
        Args:
            name: Unique name for the API
            config: API configuration
        
        Returns:
            The managed API instance
        """
        with self._lock:
            if name in self._apis:
                logger.warning(f"API '{name}' already registered, replacing")
            
            managed = ManagedAPI(name, config)
            self._apis[name] = managed
            logger.info(f"API '{name}' registered")
            return managed
    
    def register_provider(self, provider: str, api_key: Optional[str] = None) -> ManagedAPI:
        """
        Register a known AI provider with preset configuration.
        
        Args:
            provider: Provider name (openai, gemini, anthropic, perplexity, deepseek, ollama)
            api_key: API key for the provider
        
        Returns:
            The managed API instance
        """
        provider_lower = provider.lower()
        
        presets = {
            "openai": lambda: APIConfigPresets.openai(api_key),
            "gemini": lambda: APIConfigPresets.gemini(api_key),
            "google": lambda: APIConfigPresets.gemini(api_key),
            "anthropic": lambda: APIConfigPresets.anthropic(api_key),
            "claude": lambda: APIConfigPresets.anthropic(api_key),
            "perplexity": lambda: APIConfigPresets.perplexity(api_key),
            "deepseek": lambda: APIConfigPresets.deepseek(api_key),
            "ollama": lambda: APIConfigPresets.ollama(),
        }
        
        config_fn = presets.get(provider_lower)
        if config_fn is None:
            # Use generic conservative for unknown providers
            config = APIConfig(
                api_key=api_key,
                rate_limit=AIProviderPresets.generic_conservative()
            )
        else:
            config = config_fn()
        
        return self.register_api(provider_lower, config)
    
    def get_api(self, name: str) -> ManagedAPI:
        """
        Get a registered API by name.
        
        Raises:
            APINotRegisteredError: If API is not registered
        """
        with self._lock:
            if name not in self._apis:
                raise APINotRegisteredError(f"API '{name}' is not registered")
            return self._apis[name]
    
    def unregister_api(self, name: str) -> None:
        """Unregister an API."""
        with self._lock:
            if name in self._apis:
                del self._apis[name]
                logger.info(f"API '{name}' unregistered")
    
    def list_apis(self) -> List[str]:
        """List all registered API names."""
        with self._lock:
            return list(self._apis.keys())
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all registered APIs."""
        with self._lock:
            return {
                name: api.get_stats()
                for name, api in self._apis.items()
            }
    
    def wait_for_permit(self, name: str, tokens: int = 0, 
                        timeout: Optional[float] = None) -> bool:
        """
        Wait for rate limit permit for an API.
        
        Args:
            name: API name
            tokens: Number of tokens for this request
            timeout: Maximum wait time
        
        Returns:
            True if acquired, False otherwise
        """
        api = self.get_api(name)
        
        if not api.can_execute():
            raise CircuitBreakerOpenError(
                f"Circuit breaker is open for API '{name}'",
                api_name=name
            )
        
        if api.rate_limiter:
            return api.rate_limiter.wait_for_permit(tokens, timeout)
        
        return True
    
    def release(self, name: str) -> None:
        """Release concurrent slot for an API."""
        api = self.get_api(name)
        if api.rate_limiter:
            api.rate_limiter.release_concurrent()
    
    class _AcquireContext:
        """Context manager for API access."""
        
        def __init__(self, manager: 'APIManager', name: str, 
                     tokens: int, timeout: Optional[float]):
            self.manager = manager
            self.name = name
            self.tokens = tokens
            self.timeout = timeout
            self.acquired = False
            self.start_time: float = 0.0
        
        def __enter__(self) -> 'APIManager._AcquireContext':
            self.start_time = time.monotonic()
            self.acquired = self.manager.wait_for_permit(
                self.name, self.tokens, self.timeout
            )
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            latency_ms = (time.monotonic() - self.start_time) * 1000
            api = self.manager.get_api(self.name)
            
            if exc_type is None:
                api.record_success(latency_ms, self.tokens)
            else:
                api.record_failure(exc_val)
            
            if self.acquired:
                self.manager.release(self.name)
            
            return False
    
    def acquire(self, name: str, tokens: int = 0, 
                timeout: Optional[float] = None) -> '_AcquireContext':
        """
        Get a context manager for API access.
        
        Example:
            >>> with manager.acquire("openai", tokens=1000):
            ...     response = make_api_call()
        """
        return self._AcquireContext(self, name, tokens, timeout)
    
    class _AsyncAcquireContext:
        """Async context manager for API access."""
        
        def __init__(self, manager: 'APIManager', name: str, 
                     tokens: int, timeout: Optional[float]):
            self.manager = manager
            self.name = name
            self.tokens = tokens
            self.timeout = timeout
            self.acquired = False
            self.start_time: float = 0.0
        
        async def __aenter__(self) -> 'APIManager._AsyncAcquireContext':
            self.start_time = time.monotonic()
            api = self.manager.get_api(self.name)
            
            if not api.can_execute():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is open for API '{self.name}'",
                    api_name=self.name
                )
            
            if api.rate_limiter:
                self.acquired = await api.rate_limiter.async_wait_for_permit(
                    self.tokens, self.timeout
                )
            else:
                self.acquired = True
            
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
            latency_ms = (time.monotonic() - self.start_time) * 1000
            api = self.manager.get_api(self.name)
            
            if exc_type is None:
                api.record_success(latency_ms, self.tokens)
            else:
                api.record_failure(exc_val)
            
            if self.acquired:
                self.manager.release(self.name)
    
    def async_acquire(self, name: str, tokens: int = 0, 
                      timeout: Optional[float] = None) -> '_AsyncAcquireContext':
        """
        Get an async context manager for API access.
        
        Example:
            >>> async with manager.async_acquire("openai", tokens=1000):
            ...     response = await make_async_api_call()
        """
        return self._AsyncAcquireContext(self, name, tokens, timeout)


# =============================================================================
# UTILITY FUNCTIONS (Backward compatible with api_rate_limiter.py)
# =============================================================================

def create_limiter_for_provider(provider: str) -> APIRateLimiter:
    """
    Create a rate limiter with preset configuration for a known provider.
    
    Args:
        provider: Provider name (openai, gemini, claude, perplexity, deepseek, ollama)
    
    Returns:
        Configured APIRateLimiter instance
    """
    provider_lower = provider.lower()
    
    presets = {
        "openai": AIProviderPresets.openai_gpt4_turbo,
        "openai-gpt4": AIProviderPresets.openai_gpt4,
        "openai-gpt4-turbo": AIProviderPresets.openai_gpt4_turbo,
        "openai-gpt35": AIProviderPresets.openai_gpt35_turbo,
        "gemini": AIProviderPresets.google_gemini,
        "gemini-pro": AIProviderPresets.google_gemini_pro,
        "google": AIProviderPresets.google_gemini,
        "claude": AIProviderPresets.anthropic_claude,
        "anthropic": AIProviderPresets.anthropic_claude,
        "perplexity": AIProviderPresets.perplexity,
        "deepseek": AIProviderPresets.deepseek,
        "ollama": AIProviderPresets.ollama_local,
        "grok": AIProviderPresets.generic_conservative,
    }
    
    config_fn = presets.get(provider_lower, AIProviderPresets.generic_conservative)
    return APIRateLimiter(config_fn())


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for a text string.
    
    This is a rough estimation. For accurate counts, use tiktoken.
    """
    chars_per_token = 4
    
    if "claude" in model.lower():
        chars_per_token = 3.5
    elif "gemini" in model.lower():
        chars_per_token = 4
    
    return max(1, int(len(text) / chars_per_token))


# --- Module-level convenience ---

_default_limiter: Optional[APIRateLimiter] = None
_default_manager: Optional[APIManager] = None


def set_default_limiter(limiter: APIRateLimiter) -> None:
    """Set the module-level default rate limiter."""
    global _default_limiter
    _default_limiter = limiter


def get_default_limiter() -> Optional[APIRateLimiter]:
    """Get the module-level default rate limiter."""
    return _default_limiter


def wait_for_rate_limit(tokens: int = 0) -> bool:
    """Wait for the default rate limiter."""
    if _default_limiter is None:
        raise ValueError("No default rate limiter configured. Call set_default_limiter first.")
    return _default_limiter.wait_for_permit(tokens)


def get_default_manager() -> APIManager:
    """Get or create the default API manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = APIManager()
    return _default_manager


def set_default_manager(manager: APIManager) -> None:
    """Set the module-level default API manager."""
    global _default_manager
    _default_manager = manager


# =============================================================================
# DECORATORS
# =============================================================================

T = TypeVar('T')


def with_rate_limit(limiter: Optional[APIRateLimiter] = None, tokens: int = 0):
    """
    Decorator for rate-limited functions.
    
    Example:
        >>> @with_rate_limit(limiter, tokens=500)
        ... def call_api(prompt: str) -> str:
        ...     return api.generate(prompt)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            rate_limiter = limiter or _default_limiter
            if rate_limiter is None:
                raise ValueError("No rate limiter provided and no default configured")
            
            with rate_limiter.acquire_context(tokens=tokens):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator for functions with retry logic.
    
    Example:
        >>> @with_retry(RetryConfig(max_retries=3))
        ... def call_api(prompt: str) -> str:
        ...     return api.generate(prompt)
    """
    retry_config = config or RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(1, retry_config.max_retries + 2):
                try:
                    return func(*args, **kwargs)
                except retry_config.retryable_exceptions as e:
                    last_exception = e
                    if attempt > retry_config.max_retries:
                        break
                    
                    delay = retry_config.get_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
            
            raise RetryExhaustedError(
                f"All {retry_config.max_retries} retries exhausted",
                attempts=retry_config.max_retries,
                last_exception=last_exception
            )
        
        return wrapper
    return decorator


def with_circuit_breaker(breaker: CircuitBreaker):
    """
    Decorator for functions protected by a circuit breaker.
    
    Example:
        >>> cb = CircuitBreaker(CircuitBreakerConfig())
        >>> @with_circuit_breaker(cb)
        ... def call_api(prompt: str) -> str:
        ...     return api.generate(prompt)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not breaker.can_execute():
                raise CircuitBreakerOpenError(
                    "Circuit breaker is open",
                    api_name="decorated_function"
                )
            
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure(e)
                raise
        
        return wrapper
    return decorator


def with_managed_api(manager: Optional[APIManager] = None, api_name: str = "", 
                     tokens: int = 0):
    """
    Decorator for functions using a managed API.
    
    Example:
        >>> @with_managed_api(api_name="openai", tokens=500)
        ... def call_openai(prompt: str) -> str:
        ...     return openai_client.generate(prompt)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            api_manager = manager or _default_manager
            if api_manager is None:
                raise ValueError("No API manager provided and no default configured")
            
            with api_manager.acquire(api_name, tokens=tokens):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator
