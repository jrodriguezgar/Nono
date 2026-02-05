# -*- coding: utf-8 -*-
"""
Connector Module - Unified AI Service Connectors

This module provides:
- connector_genai: Unified interface for multiple AI providers
- api_manager: Comprehensive API management with rate limiting, circuit breakers, and retries
"""

try:
    # Package imports (when used as part of nono package)
    from .connector_genai import (
        SSLVerificationMode,
        configure_ssl_verification,
        ResponseFormat,
        GenerativeAIService,
        OpenAICompatibleService,
        install_library,
        convert_json_schema,
    )
    
    from .genai_batch_processing import GeminiBatchService, OpenAIBatchService

    from .api_manager import (
        # Rate Limiting (backward compatible)
        APIRateLimiter,
        RateLimitConfig,
        RateLimitAlgorithm,
        RateLimitExceededAction,
        RateLimitScope,
        TimeUnit,
        AIProviderPresets,
        RateLimitError,
        RateLimitExceededError,
        RateLimitConfigError,
        create_limiter_for_provider,
        estimate_tokens,
        set_default_limiter,
        get_default_limiter,
        wait_for_rate_limit,
        # New API Manager features
        APIManager,
        APIConfig,
        APIConfigPresets,
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerState,
        CircuitBreakerOpenError,
        RetryConfig,
        RetryStrategy,
        RetryExhaustedError,
        ManagedAPI,
        APIMetrics,
        APIStatus,
        APIManagerError,
        APINotRegisteredError,
        get_default_manager,
        set_default_manager,
        with_rate_limit,
        with_retry,
        with_circuit_breaker,
        with_managed_api,
    )
except ImportError:
    # Direct imports (when used standalone)
    from connector_genai import (
        SSLVerificationMode,
        configure_ssl_verification,
        ResponseFormat,
        GenerativeAIService,
        OpenAICompatibleService,
        install_library,
        convert_json_schema,
    )
    
    from genai_batch_processing import GeminiBatchService, OpenAIBatchService

    from api_manager import (
        # Rate Limiting (backward compatible)
        APIRateLimiter,
        RateLimitConfig,
        RateLimitAlgorithm,
        RateLimitExceededAction,
        RateLimitScope,
        TimeUnit,
        AIProviderPresets,
        RateLimitError,
        RateLimitExceededError,
        RateLimitConfigError,
        create_limiter_for_provider,
        estimate_tokens,
        set_default_limiter,
        get_default_limiter,
        wait_for_rate_limit,
        # New API Manager features
        APIManager,
        APIConfig,
        APIConfigPresets,
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerState,
        CircuitBreakerOpenError,
        RetryConfig,
        RetryStrategy,
        RetryExhaustedError,
        ManagedAPI,
        APIMetrics,
        APIStatus,
        APIManagerError,
        APINotRegisteredError,
        get_default_manager,
        set_default_manager,
        with_rate_limit,
        with_retry,
        with_circuit_breaker,
        with_managed_api,
    )

__all__ = [
    # SSL
    "SSLVerificationMode",
    "configure_ssl_verification",
    # Response
    "ResponseFormat",
    # AI Services
    "GenerativeAIService",
    "OpenAICompatibleService",
    "GeminiBatchService",
    "OpenAIBatchService",
    # Rate Limiting (backward compatible)
    "APIRateLimiter", 
    "RateLimitConfig",
    "RateLimitAlgorithm",
    "RateLimitExceededAction",
    "RateLimitScope",
    "TimeUnit",
    "AIProviderPresets",
    "RateLimitError",
    "RateLimitExceededError",
    "RateLimitConfigError",
    "create_limiter_for_provider",
    "estimate_tokens",
    "set_default_limiter",
    "get_default_limiter",
    "wait_for_rate_limit",
    # API Manager
    "APIManager",
    "APIConfig",
    "APIConfigPresets",
    "ManagedAPI",
    "APIMetrics",
    "APIStatus",
    "APIManagerError",
    "APINotRegisteredError",
    "get_default_manager",
    "set_default_manager",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "CircuitBreakerOpenError",
    # Retry
    "RetryConfig",
    "RetryStrategy",
    "RetryExhaustedError",
    # Decorators
    "with_rate_limit",
    "with_retry",
    "with_circuit_breaker",
    "with_managed_api",
    # Utility
    "install_library",
    "convert_json_schema",
]
