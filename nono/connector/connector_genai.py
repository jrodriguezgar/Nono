"""
Connector Gen AI - Unified Generative AI Services Connector

Provides a unified interface for connecting to multiple generative AI services
including OpenAI, Google Gemini, Perplexity, DeepSeek, Grok (xAI), Groq, Cerebras,
NVIDIA, Microsoft Foundry, Vercel AI SDK, and Ollama.
Features built-in rate limiting (Token Bucket), SSL configuration management,
and auto-installation of required dependencies.

Supported Providers:
    - OpenAIService: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
    - GeminiService: Gemini 1.5 Flash, Gemini 1.5 Pro, Gemini Pro
    - PerplexityService: Sonar, LLaMA-3 variants, Mixtral
    - DeepSeekService: DeepSeek Chat, DeepSeek Coder
    - GrokService: Grok-1 (xAI)
    - GroqService: LLaMA, Qwen, Kimi via Groq's ultra-fast LPU inference
    - CerebrasService: LLaMA models via Cerebras Wafer-Scale Engine
    - NvidiaService: Models via NVIDIA NIM inference platform
    - FoundryService: Models via Microsoft Foundry (GitHub Models)
    - VercelAIService: Provider-agnostic via Vercel AI SDK (OpenAI, Anthropic, Gemini)
    - OllamaService: Any locally hosted model

Example:
    >>> from connector_genai import GeminiService, ResponseFormat
    >>> client = GeminiService(model_name="gemini-3-flash-preview", api_key="...")
    >>> messages = [{"role": "user", "content": "Hello!"}]
    >>> response = client.generate_completion(messages, response_format=ResponseFormat.TEXT)

Author: DatamanEdge
License: MIT
Date: 2026-02-02
Version: 1.2.0
"""

import sys
import subprocess
import ssl
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import csv
import json
import threading
from typing import Optional, Any, Callable
from datetime import datetime, timedelta

# --- SSL Configuration ---

class SSLVerificationMode(Enum):
    """
    Enumeration for SSL verification modes.
    
    INSECURE: Disables SSL verification (for development/testing only)
    CERTIFI: Uses certifi package for certificate validation (recommended for production)
    CUSTOM: Uses a custom certificate file path
    """
    INSECURE = "insecure"
    CERTIFI = "certifi"
    CUSTOM = "custom"


def configure_ssl_verification(mode: SSLVerificationMode = SSLVerificationMode.INSECURE, 
                               custom_cert_path: str | None = None) -> None:
    """
    Configures SSL certificate verification for the application.
    
    Args:
        mode: SSL verification mode (INSECURE, CERTIFI, or CUSTOM)
        custom_cert_path: Path to custom certificate file (required if mode=CUSTOM)
    
    Returns:
        None
    
    Raises:
        ValueError: If CUSTOM mode is selected but no certificate path is provided
        FileNotFoundError: If custom certificate file does not exist
    
    Example Usage:
        # Option 1: Insecure mode (development only)
        configure_ssl_verification(SSLVerificationMode.INSECURE)
        
        # Option 2: Use certifi package (recommended for production)
        configure_ssl_verification(SSLVerificationMode.CERTIFI)
        
        # Option 3: Use custom corporate certificate
        configure_ssl_verification(SSLVerificationMode.CUSTOM, 
                                  custom_cert_path='C:/path/to/corporate-cert.crt')
    
    Cost: O(1)
    """
    import os
    import warnings
    
    if mode == SSLVerificationMode.INSECURE:
        # Option 1: Disable SSL verification (INSECURE - use only for development/testing)
        print("⚠️  SSL verification DISABLED - This is insecure and should only be used for development!")
        
        # Disable SSL verification globally
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Suppress SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Set environment variables for various libraries
        os.environ['PYTHONHTTPSVERIFY'] = '0'
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        
    elif mode == SSLVerificationMode.CERTIFI:
        # Option 2: Use certifi package (RECOMMENDED for production)
        try:
            import certifi
            cert_path = certifi.where()
            print(f"✓ SSL verification enabled using certifi: {cert_path}")
            
            # Set environment variables to use certifi certificates
            os.environ['REQUESTS_CA_BUNDLE'] = cert_path
            os.environ['SSL_CERT_FILE'] = cert_path
            os.environ['CURL_CA_BUNDLE'] = cert_path
            
        except ImportError:
            print("⚠️  certifi package not found. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "certifi"])
                import certifi
                cert_path = certifi.where()
                os.environ['REQUESTS_CA_BUNDLE'] = cert_path
                os.environ['SSL_CERT_FILE'] = cert_path
                os.environ['CURL_CA_BUNDLE'] = cert_path
                print(f"✓ certifi installed and configured: {cert_path}")
            except Exception as e:
                print(f"❌ Error installing certifi: {e}")
                print("⚠️  Falling back to system certificates")
                
    elif mode == SSLVerificationMode.CUSTOM:
        # Option 3: Use custom certificate file
        if not custom_cert_path:
            raise ValueError("custom_cert_path must be provided when using CUSTOM mode")
        
        if not os.path.exists(custom_cert_path):
            raise FileNotFoundError(f"Certificate file not found: {custom_cert_path}")
        
        print(f"✓ SSL verification enabled using custom certificate: {custom_cert_path}")
        
        # Set environment variables to use custom certificate
        os.environ['REQUESTS_CA_BUNDLE'] = custom_cert_path
        os.environ['SSL_CERT_FILE'] = custom_cert_path
        os.environ['CURL_CA_BUNDLE'] = custom_cert_path
    
    else:
        raise ValueError(f"Invalid SSL verification mode: {mode}")


# Configure SSL by default (INSECURE mode for development)
# To change the mode, call configure_ssl_verification() with the desired mode before using any AI services
configure_ssl_verification(SSLVerificationMode.INSECURE)


# --- Dependency Management Utility ---

def install_library(library_name: str, import_name: str | None = None, package_name: str | None = None) -> bool:
    """
    Checks if a library is installed and, if not, attempts to install it via pip.
    
    Args:
        library_name: The name to use for import check (e.g., 'google.genai')
        import_name: Deprecated, use library_name instead
        package_name: The pip package name to install (e.g., 'google-genai'). 
                      If not specified, uses library_name for pip install.
    """
    try:
        check_name = import_name if import_name else library_name
        __import__(check_name)
        return True
    except ImportError:
        pip_name = package_name if package_name else library_name
        print(f"Library '{library_name}' not found. Attempting to install '{pip_name}'...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"Library '{pip_name}' installed successfully.")
            return True
        except subprocess.CalledProcessError:
            print(f"Error: Failed to install '{pip_name}'.")
            return False

# Ensure required libraries are installed before importing
for lib in ["json", "urllib3", "requests"]:
    install_library(lib)

# --- Initial Setup ---
import requests
import urllib3
if install_library("requests"):
    HTTP_SESSION = requests.Session()
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
else:
    raise ImportError("The 'requests' library is required and could not be installed.")

# Disable SSL verification (for development only)
import ssl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context # Create SSL context without verification

# Configure SSL verification
import os
import certifi
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

# --- API Manager ---
# Import the professional API management module
try:
    from .api_manager import (
        APIRateLimiter,
        RateLimitConfig,
        AIProviderPresets,
        RateLimitAlgorithm,
        RateLimitExceededAction,
        RateLimitExceededError,
        create_limiter_for_provider,
        estimate_tokens,
        APIManager,
        APIConfig,
        APIConfigPresets,
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerState,
        RetryConfig,
        RetryStrategy,
        ManagedAPI,
        APIMetrics
    )
except ImportError:
    from api_manager import (
        APIRateLimiter,
        RateLimitConfig,
        AIProviderPresets,
        RateLimitAlgorithm,
        RateLimitExceededAction,
        RateLimitExceededError,
        create_limiter_for_provider,
        estimate_tokens,
        APIManager,
        APIConfig,
        APIConfigPresets,
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerState,
        RetryConfig,
        RetryStrategy,
        ManagedAPI,
        APIMetrics
    )

# --- Model Features (prompt_size) ---

# Cache for model features loaded from CSV
_MODEL_FEATURES_CACHE: dict[str, int] | None = None


def _load_model_features() -> dict[str, int]:
    """
    Loads model features from model_features.csv.
    
    Returns:
        Dictionary with key "provider/model" or "model" and value prompt_size.
    """
    global _MODEL_FEATURES_CACHE
    if _MODEL_FEATURES_CACHE is not None:
        return _MODEL_FEATURES_CACHE
    
    csv_path = os.path.join(os.path.dirname(__file__), "model_features.csv")
    features: dict[str, int] = {}
    
    if not os.path.exists(csv_path):
        _MODEL_FEATURES_CACHE = features
        return features
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            provider = row.get('provider', '').strip().lower()
            model = row.get('model', '').strip()
            prompt_size_str = row.get('prompt_size', '').strip()
            
            if model and prompt_size_str:
                try:
                    prompt_size = int(prompt_size_str)
                    # Store both with and without provider prefix
                    if provider:
                        features[f"{provider}/{model}"] = prompt_size
                    features[model] = prompt_size
                except ValueError:
                    pass
    
    _MODEL_FEATURES_CACHE = features
    return features


def get_prompt_size(provider: str, model: str, default: int = 500_000) -> int:
    """
    Gets the maximum prompt size for a model from model_features.csv.
    
    Args:
        provider: Provider name (openrouter, google, openai, etc.)
        model: Model name
        default: Default value if model is not found
    
    Returns:
        Maximum prompt size in characters
    
    Example:
        >>> get_prompt_size('openrouter', 'openrouter/auto')
        2000000
        >>> get_prompt_size('google', 'gemini-3-flash-preview')
        1048576
    """
    features = _load_model_features()
    provider_lower = provider.lower()
    
    # Try with full path first: provider/model
    key = f"{provider_lower}/{model}"
    if key in features:
        return features[key]
    
    # Try just the model name
    if model in features:
        return features[model]
    
    return default


# --- Rate Limits (from CSV) ---

# Cache for rate limits loaded from CSV
_RATE_LIMITS_CACHE: dict[str, dict[str, int | None]] | None = None


def _parse_limit(value: str | None) -> int | None:
    """
    Parses a limit value from string to int.
    
    Supports values like '7000', '7K', '500K', '-', '' or None.
    
    Args:
        value: String value from CSV ('-', '', number, or number with K suffix)
    
    Returns:
        int or None if no limit
    
    Example:
        >>> _parse_limit('30')
        30
        >>> _parse_limit('7K')
        7000
        >>> _parse_limit('-')
        None
    """
    if not value or value.strip() in ('-', ''):
        return None
    try:
        cleaned = value.strip().upper()
        if cleaned.endswith('K'):
            return int(float(cleaned[:-1]) * 1000)
        return int(cleaned)
    except ValueError:
        return None


def _load_rate_limits() -> dict[str, dict[str, int | None]]:
    """
    Loads rate limits from model_rate_limits.csv.
    
    Returns:
        Dictionary with key "provider/model" and value dict with rpm, rpd, tpm, tpd.
    """
    global _RATE_LIMITS_CACHE
    if _RATE_LIMITS_CACHE is not None:
        return _RATE_LIMITS_CACHE
    
    csv_path = os.path.join(os.path.dirname(__file__), "model_rate_limits.csv")
    limits: dict[str, dict[str, int | None]] = {}
    
    if not os.path.exists(csv_path):
        _RATE_LIMITS_CACHE = limits
        return limits
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            provider = row.get('provider', '').strip().lower()
            model = row.get('model', '').strip()
            
            if model:
                limit_data = {
                    'rpm': _parse_limit(row.get('rpm')),
                    'rpd': _parse_limit(row.get('rpd')),
                    'tpm': _parse_limit(row.get('tpm')),
                    'tpd': _parse_limit(row.get('tpd'))
                }
                # Store with provider prefix
                if provider:
                    limits[f"{provider}/{model}"] = limit_data
                limits[model] = limit_data
    
    _RATE_LIMITS_CACHE = limits
    return limits


def get_rate_limit_config(provider: str, model: str) -> RateLimitConfig | None:
    """
    Gets rate limit configuration for a model from model_rate_limits.csv.
    
    Args:
        provider: Provider name (groq, openrouter, google, etc.)
        model: Model name
    
    Returns:
        RateLimitConfig with model limits, or None if not found.
    
    Example:
        >>> config = get_rate_limit_config('groq', 'llama-3.3-70b-versatile')
        >>> config.rpm
        30
        >>> config.tpm
        12000
    """
    limits = _load_rate_limits()
    provider_lower = provider.lower()
    
    # Try with full path first: provider/model
    key = f"{provider_lower}/{model}"
    limit_data = limits.get(key) or limits.get(model)
    
    if not limit_data:
        return None
    
    # Only create config if we have at least one limit
    if not any(limit_data.values()):
        return None
    
    return RateLimitConfig(
        rpm=limit_data.get('rpm'),
        rpd=limit_data.get('rpd'),
        tpm=limit_data.get('tpm'),
        tpd=limit_data.get('tpd')
    )


# --- JSON Schema Conversion ---

def convert_json_schema(input_schema: dict, output_title: str = "perplexity") -> dict:
    """Converts a simplified JSON schema to a more detailed one."""
    if "properties" not in input_schema:
        raise ValueError("Input schema must contain a 'properties' key.")

    output_schema = {
        "properties": {},
        "required": input_schema.get("required", []),
        "type": input_schema.get("type", "object"),
        "title": output_title,
    }

    for property_name, property_details in input_schema["properties"].items():
        updated_property_details = property_details.copy()
        updated_property_details["title"] = property_name
        output_schema["properties"][property_name] = updated_property_details

    return output_schema

# --- Response Format Enum ---

class ResponseFormat(Enum):
    TEXT = "text"
    TABLE = "table"
    XML = "xml"
    JSON = "json"
    CSV = "csv"


# --- Rate Limit Status ---

@dataclass
class RateLimitStatus:
    """
    Represents the current rate limit status for an API service.
    
    Attributes:
        configured_rpm: Configured requests per minute limit
        configured_rpd: Configured requests per day limit
        configured_tpm: Configured tokens per minute limit
        configured_tpd: Configured tokens per day limit
        remaining_requests: Remaining requests in current window
        remaining_tokens: Remaining tokens in current window
        reset_time: Time until limits reset (seconds)
        credits_remaining: Remaining credits (for paid APIs)
        credits_used: Credits used in current period
        is_free_tier: Whether using free tier
        rate_limit_tier: Rate limit tier name (if applicable)
        usage_percentage: Percentage of limit used (0-100)
        last_updated: Timestamp of last status update
        raw_response: Raw response from API (for debugging)
    """
    # Configured limits
    configured_rpm: int | None = None
    configured_rpd: int | None = None
    configured_tpm: int | None = None
    configured_tpd: int | None = None
    
    # Remaining capacity
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    reset_time: float | None = None
    
    # Credits/billing info
    credits_remaining: float | None = None
    credits_used: float | None = None
    is_free_tier: bool = False
    rate_limit_tier: str | None = None
    
    # Usage metrics
    usage_percentage: float | None = None
    last_updated: datetime | None = None
    
    # Debug
    raw_response: dict | None = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
        
        # Calculate usage percentage if we have the data
        if self.usage_percentage is None and self.configured_rpm and self.remaining_requests is not None:
            used = self.configured_rpm - self.remaining_requests
            self.usage_percentage = (used / self.configured_rpm) * 100
    
    @property
    def is_rate_limited(self) -> bool:
        """Returns True if currently rate limited (no remaining capacity)."""
        if self.remaining_requests is not None and self.remaining_requests <= 0:
            return True
        if self.remaining_tokens is not None and self.remaining_tokens <= 0:
            return True
        return False
    
    @property
    def has_credits(self) -> bool:
        """Returns True if there are credits remaining (or if free tier)."""
        if self.is_free_tier:
            return True
        if self.credits_remaining is not None:
            return self.credits_remaining > 0
        return True  # Assume has credits if unknown
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "configured": {
                "rpm": self.configured_rpm,
                "rpd": self.configured_rpd,
                "tpm": self.configured_tpm,
                "tpd": self.configured_tpd,
            },
            "remaining": {
                "requests": self.remaining_requests,
                "tokens": self.remaining_tokens,
                "reset_time_seconds": self.reset_time,
            },
            "credits": {
                "remaining": self.credits_remaining,
                "used": self.credits_used,
                "is_free_tier": self.is_free_tier,
            },
            "tier": self.rate_limit_tier,
            "usage_percentage": self.usage_percentage,
            "is_rate_limited": self.is_rate_limited,
            "has_credits": self.has_credits,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }
    
    def __repr__(self) -> str:
        parts = []
        if self.configured_rpm:
            remaining = f" ({self.remaining_requests} left)" if self.remaining_requests is not None else ""
            parts.append(f"RPM={self.configured_rpm}{remaining}")
        if self.credits_remaining is not None:
            parts.append(f"Credits=${self.credits_remaining:.2f}")
        if self.rate_limit_tier:
            parts.append(f"Tier={self.rate_limit_tier}")
        if self.is_rate_limited:
            parts.append("⚠️ RATE LIMITED")
        return f"RateLimitStatus({', '.join(parts)})"


# --- Base Class ---

class GenerativeAIService(ABC):
    """
    Abstract base class for interacting with generative AI services with rate limiting.
    
    Provides properties to inspect the current configuration:
        - model_name: Name of the model being used
        - api_key: API key (masked for security)
        - max_input_chars: Maximum input characters allowed
        - provider: Name of the AI provider
        - rate_limit: Rate limit configuration and stats (delegated to api_manager)
        - config: Dictionary with all configuration parameters
    """
    # Class attribute for subclasses to define their model input limits
    _MAX_INPUT_CHARS: dict[str, int] = {}
    
    TEMPERATURE_RECOMMENDATIONS = {
        "coding": 0.0,           # Maximum precision, deterministic code
        "math": 0.0,             # Exact mathematical answers
        "data_cleaning": 0.1,    # High precision for data transformations
        "data_analysis": 0.3,    # Consistency in analysis, some flexibility
        "translation": 0.3,      # Precise and faithful translations
        "conversation": 0.7,     # Balance between coherence and naturalness
        "creative": 1.0,         # Higher variability for creative content
        "poetry": 1.2,           # High creativity for artistic expression
        "default": 0.7           # Balanced default value for general use
    }

    @classmethod
    def get_recommended_temperature(cls, use_case: str) -> float:
        return cls.TEMPERATURE_RECOMMENDATIONS.get(use_case.lower(), 0.7)

    @classmethod
    def _resolve_temperature(cls, temperature: float | str) -> float:
        """
        Resolves temperature from a numeric value or string use case name.
        
        Args:
            temperature: Numeric value (0.0-2.0) or use case name
                        (coding, math, data_cleaning, etc.)
        
        Returns:
            Temperature value as float
        
        Example:
            >>> GenerativeAIService._resolve_temperature(0.5)
            0.5
            >>> GenerativeAIService._resolve_temperature("coding")
            0.0
            >>> GenerativeAIService._resolve_temperature("data_cleaning")
            0.1
        """
        if isinstance(temperature, str):
            return cls.TEMPERATURE_RECOMMENDATIONS.get(temperature.lower(), 0.7)
        return temperature

    @classmethod
    def get_max_input_chars(cls, model_name: str) -> int:
        """Returns the maximum input characters for a given model."""
        if hasattr(cls, "_MAX_INPUT_CHARS"):
            # Return lookup or a reasonable default (matches __init__ defaults generally)
            return cls._MAX_INPUT_CHARS.get(model_name, 120_000) 
        return 120_000

    def __init__(self, model_name: str, max_input_chars: int,
                 api_key: Optional[str] = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        if not model_name:
            raise ValueError("A model_name must be provided.")

        self._model_name = model_name
        self._api_key = api_key
        self._max_input_chars = max_input_chars
        
        # Use provided config or create default (1 RPS)
        self._rate_limit_config = rate_limit_config or RateLimitConfig(rps=1.0)
        
        self.rate_limiter = APIRateLimiter(self._rate_limit_config)

    # --- Configuration Properties ---
    
    @property
    def model_name(self) -> str:
        """Returns the name of the model being used."""
        return self._model_name
    
    @property
    def api_key(self) -> Optional[str]:
        """Returns the API key (for internal use)."""
        return self._api_key
    
    @property
    def api_key_masked(self) -> str:
        """Returns the API key masked for security (shows first 4 and last 4 characters)."""
        if not self._api_key:
            return "Not set"
        if len(self._api_key) <= 8:
            return "****"
        return f"{self._api_key[:4]}...{self._api_key[-4:]}"
    
    @property
    def max_input_chars(self) -> int:
        """Returns the maximum number of input characters allowed for the model."""
        return self._max_input_chars
    
    @property
    def rate_limit(self) -> dict:
        """
        Returns the rate limit configuration and current stats.
        
        Delegates to the underlying APIRateLimiter from api_manager.
        
        Returns:
            Dictionary with rate limit configuration and statistics:
            - config: Current rate limit configuration (rpm, rps, tpm, etc.)
            - stats: Current limiter statistics from api_manager
            - summary: Human-readable summary of limits
        
        Example:
            >>> client.rate_limit
            {
                'config': {'rps': 0.25, 'algorithm': 'token_bucket', ...},
                'stats': {'limiters': {...}, ...},
                'summary': 'RPS=0.25 (15.0 req/min)'
            }
        """
        config = self._rate_limit_config
        stats = self.rate_limiter.get_stats()
        
        # Build configuration dict
        config_dict = {
            "algorithm": config.algorithm.value,
            "action": config.action.value,
            "max_wait_time": config.max_wait_time,
        }
        
        # Add configured limits
        if config.rps is not None:
            config_dict["rps"] = config.rps
        if config.rpm is not None:
            config_dict["rpm"] = config.rpm
        if config.rpd is not None:
            config_dict["rpd"] = config.rpd
        if config.tpm is not None:
            config_dict["tpm"] = config.tpm
        if config.tpd is not None:
            config_dict["tpd"] = config.tpd
        if config.concurrent_limit is not None:
            config_dict["concurrent_limit"] = config.concurrent_limit
        if config.burst_size is not None:
            config_dict["burst_size"] = config.burst_size
        
        # Build human-readable summary
        summary_parts = []
        if config.rps:
            summary_parts.append(f"RPS={config.rps} ({config.rps * 60:.1f} req/min)")
        if config.rpm:
            summary_parts.append(f"RPM={config.rpm}")
        if config.rpd:
            summary_parts.append(f"RPD={config.rpd}")
        if config.tpm:
            summary_parts.append(f"TPM={config.tpm}")
        if config.concurrent_limit:
            summary_parts.append(f"Concurrent={config.concurrent_limit}")
        
        return {
            "config": config_dict,
            "stats": stats,
            "summary": ", ".join(summary_parts) or "No limits configured"
        }
    
    @property
    def rate_limit_config(self) -> RateLimitConfig:
        """Returns the underlying RateLimitConfig object for advanced usage."""
        return self._rate_limit_config
    
    @property
    def provider(self) -> str:
        """Returns the name of the AI provider (derived from class name)."""
        class_name = self.__class__.__name__
        return class_name.replace("Service", "")
    
    @property
    def config(self) -> dict:
        """Returns a dictionary with all configuration parameters."""
        return self.get_config()
    
    def get_config(self, include_api_key: bool = False, include_rate_stats: bool = False) -> dict:
        """
        Returns a dictionary with the current configuration.
        
        Args:
            include_api_key: If True, includes the full API key. 
                           If False (default), includes masked version.
            include_rate_stats: If True, includes full rate limiter statistics.
                              If False (default), includes only summary.
        
        Returns:
            Dictionary with configuration parameters.
        
        Example:
            >>> client = GeminiService(model_name="gemini-3-flash-preview", api_key="...")
            >>> print(client.get_config())
            {
                'provider': 'Gemini',
                'model_name': 'gemini-3-flash-preview',
                'max_input_chars': 4000000,
                'rate_limit': 'RPS=0.25 (15.0 req/min)',
                'api_key': 'abc1...xyz9'
            }
        """
        rate_info = self.rate_limit
        
        config_dict = {
            "provider": self.provider,
            "model_name": self.model_name,
            "max_input_chars": self.max_input_chars,
            "rate_limit": rate_info if include_rate_stats else rate_info["summary"],
        }
        
        if include_api_key:
            config_dict["api_key"] = self._api_key
        else:
            config_dict["api_key"] = self.api_key_masked
            
        return config_dict
    
    def get_rate_limit_status(self) -> RateLimitStatus:
        """
        Returns the current rate limit status for the service.
        
        This base implementation returns status based on local rate limiter stats.
        Subclasses can override this to query the provider's API for real-time limits.
        
        Returns:
            RateLimitStatus object with configured and remaining limits
        
        Example:
            >>> status = client.get_rate_limit_status()
            >>> print(status.is_rate_limited)
            False
            >>> print(status.remaining_requests)
            45
        """
        config = self._rate_limit_config
        stats = self.rate_limiter.get_stats()
        
        # Extract remaining capacity from limiter stats
        remaining_requests = None
        remaining_tokens = None
        reset_time = None
        
        if "limiters" in stats:
            if "rpm" in stats["limiters"]:
                rpm_stats = stats["limiters"]["rpm"]
                if "current_tokens" in rpm_stats:
                    remaining_requests = int(rpm_stats["current_tokens"])
                elif "current_count" in rpm_stats and config.rpm:
                    remaining_requests = config.rpm - rpm_stats["current_count"]
                if "time_remaining" in rpm_stats:
                    reset_time = rpm_stats["time_remaining"]
            
            if "tpm" in stats["limiters"]:
                tpm_stats = stats["limiters"]["tpm"]
                if "current_tokens" in tpm_stats:
                    remaining_tokens = int(tpm_stats["current_tokens"])
                elif "current_count" in tpm_stats and config.tpm:
                    remaining_tokens = config.tpm - tpm_stats["current_count"]
        
        return RateLimitStatus(
            configured_rpm=config.rpm,
            configured_rpd=config.rpd,
            configured_tpm=config.tpm,
            configured_tpd=config.tpd,
            remaining_requests=remaining_requests,
            remaining_tokens=remaining_tokens,
            reset_time=reset_time,
            rate_limit_tier=self.provider,
        )
    
    def __repr__(self) -> str:
        """Returns a string representation of the service configuration."""
        return (f"{self.__class__.__name__}("
                f"model='{self.model_name}', "
                f"max_chars={self.max_input_chars:,}, "
                f"rate_limit='{self.rate_limit['summary']}')")
    
    def __str__(self) -> str:
        """Returns a human-readable string describing the service."""
        return (f"{self.provider} Service using model '{self.model_name}' "
                f"(max {self.max_input_chars:,} chars, {self.rate_limit['summary']})")

    def _validate_messages_length(self, messages: list[dict[str, str]]) -> None:
        total_length = sum(len(m.get("content", "")) for m in messages)
        if total_length > self.max_input_chars:
            raise ValueError(
                f"Total message length too long for model '{self.model_name}'. "
                f"Maximum allowed characters: {self.max_input_chars}, "
                f"but messages have {total_length} characters."
            )

    def _format_messages_for_response(self, messages: list[dict[str, str]],
                                   response_format: ResponseFormat,
                                   json_schema: dict | None = None) -> list[dict[str, str]]:
        formatted_messages = [msg.copy() for msg in messages]

        instruction = ""
        if response_format == ResponseFormat.TABLE:
            instruction = "\n\nPlease provide the output in a markdown table format."
        elif response_format == ResponseFormat.CSV:
            instruction = "\n\nPlease provide the output in CSV format. Ensure the first line is the header."
        elif response_format == ResponseFormat.XML:
            instruction = "\n\nPlease provide the output in XML format. Ensure the response is valid XML."
        elif response_format == ResponseFormat.JSON:
            if json_schema:
                try:
                    schema_str = json.dumps(json_schema, indent=2)
                    instruction = (f"\n\nPlease provide the output in JSON format, "
                                 f"adhering to the following schema:\n```json\n{schema_str}\n```\n"
                                 f"Ensure the response is valid JSON.")
                except TypeError as e:
                    print(f"Warning: Could not serialize JSON schema. Error: {e}")
                    instruction = "\n\nPlease provide the output in JSON format. Ensure the response is valid JSON."
            else:
                instruction = "\n\nPlease provide the output in JSON format. Ensure the response is valid JSON."

        if instruction:
             # Try to append to system message first if available
             system_message_index = -1
             for i, msg in enumerate(formatted_messages):
                 if msg.get("role") == "system":
                     system_message_index = i
                     break
             
             if system_message_index != -1:
                 formatted_messages[system_message_index]["content"] += instruction
             else:
                 # Fallback: append to the last user message
                 last_user_message_index = -1
                 for i, msg in enumerate(reversed(formatted_messages)):
                    if msg.get("role") == "user":
                        last_user_message_index = len(formatted_messages) - 1 - i
                        break

                 if last_user_message_index != -1:
                     formatted_messages[last_user_message_index]["content"] += instruction
                 else:
                     # Fallback 2: If no system and no user message found (rare), create a user message logic or raise.
                     # But original code raised if no user message found.
                     # Let's keep original behavior if specific mapping fails or just ensure user message approach is safe.
                     raise ValueError("No 'user' message found in the input messages to append formatting instructions to.")

        return formatted_messages

    @abstractmethod
    def generate_completion(self, messages: list[dict[str, str]], 
                          temperature: float | str = 0.7,
                          max_tokens: int | None = None,
                          top_p: float | None = None,
                          frequency_penalty: float | None = None,
                          presence_penalty: float | None = None,
                          stop: list[str] | None = None,
                          response_format: ResponseFormat = ResponseFormat.JSON,
                          json_schema: dict | None = None,
                          use_case: str | None = None,
                          **kwargs) -> str:
        pass

# --- OpenAI Compatible Services ---

class OpenAICompatibleService(GenerativeAIService):
    """Base class for OpenAI-compatible API services."""
    
    def __init__(self, model_name: str, api_key: str | None, base_url: str,
                 max_input_chars: int,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        super().__init__(model_name, max_input_chars, api_key, rate_limit_config)
        self._base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    @property
    def base_url(self) -> str:
        """Returns the base URL for the API endpoint."""
        return self._base_url
    
    def get_config(self, include_api_key: bool = False, include_rate_stats: bool = False) -> dict:
        """Returns extended configuration including base_url."""
        config_dict = super().get_config(include_api_key, include_rate_stats)
        config_dict["base_url"] = self.base_url
        return config_dict

    def generate_completion(self, messages: list[dict[str, str]], 
                          temperature: float | str = 0.7,
                          max_tokens: int | None = None,
                          top_p: float | None = None,
                          frequency_penalty: float | None = None,
                          presence_penalty: float | None = None,
                          stop: list[str] | None = None,
                          response_format: ResponseFormat = ResponseFormat.JSON,
                          json_schema: dict | None = None,
                          use_case: str | None = None,
                          **kwargs) -> str:
        temperature = self._resolve_temperature(temperature)
        formatted_messages = self._format_messages_for_response(messages, response_format, json_schema)
        
        # Ensure a system message exists (default behavior)
        if not any(msg.get("role") == "system" for msg in formatted_messages):
             formatted_messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
        
        self._validate_messages_length(formatted_messages)
        self.rate_limiter.wait_for_permit()

        endpoint = f"{self._base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": temperature
        }

        # Add optional parameters if provided
        if max_tokens is not None: payload["max_tokens"] = max_tokens
        if top_p is not None: payload["top_p"] = top_p
        if frequency_penalty is not None: payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None: payload["presence_penalty"] = presence_penalty
        if stop is not None: payload["stop"] = stop
        
        # Include any extra supported kwargs
        for key in kwargs:
             payload[key] = kwargs[key]

        if response_format == ResponseFormat.JSON:
            payload["response_format"] = {"type": "json_object"}

        # OpenAI strictly enforces 'system', 'user', 'assistant' roles
        # Ensure potential 'model' role (from Gemini/internal) is mapped to 'assistant'
        # Also ensure 'system' maps to 'system' (already standard)
        for msg in payload["messages"]:
            if msg.get("role") == "model":
                msg["role"] = "assistant"
            # Some providers like Grok might be strict about role names if different standard used? 
            # Usually they follow OpenAI, so 'system' is correct.

        try:
            response = HTTP_SESSION.post(endpoint, headers=self.headers, json=payload, timeout=90, verify=False)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to {self._base_url}: {e}")
            raise
        except (KeyError, IndexError) as e:
            print(f"Unexpected API response format: {e}")
            raise

# --- Concrete Implementations ---

class OpenAIService(OpenAICompatibleService):
    _PROVIDER_NAME = "openai"

    def __init__(self, model_name: str, api_key: str | None = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        # Resolve API key if not provided
        if api_key is None:
            api_key = resolve_api_key_for_provider(self._PROVIDER_NAME)
        
        super().__init__(
            model_name,
            api_key,
            "https://api.openai.com/v1",
            get_prompt_size(self._PROVIDER_NAME, model_name),
            rate_limit_config
        )

class PerplexityService(OpenAICompatibleService):
    _PROVIDER_NAME = "perplexity"

    def __init__(self, model_name: str, api_key: str | None = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        # Resolve API key if not provided
        if api_key is None:
            api_key = resolve_api_key_for_provider(self._PROVIDER_NAME)
        
        super().__init__(
            model_name,
            api_key,
            "https://api.perplexity.ai",
            get_prompt_size(self._PROVIDER_NAME, model_name),
            rate_limit_config
        )

    def generate_completion(self, messages: list[dict[str, str]], 
                          temperature: float | str = 0.7,
                          max_tokens: int | None = None,
                          top_p: float | None = None,
                          frequency_penalty: float | None = None,
                          presence_penalty: float | None = None,
                          stop: list[str] | None = None,
                          response_format: ResponseFormat = ResponseFormat.JSON,
                          json_schema: dict | None = None,
                          use_case: str | None = None,
                          **kwargs) -> str:
        temperature = self._resolve_temperature(temperature)
        if response_format == ResponseFormat.JSON and json_schema:
            processed_messages = [msg.copy() for msg in messages]
        else:
            processed_messages = self._format_messages_for_response(messages, response_format, json_schema)

        if not any(msg.get("role") == "system" for msg in processed_messages):
            processed_messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})

        self._validate_messages_length(processed_messages)
        self.rate_limiter.wait_for_permit()

        endpoint = f"{self._base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": temperature
        }

        # Add optional parameters specific to Perplexity/OpenAI standards
        if max_tokens is not None: payload["max_tokens"] = max_tokens
        if top_p is not None: payload["top_p"] = top_p
        if frequency_penalty is not None: payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None: payload["presence_penalty"] = presence_penalty
        
        # Perplexity specific parameters (pass via kwargs for advanced use)
        # return_citations, search_domain_filter, return_images, return_related_questions
        for key in kwargs:
             payload[key] = kwargs[key]

        if response_format == ResponseFormat.JSON:
            if json_schema:
                json_schema = convert_json_schema(json_schema)
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"schema": json_schema}
                }
            else:
                payload["response_format"] = {"type": "json_object"}

        try:
            response = HTTP_SESSION.post(endpoint, headers=self.headers, json=payload, timeout=90, verify=False)
            response.raise_for_status()
            data = response.json()
            content = data['choices'][0]['message']['content'].strip()
            if content.startswith("```json") and content.endswith("```"):
                return content[len("```json"): -len("```")].strip()
            return content
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to {endpoint}: {e}")
            raise
        except (KeyError, IndexError) as e:
            print(f"Unexpected API response format from Perplexity: {e}")
            raise

class DeepSeekService(OpenAICompatibleService):
    _PROVIDER_NAME = "deepseek"

    def __init__(self, model_name: str, api_key: str | None = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        # Resolve API key if not provided
        if api_key is None:
            api_key = resolve_api_key_for_provider(self._PROVIDER_NAME)
        
        super().__init__(
            model_name,
            api_key,
            "https://api.deepseek.com",
            get_prompt_size(self._PROVIDER_NAME, model_name),
            rate_limit_config
        )

class GrokService(OpenAICompatibleService):
    _PROVIDER_NAME = "grok"

    def __init__(self, model_name: str, api_key: str | None = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        # Resolve API key if not provided
        if api_key is None:
            api_key = resolve_api_key_for_provider(self._PROVIDER_NAME)
        
        super().__init__(
            model_name,
            api_key,
            "https://api.x.ai/v1",
            get_prompt_size(self._PROVIDER_NAME, model_name),
            rate_limit_config
        )


class GroqService(OpenAICompatibleService):
    """
    Groq AI service - Ultra-fast inference using Groq's LPU technology.
    
    Groq provides extremely fast inference for open-source models like LLaMA, Mixtral, etc.
    Uses OpenAI-compatible API endpoint.
    
    Supported models (as of 2026):
        - llama-3.3-70b-versatile: LLaMA 3.3 70B (general purpose)
        - llama-3.1-8b-instant: LLaMA 3.1 8B (fast, lightweight)
        - meta-llama/llama-4-maverick-17b-128e-instruct: LLaMA 4 Maverick
        - meta-llama/llama-4-scout-17b-16e-instruct: LLaMA 4 Scout
        - qwen/qwen3-32b: Qwen 3 32B
        - moonshotai/kimi-k2-instruct: Kimi K2
        - groq/compound: Groq Compound (multi-model)
        - groq/compound-mini: Groq Compound Mini
    
    Rate Limits (free tier):
        - Most models: 30 RPM, 6K-30K TPM
        - See model_rate_limits.csv for detailed limits per model
    """
    _PROVIDER_NAME = "groq"

    def __init__(self, model_name: str, 
                 api_key: str | None = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        """
        Initialize Groq service.
        
        Args:
            model_name: Model to use
            api_key: Groq API key (get from console.groq.com)
            rate_limit_config: Rate limit configuration (loaded from CSV if None)
        """
        # Resolve API key if not provided
        if api_key is None:
            api_key = resolve_api_key_for_provider(self._PROVIDER_NAME)
        
        # Load rate limit from CSV if not provided
        if rate_limit_config is None:
            rate_limit_config = get_rate_limit_config(self._PROVIDER_NAME, model_name)
        
        super().__init__(
            model_name,
            api_key,
            "https://api.groq.com/openai/v1",
            get_prompt_size(self._PROVIDER_NAME, model_name),
            rate_limit_config
        )


class NvidiaService(OpenAICompatibleService):
    """
    NVIDIA AI service - Access to NVIDIA NIM models via OpenAI-compatible API.
    
    NVIDIA provides inference for a wide variety of models through their
    NIM (NVIDIA Inference Microservice) platform.
    
    Supported models (as of 2026):
        - qwen/qwq-32b: Qwen QWQ 32B
        - meta/llama-3.3-70b-instruct: LLaMA 3.3 70B Instruct
        - deepseek-ai/deepseek-r1: DeepSeek R1
        - nvidia/llama-3.1-nemotron-70b-instruct: Nemotron 70B
    
    API Documentation: https://build.nvidia.com
    
    Usage:
        >>> from connector_genai import NvidiaService
        >>> client = NvidiaService(model_name="qwen/qwq-32b", api_key="nvapi-...")
        >>> response = client.generate_completion(
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """
    _PROVIDER_NAME = "nvidia"

    def __init__(self, model_name: str, 
                 api_key: str | None = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        """
        Initialize NVIDIA NIM service.
        
        Args:
            model_name: Model to use (e.g., 'qwen/qwq-32b')
            api_key: NVIDIA API key (get from build.nvidia.com)
            rate_limit_config: Rate limit configuration (loaded from CSV if None)
        """
        # Resolve API key if not provided
        if api_key is None:
            api_key = resolve_api_key_for_provider(self._PROVIDER_NAME)
        
        # Load rate limit from CSV if not provided
        if rate_limit_config is None:
            rate_limit_config = get_rate_limit_config(self._PROVIDER_NAME, model_name)
        
        super().__init__(
            model_name,
            api_key,
            "https://integrate.api.nvidia.com/v1",
            get_prompt_size(self._PROVIDER_NAME, model_name),
            rate_limit_config
        )


class CerebrasService(GenerativeAIService):
    """
    Cerebras AI service - Ultra-fast inference using Cerebras Wafer-Scale Engine.
    
    Cerebras provides extremely fast inference for open-source models using their
    custom Wafer-Scale Engine hardware. Uses the native cerebras-cloud-sdk.
    
    Supported models (as of 2026):
        - gpt-oss-120b: GPT-OSS 120B (65K context)
        - llama-3.3-70b: LLaMA 3.3 70B (65K context)
        - llama3.1-8b: LLaMA 3.1 8B (8K context, lightweight)
        - qwen-3-32b: Qwen 3 32B (65K context)
        - zai-glm-4.7: ZAI-GLM 4.7 (64K context, preview)
    
    Rate Limits (free tier):
        - Most models: 30 RPM, 14.4K RPD, 64K TPM, 1M TPD
        - zai-glm-4.7 (preview): 10 RPM, 100 RPD, 60K TPM, 500K TPD
        - See model_rate_limits.csv for detailed limits per model
    
    API Documentation: https://inference-docs.cerebras.ai
    
    Usage:
        >>> from connector_genai import CerebrasService
        >>> client = CerebrasService(model_name="llama-3.3-70b", api_key="csk-...")
        >>> response = client.generate_completion(
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """
    _PROVIDER_NAME = "cerebras"

    def __init__(self, model_name: str, 
                 api_key: str | None = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        """
        Initialize Cerebras service.
        
        Args:
            model_name: Model to use (e.g., 'llama-3.3-70b')
            api_key: Cerebras API key (get from cloud.cerebras.ai)
            rate_limit_config: Rate limit configuration (loaded from CSV if None)
        """
        # Resolve API key if not provided
        if api_key is None:
            api_key = resolve_api_key_for_provider(self._PROVIDER_NAME)
        
        # Load rate limit from CSV if not provided
        if rate_limit_config is None:
            rate_limit_config = get_rate_limit_config(self._PROVIDER_NAME, model_name)
        
        super().__init__(
            model_name,
            get_prompt_size(self._PROVIDER_NAME, model_name),
            api_key,
            rate_limit_config
        )

        # Lazy: SDK client created on first generate_completion call
        self.client = None

    def _ensure_client(self) -> None:
        """Lazy initialization of the Cerebras SDK client."""
        if self.client is not None:
            return
        
        if not install_library("cerebras.cloud.sdk", package_name="cerebras-cloud-sdk"):
            raise ImportError(
                "The 'cerebras-cloud-sdk' library is required for CerebrasService "
                "and could not be installed."
            )

        from cerebras.cloud.sdk import Cerebras
        self.client = Cerebras(api_key=self._api_key)

    def generate_completion(self, messages: list[dict[str, str]], 
                          temperature: float | str = 0.7,
                          max_tokens: int | None = None,
                          top_p: float | None = None,
                          frequency_penalty: float | None = None,
                          presence_penalty: float | None = None,
                          stop: list[str] | None = None,
                          response_format: ResponseFormat = ResponseFormat.JSON,
                          json_schema: dict | None = None,
                          use_case: str | None = None,
                          **kwargs) -> str:
        """
        Generate a completion using the Cerebras SDK.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature value or use case name (e.g., 'coding', 'creative')
            max_tokens: Maximum number of completion tokens
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty (not supported by Cerebras, ignored)
            presence_penalty: Presence penalty (not supported by Cerebras, ignored)
            stop: Stop sequences
            response_format: ResponseFormat.JSON or ResponseFormat.TEXT
            json_schema: JSON schema for structured output
            use_case: Use case name for automatic temperature selection
        
        Returns:
            Generated text response
        """
        self._ensure_client()
        temperature = self._resolve_temperature(temperature)
        
        # Format messages for response format
        if response_format == ResponseFormat.JSON and json_schema:
            processed_messages = [msg.copy() for msg in messages]
        else:
            processed_messages = self._format_messages_for_response(messages, response_format, json_schema)
        
        # Ensure system message exists
        if not any(msg.get("role") == "system" for msg in processed_messages):
            processed_messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
        
        self._validate_messages_length(processed_messages)
        self.rate_limiter.wait_for_permit()

        # Build SDK call parameters
        sdk_params: dict[str, Any] = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": temperature,
            "stream": False,
        }
        
        if max_tokens is not None:
            sdk_params["max_completion_tokens"] = max_tokens
        if top_p is not None:
            sdk_params["top_p"] = top_p
        if stop is not None:
            sdk_params["stop"] = stop
        
        # Handle JSON response format
        if response_format == ResponseFormat.JSON:
            if json_schema:
                json_schema = convert_json_schema(json_schema)
                sdk_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"schema": json_schema}
                }
            else:
                sdk_params["response_format"] = {"type": "json_object"}

        try:
            completion = self.client.chat.completions.create(**sdk_params)
            content = completion.choices[0].message.content or ""
            content = content.strip()
            if content.startswith("```json") and content.endswith("```"):
                return content[len("```json"): -len("```")].strip()
            return content
        except Exception as e:
            print(f"Error with Cerebras API: {e}")
            raise


class OpenRouterService(OpenAICompatibleService):
    """
    OpenRouter AI service - Unified access to 300+ AI models from multiple providers.
    
    OpenRouter provides a single API endpoint to access models from OpenAI, Anthropic,
    Google, Meta, Mistral, and many other providers. It uses OpenAI-compatible API format.
    
    Features:
        - Access to 300+ models through one API
        - Automatic fallback between providers
        - Cost optimization and routing
        - Provider preferences and filtering
    
    API Documentation: https://openrouter.ai/docs
    
    Example models:
        - openai/gpt-4o, openai/gpt-4o-mini
        - anthropic/claude-3.5-sonnet, anthropic/claude-3-opus
        - google/gemini-2.0-flash, google/gemini-pro
        - meta-llama/llama-3.3-70b-instruct
        - mistralai/mistral-large
        - deepseek/deepseek-chat
    
    Usage:
        >>> from connector_genai import OpenRouterService
        >>> 
        >>> client = OpenRouterService(
        ...     model_name="anthropic/claude-3.5-sonnet",
        ...     api_key="sk-or-...",
        ...     app_name="MyApp",
        ...     app_url="https://myapp.com"
        ... )
        >>> response = client.generate_completion(
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """
    _PROVIDER_NAME = "openrouter"

    def __init__(self, 
                 model_name: str, 
                 api_key: str | None = None,
                 app_name: str | None = None,
                 app_url: str | None = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        """
        Initialize OpenRouter service.
        
        Args:
            model_name: Model to use in format "provider/model" (e.g., "anthropic/claude-3.5-sonnet")
            api_key: OpenRouter API key (get from https://openrouter.ai/keys)
            app_name: Your application name (shown in OpenRouter dashboard)
            app_url: Your application URL (used for rankings and tracking)
            rate_limit_config: Rate limit configuration (loaded from CSV if None)
        """
        self._app_name = app_name
        self._app_url = app_url
        
        # Resolve API key if not provided
        if api_key is None:
            api_key = resolve_api_key_for_provider(self._PROVIDER_NAME)
        
        # Load rate limit from CSV if not provided
        if rate_limit_config is None:
            rate_limit_config = get_rate_limit_config(self._PROVIDER_NAME, model_name)
        
        super().__init__(
            model_name,
            api_key,
            "https://openrouter.ai/api/v1",
            get_prompt_size(self._PROVIDER_NAME, model_name),
            rate_limit_config
        )
        
        # Add OpenRouter-specific headers
        if app_url:
            self.headers["HTTP-Referer"] = app_url
        if app_name:
            self.headers["X-Title"] = app_name
    
    @property
    def app_name(self) -> str | None:
        """Returns the application name."""
        return self._app_name
    
    @property
    def app_url(self) -> str | None:
        """Returns the application URL."""
        return self._app_url
    
    def get_config(self, include_api_key: bool = False, include_rate_stats: bool = False) -> dict:
        """Returns extended configuration including app info."""
        config_dict = super().get_config(include_api_key, include_rate_stats)
        config_dict["app_name"] = self.app_name
        config_dict["app_url"] = self.app_url
        return config_dict
    
    def generate_completion(self, messages: list[dict[str, str]], 
                          temperature: float | str = 0.7,
                          max_tokens: int | None = None,
                          top_p: float | None = None,
                          frequency_penalty: float | None = None,
                          presence_penalty: float | None = None,
                          stop: str | list[str] | None = None,
                          response_format: ResponseFormat = ResponseFormat.JSON,
                          json_schema: dict | None = None,
                          use_case: str | None = None,
                          # OpenRouter-specific parameters
                          provider_preferences: dict | None = None,
                          top_k: int | None = None,
                          repetition_penalty: float | None = None,
                          min_p: float | None = None,
                          top_a: float | None = None,
                          seed: int | None = None,
                          logit_bias: dict[int, float] | None = None,
                          top_logprobs: int | None = None,
                          tools: list[dict] | None = None,
                          tool_choice: str | dict | None = None,
                          plugins: list[dict] | None = None,
                          prediction: dict | None = None,
                          user: str | None = None,
                          **kwargs) -> str:
        """
        Generate completion using OpenRouter API.
        
        Args:
            messages: List of messages in OpenAI format
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter (0, 1]
            frequency_penalty: Frequency penalty (-2 to 2)
            presence_penalty: Presence penalty (-2 to 2)
            stop: Stop sequences (string or list)
            response_format: Response format (TEXT, JSON, etc.)
            json_schema: JSON schema for structured output
            use_case: Optional use case identifier
            provider_preferences: OpenRouter provider routing preferences
            top_k: Top-k sampling (not available for OpenAI models)
            repetition_penalty: Repetition penalty (0, 2]
            min_p: Min-p sampling [0, 1]
            top_a: Top-a sampling [0, 1]
            seed: Random seed for reproducibility
            logit_bias: Token logit bias dictionary
            top_logprobs: Number of top logprobs to return
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy ('none', 'auto', or specific)
            plugins: OpenRouter plugins (web, file-parser, response-healing)
            prediction: Predicted output for latency optimization
            user: Stable identifier for end-user (abuse prevention)
            **kwargs: Additional OpenRouter-specific parameters:
                - transforms: List of prompt transforms
                - route: Routing strategy ("fallback")
                - models: Fallback model list
                - debug: Debug options (streaming only)
        
        Returns:
            Generated text response
        """
        temperature = self._resolve_temperature(temperature)
        formatted_messages = self._format_messages_for_response(messages, response_format, json_schema)
        
        if not any(msg.get("role") == "system" for msg in formatted_messages):
            formatted_messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
        
        self._validate_messages_length(formatted_messages)
        self.rate_limiter.wait_for_permit()

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": formatted_messages,
            "temperature": temperature,
        }

        # Standard OpenAI parameters
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if stop is not None:
            payload["stop"] = stop

        # Advanced sampling parameters
        if top_k is not None:
            payload["top_k"] = top_k
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        if min_p is not None:
            payload["min_p"] = min_p
        if top_a is not None:
            payload["top_a"] = top_a
        if seed is not None:
            payload["seed"] = seed
        if logit_bias is not None:
            payload["logit_bias"] = logit_bias
        if top_logprobs is not None:
            payload["top_logprobs"] = top_logprobs

        # Tool calling
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        # Response format
        if response_format == ResponseFormat.JSON:
            if json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": json_schema.get("title", "response"),
                        "strict": True,
                        "schema": json_schema
                    }
                }
            else:
                payload["response_format"] = {"type": "json_object"}
        
        # OpenRouter-specific parameters
        if provider_preferences is not None:
            payload["provider"] = provider_preferences
        if plugins is not None:
            payload["plugins"] = plugins
        if prediction is not None:
            payload["prediction"] = prediction
        if user is not None:
            payload["user"] = user
        
        # Additional OpenRouter parameters from kwargs
        for key in ["transforms", "route", "models", "debug"]:
            if key in kwargs:
                payload[key] = kwargs[key]

        try:
            response = HTTP_SESSION.post(
                f"{self._base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=120,
                verify=False
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_body = e.response.json()
                if "error" in error_body:
                    error_detail = f": {error_body['error'].get('message', str(error_body['error']))}"
            except (ValueError, KeyError):
                pass
            print(f"OpenRouter API error{error_detail}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"OpenRouter request failed: {e}")
            raise
    
    @staticmethod
    def get_model_providers() -> list[str]:
        """Returns a list of supported providers on OpenRouter."""
        return [
            "openai", "anthropic", "google", "meta-llama", "mistralai",
            "deepseek", "qwen", "cohere", "perplexity", "groq", 
            "together", "fireworks-ai", "01-ai", "nvidia"
        ]
    
    def get_openai_client(self) -> Any:
        """
        Returns an OpenAI client configured for OpenRouter.
        
        This allows using the OpenAI SDK directly with OpenRouter's API.
        Useful for advanced features like streaming, async operations, etc.
        
        Returns:
            OpenAI client instance configured for OpenRouter
            
        Example:
            service = OpenRouterService(api_key="sk-or-...")
            client = service.get_openai_client()
            
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://myapp.com",
                    "X-Title": "My App"
                },
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello!"}]
            )
            print(completion.choices[0].message.content)
        """
        try:
            from openai import OpenAI
        except ImportError:
            install_library("openai")
            from openai import OpenAI
        
        return OpenAI(
            base_url=self._base_url,
            api_key=self._api_key
        )
    
    @property
    def extra_headers(self) -> dict[str, str]:
        """
        Returns extra headers for use with OpenAI client.
        
        Use these with the `extra_headers` parameter when calling
        the OpenAI client methods.
        
        Example:
            client = service.get_openai_client()
            completion = client.chat.completions.create(
                extra_headers=service.extra_headers,
                model="anthropic/claude-3.5-sonnet",
                messages=[...]
            )
        """
        headers = {}
        if self._app_url:
            headers["HTTP-Referer"] = self._app_url
        if self._app_name:
            headers["X-Title"] = self._app_name
        return headers
    
    def fetch_api_key_info(self) -> dict[str, Any]:
        """
        Fetches API key information from OpenRouter including rate limits and credits.
        
        Makes a GET request to https://openrouter.ai/api/v1/key to retrieve:
        - Rate limit information
        - Credits remaining
        - Usage statistics
        
        Returns:
            Dictionary with API key information from OpenRouter
            
        Raises:
            requests.exceptions.RequestException: If the API call fails
            
        Example:
            >>> service = OpenRouterService(api_key="sk-or-...")
            >>> info = service.fetch_api_key_info()
            >>> print(info)
            {
                "data": {
                    "label": "My API Key",
                    "usage": 1.50,
                    "limit": 100.0,
                    "is_free_tier": False,
                    "rate_limit": {
                        "requests": 200,
                        "interval": "10s"
                    }
                }
            }
        """
        try:
            response = HTTP_SESSION.get(
                f"{self._base_url.replace('/v1', '')}/api/v1/key",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=30,
                verify=False
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch OpenRouter API key info: {e}")
            raise
    
    def get_rate_limit_status(self) -> RateLimitStatus:
        """
        Returns the current rate limit status by querying OpenRouter's API.
        
        This method fetches real-time rate limit and credit information
        from OpenRouter's /api/v1/key endpoint.
        
        Returns:
            RateLimitStatus object with configured limits, remaining capacity,
            and credit information from OpenRouter
            
        Example:
            >>> service = OpenRouterService(api_key="sk-or-...")
            >>> status = service.get_rate_limit_status()
            >>> print(status)
            RateLimitStatus(RPM=200, Credits=$98.50, Tier=explorer)
            >>> print(status.is_rate_limited)
            False
            >>> print(status.credits_remaining)
            98.5
        """
        # First get local limiter status
        base_status = super().get_rate_limit_status()
        
        # Try to fetch real-time info from OpenRouter
        try:
            api_info = self.fetch_api_key_info()
            data = api_info.get("data", {})
            
            # Parse rate limit info
            rate_limit_info = data.get("rate_limit", {})
            requests_limit = rate_limit_info.get("requests")
            interval = rate_limit_info.get("interval", "60s")
            
            # Convert interval to RPM
            configured_rpm = None
            if requests_limit:
                # Parse interval like "10s", "60s", "1m"
                if interval.endswith("s"):
                    seconds = int(interval[:-1])
                    configured_rpm = int(requests_limit * (60 / seconds))
                elif interval.endswith("m"):
                    minutes = int(interval[:-1])
                    configured_rpm = int(requests_limit / minutes)
                else:
                    configured_rpm = requests_limit
            
            # Extract credit/usage info
            credits_used = data.get("usage", 0.0)
            credits_limit = data.get("limit")
            credits_remaining = None
            if credits_limit is not None and credits_limit > 0:
                credits_remaining = credits_limit - credits_used
            elif credits_limit == 0 or credits_limit is None:
                # Unlimited or pay-as-you-go
                credits_remaining = None
            
            is_free_tier = data.get("is_free_tier", False)
            
            # Determine tier based on rate limit
            tier = "free" if is_free_tier else "paid"
            if configured_rpm and configured_rpm >= 500:
                tier = "pro"
            elif configured_rpm and configured_rpm >= 200:
                tier = "explorer"
            
            return RateLimitStatus(
                configured_rpm=configured_rpm or base_status.configured_rpm,
                configured_rpd=base_status.configured_rpd,
                configured_tpm=base_status.configured_tpm,
                configured_tpd=base_status.configured_tpd,
                remaining_requests=base_status.remaining_requests,
                remaining_tokens=base_status.remaining_tokens,
                reset_time=base_status.reset_time,
                credits_remaining=credits_remaining,
                credits_used=credits_used,
                is_free_tier=is_free_tier,
                rate_limit_tier=tier,
                raw_response=api_info,
            )
        except Exception as e:
            # If API call fails, return base status with error info
            print(f"Warning: Could not fetch OpenRouter API info: {e}")
            return base_status


class GeminiService(GenerativeAIService):
    """
    Google Gemini AI service using the new google-genai SDK.
    
    This service uses the modern google-genai package (google.genai) which replaces
    the deprecated google-generativeai package.
    """
    _PROVIDER_NAME = "google"
    _DEFAULT_RPM = 15  # 15 requests/min for most Gemini models

    def __init__(self, model_name: str, api_key: str | None = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        # Resolve API key if not provided
        if api_key is None:
            api_key = resolve_api_key_for_provider(self._PROVIDER_NAME)
        
        # Default rate limit: 15 RPM for Gemini
        if rate_limit_config is None:
            rate_limit_config = RateLimitConfig(rpm=self._DEFAULT_RPM)
        
        super().__init__(
            model_name,
            get_prompt_size(self._PROVIDER_NAME, model_name),
            api_key,
            rate_limit_config
        )

        if not install_library("google.genai", package_name="google-genai"):
            raise ImportError(
                "The 'google-genai' library is required for GeminiService "
                "and could not be installed."
            )

        from google import genai
        from google.genai import types
        self.genai = genai
        self.types = types
        self.client = genai.Client(api_key=api_key)

    def generate_completion(self, messages: list[dict[str, str]], 
                          temperature: float | str = 0.7,
                          max_tokens: int | None = None,
                          top_p: float | None = None,
                          frequency_penalty: float | None = None,
                          presence_penalty: float | None = None,
                          stop: list[str] | None = None,
                          response_format: ResponseFormat = ResponseFormat.JSON,
                          json_schema: dict | None = None,
                          use_case: str | None = None,
                          **kwargs) -> str:
        temperature = self._resolve_temperature(temperature)
        # Build GenerateContentConfig
        config_params: dict[str, Any] = {"temperature": temperature}

        # Map standard parameters to Gemini config
        if max_tokens is not None: 
            config_params["max_output_tokens"] = max_tokens
        if top_p is not None: 
            config_params["top_p"] = top_p
        if stop is not None: 
            config_params["stop_sequences"] = stop
        
        # Handle top_k which is specific to Gemini
        if "top_k" in kwargs: 
            config_params["top_k"] = kwargs["top_k"]

        # Extract system instruction from messages
        processed_messages_for_api = []
        system_instruction_content = ""

        for msg in messages:
            if msg.get("role") == "system":
                system_instruction_content += msg.get("content", "") + "\n"
            else:
                processed_messages_for_api.append(msg.copy())

        # Set system instruction
        if system_instruction_content:
            config_params["system_instruction"] = system_instruction_content.strip()
        else:
            config_params["system_instruction"] = "You are a helpful assistant."

        # Handle response format
        if response_format == ResponseFormat.JSON and json_schema:
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = json_schema
        elif response_format == ResponseFormat.JSON:
            config_params["response_mime_type"] = "application/json"
            processed_messages_for_api = self._format_messages_for_response(
                processed_messages_for_api, response_format, json_schema
            )
        else:
            processed_messages_for_api = self._format_messages_for_response(
                processed_messages_for_api, response_format, json_schema
            )

        # Validate message length BEFORE converting to Content objects
        self._validate_messages_length(processed_messages_for_api)

        # Build contents in the format expected by google-genai
        # The SDK accepts a list of Content objects or simpler formats
        gemini_contents = []
        for msg in processed_messages_for_api:
            role = msg.get("role")
            content = msg.get("content")
            if role and content:
                # Skip system messages (already handled via system_instruction)
                if role == "system": 
                    continue
                
                # Map roles: 'assistant' -> 'model', 'user' -> 'user'
                mapped_role = "user" if role == "user" else "model"
                gemini_contents.append(
                    self.types.Content(
                        role=mapped_role,
                        parts=[self.types.Part.from_text(text=content)]
                    )
                )

        self.rate_limiter.wait_for_permit()

        try:
            # Create config object
            config = self.types.GenerateContentConfig(**config_params)
            
            # Call the new API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=gemini_contents,
                config=config
            )
            return response.text or ""
        except Exception as e:
            print(f"An error occurred with the Gemini API: {e}")
            raise


class FoundryService(GenerativeAIService):
    """
    Microsoft Foundry Models service - Access to GitHub-hosted AI models.
    
    Provides access to models hosted on Microsoft Foundry (GitHub Models)
    using the azure-ai-inference SDK.
    
    Supported models (as of 2026):
        - openai/gpt-5: GPT-5
        - openai/gpt-4o: GPT-4o
        - meta-llama/llama-3.3-70b-instruct: LLaMA 3.3 70B
        - deepseek/deepseek-r1: DeepSeek R1
        - mistralai/mistral-large-2411: Mistral Large
    
    Rate Limits:
        - See model_rate_limits.csv for detailed limits per model
    
    API Documentation: https://docs.github.com/en/github-models
    
    Usage:
        >>> from connector_genai import FoundryService
        >>> client = FoundryService(model_name="openai/gpt-5", api_key="ghp_...")
        >>> response = client.generate_completion(
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """
    _PROVIDER_NAME = "foundry"
    _DEFAULT_ENDPOINT = "https://models.github.ai/inference"

    def __init__(self, model_name: str,
                 api_key: str | None = None,
                 endpoint: str | None = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        """
        Initialize Microsoft Foundry Models service.
        
        Args:
            model_name: Model to use (e.g., 'openai/gpt-5')
            api_key: GitHub token or Azure key (uses GITHUB_TOKEN env var if None)
            endpoint: API endpoint (defaults to https://models.github.ai/inference)
            rate_limit_config: Rate limit configuration (loaded from CSV if None)
        """
        # Resolve API key: try provider key, then GITHUB_TOKEN env var
        if api_key is None:
            try:
                api_key = resolve_api_key_for_provider(self._PROVIDER_NAME)
            except (ValueError, KeyError):
                import os
                api_key = os.environ.get("GITHUB_TOKEN")
                if not api_key:
                    raise ValueError(
                        "No API key found for Foundry. Set GITHUB_TOKEN env var "
                        "or add 'foundry' entry to apikeys.csv."
                    )
        
        self._endpoint = endpoint or self._DEFAULT_ENDPOINT
        
        # Load rate limit from CSV if not provided
        if rate_limit_config is None:
            rate_limit_config = get_rate_limit_config(self._PROVIDER_NAME, model_name)
        
        super().__init__(
            model_name,
            get_prompt_size(self._PROVIDER_NAME, model_name),
            api_key,
            rate_limit_config
        )

        # Lazy: SDK client created on first generate_completion call
        self.client = None

    def _ensure_client(self) -> None:
        """Lazy initialization of the Azure AI Inference client."""
        if self.client is not None:
            return
        
        if not install_library("azure.ai.inference", package_name="azure-ai-inference"):
            raise ImportError(
                "The 'azure-ai-inference' library is required for FoundryService "
                "and could not be installed."
            )
        
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential
        
        self.client = ChatCompletionsClient(
            endpoint=self._endpoint,
            credential=AzureKeyCredential(self._api_key),
        )

    @property
    def endpoint(self) -> str:
        """Returns the API endpoint URL."""
        return self._endpoint

    def get_config(self, include_api_key: bool = False, include_rate_stats: bool = False) -> dict:
        """Returns extended configuration including endpoint."""
        config_dict = super().get_config(include_api_key, include_rate_stats)
        config_dict["endpoint"] = self.endpoint
        return config_dict

    def generate_completion(self, messages: list[dict[str, str]], 
                          temperature: float | str = 0.7,
                          max_tokens: int | None = None,
                          top_p: float | None = None,
                          frequency_penalty: float | None = None,
                          presence_penalty: float | None = None,
                          stop: list[str] | None = None,
                          response_format: ResponseFormat = ResponseFormat.JSON,
                          json_schema: dict | None = None,
                          use_case: str | None = None,
                          **kwargs) -> str:
        """
        Generate a completion using the Azure AI Inference SDK.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature value or use case name
            max_tokens: Maximum number of completion tokens
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop: Stop sequences
            response_format: ResponseFormat.JSON or ResponseFormat.TEXT
            json_schema: JSON schema for structured output
            use_case: Use case name for automatic temperature selection
        
        Returns:
            Generated text response
        """
        self._ensure_client()
        temperature = self._resolve_temperature(temperature)
        
        from azure.ai.inference.models import (
            SystemMessage, UserMessage, AssistantMessage
        )
        
        # Format messages for response format
        if response_format == ResponseFormat.JSON and json_schema:
            processed_messages = [msg.copy() for msg in messages]
        else:
            processed_messages = self._format_messages_for_response(
                messages, response_format, json_schema
            )
        
        # Ensure system message exists
        if not any(msg.get("role") == "system" for msg in processed_messages):
            processed_messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
        
        self._validate_messages_length(processed_messages)
        self.rate_limiter.wait_for_permit()

        # Convert to Azure AI Inference message objects
        sdk_messages = []
        for msg in processed_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                sdk_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                sdk_messages.append(AssistantMessage(content=content))
            else:
                sdk_messages.append(UserMessage(content=content))

        # Build SDK call parameters
        sdk_params: dict[str, Any] = {
            "messages": sdk_messages,
            "model": self.model_name,
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            sdk_params["max_tokens"] = max_tokens
        if top_p is not None:
            sdk_params["top_p"] = top_p
        if frequency_penalty is not None:
            sdk_params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            sdk_params["presence_penalty"] = presence_penalty
        if stop is not None:
            sdk_params["stop"] = stop
        
        # Handle JSON response format
        if response_format == ResponseFormat.JSON:
            if json_schema:
                json_schema = convert_json_schema(json_schema)
                sdk_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"schema": json_schema}
                }
            else:
                sdk_params["response_format"] = {"type": "json_object"}

        try:
            response = self.client.complete(**sdk_params)
            content = response.choices[0].message.content or ""
            content = content.strip()
            if content.startswith("```json") and content.endswith("```"):
                return content[len("```json"): -len("```")].strip()
            return content
        except Exception as e:
            print(f"Error with Foundry Models API: {e}")
            raise


class VercelAIService(GenerativeAIService):
    """
    Vercel AI SDK service - Provider-agnostic AI interface.
    
    Wraps the Python port of Vercel's AI SDK (ai-sdk-python) providing
    zero-configuration, provider-agnostic text and structured output generation.
    Supports OpenAI, Anthropic, and Gemini providers through a unified API.
    
    Supported providers:
        - openai: GPT-4o, GPT-4o-mini, GPT-5, etc.
        - anthropic: Claude 3.5 Sonnet, Claude 3 Opus, etc.
        - gemini: Gemini 1.5 Flash, Gemini 1.5 Pro, etc.
    
    Features:
        - Provider-agnostic: swap providers without changing code
        - Pydantic-based structured output via generate_object
        - Streaming support via stream_text
        - Built-in tool calling
    
    API Documentation: https://pythonaisdk.mintlify.app
    
    Usage:
        >>> from connector_genai import VercelAIService
        >>> client = VercelAIService(
        ...     model_name="gpt-4o-mini",
        ...     sdk_provider="openai",
        ...     api_key="sk-..."
        ... )
        >>> response = client.generate_completion(
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """
    _PROVIDER_NAME = "vercel"
    _SUPPORTED_SDK_PROVIDERS = ("openai", "anthropic", "gemini")

    def __init__(self, model_name: str,
                 sdk_provider: str = "openai",
                 api_key: str | None = None,
                 rate_limit_config: Optional[RateLimitConfig] = None,
                 **model_kwargs: Any):
        """
        Initialize Vercel AI SDK service.
        
        Args:
            model_name: Model to use (e.g., 'gpt-4o-mini', 'claude-3-sonnet-20240229')
            sdk_provider: SDK provider name ('openai', 'anthropic', 'gemini')
            api_key: API key for the chosen provider
            rate_limit_config: Rate limit configuration
            **model_kwargs: Extra kwargs passed to the SDK provider factory
                           (e.g., temperature, top_p as defaults)
        """
        if sdk_provider not in self._SUPPORTED_SDK_PROVIDERS:
            raise ValueError(
                f"Unsupported SDK provider '{sdk_provider}'. "
                f"Choose from: {', '.join(self._SUPPORTED_SDK_PROVIDERS)}"
            )
        
        self._sdk_provider_name = sdk_provider
        self._model_kwargs = model_kwargs
        
        # Resolve API key if not provided
        if api_key is None:
            try:
                api_key = resolve_api_key_for_provider(sdk_provider)
            except (ValueError, KeyError):
                pass  # SDK will use env vars (OPENAI_API_KEY, etc.)
        
        super().__init__(
            model_name,
            get_prompt_size(self._PROVIDER_NAME, model_name),
            api_key,
            rate_limit_config
        )

        # Lazy: SDK model created on first generate_completion call
        self._sdk_model = None

    def _ensure_model(self) -> None:
        """Lazy initialization of the Vercel AI SDK model."""
        if self._sdk_model is not None:
            return
        
        if not install_library("ai_sdk", package_name="ai-sdk-python"):
            raise ImportError(
                "The 'ai-sdk-python' library is required for VercelAIService "
                "and could not be installed."
            )
        
        import ai_sdk
        
        factory_kwargs: dict[str, Any] = {}
        if self._api_key:
            factory_kwargs["api_key"] = self._api_key
        factory_kwargs.update(self._model_kwargs)
        
        if self._sdk_provider_name == "openai":
            self._sdk_model = ai_sdk.openai(self.model_name, **factory_kwargs)
        elif self._sdk_provider_name == "anthropic":
            self._sdk_model = ai_sdk.anthropic(self.model_name, **factory_kwargs)
        elif self._sdk_provider_name == "gemini":
            self._sdk_model = ai_sdk.gemini(self.model_name, **factory_kwargs)

    @property
    def sdk_provider(self) -> str:
        """Returns the SDK provider name."""
        return self._sdk_provider_name

    def get_config(self, include_api_key: bool = False, include_rate_stats: bool = False) -> dict:
        """Returns extended configuration including SDK provider."""
        config_dict = super().get_config(include_api_key, include_rate_stats)
        config_dict["sdk_provider"] = self.sdk_provider
        return config_dict

    def generate_completion(self, messages: list[dict[str, str]], 
                          temperature: float | str = 0.7,
                          max_tokens: int | None = None,
                          top_p: float | None = None,
                          frequency_penalty: float | None = None,
                          presence_penalty: float | None = None,
                          stop: list[str] | None = None,
                          response_format: ResponseFormat = ResponseFormat.JSON,
                          json_schema: dict | None = None,
                          use_case: str | None = None,
                          **kwargs) -> str:
        """
        Generate a completion using the Vercel AI SDK.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature value or use case name
            max_tokens: Maximum number of completion tokens
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop: Stop sequences
            response_format: ResponseFormat.JSON or ResponseFormat.TEXT
            json_schema: JSON schema for structured output
            use_case: Use case name for automatic temperature selection
        
        Returns:
            Generated text response
        """
        self._ensure_model()
        temperature = self._resolve_temperature(temperature)
        
        import ai_sdk
        from ai_sdk.types import CoreSystemMessage, CoreUserMessage, CoreAssistantMessage
        
        # Format messages for response format
        if response_format == ResponseFormat.JSON and json_schema:
            processed_messages = [msg.copy() for msg in messages]
        else:
            processed_messages = self._format_messages_for_response(
                messages, response_format, json_schema
            )
        
        # Ensure system message exists
        if not any(msg.get("role") == "system" for msg in processed_messages):
            processed_messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
        
        self._validate_messages_length(processed_messages)
        self.rate_limiter.wait_for_permit()

        # Convert to Vercel AI SDK message objects
        sdk_messages = []
        system_prompt = None
        for msg in processed_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "assistant":
                sdk_messages.append(CoreAssistantMessage(content=content))
            else:
                sdk_messages.append(CoreUserMessage(content=content))

        # Build SDK call parameters
        sdk_params: dict[str, Any] = {
            "model": self._sdk_model,
            "temperature": temperature,
        }
        
        if system_prompt:
            sdk_params["system"] = system_prompt
        if sdk_messages:
            sdk_params["messages"] = sdk_messages
        if max_tokens is not None:
            sdk_params["max_tokens"] = max_tokens
        if top_p is not None:
            sdk_params["top_p"] = top_p
        if frequency_penalty is not None:
            sdk_params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            sdk_params["presence_penalty"] = presence_penalty
        if stop is not None:
            sdk_params["stop"] = stop
        
        # Handle JSON response format
        if response_format == ResponseFormat.JSON:
            if json_schema:
                sdk_params["response_format"] = {"type": "json_object"}

        try:
            result = ai_sdk.generate_text(**sdk_params)
            content = result.text or ""
            content = content.strip()
            if content.startswith("```json") and content.endswith("```"):
                return content[len("```json"): -len("```")].strip()
            return content
        except Exception as e:
            print(f"Error with Vercel AI SDK ({self._sdk_provider_name}): {e}")
            raise


class OllamaService(GenerativeAIService):
    """Ollama local AI service for running models locally."""
    _DEFAULT_MAX_CHARS = 200_000

    def __init__(self, model_name: str, host: str = "http://localhost:11434",
                 rate_limit_config: Optional[RateLimitConfig] = None):
        super().__init__(model_name, self._DEFAULT_MAX_CHARS, api_key=None,
                         rate_limit_config=rate_limit_config)
        self._host = host.rstrip('/')
    
    @property
    def host(self) -> str:
        """Returns the Ollama host URL."""
        return self._host
    
    def get_config(self, include_api_key: bool = False, include_rate_stats: bool = False) -> dict:
        """Returns extended configuration including host URL."""
        config_dict = super().get_config(include_api_key, include_rate_stats)
        config_dict["host"] = self.host
        # Remove api_key as Ollama doesn't use it
        config_dict.pop("api_key", None)
        return config_dict

    def generate_completion(self, messages: list[dict[str, str]], 
                          temperature: float | str = 0.7,
                          max_tokens: int | None = None,
                          top_p: float | None = None,
                          frequency_penalty: float | None = None,
                          presence_penalty: float | None = None,
                          stop: list[str] | None = None,
                          response_format: ResponseFormat = ResponseFormat.JSON,
                          json_schema: dict | None = None,
                          use_case: str | None = None,
                          **kwargs) -> str:
        temperature = self._resolve_temperature(temperature)
        formatted_messages = self._format_messages_for_response(messages, response_format, json_schema)
        self._validate_messages_length(formatted_messages)
        self.rate_limiter.wait_for_permit()

        endpoint = f"{self._host}/api/chat"
        
        # Build options dictionary
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None: options["num_predict"] = max_tokens
        if top_p is not None: options["top_p"] = top_p
        if stop is not None: options["stop"] = stop
        if frequency_penalty is not None: options["repeat_penalty"] = frequency_penalty # Approximate mapping
        
        # Specific Ollama params
        if "top_k" in kwargs: options["top_k"] = kwargs["top_k"]
        if "seed" in kwargs: options["seed"] = kwargs["seed"]
        if "num_ctx" in kwargs: options["num_ctx"] = kwargs["num_ctx"]

        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "stream": False,
            "options": options
        }
        try:
            response = HTTP_SESSION.post(endpoint, json=payload, timeout=180, verify=False)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama at {self._host}. Is it running? Error: {e}")
            raise
        except KeyError as e:
            print(f"Unexpected response format from Ollama: {e}")
            raise


# --- Helper Functions for Loading Configuration ---

def load_api_keys_from_csv(csv_path: str | None = None) -> dict[str, dict[str, str]]:
    """
    Loads API keys from apikeys.csv file.
    
    Args:
        csv_path: Path to CSV file. If None, uses default path.
    
    Returns:
        Dictionary with service name as key and dict with 'user' and 'apikey' as value.
    
    Example:
        >>> keys = load_api_keys_from_csv()
        >>> print(keys['gemini']['apikey'][:10])
        'AIzaSy...'
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "apikeys.csv")
    
    providers: dict[str, dict[str, str]] = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        import csv
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            provider = row.get('provider', '').lower()
            apikey = row.get('apikey', '')
            user = row.get('user', '')
            
            # Only add if has valid API key (not empty, not "Leaked")
            if provider and apikey and apikey != 'Leaked':
                providers[provider] = {'user': user, 'apikey': apikey}
    
    return providers


def load_models_from_csv(csv_path: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """
    Loads models and their limits from model_rate_limits.csv.
    
    Args:
        csv_path: Path to CSV file. If None, uses default path.
    
    Returns:
        Dictionary with provider as key and list of models as value.
        Each model is a dict with: model, rpm, rpd, tpm, tpd
    
    Example:
        >>> models = load_models_from_csv()
        >>> print(models['GROQ'][0]['model'])
        'allam-2-7b'
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "model_rate_limits.csv")
    
    models: dict[str, list[dict[str, Any]]] = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            provider = row.get('provider', '').upper()
            model = row.get('model', '')
            
            if not provider or not model:
                continue
            
            config = {
                'model': model,
                'rpm': _parse_limit(row.get('rpm', '30')) or 30,
                'rpd': _parse_limit(row.get('rpd')),
                'tpm': _parse_limit(row.get('tpm')),
                'tpd': _parse_limit(row.get('tpd'))
            }
            
            if provider not in models:
                models[provider] = []
            models[provider].append(config)
    
    return models


def get_service_for_provider(
    provider: str, 
    model: str, 
    apikey: str,
    rate_limit_config: Optional[RateLimitConfig] = None
) -> GenerativeAIService:
    """
    Factory function that returns the appropriate service for each provider.
    
    Args:
        provider: Provider name (gemini, groq, openrouter, openai, etc.)
        model: Model name to use
        apikey: Provider API key
        rate_limit_config: Optional rate limiting configuration
    
    Returns:
        Instance of the corresponding service
    
    Raises:
        ValueError: If provider is not supported
    
    Example:
        >>> service = get_service_for_provider('google', 'gemini-3-flash-preview', 'AIza...')
        >>> print(service.provider)
        'Google'
    """
    provider_lower = provider.lower()
    
    if provider_lower == 'google':
        return GeminiService(model_name=model, api_key=apikey, rate_limit_config=rate_limit_config)
    elif provider_lower == 'groq':
        return GroqService(model_name=model, api_key=apikey, rate_limit_config=rate_limit_config)
    elif provider_lower == 'openrouter':
        return OpenRouterService(model_name=model, api_key=apikey, rate_limit_config=rate_limit_config)
    elif provider_lower == 'openai':
        return OpenAIService(model_name=model, api_key=apikey, rate_limit_config=rate_limit_config)
    elif provider_lower == 'perplexity':
        return PerplexityService(model_name=model, api_key=apikey, rate_limit_config=rate_limit_config)
    elif provider_lower == 'deepseek':
        return DeepSeekService(model_name=model, api_key=apikey, rate_limit_config=rate_limit_config)
    elif provider_lower == 'grok':
        return GrokService(model_name=model, api_key=apikey, rate_limit_config=rate_limit_config)
    elif provider_lower == 'cerebras':
        return CerebrasService(model_name=model, api_key=apikey, rate_limit_config=rate_limit_config)
    elif provider_lower == 'nvidia':
        return NvidiaService(model_name=model, api_key=apikey, rate_limit_config=rate_limit_config)
    elif provider_lower == 'foundry':
        return FoundryService(model_name=model, api_key=apikey, rate_limit_config=rate_limit_config)
    elif provider_lower == 'vercel':
        return VercelAIService(model_name=model, api_key=apikey, rate_limit_config=rate_limit_config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def resolve_api_key_for_provider(provider: str) -> str:
    """
    Resolves the API key for a specific provider.
    
    Search order:
        1. Keyring (OS credential store)
        2. CSV file (apikeys.csv)
    
    Args:
        provider: Nombre del proveedor (gemini, openrouter, groq, etc.)
    
    Returns:
        API key string
    
    Raises:
        ValueError: Si no se encuentra la API key
    
    Example:
        >>> key = resolve_api_key_for_provider('openrouter')
        >>> print(key[:10])
        'sk-or-v1-...'
    """
    import logging
    logger = logging.getLogger("connector_genai")
    
    provider_lower = provider.lower()
    
    # Ollama no requiere API key
    if provider_lower == "ollama":
        return "ollama-local"
    
    # Define aliases for keyring/CSV lookup (try both names)
    lookup_names = [provider_lower]
    if provider_lower == "google":
        lookup_names.append("gemini")  # Also try 'gemini' as legacy alias
    
    # 1. Try keyring first (OS credential store)
    try:
        if install_library("keyring"):
            import keyring
            key = keyring.get_password(provider_lower, "api_key")
            if key:
                logger.info(f"API key loaded from keyring for '{provider_lower}'")
                return key
    except Exception as e:
        logger.debug(f"Keyring lookup failed for '{provider_lower}': {e}")
    
    # 2. Try CSV file (apikeys.csv)
    try:
        api_keys = load_api_keys_from_csv()
        if provider_lower in api_keys:
            key = api_keys[provider_lower].get('apikey')
            if key:
                logger.info(f"API key loaded from CSV for '{provider_lower}'")
                return key
    except Exception as e:
        logger.debug(f"Could not load API keys from CSV: {e}")
    
    raise ValueError(f"API key not found for provider '{provider_lower}'. "
                     f"Please add it to keyring or apikeys.csv")


# --- Main Example ---

def load_all_api_keys_from_csv(csv_path: str | None = None) -> list[dict[str, str]]:
    """
    Loads ALL API keys from apikeys.csv file (including duplicates per provider).
    
    Args:
        csv_path: Path to CSV file. If None, uses default path.
    
    Returns:
        List of dictionaries with 'service', 'user' and 'apikey'.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "apikeys.csv")
    
    entries: list[dict[str, str]] = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        import csv
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            provider = row.get('provider', '').lower()
            apikey = row.get('apikey', '')
            user = row.get('user', '')
            
            # Only add if has valid API key (not empty, not "Leaked")
            if provider and apikey and apikey != 'Leaked':
                entries.append({'provider': provider, 'user': user, 'apikey': apikey})
    
    return entries


def interactive_select(options: list[str], prompt: str) -> int:
    """
    Shows numbered options and allows user to select one.
    
    Args:
        options: List of options to display
        prompt: Message for the user
    
    Returns:
        Index of selected option (0-based)
    """
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")
    
    while True:
        try:
            choice = input("\n👉 Select an option: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
            print(f"❌ Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("❌ Please enter a valid number")


if __name__ == "__main__":
    """
    Interactive mode: Select provider, user, model, prompt and execute.
    
    Flow:
    1. Select provider and user
    2. Select model
    3. Enter prompt
    4. Confirm execution and show result
    """
    
    print("🚀 Connector GenAI - Interactive Mode")
    print("=" * 60)
    
    # 1. Load all API keys (including multiple users per provider)
    print("\n📂 Loading configurations...")
    
    try:
        all_api_entries = load_all_api_keys_from_csv()
        if not all_api_entries:
            print("❌ No valid API keys found in apikeys.csv")
            sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ Error loading apikeys.csv: {e}")
        sys.exit(1)
    
    try:
        models_by_provider = load_models_from_csv()
    except FileNotFoundError:
        models_by_provider = {}
    
    # 2. STEP 1: Select provider and user
    print("\n" + "=" * 60)
    print("📋 STEP 1: Select Provider and User")
    print("=" * 60)
    
    options_provider_user = [
        f"{entry['provider'].upper()} - {entry['user']}" 
        for entry in all_api_entries
    ]
    
    selected_idx = interactive_select(
        options_provider_user, 
        "Available providers and users:"
    )
    
    selected_entry = all_api_entries[selected_idx]
    selected_provider = selected_entry['provider']
    selected_user = selected_entry['user']
    selected_apikey = selected_entry['apikey']
    
    print(f"\n✅ Selected: {selected_provider.upper()} ({selected_user})")
    
    # 3. STEP 2: Select model
    print("\n" + "=" * 60)
    print("📋 STEP 2: Select Model")
    print("=" * 60)
    
    # Get models from CSV or use defaults
    csv_provider_key = selected_provider.upper()
    available_models: list[str] = []
    model_configs: dict[str, dict[str, Any]] = {}
    model_display_options: list[str] = []
    
    if csv_provider_key in models_by_provider:
        for mc in models_by_provider[csv_provider_key]:
            model_name = mc['model']
            available_models.append(model_name)
            model_configs[model_name] = mc
            
            # Create string with rate limits for display
            limits_parts: list[str] = []
            if mc.get('rpm'):
                limits_parts.append(f"RPM:{mc['rpm']}")
            if mc.get('tpm'):
                tpm_display = mc['tpm'] // 1000 if mc['tpm'] >= 1000 else mc['tpm']
                tpm_suffix = "K" if mc['tpm'] >= 1000 else ""
                limits_parts.append(f"TPM:{tpm_display}{tpm_suffix}")
            
            limits_str = f" [{', '.join(limits_parts)}]" if limits_parts else ""
            model_display_options.append(f"{model_name}{limits_str}")
    
    # If no models in CSV, exit the program
    if not available_models:
        print(f"\n❌ No models configured in model_rate_limits.csv for {selected_provider.upper()}")
        print("   Add models to the CSV file and run again.")
        sys.exit(1)
    
    selected_model_idx = interactive_select(
        model_display_options,
        f"Available models for {selected_provider.upper()}:"
    )
    
    selected_model = available_models[selected_model_idx]
    
    # Show selected model info with rate limits
    if selected_model in model_configs:
        mc = model_configs[selected_model]
        limits_info: list[str] = []
        if mc.get('rpm'):
            limits_info.append(f"RPM: {mc['rpm']}")
        if mc.get('rpd'):
            limits_info.append(f"RPD: {mc['rpd']:,}")
        if mc.get('tpm'):
            limits_info.append(f"TPM: {mc['tpm']:,}")
        if mc.get('tpd'):
            limits_info.append(f"TPD: {mc['tpd']:,}")
        
        limits_display = f" ({' | '.join(limits_info)})" if limits_info else ""
        print(f"\n✅ Model selected: {selected_model}{limits_display}")
    else:
        print(f"\n✅ Model selected: {selected_model}")
    
    # 4. STEP 3: Enter prompt
    print("\n" + "=" * 60)
    print("📋 STEP 3: Enter Prompt")
    print("=" * 60)
    
    print("\n📝 Type your prompt (Enter to finish, empty line + Enter for multiline):")
    print("   (Type 'END' on an empty line to finish a multiline prompt)")
    
    lines: list[str] = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        if not lines and line:  # First non-empty line
            lines.append(line)
            break  # For single-line prompts
        lines.append(line)
    
    prompt_text = '\n'.join(lines).strip()
    
    if not prompt_text:
        print("❌ Prompt cannot be empty")
        sys.exit(1)
    
    print(f"\n📋 Prompt entered ({len(prompt_text)} characters):")
    preview = prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text
    print(f"   \"{preview}\"")
    
    # 5. STEP 4: Confirm execution
    print("\n" + "=" * 60)
    print("📋 STEP 4: Confirm Execution")
    print("=" * 60)
    
    # Prepare rate limits from CSV
    rate_config = None
    rate_limits_info: list[str] = []
    
    if selected_model in model_configs:
        mc = model_configs[selected_model]
        rpm_val = mc.get('rpm')
        rpd_val = mc.get('rpd')
        tpm_val = mc.get('tpm')
        tpd_val = mc.get('tpd')
        
        rate_config = RateLimitConfig(
            rpm=rpm_val,
            rpd=rpd_val,
            tpm=tpm_val,
            tpd=tpd_val
        )
        
        # Prepare info for display
        if rpm_val:
            rate_limits_info.append(f"RPM: {rpm_val:,}")
        if rpd_val:
            rate_limits_info.append(f"RPD: {rpd_val:,}")
        if tpm_val:
            rate_limits_info.append(f"TPM: {tpm_val:,}")
        if tpd_val:
            rate_limits_info.append(f"TPD: {tpd_val:,}")
    
    print("\n📊 Configuration summary:")
    print(f"   🔹 Provider: {selected_provider.upper()}")
    print(f"   🔹 User: {selected_user}")
    print(f"   🔹 Model: {selected_model}")
    if rate_limits_info:
        print(f"   🔹 Rate Limits: {' | '.join(rate_limits_info)}")
    else:
        print(f"   🔹 Rate Limits: Not configured (using defaults)")
    print(f"   🔹 Prompt: {len(prompt_text)} characters")
    
    confirm = input("\n❓ Execute the query? (y/n): ").strip().lower()
    
    if confirm not in ['y', 'yes']:
        print("\n❌ Execution cancelled by user")
        sys.exit(0)
    
    # 6. Execute the query
    print("\n" + "=" * 60)
    print("🔄 Executing query...")
    print("=" * 60)
    
    try:
        
        # Create service
        service = get_service_for_provider(
            provider=selected_provider,
            model=selected_model,
            apikey=selected_apikey,
            rate_limit_config=rate_config
        )
        
        print(f"   📡 Service: {service}")
        if rate_limits_info:
            print(f"   📊 Rate Limits applied: {' | '.join(rate_limits_info)}")
        print("   ⏳ Sending request...")
        
        # Make the query
        messages = [{"role": "user", "content": prompt_text}]
        start_time = time.time()
        
        response = service.generate_completion(
            messages=messages,
            response_format=ResponseFormat.TEXT,
            temperature=0.7
        )
        
        elapsed_time = time.time() - start_time
        
        # Show result
        print("\n" + "=" * 60)
        print("✅ RESULT")
        print("=" * 60)
        print(f"\n⏱️  Response time: {elapsed_time:.2f} seconds")
        print(f"📏 Response length: {len(response)} characters")
        print("\n" + "-" * 60)
        print("📤 RESPONSE:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        
        # Show updated rate limit stats
        print("\n📈 Rate Limit Usage:")
        rate_stats = service.rate_limiter.get_stats()
        if "limiters" in rate_stats:
            for limiter_name, limiter_stats in rate_stats["limiters"].items():
                if isinstance(limiter_stats, dict):
                    # Get current usage (different fields for different limiter types)
                    current = limiter_stats.get("current_tokens", 
                              limiter_stats.get("current_count", 
                              limiter_stats.get("total_acquired", "?")))
                    # Get capacity (burst_size for token bucket, limit for sliding window)
                    capacity = limiter_stats.get("burst_size", 
                               limiter_stats.get("limit", "?"))
                    # Format nicely
                    if isinstance(current, float):
                        current = f"{current:.1f}"
                    if isinstance(capacity, float):
                        capacity = f"{capacity:.0f}"
                    print(f"   {limiter_name.upper()}: {current}/{capacity}")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        sys.exit(1)
    
    print("\n✨ Execution completed.")
