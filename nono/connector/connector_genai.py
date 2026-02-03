# -*- coding: utf-8 -*-
"""
Connector Gen AI - Unified Generative AI Services Connector

Provides a unified interface for connecting to multiple generative AI services
including OpenAI, Google Gemini, Perplexity, DeepSeek, Grok (xAI), and Ollama.
Features built-in rate limiting (Token Bucket), SSL configuration management,
and auto-installation of required dependencies.

Supported Providers:
    - OpenAIService: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
    - GeminiService: Gemini 1.5 Flash, Gemini 1.5 Pro, Gemini Pro
    - PerplexityService: Sonar, LLaMA-3 variants, Mixtral
    - DeepSeekService: DeepSeek Chat, DeepSeek Coder
    - GrokService: Grok-1
    - OllamaService: Any locally hosted model

Example:
    >>> from connector_genai import GeminiService, ResponseFormat
    >>> client = GeminiService(model_name="gemini-1.5-flash", api_key="...")
    >>> messages = [{"role": "user", "content": "Hello!"}]
    >>> response = client.generate_completion(messages, response_format=ResponseFormat.TEXT)

Author: DatamanEdge
License: MIT
Date: 2026-02-02
Version: 1.0.0
"""

import sys
import subprocess
import ssl
import time
from abc import ABC, abstractmethod
from enum import Enum
import json
import threading
from typing import Optional
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
                               custom_cert_path: str = None) -> None:
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
                                  custom_cert_path=r'C:\path\to\corporate-cert.crt')
    
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

def install_library(library_name: str, import_name: str = None, package_name: str = None) -> bool:
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
    requests.packages.urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
else:
    raise ImportError("The 'requests' library is required and could not be installed.")

# Disable SSL verification (for development only)
import ssl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context # Crear contexto SSL sin verificación

# Configure SSL verification
import os
import certifi
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

# --- Rate Limiter ---

class RateLimiter:
    """
    Token bucket rate limiter for controlling API request frequency.
    """
    def __init__(self, requests_per_second: float, burst: int = 1):
        self.requests_per_second = requests_per_second
        self.tokens = burst
        self.max_tokens = burst
        self.last_refill = datetime.now()
        self.lock = threading.Lock()

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = datetime.now()
        elapsed = (now - self.last_refill).total_seconds()
        new_tokens = elapsed * self.requests_per_second
        with self.lock:
            self.tokens = min(self.max_tokens, self.tokens + new_tokens)
            self.last_refill = now

    def acquire(self) -> bool:
        """Attempt to acquire a token for making a request."""
        self._refill()
        with self.lock:
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def wait_for_token(self):
        """Wait until a token is available."""
        while not self.acquire():
            time.sleep(0.1)  # Small sleep to avoid busy waiting

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

# --- Base Class ---

class GenerativeAIService(ABC):
    """
    Abstract base class for interacting with generative AI services with rate limiting.
    """
    TEMPERATURE_RECOMMENDATIONS = {
        "coding": 0.0,           # Máxima precisión, código determinístico
        "math": 0.0,             # Respuestas matemáticas exactas
        "data_cleaning": 0.1,    # Alta precisión para transformaciones de datos
        "data_analysis": 0.3,    # Consistencia en análisis, algo de flexibilidad
        "translation": 0.3,      # Traducciones precisas y fieles al original
        "conversation": 0.7,     # Balance entre coherencia y naturalidad
        "creative": 1.0,         # Mayor variabilidad para contenido creativo
        "poetry": 1.2,           # Alta creatividad para expresión artística
        "default": 0.7           # Valor por defecto balanceado para uso general
    }

    @classmethod
    def get_recommended_temperature(cls, use_case: str) -> float:
        return cls.TEMPERATURE_RECOMMENDATIONS.get(use_case.lower(), 0.7)

    @classmethod
    def get_max_input_chars(cls, model_name: str) -> int:
        """Returns the maximum input characters for a given model."""
        if hasattr(cls, "_MAX_INPUT_CHARS"):
            # Return lookup or a reasonable default (matches __init__ defaults generally)
            return cls._MAX_INPUT_CHARS.get(model_name, 120_000) 
        return 120_000

    def __init__(self, model_name: str, max_input_chars: int,
                 requests_per_second: float = 1.0, api_key: Optional[str] = None):
        if not model_name:
            raise ValueError("A model_name must be provided.")

        self.model_name = model_name
        self.api_key = api_key
        self.max_input_chars = max_input_chars
        self.rate_limiter = RateLimiter(requests_per_second)

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
                          temperature: float = 0.7,
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
    def __init__(self, model_name: str, api_key: str, base_url: str,
                 max_input_chars: int, requests_per_second: float = 1.0):
        super().__init__(model_name, max_input_chars, requests_per_second, api_key)
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_completion(self, messages: list[dict[str, str]], 
                          temperature: float = 0.7,
                          max_tokens: int | None = None,
                          top_p: float | None = None,
                          frequency_penalty: float | None = None,
                          presence_penalty: float | None = None,
                          stop: list[str] | None = None,
                          response_format: ResponseFormat = ResponseFormat.JSON,
                          json_schema: dict | None = None,
                          use_case: str | None = None,
                          **kwargs) -> str:
        formatted_messages = self._format_messages_for_response(messages, response_format, json_schema)
        
        # Ensure a system message exists (default behavior)
        if not any(msg.get("role") == "system" for msg in formatted_messages):
             formatted_messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
        
        self._validate_messages_length(formatted_messages)
        self.rate_limiter.wait_for_token()

        endpoint = f"{self.base_url}/chat/completions"
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
            print(f"Error connecting to {self.base_url}: {e}")
            raise
        except (KeyError, IndexError) as e:
            print(f"Unexpected API response format: {e}")
            raise

# --- Concrete Implementations ---

class OpenAIService(OpenAICompatibleService):
    _MAX_INPUT_CHARS = {
        "gpt-4o": 500_000,
        "gpt-4o-mini": 120_000,
        "gpt-4-turbo": 500_000,
        "gpt-3.5-turbo": 60_000
    }

    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str | None = None,
                 requests_per_second: float = 1.0):
        super().__init__(
            model_name,
            api_key,
            "https://api.openai.com/v1",
            self.get_max_input_chars(model_name),
            requests_per_second
        )
    
    @classmethod
    def get_max_input_chars(cls, model_name: str) -> int:
        return cls._MAX_INPUT_CHARS.get(model_name, 60_000)

class PerplexityService(OpenAICompatibleService):
    _MAX_INPUT_CHARS = {
        "sonar": 120_000,
        "llama-3-sonar-large-32k-online": 120_000,
        "llama-3-sonar-small-32k-online": 120_000,
        "llama-3-8b-instruct": 30_000,
        "llama-3-70b-instruct": 120_000,
        "mixtral-8x7b-instruct": 120_000,
        "mistral-7b-instruct": 30_000
    }

    def __init__(self, model_name: str = "llama-3-sonar-large-32k-online", api_key: str | None = None,
                 requests_per_second: float = 1.0):
        super().__init__(
            model_name,
            api_key,
            "https://api.perplexity.ai",
            self._MAX_INPUT_CHARS.get(model_name, 120_000),
            requests_per_second
        )

    def generate_completion(self, messages: list[dict[str, str]], 
                          temperature: float = 0.7,
                          max_tokens: int | None = None,
                          top_p: float | None = None,
                          frequency_penalty: float | None = None,
                          presence_penalty: float | None = None,
                          stop: list[str] | None = None,
                          response_format: ResponseFormat = ResponseFormat.JSON,
                          json_schema: dict | None = None,
                          use_case: str | None = None,
                          **kwargs) -> str:
        if response_format == ResponseFormat.JSON and json_schema:
            processed_messages = [msg.copy() for msg in messages]
        else:
            processed_messages = self._format_messages_for_response(messages, response_format, json_schema)

        if not any(msg.get("role") == "system" for msg in processed_messages):
            processed_messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})

        self._validate_messages_length(processed_messages)
        self.rate_limiter.wait_for_token()

        endpoint = f"{self.base_url}/chat/completions"
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
    _MAX_INPUT_CHARS = {
        "deepseek-chat": 500_000,
        "deepseek-coder": 500_000
    }

    def __init__(self, model_name: str = "deepseek-chat", api_key: str | None = None,
                 requests_per_second: float = 1.0):
        super().__init__(
            model_name,
            api_key,
            "https://api.deepseek.com",
            self._MAX_INPUT_CHARS.get(model_name, 500_000),
            requests_per_second
        )

class GrokService(OpenAICompatibleService):
    _MAX_INPUT_CHARS = {"grok-1": 30_000}

    def __init__(self, model_name: str = "grok-1", api_key: str | None = None,
                 requests_per_second: float = 1.0):
        super().__init__(
            model_name,
            api_key,
            "https://api.x.ai/v1",
            self._MAX_INPUT_CHARS.get(model_name, 30_000),
            requests_per_second
        )

class GeminiService(GenerativeAIService):
    """
    Google Gemini AI service using the new google-genai SDK.
    
    This service uses the modern google-genai package (google.genai) which replaces
    the deprecated google-generativeai package.
    """
    _MAX_INPUT_CHARS = {
        "gemini-2.5-flash": 4_000_000,
        "gemini-2.5-pro": 4_000_000,
        "gemini-1.5-flash": 4_000_000,
        "gemini-1.5-pro": 4_000_000,
        "gemini-pro": 120_000
    }
    _DEFAULT_REQUESTS_PER_SECOND = 0.25  # 15 requests/min = 0.25 requests/sec

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str | None = None,
                 requests_per_second: float = _DEFAULT_REQUESTS_PER_SECOND):
        super().__init__(
            model_name,
            self._MAX_INPUT_CHARS.get(model_name, 120_000),
            requests_per_second,
            api_key
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
                          temperature: float = 0.7,
                          max_tokens: int | None = None,
                          top_p: float | None = None,
                          frequency_penalty: float | None = None,
                          presence_penalty: float | None = None,
                          stop: list[str] | None = None,
                          response_format: ResponseFormat = ResponseFormat.JSON,
                          json_schema: dict | None = None,
                          use_case: str | None = None,
                          **kwargs) -> str:
        # Build GenerateContentConfig
        config_params = {"temperature": temperature}

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

        self.rate_limiter.wait_for_token()

        try:
            # Create config object
            config = self.types.GenerateContentConfig(**config_params)
            
            # Call the new API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=gemini_contents,
                config=config
            )
            return response.text
        except Exception as e:
            print(f"An error occurred with the Gemini API: {e}")
            raise


class OllamaService(GenerativeAIService):
    _MAX_INPUT_CHARS = 200_000

    def __init__(self, model_name: str = "llama3", host: str = "http://localhost:11434",
                 requests_per_second: float = 1.0):
        super().__init__(model_name, self._MAX_INPUT_CHARS, requests_per_second, api_key=None)
        self.host = host.rstrip('/')

    def generate_completion(self, messages: list[dict[str, str]], 
                          temperature: float = 0.7,
                          max_tokens: int | None = None,
                          top_p: float | None = None,
                          frequency_penalty: float | None = None,
                          presence_penalty: float | None = None,
                          stop: list[str] | None = None,
                          response_format: ResponseFormat = ResponseFormat.JSON,
                          json_schema: dict | None = None,
                          use_case: str | None = None,
                          **kwargs) -> str:
        formatted_messages = self._format_messages_for_response(messages, response_format, json_schema)
        self._validate_messages_length(formatted_messages)
        self.rate_limiter.wait_for_token()

        endpoint = f"{self.host}/api/chat"
        
        # Build options dictionary
        options = {"temperature": temperature}
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
            print(f"Error connecting to Ollama at {self.host}. Is it running? Error: {e}")
            raise
        except KeyError as e:
            print(f"Unexpected response format from Ollama: {e}")
            raise
        