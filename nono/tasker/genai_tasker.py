"""
GenAI Tasker - Generative AI Task Execution Module

This module provides a unified interface for executing AI tasks across multiple
providers (Gemini, OpenAI, Perplexity, Ollama). It follows S.O.L.I.D principles
and supports task-based execution with JSON configuration files.

Author: DatamanEdge
License: MIT
Version: 1.0.0
"""

import os
import json
import logging
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("GenAITaskExecutor")


def msg_log(message: str, level: int = logging.INFO) -> None:
    """
    Log a message and print it to the console with timestamp.
    
    Args:
        message: The message to log and print.
        level: The logging level (default: INFO).
    
    Returns:
        None
    """
    logger.log(level, message)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - [{logging.getLevelName(level)}] {message}")


def event_log(message: Optional[str] = None, level: int = logging.INFO):
    """
    Decorator for managing log messages at function entry and exit.
    
    Automatically logs when a function starts, completes, or raises an error.
    
    Args:
        message: Custom message to display. Defaults to function name.
        level: The logging level (default: INFO).
    
    Returns:
        Callable: Decorated function wrapper.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine the message to use
            act_msg = message if message else f"Execution of {func.__name__}"
            
            msg_log(f"Starting: {act_msg}", level)
            try:
                result = func(*args, **kwargs)
                msg_log(f"Completed: {act_msg}", level)
                return result
            except Exception as e:
                msg_log(f"Error in {act_msg}: {str(e)}", logging.ERROR)
                raise
        return wrapper
    return decorator


class AIProvider(Enum):
    """Enumeration of supported AI service providers."""
    
    GOOGLE = "google"
    OPENAI = "openai"
    PERPLEXITY = "perplexity"
    DEEPSEEK = "deepseek"
    GROK = "grok"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


@dataclass
class AIConfiguration:
    """
    Configuration dataclass for AI service settings.
    
    Attributes:
        provider: The AI provider to use.
        model_name: Name of the model.
        api_key: API key for authentication.
        temperature: Sampling temperature (default: 0.7).
        max_tokens: Maximum tokens in response (default: 2048).
        system_instruction: Optional system prompt instruction.
    """
    
    provider: AIProvider
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 2048
    system_instruction: Optional[str] = None


class BaseAIClient(ABC):
    """
    Abstract base class for AI service wrappers.
    
    Provides a unified interface for content generation across different AI providers.
    Subclasses must initialize the `service` attribute with the appropriate connector.
    """
    
    def __init__(self, config: AIConfiguration) -> None:
        """
        Initialize the AI client with the given configuration.
        
        Args:
            config: AIConfiguration instance with provider settings.
        """
        self.config = config
        self.service = None  # To be initialized by subclasses

    @event_log(message="AI Content Generation")
    def generate_content(
        self,
        input_data: Union[str, List[Dict[str, Any]]],
        json_schema: Optional[Dict[str, Any]] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate content based on the prompt or message list.
        
        Args:
            input_data: String prompt or list of message dictionaries.
            json_schema: Optional JSON schema for structured output.
            config_overrides: Optional runtime configuration overrides.
        
        Returns:
            Generated content as a string.
        
        Raises:
            ValueError: If input_data is not a string or list of dicts.
        """
        # Normalize input to messages list
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        elif isinstance(input_data, list):
            # Create a copy to avoid modifying original list
            messages = [msg.copy() for msg in input_data]
        else:
            raise ValueError("Input data must be a string or a list of message dictionaries.")
        
        # Add local system instruction if present in config ONLY if not already in messages
        # (This allows task-specific system prompts to take precedence if passed in messages)
        if self.config.system_instruction and not any(msg.get("role") == "system" for msg in messages):
             messages.insert(0, {"role": "system", "content": self.config.system_instruction})

        try:
            overrides = config_overrides or {}
            
            # Determine effective configuration
            effective_temperature = overrides.get("temperature", self.config.temperature)
            effective_max_tokens = overrides.get("max_tokens", self.config.max_tokens)
            override_model = overrides.get("model_name")
            
            # Handle temperature preset (string) or float
            if isinstance(effective_temperature, str):
                try:
                    effective_temperature = connector_genai.GenerativeAIService.get_recommended_temperature(effective_temperature)
                except Exception:
                    logger.warning(f"Unknown temperature preset '{effective_temperature}', using default 0.7")
                    effective_temperature = 0.7
            
            # Handle model override
            original_model_name = None
            if override_model and self.service and hasattr(self.service, 'model_name') and self.service.model_name != override_model:
                logger.info(f"Overriding model: {self.service.model_name} -> {override_model}")
                original_model_name = self.service.model_name
                self.service.model_name = override_model

            logger.info(f"Sending request to {self.config.provider.value} ({self.service.model_name if self.service else self.config.model_name})...")
            
            # Call generate_completion on the service
            if not self.service:
                raise ValueError("AI Service not initialized.")
            
            # Determine response format
            resp_fmt = connector_genai.ResponseFormat.JSON if json_schema else connector_genai.ResponseFormat.TEXT
            
            if overrides and "response_format" in overrides:
                 try:
                     # Attempt to use the provided format
                     fmt_str = overrides["response_format"].lower()
                     resp_fmt = connector_genai.ResponseFormat(fmt_str)
                 except ValueError:
                     logger.warning(f"Invalid response_format '{overrides['response_format']}', using default logic.")

            try:
                # Prepare extra kwargs for generate_completion
                # We copy overrides and remove keys that are passed as explicit arguments
                call_kwargs = overrides.copy() if overrides else {}
                
                # These are handled explicitly via arguments or logic, so remove from kwargs
                for key_to_remove in ["temperature", "max_tokens", "model_name", "provider", "api_key", "response_format"]:
                    if key_to_remove in call_kwargs:
                        del call_kwargs[key_to_remove]

                response_text = self.service.generate_completion(
                    messages=messages,
                    temperature=effective_temperature,
                    max_tokens=effective_max_tokens,
                    response_format=resp_fmt,
                    json_schema=json_schema,
                    **call_kwargs # Pass remaining overrides (top_p, stop, etc.)
                )
            finally:
                # Restore original model name if it was changed
                if original_model_name:
                    self.service.model_name = original_model_name
            
            return response_text
        except Exception as e:
            logger.error(f"{self.config.provider.value} API Error via Connector: {e}")
            raise


# Import the connector_genai module (shared across subprojects)
try:
    from ..connector import connector_genai
except ImportError:
    # Fallback for when running as script - add parent dir to path
    import sys
    import os
    _nono_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _nono_dir not in sys.path:
        sys.path.insert(0, _nono_dir)
    try:
        from connector import connector_genai
    except ImportError:
        # Try absolute path for safety if run from root
        from nono.connector import connector_genai

# Import jinjapromptpy for prompt building (required dependency)
try:
    from .jinja_prompt_builder import TaskPromptBuilder, build_prompt, build_prompts
except ImportError:
    try:
        from jinja_prompt_builder import TaskPromptBuilder, build_prompt, build_prompts
    except ImportError:
        from nono.tasker.jinja_prompt_builder import TaskPromptBuilder, build_prompt, build_prompts

# Lazy load jsonschema
HAS_JSONSCHEMA = False
if connector_genai.install_library("jsonschema"):
    import jsonschema
    HAS_JSONSCHEMA = True
else:
    logger.warning("jsonschema not found. Input validation will be skipped.")

class GeminiClient(BaseAIClient):
    """Client for Google's Gemini Models using connector_genai."""
    
    def __init__(self, config: AIConfiguration):
        super().__init__(config)
        
        # Use connector_genai's install_library for lazy loading google-genai
        if not connector_genai.install_library("google.genai", package_name="google-genai"):
             logger.critical("google-genai library is required. Install it using: pip install google-genai")
             raise ImportError("google-genai library is required.")

        # Initialize the service via connector_genai.GeminiService
        try:
             # Map configuration to GeminiService parameters
             self.service = connector_genai.GeminiService(
                 model_name=self.config.model_name,
                 api_key=self.config.api_key
             )
        except Exception as e:
            logger.critical(f"Failed to initialize GeminiService: {e}")
            raise



class OpenAIClient(BaseAIClient):
    """Client for OpenAI Models using connector_genai."""
    
    def __init__(self, config: AIConfiguration):
        super().__init__(config)
        
        try:
             self.service = connector_genai.OpenAIService(
                 model_name=self.config.model_name,
                 api_key=self.config.api_key
             )
        except Exception as e:
            logger.critical(f"Failed to initialize OpenAIService: {e}")
            raise


class PerplexityClient(BaseAIClient):
    """Client for Perplexity Models using connector_genai."""
    
    def __init__(self, config: AIConfiguration):
        super().__init__(config)
        
        try:
             self.service = connector_genai.PerplexityService(
                 model_name=self.config.model_name,
                 api_key=self.config.api_key
             )
        except Exception as e:
            logger.critical(f"Failed to initialize PerplexityService: {e}")
            raise


class DeepSeekClient(BaseAIClient):
    """Client for DeepSeek Models using connector_genai."""

    def __init__(self, config: AIConfiguration):
        super().__init__(config)
        
        try:
            self.service = connector_genai.DeepSeekService(
                model_name=self.config.model_name,
                api_key=self.config.api_key
            )
        except Exception as e:
            logger.critical(f"Failed to initialize DeepSeekService: {e}")
            raise


class GrokClient(BaseAIClient):
    """Client for Grok (xAI) Models using connector_genai."""

    def __init__(self, config: AIConfiguration):
        super().__init__(config)
        
        try:
            self.service = connector_genai.GrokService(
                model_name=self.config.model_name,
                api_key=self.config.api_key
            )
        except Exception as e:
            logger.critical(f"Failed to initialize GrokService: {e}")
            raise


class GroqClient(BaseAIClient):
    """Client for Groq LPU inference using connector_genai."""

    def __init__(self, config: AIConfiguration):
        super().__init__(config)
        
        try:
            self.service = connector_genai.GroqService(
                model_name=self.config.model_name,
                api_key=self.config.api_key
            )
        except Exception as e:
            logger.critical(f"Failed to initialize GroqService: {e}")
            raise


class OpenRouterClient(BaseAIClient):
    """Client for OpenRouter unified API using connector_genai."""

    def __init__(self, config: AIConfiguration):
        super().__init__(config)
        
        try:
            self.service = connector_genai.OpenRouterService(
                model_name=self.config.model_name,
                api_key=self.config.api_key
            )
        except Exception as e:
            logger.critical(f"Failed to initialize OpenRouterService: {e}")
            raise


class OllamaClient(BaseAIClient):
    """Client for Ollama Models using connector_genai."""

    def __init__(self, config: AIConfiguration):
        super().__init__(config)
        
        try:
            self.service = connector_genai.OllamaService(
                model_name=self.config.model_name
            )
        except Exception as e:
            logger.critical(f"Failed to initialize OllamaService: {e}")
            raise


class TaskExecutor:
    """
    Main executor class for managing AI tasks.
    
    Follows S.O.L.I.D principles by separating configuration, usage, and implementation.
    Supports task-based execution with JSON configuration files and automatic batching.
    
    Provider and model are passed as parameters. API key resolution is delegated
    to connector_genai (keyring -> CSV fallback).
    
    Attributes:
        config: AIConfiguration instance for the executor.
        service: Active AI service based on the configured provider.
    """
    
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float | str = 0.7,
        max_tokens: int = 2048
    ) -> None:
        """
        Initialize the TaskExecutor.
        
        Args:
            provider: AI provider name (google, openai, groq, openrouter, etc.). Required.
            model: Model name to use. Required.
            api_key: Optional API key. If None, resolved via connector_genai.
            temperature: Sampling temperature (float or use case name like "coding").
            max_tokens: Maximum tokens in response.
        """
        # Resolve temperature if string
        if isinstance(temperature, str):
            final_temperature = connector_genai.GenerativeAIService.get_recommended_temperature(temperature)
        else:
            final_temperature = float(temperature)
        
        # Normalize provider name (gemini -> google alias)
        provider_name = provider.lower()
        
        # Resolve provider enum
        try:
            provider_enum = AIProvider(provider_name)
        except ValueError:
            logger.warning(f"Unknown provider '{provider}', defaulting to GOOGLE")
            provider_enum = AIProvider.GOOGLE
        
        # Resolve API key via connector_genai if not provided
        final_api_key = api_key or connector_genai.resolve_api_key_for_provider(provider_enum.value)
        
        self.config = AIConfiguration(
            provider=provider_enum,
            model_name=model,
            api_key=final_api_key,
            temperature=final_temperature,
            max_tokens=max_tokens
        )
        
        # Create the AI service directly via connector_genai
        self.service = connector_genai.get_service_for_provider(
            provider=self.config.provider.value,
            model=self.config.model_name,
            apikey=self.config.api_key
        )
        logger.info(f"TaskExecutor initialized: {self.config.provider.value}/{self.config.model_name}")

    @event_log(message="Task Execution")
    def execute(
        self,
        input_data: Union[str, List[Dict[str, Any]]],
        output_schema: Optional[Dict[str, Any]] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute the task with the given input.
        
        Args:
            input_data: String prompt or list of message dictionaries.
            output_schema: Optional JSON schema for structured responses.
            config_overrides: Optional runtime overrides (temperature, model, etc.).
        
        Returns:
            Generated response as a string.
        """
        if not input_data:
            logger.warning("Empty input data provided.")
            return ""
        
        # Determine which service to use
        active_service = self.service
        
        # Check if we need to switch provider for this call
        if config_overrides and "provider" in config_overrides:
            new_provider = config_overrides["provider"]
            if isinstance(new_provider, AIProvider):
                new_provider = new_provider.value
            
            if new_provider != self.config.provider.value:
                logger.info(f"Switching provider for this task: {self.config.provider.value} -> {new_provider}")
                
                # Resolve API key for the new provider
                api_key = config_overrides.get("api_key")
                if not api_key:
                    api_key = connector_genai.resolve_api_key_for_provider(new_provider)
                
                model_name = config_overrides.get("model_name", config_overrides.get("model", "default"))
                
                # Create temporary service via connector_genai
                active_service = connector_genai.get_service_for_provider(
                    provider=new_provider,
                    model=model_name,
                    apikey=api_key
                )
        
        # Generate content using the service
        return self._generate_content(active_service, input_data, output_schema, config_overrides)
    
    def _generate_content(
        self,
        service: connector_genai.GenerativeAIService,
        input_data: Union[str, List[Dict[str, Any]]],
        json_schema: Optional[Dict[str, Any]] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate content using the provided service.
        
        Args:
            service: The AI service to use for generation.
            input_data: String prompt or list of message dictionaries.
            json_schema: Optional JSON schema for structured output.
            config_overrides: Optional runtime configuration overrides.
        
        Returns:
            Generated content as a string.
        """
        # Normalize input to messages list
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        elif isinstance(input_data, list):
            messages = [msg.copy() for msg in input_data]
        else:
            raise ValueError("Input data must be a string or a list of message dictionaries.")
        
        overrides = config_overrides or {}
        
        # Determine effective temperature
        effective_temperature = overrides.get("temperature", self.config.temperature)
        if isinstance(effective_temperature, str):
            effective_temperature = connector_genai.GenerativeAIService.get_recommended_temperature(effective_temperature)
        
        # Determine effective max_tokens
        effective_max_tokens = overrides.get("max_tokens", self.config.max_tokens)
        
        # Determine response format
        resp_fmt = connector_genai.ResponseFormat.JSON if json_schema else connector_genai.ResponseFormat.TEXT
        if "response_format" in overrides:
            try:
                fmt_str = overrides["response_format"].lower()
                resp_fmt = connector_genai.ResponseFormat(fmt_str)
            except ValueError:
                logger.warning(f"Invalid response_format '{overrides['response_format']}', using default.")
        
        # Prepare extra kwargs (remove keys handled explicitly)
        call_kwargs = {k: v for k, v in overrides.items() 
                       if k not in ["temperature", "max_tokens", "model_name", "model", 
                                    "provider", "api_key", "response_format"]}
        
        logger.info(f"Sending request to {service.provider} ({service.model_name})...")
        
        try:
            response_text = service.generate_completion(
                messages=messages,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
                response_format=resp_fmt,
                json_schema=json_schema,
                **call_kwargs
            )
            return response_text
        except Exception as e:
            logger.error(f"{service.provider} API Error: {e}")
            raise

    def _merge_batched_results(
        self,
        results: List[Any],
        output_format: Optional[str] = None
    ) -> Any:
        """
        Merge results from batched execution based on output format.
        
        Args:
            results: List of results from individual batch executions.
            output_format: Expected output format (json, csv, table, xml, text).
        
        Returns:
            Merged results in the specified format.
        """
        if not results:
            return ""
        
        # Determine format if not provided (try to guess from first result)
        if not output_format and results:
             if isinstance(results[0], str) and results[0].strip().startswith("{"):
                 output_format = "json"
             else:
                 output_format = "text"

        if output_format == "json":
             # Try to merge JSONs
             merged_list = []
             merged_dict = {}
             is_list_mode = False
             
             for res in results:
                 try:
                     parsed = json.loads(res) if isinstance(res, str) else res
                     
                     if isinstance(parsed, list):
                         is_list_mode = True
                         merged_list.extend(parsed)
                     elif isinstance(parsed, dict):
                         # Merge dicts
                         for k, v in parsed.items():
                             if k in merged_dict:
                                 if isinstance(merged_dict[k], list) and isinstance(v, list):
                                     merged_dict[k].extend(v)
                                 # else: overwrite or complex merge strategy not implemented
                             else:
                                 merged_dict[k] = v
                 except Exception:
                     pass # Ignore parse errors in chunks
             
             if is_list_mode and not merged_dict:
                 return json.dumps(merged_list, indent=2, ensure_ascii=False)
             return json.dumps(merged_dict, indent=2, ensure_ascii=False)
        
        elif output_format == "csv":
             # Merge CSVs handling headers
             full_csv = []
             header_found = None
             
             for res in results:
                 if not isinstance(res, str): continue
                 lines = res.strip().splitlines()
                 if not lines: continue
                 
                 if not header_found:
                     header_found = lines[0]
                     full_csv.append(header_found)
                     full_csv.extend(lines[1:])
                 else:
                     # Check if first line matches header
                     start_idx = 1 if lines[0] == header_found else 0
                     full_csv.extend(lines[start_idx:])
             return "\n".join(full_csv)

        elif output_format == "table":
             # Merge Markdown Tables handling headers
             full_table = []
             header_lines = []
             
             for res in results:
                 if not isinstance(res, str): continue
                 lines = res.strip().splitlines()
                 result_lines = [l for l in lines if l.strip()] # Filter empty lines
                 if not result_lines: continue
                 
                 # Basic heuristic: A markdown table usually starts with a header row `| ... |` 
                 # followed by separator row `| --- |`.
                 if len(result_lines) >= 2 and "|" in result_lines[1] and "---" in result_lines[1]:
                     current_header = result_lines[:2]
                     current_body = result_lines[2:]
                     
                     if not header_lines:
                         header_lines = current_header
                         full_table.extend(header_lines)
                         full_table.extend(current_body)
                     else:
                         # Ensure headers match before skipping, otherwise just append everything (safety fallback)
                         if current_header[0] == header_lines[0] and current_header[1] == header_lines[1]:
                             full_table.extend(current_body)
                         else:
                             # Mismatched headers? Just append with a spacer, effectively making a new table
                             full_table.append("") 
                             full_table.extend(current_header)
                             full_table.extend(current_body)
                 else:
                     # Not a standard table? Just append
                     if full_table: full_table.append("")
                     full_table.extend(result_lines)
            
             return "\n".join(full_table)

        elif output_format == "xml":
             # Simple concatenation for XML fragments
             # Ideally one should strip root tags if they exist, but generic XML merging is hard.
             # We assume the prompt asks for list items or the user handles the wrapping.
             return "\n".join([str(r) for r in results])

        else:
             # Text: simple concatenation
             return "\n\n".join([str(r) for r in results])

    def _get_active_service_class(self, provider: AIProvider) -> Optional[type]:
        """
        Get the service class for a provider.
        
        Args:
            provider: The AI provider.
        
        Returns:
            The service class, or None if not found.
        """
        service_map = {
            AIProvider.GOOGLE: connector_genai.GeminiService,
            AIProvider.OPENAI: connector_genai.OpenAIService,
            AIProvider.PERPLEXITY: connector_genai.PerplexityService,
            AIProvider.DEEPSEEK: connector_genai.DeepSeekService,
            AIProvider.GROK: connector_genai.GrokService,
            AIProvider.GROQ: connector_genai.GroqService,
            AIProvider.OPENROUTER: connector_genai.OpenRouterService,
            AIProvider.OLLAMA: connector_genai.OllamaService,
        }
        return service_map.get(provider)

    def run_json_task(
        self,
        task_file: str,
        data_input: Optional[Any] = None,
        **data_inputs: Any
    ) -> str:
        """
        Execute a task defined in a JSON file with automatic batching support.
        
        Multiple data inputs are supported using named placeholders in the prompt template.
        Use {data_input_json} for the main input, and {placeholder_name} for additional inputs.
        Pass additional inputs as keyword arguments matching the placeholder names.
        
        Example:
            executor.run_json_task(
                "task.json",
                main_data,                    # -> {data_input_json}
                categories=["A", "B", "C"],   # -> {categories}
                context="Some context"        # -> {context}
            )
        
        Args:
            task_file: Path to task JSON file. Required.
            data_input: Primary input data for the task (maps to {data_input_json}).
            **data_inputs: Additional named inputs (maps to {name} placeholders).
        
        Returns:
            Task execution result as a string.
        
        Raises:
            FileNotFoundError: If task file does not exist.
            ValueError: If no valid task definition is available.
        """
        if not os.path.exists(task_file):
            raise FileNotFoundError(f"Task file not found: {task_file}")
        
        with open(task_file, 'r', encoding='utf-8') as f:
            task_def = json.load(f)
        
        data = data_input

        # Parse new simplified schema structure
        prompts = task_def.get("prompts", {})
        system_prompt = prompts.get("system", "")
        user_prompt_template = prompts.get("user", "")
        assistant_prompt = prompts.get("assistant", "")
        
        output_schema = task_def.get("output_schema")
        input_schema = task_def.get("input_schema")

        # Parse AI Configuration overrides from task definition
        genai_config = task_def.get("genai", {})
        config_overrides = {}
        if "temperature" in genai_config:
            temp_val = genai_config["temperature"]
            if isinstance(temp_val, str):
                # Convert enumerations to recommended float values using the connector logic
                try:
                    config_overrides["temperature"] = connector_genai.GenerativeAIService.get_recommended_temperature(temp_val)
                except Exception:
                    logger.warning(f"Unknown temperature preset '{temp_val}', using default 0.7")
                    config_overrides["temperature"] = 0.7
            else:
                config_overrides["temperature"] = temp_val

        if "max_tokens" in genai_config:
            config_overrides["max_tokens"] = genai_config["max_tokens"]
        if "model" in genai_config:
            config_overrides["model_name"] = genai_config["model"]
            
        # Handle Provider Override
        if "provider" in genai_config:
            try:
                new_provider = AIProvider(genai_config["provider"].lower())
                if new_provider != self.config.provider:
                    config_overrides["provider"] = new_provider
                    # We need to resolve the key for this new provider
                    config_overrides["api_key"] = connector_genai.resolve_api_key_for_provider(new_provider.value)
            except ValueError:
                logger.warning(f"Unknown provider in task: {genai_config['provider']}")
        
        # Capture all other unknown parameters from task config for extra kwargs
        # (e.g. top_p, top_k, stop, frequency_penalty, etc.)
        known_keys = {"temperature", "max_tokens", "model", "provider"}
        for k, v in genai_config.items():
            if k not in known_keys:
                config_overrides[k] = v

        # Validate Input if schema exists and library is available
        if input_schema:
            if HAS_JSONSCHEMA:
                try:
                    jsonschema.validate(instance=data, schema=input_schema)  # type: ignore[possibly-undefined]
                    logger.info("Input data validation successful.")
                except jsonschema.ValidationError as ve:  # type: ignore[possibly-undefined]
                    logger.error(f"Input data validation failed: {ve.message}")
                    raise ValueError(f"Input data does not match task input_schema: {ve.message}")
            else:
                logger.debug("Skipping input validation (jsonschema not installed).")

        # ====================================================================
        # Use jinjapromptpy for prompt building with automatic batching
        # ====================================================================
        
        # Initialize the prompt builder
        prompt_builder = TaskPromptBuilder()
        
        # Determine max tokens for batching
        max_tokens = config_overrides.get("max_tokens", self.config.max_tokens)
        
        # Build the Jinja2 template from user_prompt_template
        # jinjapromptpy handles batching automatically based on token limits
        user_prompts = prompt_builder.from_task_definition(
            task_def=task_def,
            data=data,
            max_tokens=max_tokens,
            **data_inputs  # Pass additional named variables (categories, context, etc.)
        )
        
        logger.info(f"Generated {len(user_prompts)} prompt batch(es) using jinjapromptpy")
        
        # Execute each prompt batch
        def execute_prompt(user_content: str) -> str:
            """Execute a single prompt with the AI."""
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": user_content})

            if assistant_prompt:
                messages.append({"role": "assistant", "content": assistant_prompt})

            return self.execute(messages, output_schema=output_schema, config_overrides=config_overrides)
        
        # Execute all batches
        if len(user_prompts) == 1:
            # Single prompt - direct execution
            return execute_prompt(user_prompts[0])
        else:
            # Multiple prompts - batch execution with result merging
            results = []
            for i, prompt in enumerate(user_prompts):
                logger.info(f"Executing batch {i + 1}/{len(user_prompts)}...")
                try:
                    result = execute_prompt(prompt)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing batch {i + 1}: {e}")
                    raise
            
            # Merge results based on format
            output_fmt = config_overrides.get("response_format", "json" if output_schema else "text")
            return self._merge_batched_results(results, output_fmt)


