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
    
    GEMINI = "gemini"
    OPENAI = "openai"
    PERPLEXITY = "perplexity"
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


# Import the connector_genai module
try:
    from .connector import connector_genai
except ImportError:
    # Fallback for when running as script depending on path
    try:
        from connector import connector_genai
    except ImportError:
         # Try absolute path for safety if run from root
         from nono.genai_tasker.connector import connector_genai

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
    
    Attributes:
        default_task_definition: Loaded task definition from initialization.
        config: AIConfiguration instance for the executor.
        client: Active AI client based on the configured provider.
    """
    
    def __init__(
        self,
        task_or_config_file: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> None:
        """
        Initialize the TaskExecutor.
        
        Args:
            task_or_config_file: Path to task definition or config JSON file.
            api_key: Optional API key override.
        """
        self.default_task_definition: Optional[Dict[str, Any]] = None
        self.config = self._load_config(task_or_config_file, api_key)
        self.client = self._create_client()

    def _load_config(self, path: Optional[str], api_key: Optional[str]) -> AIConfiguration:
        """Load configuration from file or use defaults."""
        data: Dict[str, Any] = {}
        if path:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        loaded_json = json.load(f)
                        
                        # Check if it is a Task Definition (has "task" and "genai" sections)
                        if "task" in loaded_json and "genai" in loaded_json:
                            self.default_task_definition = loaded_json
                            # Use the genai section as the base configuration
                            data = loaded_json["genai"]
                            logger.info(f"Loaded Task Definition from {path}")
                        else:
                            # Treat as standard configuration file
                            data = loaded_json
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in configuration file: {path}")
                    raise
            else:
                logger.warning(f"Configuration file not found: {path}. Using defaults.")
        
        try:
            # Determine provider (default to Gemini if not set)
            # Support both 'provider' (std) and 'ai_service' (legacy)
            provider_str = data.get("provider", data.get("ai_service", "gemini")).lower()
            try:
                provider = AIProvider(provider_str)
            except ValueError:
                logger.warning(f"Unknown provider '{provider_str}', defaulting to GEMINI")
                provider = AIProvider.GEMINI

            # Resolve API Key based on provider
            # env_var_map removed as environment variables support is disabled
            
            final_api_key = api_key
            
            # Ollama does not strictly require an API key if running locally
            if provider == AIProvider.OLLAMA and not final_api_key:
                 final_api_key = "ollama-local"

            # Fallback: Try to read from a local file if needed (specific to this project structure)
            if not final_api_key:
                # Try provider specific key file
                key_files = [f"{provider.value}_api_key.txt", "google_ai_api_key.txt", "apikey.txt"]
                
                search_dirs = []
                if path:
                     search_dirs.append(os.path.dirname(os.path.dirname(path)))
                
                # Always check relative to module location
                current_module_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_module_dir)
                search_dirs.append(project_root)

                for search_dir in search_dirs:
                    for kf_name in key_files:
                        key_path = os.path.join(search_dir, kf_name)
                        if os.path.exists(key_path):
                            with open(key_path, 'r') as kf:
                                final_api_key = kf.read().strip()
                            break
                    if final_api_key:
                        break

            if not final_api_key:
                raise ValueError(f"API Key must be provided via argument or key file.")
            
            # Handle temperature preset (string) or float
            temp_val = data.get("temperature", 0.7)
            final_temperature = 0.7
            if isinstance(temp_val, str):
                try:
                     final_temperature = connector_genai.GenerativeAIService.get_recommended_temperature(temp_val)
                except Exception:
                     logger.warning(f"Unknown temperature preset '{temp_val}', using default 0.7")
            else:
                 final_temperature = float(temp_val)

            # Resolve Model Name: Support 'model' (std) and 'model_name' (legacy)
            model_val = data.get("model", data.get("model_name", "gemini-1.5-flash"))

            # Resolve Max Tokens: Support 'max_tokens' (std) and 'max_prompt_length' (legacy)
            max_tokens_val = data.get("max_tokens", data.get("max_prompt_length", 2048))

            return AIConfiguration(
                provider=provider,
                model_name=model_val,
                api_key=final_api_key,
                temperature=final_temperature,
                max_tokens=max_tokens_val,
                system_instruction=data.get("system_instruction", None)
            )
        except Exception as e:
             logger.error(f"Error loading configuration: {e}")
             raise

    def _create_client(self, config: Optional[AIConfiguration] = None) -> BaseAIClient:
        """
        Factory method to instantiate the correct client based on provider.
        
        Args:
            config: Optional configuration override. Uses self.config if None.
        
        Returns:
            Appropriate AI client instance for the provider.
        
        Raises:
            ValueError: If the provider is not supported.
        """
        cfg = config if config else self.config
        
        client_map = {
            AIProvider.GEMINI: GeminiClient,
            AIProvider.OPENAI: OpenAIClient,
            AIProvider.PERPLEXITY: PerplexityClient,
            AIProvider.OLLAMA: OllamaClient,
        }
        
        client_class = client_map.get(cfg.provider)
        if client_class is None:
            raise ValueError(f"Unsupported provider: {cfg.provider}")
        
        return client_class(cfg)

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
        
        # Check if we need to switch temporary client due to provider change
        active_client = self.client
        if config_overrides and "provider" in config_overrides:
             if config_overrides["provider"] != self.config.provider:
                 logger.info(f"Switching provider for this task: {self.config.provider.value} -> {config_overrides['provider'].value}")
                 
                 # Create temporary config for provider switch
                 temp_config = AIConfiguration(
                     provider=config_overrides["provider"],
                     model_name=config_overrides.get("model_name", "default"),
                     api_key=config_overrides.get("api_key", ""),
                     temperature=config_overrides.get("temperature", self.config.temperature),
                     max_tokens=config_overrides.get("max_tokens", self.config.max_tokens),
                     system_instruction=self.config.system_instruction
                 )
                 active_client = self._create_client(temp_config)

        return active_client.generate_content(input_data, output_schema, config_overrides)

    def _resolve_api_key_for_provider(self, provider: AIProvider) -> str:
        """
        Find API key for a specific provider dynamically.
        
        Args:
            provider: The AI provider to find the key for.
        
        Returns:
            API key string, or empty string if not found.
        """
        if provider == AIProvider.OLLAMA:
            return "ollama-local"
        
        # Similar logic to _load_config but simplified for dynamic load
        # We assume standard paths relative to current file or project root
        key_files = [f"{provider.value}_api_key.txt", "apikey.txt"]
        if provider == AIProvider.GEMINI: key_files.append("google_ai_api_key.txt")
        
        # Search paths: 1. same dir as config (if we knew it), 2. project root (nono/)
        # We'll assume project root is parent of genai_tasker
        current_dir = os.path.dirname(os.path.abspath(__file__)) # genai_tasker/
        project_root = os.path.dirname(current_dir) # nono/
        
        for kf in key_files:
            p = os.path.join(project_root, kf)
            if os.path.exists(p):
                with open(p, 'r') as f: return f.read().strip()
                
        # If not found, maybe check env vars or raise
        logger.warning(f"Could not find API key file for {provider.value} in {project_root}")
        return ""

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
            AIProvider.GEMINI: connector_genai.GeminiService,
            AIProvider.OPENAI: connector_genai.OpenAIService,
            AIProvider.PERPLEXITY: connector_genai.PerplexityService,
            AIProvider.OLLAMA: connector_genai.OllamaService,
        }
        return service_map.get(provider)

    def run_json_task(
        self,
        task_source: Union[str, Any],
        data_input: Optional[Any] = None,
        **data_inputs: Any
    ) -> str:
        """
        Execute a task defined in a JSON file with automatic batching support.
        
        Supports two usage patterns:
            - run_json_task("path/to/task.json", data): Load task from file.
            - run_json_task(data): Use task definition from __init__.
        
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
            task_source: Path to task JSON file or data (if task bound at init).
            data_input: Primary input data for the task (maps to {data_input_json}).
            **data_inputs: Additional named inputs (maps to {name} placeholders).
        
        Returns:
            Task execution result as a string.
        
        Raises:
            FileNotFoundError: If task file does not exist.
            ValueError: If no valid task definition is available.
        """
        task_def = None
        data = None

        if isinstance(task_source, str) and (task_source.endswith('.json') or os.path.exists(task_source)):
             # Usage 1: Path provided
             if not os.path.exists(task_source):
                  raise FileNotFoundError(f"Task file not found: {task_source}")
             with open(task_source, 'r', encoding='utf-8') as f:
                task_def = json.load(f)
             data = data_input
        else:
             # Usage 2: Data provided as first arg, or task_source is not a file path
             if self.default_task_definition:
                 task_def = self.default_task_definition
                 data = task_source # First arg is the data
             else:
                 # Check if user reversed args accidentally or passed bad path
                 if data_input is not None:
                      # If two args, and first isn't valid file, assume it's data_input? No, standard is path, data.
                      raise ValueError("First argument must be a valid task file path.")
                 else:
                      raise ValueError("No task definition file loaded in init, and no valid file path provided.")

        if task_def is None:
             raise ValueError("Failed to load task definition.")

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
                    config_overrides["api_key"] = self._resolve_api_key_for_provider(new_provider)
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
                    jsonschema.validate(instance=data, schema=input_schema)
                    logger.info("Input data validation successful.")
                except jsonschema.ValidationError as ve:
                    logger.error(f"Input data validation failed: {ve.message}")
                    raise ValueError(f"Input data does not match task input_schema: {ve.message}")
            else:
                logger.debug("Skipping input validation (jsonschema not installed).")

        # Prepare content function to handle batching
        def execute_chunk(chunk_data):
            # Use compact JSON to match size calculation and save tokens
            data_str = json.dumps(chunk_data, indent=None, ensure_ascii=False, separators=(',', ':'))
            user_content = user_prompt_template.replace("{data_input_json}", data_str)
            
            # Replace additional named placeholders from data_inputs
            for placeholder_name, placeholder_value in data_inputs.items():
                placeholder_key = "{" + placeholder_name + "}"
                if placeholder_key in user_content:
                    # Convert value to JSON string if it's a complex type
                    if isinstance(placeholder_value, (dict, list)):
                        value_str = json.dumps(placeholder_value, indent=None, ensure_ascii=False, separators=(',', ':'))
                    else:
                        value_str = str(placeholder_value)
                    user_content = user_content.replace(placeholder_key, value_str)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": user_content})

            if assistant_prompt:
                messages.append({"role": "assistant", "content": assistant_prompt})

            return self.execute(messages, output_schema=output_schema, config_overrides=config_overrides)

        # Automatic Batching / Throttling logic
        # 1. Check if user specified explicit batch_size
        batch_size = genai_config.get("batch_size", 0)
        
        # 2. Smart Throttling: If no batch_size, calculate using model context window
        if batch_size <= 0 and isinstance(data, list) and data:
             # Identify effective provider and model
             eff_provider = config_overrides.get("provider", self.config.provider)
             eff_model = config_overrides.get("model_name", self.config.model_name)
             
             svc_class = self._get_active_service_class(eff_provider)
             if svc_class:
                 max_chars = svc_class.get_max_input_chars(eff_model)
                 
                 # Calculate Static Prompt Overhead
                 # (approximation of system + user template + extra json overhead)
                 # We use a safety buffer of 0.9 (10% reserve)
                 overhead = len(system_prompt) + len(user_prompt_template) + len(assistant_prompt) + 100
                 available_chars = int((max_chars * 0.9) - overhead)
                 
                 if available_chars > 100: # Ensure we have some space
                     logger.info(f"Auto-Throttling enabled. Max Chars: {max_chars}, Available: {available_chars}")
                     
                     results = []
                     current_batch = []
                     current_batch_size = 2 # Start with 2 for brackets []
                     
                     for item in data:
                         # Serialize item exactly as it will appear in the chunk (unindented, compact separators)
                         item_str = json.dumps(item, ensure_ascii=False, separators=(',', ':'))
                         
                         # Check if item itself fits (plus comma overhead if not first)
                         comma_overhead = 1 if current_batch else 0
                         item_total_len = len(item_str) + comma_overhead
                         
                         # If adding this item exceeds available space...
                         if current_batch_size + item_total_len > available_chars:
                             if current_batch:
                                 # Execute current batch (Guarantee: items are never split, only grouped)
                                 logger.info(f"Executing Batch: {len(current_batch)} items (Size: {current_batch_size})")
                                 results.append(execute_chunk(current_batch))
                                 current_batch = []
                                 current_batch_size = 2 # Reset
                                 comma_overhead = 0 # Reset overhead for first item of new batch
                                 item_total_len = len(item_str) # Recalculate without comma
                             
                             # If a single item is too big even for an empty batch
                             if item_total_len + 2 > available_chars: # +2 for brackets
                                 logger.warning(f"Single item size ({item_total_len}) exceeds context window ({available_chars}). Sending anyway (truncation risk).")
                         
                         current_batch.append(item)
                         current_batch_size += item_total_len
                     
                     # Process final batch
                     if current_batch:
                         logger.info(f"Executing Final Batch: {len(current_batch)} items")
                         results.append(execute_chunk(current_batch))
                     
                     # Merge results based on format
                     output_fmt = config_overrides.get("response_format", "json" if output_schema else "text")
                     return self._merge_batched_results(results, output_fmt)

        if batch_size > 0 and isinstance(data, list):
            logger.info(f"Executing task with batch size: {batch_size}")
            results = []
            total_items = len(data)
            
            for i in range(0, total_items, batch_size):
                chunk = data[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(total_items + batch_size - 1)//batch_size} ({len(chunk)} items)...")
                try:
                    res_chunk = execute_chunk(chunk)
                    results.append(res_chunk)
                except Exception as e:
                    logger.error(f"Error processing batch starting at index {i}: {e}")
                    # Policy: Continue or Raise? Defaulting to Raise to ensure data integrity
                    raise
            
            # Merge results based on format
            output_fmt = config_overrides.get("response_format", "json" if output_schema else "text")
            return self._merge_batched_results(results, output_fmt)
        
        # Standard execution (single batch)
        return execute_chunk(data)


