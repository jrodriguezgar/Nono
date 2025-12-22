# -*- coding: utf-8 -*-
"""
A module for connecting to various generative AI services
through a unified interface. It includes auto-installation for
required packages.
"""

import sys
import subprocess
import ssl
import time
from abc import ABC, abstractmethod
from enum import Enum

import json


# --- Dependency Management Utility ---

def install_library(library_name: str) -> bool:
    """
    Checks if a library is installed and, if not, attempts to install it via pip.

    This function makes the module more user-friendly by handling missing
    dependencies automatically.

    Args:
        library_name (str): The name of the library to check and install.

    Returns:
        bool: True if the library is available or was successfully installed,
              False otherwise.
    """
    try:
        # Why try to import first?
        # It's the quickest way to check if the library is already available.
        __import__(library_name)
    except ImportError:
        print(f"Library '{library_name}' not found. Attempting to install...")
        try:
            # Why use sys.executable?
            # It ensures that pip is run for the same Python interpreter that is
            # executing our script, avoiding environment mismatches.
            subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])
            print(f"Library '{library_name}' installed successfully.")
        except subprocess.CalledProcessError:
            print(f"Error: Failed to install '{library_name}'.")
            return False
    return True


# Ensure required libraries are installed before importing
for lib in ["json", "urllib3", "requests"]:
    install_library(lib)


# --- Initial Setup ---

# Disable SSL certificate verification globally.
# WARNING: This disables SSL certificate verification for all HTTPS connections
# made using the default context. This is generally NOT recommended for
# production environments due to significant security risks.
# It should only be used in controlled development or testing environments
# where the risks are understood and accepted.
ssl._create_default_https_context = ssl._create_unverified_context

# Ensure the 'requests' library is available before proceeding.
if install_library("requests"):
    import requests
    import urllib3 # Import urllib3 for warning suppression
    HTTP_SESSION = requests.Session()
    # Why disable warnings?
    # Disabling SSL warnings can be useful in development or specific environments
    # where self-signed certificates are used and verification is not strictly
    # necessary, to avoid verbose output. In production, consider proper SSL verification.
    requests.packages.urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
else:
    # If requests cannot be installed, the module is non-functional.
    raise ImportError("The 'requests' library is required and could not be installed.")


# --- JSON Schema Conversion ---

def convert_json_schema(input_schema: dict, output_title: str = "perplexity") -> dict:
    """Converts a simplified JSON schema to a more detailed one, with a customizable output title.

    This function takes a basic JSON schema and transforms it into a schema that includes
    'title' for each property and a 'title' for the main object. The title for the
    main object can be specified as an argument.

    Args:
        input_schema (dict): The input JSON schema in the simplified format.
                             Example:
                             {
                                 "type": "object",
                                 "properties": {
                                     "name": {"type": "string"},
                                     "address": {"type": "string"}
                                 },
                                 "required": ["name"]
                             }
        output_title (str, optional): The desired title for the main object in the output schema.
                                      Defaults to "converted_object".

    Returns:
        dict: The converted JSON schema with added 'title' fields.
              Example:
              {
                  "properties": {
                      "name": {"title": "name", "type": "string"},
                      "address": {"title": "address", "type": "string"}
                  },
                  "required": ["name"],
                  "title": "MyCustomSchemaTitle",
                  "type": "object"
              }

    Raises:
        ValueError: If the input_schema does not have a 'properties' key.

    Cost:
        O(N), where N is the number of properties in the input schema.
        This is because the function iterates through each property once.
    """
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
    """
    Defines the desired output format for the generative AI completion.
    """
    TEXT = "text"
    TABLE = "table" # Markdown table format
    XML = "xml"
    JSON = "json"


# --- Base Class ---

class GenerativeAIService(ABC):
    """
    An abstract base class for interacting with a generative AI service.

    This class defines the common interface that all concrete service
    implementations must adhere to, ensuring consistent method signatures
    for initialization and content generation. It also handles model validation,
    prompt length checks, and request delays.
    """

    # Class attribute to store available models for each service.
    # This must be overridden by concrete implementations.
    AVAILABLE_MODELS: list[str] = []

    # Temperature recommendations for different use cases
    TEMPERATURE_RECOMMENDATIONS = {
        "coding": 0.0,        # Precise, deterministic responses for code
        "math": 0.0,         # Exact calculations and mathematical operations
        "data_cleaning": 1.0, # Balance between consistency and adaptability
        "data_analysis": 1.0, # Balance for analytical tasks
        "conversation": 1.3,  # Natural language interaction
        "translation": 1.3,   # Language translation tasks
        "creative": 1.5,      # Creative and diverse outputs
        "poetry": 1.5        # Artistic and varied expressions
    }

    @classmethod
    def get_recommended_temperature(cls, use_case: str) -> float:
        """
        Get the recommended temperature setting for a specific use case.

        Args:
            use_case (str): The type of task being performed. Valid options:
                - 'coding', 'math': Precise, deterministic outputs (temp=0.0)
                - 'data_cleaning', 'data_analysis': Balanced outputs (temp=1.0)
                - 'conversation', 'translation': Natural language (temp=1.3)
                - 'creative', 'poetry': Creative writing (temp=1.5)

        Returns:
            float: The recommended temperature value for the use case.
                  Defaults to 0.7 if use case is not recognized.
        """
        use_case = use_case.lower()
        return cls.TEMPERATURE_RECOMMENDATIONS.get(use_case, 0.7)

    def __init__(self, model_name: str, max_input_chars: int,
                 delay_between_requests: float = 0.0, api_key: str | None = None):
        if not model_name:
            raise ValueError("A model_name must be provided.")
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not supported by this service. "
                f"Available models are: {', '.join(self.AVAILABLE_MODELS)}"
            )

        self.model_name = model_name
        self.api_key = api_key
        self.max_input_chars = max_input_chars
        self.delay_between_requests = delay_between_requests

    def _validate_messages_length(self, messages: list[dict[str, str]]) -> None:
        """
        Validates if the total length of messages exceeds the maximum allowed characters for the model.

        While generative AI models typically operate on token counts,
        this check uses character count as a simplified approximation
        to prevent excessively large inputs that might incur high costs
        or lead to API errors.

        Args:
            messages (list[dict[str, str]]): A list of message dictionaries,
                                             each with 'role' and 'content' keys.

        Raises:
            ValueError: If the total message length exceeds the model's maximum input characters.
        """
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
        """
        Adjusts the messages to instruct the model to produce the desired output format,
        primarily by modifying the last 'user' message's content.

        Args:
            messages (list[dict[str, str]]): The input list of message dictionaries.
            response_format (ResponseFormat): The desired format for the model's output.
            json_schema (dict | None): An optional JSON schema (as a dictionary) to guide
                                       the model when response_format is JSON.

        Returns:
            list[dict[str, str]]: A new list of messages with formatting instructions appended.

        Raises:
            ValueError: If no user message is found in the input messages for formatting.
        """
        formatted_messages = [msg.copy() for msg in messages] # Create a mutable copy

        # Find the last user message to append formatting instructions
        last_user_message_index = -1
        for i, msg in enumerate(reversed(formatted_messages)):
            if msg.get("role") == "user":
                last_user_message_index = len(formatted_messages) - 1 - i
                break

        if last_user_message_index == -1:
            raise ValueError("No 'user' message found in the input messages to apply formatting instructions.")

        current_content = formatted_messages[last_user_message_index].get("content", "")

        instruction = ""
        if response_format == ResponseFormat.TABLE:
            instruction = "\n\nPlease provide the output in a markdown table format."
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
                    print(f"Warning: Could not serialize JSON schema. Proceeding without schema. Error: {e}")
                    instruction = "\n\nPlease provide the output in JSON format. Ensure the response is valid JSON."
            else:
                instruction = "\n\nPlease provide the output in JSON format. Ensure the response is valid JSON."
        # For TEXT, no special instruction is needed.

        if instruction:
            formatted_messages[last_user_message_index]["content"] = current_content + instruction

        return formatted_messages

    @abstractmethod
    def generate_completion(self, messages: list[dict[str, str]], temperature: float = 0.7,
                            response_format: ResponseFormat = ResponseFormat.JSON,
                            json_schema: dict | None = None,
                            use_case: str | None = None) -> str:
        """
        Abstract method to generate a text completion from the AI service.

        This method must be implemented by concrete service classes.

        Args:
            messages (list[dict[str, str]]): A list of message dictionaries,
                                             each with 'role' (e.g., 'user', 'assistant', 'system')
                                             and 'content' keys.
            temperature (float): Controls the randomness of the output. Higher values
                                 make the output more random, while lower values make it
                                 more focused and deterministic. Some recommended values:
                                 - Coding/Math: 0.0 (precise, deterministic)
                                 - Data Cleaning/Analysis: 1.0 (balanced)
                                 - Conversation/Translation: 1.3 (natural)
                                 - Creative Writing/Poetry: 1.5 (varied)
            response_format (ResponseFormat): The desired format for the model's output.
                                              Defaults to JSON.
            json_schema (dict | None): An optional JSON schema (as a dictionary) to guide
                                       the model when response_format is JSON.
            use_case (str | None): Optional. If provided, the temperature will be automatically
                                  set based on the use case. Valid options:
                                  'coding', 'math', 'data_cleaning', 'data_analysis',
                                  'conversation', 'translation', 'creative', 'poetry'.

        Returns:
            str: The generated text completion, formatted as requested.

        Raises:
            Exception: Propagates any exceptions encountered during API interaction.
        """
        pass
    

# --- OpenAI Compatible Services ---

class OpenAICompatibleService(GenerativeAIService):
    """
    A base class for services that use an OpenAI-compatible API.
    """
    def __init__(self, model_name: str, api_key: str, base_url: str,
                 max_input_chars: int, delay_between_requests: float = 0.0):

        super().__init__(model_name, max_input_chars, delay_between_requests, api_key)
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_completion(self, messages: list[dict[str, str]], temperature: float = 0.7,
                            response_format: ResponseFormat = ResponseFormat.JSON,
                            json_schema: dict | None = None) -> str:
        """
        Generates a text completion using an OpenAI-compatible API.

        Args:
            messages (list[dict[str, str]]): A list of message dictionaries,
                                             each with 'role' and 'content' keys.
            temperature (float): Controls the randomness of the output.
            response_format (ResponseFormat): The desired format for the model's output.
            json_schema (dict | None): An optional JSON schema (as a dictionary) to guide
                                       the model when response_format is JSON.

        Returns:
            str: The generated text completion.

        Raises:
            ValueError: If the total message length exceeds the maximum allowed characters.
            requests.exceptions.RequestException: For network-related errors or bad HTTP responses.
            KeyError, IndexError: If the API response format is unexpected.
        """
        # For OpenAI-compatible services, we embed the JSON schema into the prompt if requested,
        # and also set the response_format in the payload to 'json_object' if applicable.
        formatted_messages = self._format_messages_for_response(messages, response_format, json_schema)
        self._validate_messages_length(formatted_messages)

        # Why add a delay?
        # To respect API rate limits and prevent flooding the service,
        # ensuring more stable and reliable interactions.
        time.sleep(self.delay_between_requests)

        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": temperature
        }

        # Add native response_format field to payload for JSON, if supported.
        # OpenAI supports "json_object" for any JSON output.
        if response_format == ResponseFormat.JSON:
            payload["response_format"] = {"type": "json_object"}

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
    """
    Connects to the official OpenAI API.
    """
    AVAILABLE_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    # Estimated maximum characters. Real limits are token-based and vary.
    # gpt-4o: ~128k tokens, gpt-3.5-turbo: ~16k tokens.
    _MAX_INPUT_CHARS = {
        "gpt-4o": 500_000,
        "gpt-4o-mini": 120_000,
        "gpt-4-turbo": 500_000,
        "gpt-3.5-turbo": 60_000
    }

    def __init__(self, model_name: str | None = None, api_key: str | None = None,
                 delay_between_requests: float | None = None):

        super().__init__(
            model_name,
            api_key,
            "https://api.openai.com/v1",
            self._MAX_INPUT_CHARS.get(model_name, 60_000), # Default to a safe limit
            delay_between_requests
        )


class PerplexityService(OpenAICompatibleService):
    """
    Connects to the Perplexity API.
    """
    AVAILABLE_MODELS: list[str] = [
        "sonar",
        "llama-3-sonar-large-32k-online",
        "llama-3-sonar-small-32k-online",
        "llama-3-8b-instruct",
        "llama-3-70b-instruct",
        "mixtral-8x7b-instruct",
        "mistral-7b-instruct"
    ]
    # Estimated maximum characters. Real limits are token-based.
    # llama-3-sonar-large-32k-online: 32k tokens.
    _MAX_INPUT_CHARS: dict[str, int] = {
        "sonar": 120_000,
        "llama-3-sonar-large-32k-online": 120_000,
        "llama-3-sonar-small-32k-online": 120_000,
        "llama-3-8b-instruct": 30_000, # Assuming 8k context
        "llama-3-70b-instruct": 120_000, # Assuming 32k context
        "mixtral-8x7b-instruct": 120_000, # Assuming 32k context
        "mistral-7b-instruct": 30_000 # Assuming 8k context
    }

    def __init__(self, model_name: str | None = None, api_key: str | None = None,
                 delay_between_requests: float | None = None) -> None:
        """
        Initializes the PerplexityService.

        Args:
            model_name (str): The name of the Perplexity model to use (default is "llama-3-sonar-large-32k-online").
            api_key (str | None): The Perplexity API key. If None, it will be resolved from environment variable.
            delay_between_requests (float): The delay in seconds between consecutive requests.
        """

        super().__init__(
            model_name,
            api_key,
            "https://api.perplexity.ai",
            self._MAX_INPUT_CHARS.get(model_name, 120_000),
            delay_between_requests
        )

    def generate_completion(self, messages: list[dict[str, str]], temperature: float = 0.7,
                            response_format: ResponseFormat = ResponseFormat.JSON,
                            json_schema: dict | None = None) -> str:
        """
        Generates a text completion using the Perplexity API, with specific handling
        for JSON schema in the payload.

        Args:
            messages (list[dict[str, str]]): A list of message dictionaries.
            temperature (float): Controls the randomness of the output.
            response_format (ResponseFormat): The desired format for the model's output.
            json_schema (dict | None): An optional JSON schema (as a dictionary) to guide
                                       the model when response_format is JSON.

        Returns:
            str: The generated text completion.

        Raises:
            ValueError: If the total message length exceeds the maximum allowed characters.
            requests.exceptions.RequestException: For network-related errors or bad HTTP responses.
            KeyError, IndexError: If the API response format is unexpected.
        """
        # Perplexity has a specific way to handle JSON schema in the payload.
        # So, we only format the messages for non-JSON formats or generic JSON
        # (where schema is not directly in payload).
        if response_format == ResponseFormat.JSON and json_schema:
            # For Perplexity, the schema goes directly into the payload.
            # No need to embed schema in prompt for this specific case.
            processed_messages = [msg.copy() for msg in messages] # Just copy messages
        else:
            # For other formats or generic JSON, use the base class's message formatting.
            processed_messages = self._format_messages_for_response(messages, response_format, json_schema)

        # Ensure a system message is present for Perplexity if not already in messages
        # Why add a system message?
        # Perplexity's API often expects a system message, even if empty, to properly
        # contextualize the conversation.
        if not any(msg.get("role") == "system" for msg in processed_messages):
            processed_messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})

        self._validate_messages_length(processed_messages)

        time.sleep(self.delay_between_requests)

        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": temperature
        }

        # Handle Perplexity's specific response_format for JSON schema
        if response_format == ResponseFormat.JSON:
            if json_schema:
                json_schema = convert_json_schema(json_schema)

                # Perplexity's specific JSON schema format from user example
                # The user's earlier working example used {"schema": p_format.model_json_schema()}
                # This translates to {"schema": json_schema} if json_schema is already the dict
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"schema": json_schema} # Correct for Perplexity
                }
            else:
                # Generic JSON object for Perplexity if no specific schema is provided
                payload["response_format"] = {"type": "json_object"}

        try:
            response = HTTP_SESSION.post(endpoint, headers=self.headers, json=payload, timeout=90, verify=False)
            response.raise_for_status()
            data = response.json()
            # Perplexity might wrap JSON in ```json blocks even with native formatting.
            content = data['choices'][0]['message']['content'].strip()
            # Attempt to extract content if it's wrapped in markdown code block
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
    """
    Connects to the DeepSeek API.
    """
    # [https://api-docs.deepseek.com/](https://api-docs.deepseek.com/)
    AVAILABLE_MODELS = ["deepseek-chat", "deepseek-coder"]
    # Estimated maximum characters. Real limits are token-based.
    # deepseek-chat/deepseek-coder: 128k context.
    _MAX_INPUT_CHARS = {
        "deepseek-chat": 500_000,
        "deepseek-coder": 500_000
    }

    def __init__(self, model_name: str | None = None, api_key: str | None = None,
                 delay_between_requests: float | None = None):

        super().__init__(
            model_name,
            api_key,
            "[https://api.deepseek.com](https://api.deepseek.com)", # Corrected base URL
            self._MAX_INPUT_CHARS.get(model_name, 500_000),
            delay_between_requests
        )


class GrokService(OpenAICompatibleService):
    """
    Connects to the Grok API (via xAI).
    """
    AVAILABLE_MODELS = ["grok-1"]
    # Estimated maximum characters for Grok-1 (8k context).
    _MAX_INPUT_CHARS = {"grok-1": 30_000}

    def __init__(self, model_name: str | None = None, api_key: str | None = None,
                 delay_between_requests: float | None = None):

        super().__init__(
            model_name,
            api_key,
            "[https://api.x.ai/v1](https://api.x.ai/v1)",
            self._MAX_INPUT_CHARS.get(model_name, 30_000),
            delay_between_requests
        )


class GeminiService(GenerativeAIService):
    """
    Connects to the Google Gemini API, auto-installing the required library.
    """
    AVAILABLE_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro-latest", "gemini-pro"]
    # Estimated maximum characters. Real limits are token-based and vary.
    # gemini-1.5 models: 1M tokens. gemini-pro: 32k tokens.
    _MAX_INPUT_CHARS = {
        "gemini-1.5-flash": 4_000_000, # Very large context
        "gemini-1.5-pro-latest": 4_000_000, # Very large context
        "gemini-pro": 120_000
    }
    # Default delay for Gemini to respect 15 requests/minute rate limit (60/15 = 4 seconds).
    _DEFAULT_DELAY = 4.0

    def __init__(self, model_name: str | None = None, api_key: str | None = None,
                 delay_between_requests: float | None = None):

        super().__init__(
            model_name,
            self._MAX_INPUT_CHARS.get(model_name, 120_000), # Default to gemini-pro limit
            delay_between_requests,
            api_key
        )

        # Why call install_library here?
        # This approach ensures the dependency is only installed when this specific
        # class is used, avoiding unnecessary installations if only other
        # services (like OpenAI) are being utilized.
        if not install_library("google-generativeai"):
            raise ImportError(
                "The 'google-generativeai' library is required for GeminiService "
                "and could not be installed. Please install it manually."
            )

        import google.generativeai as genai
        self.genai_client = genai
        genai.configure(api_key=api_key) # Configure API key here
        self.client = genai.GenerativeModel(self.model_name)

    def generate_completion(self, messages: list[dict[str, str]], temperature: float = 0.7,
                            response_format: ResponseFormat = ResponseFormat.JSON,
                            json_schema: dict | None = None) -> str:
        """
        Generates a text completion using the Google Gemini API.

        Args:
            messages (list[dict[str, str]]): A list of message dictionaries.
            temperature (float): Controls the randomness of the output.
            response_format (ResponseFormat): The desired format for the model's output.
            json_schema (dict | None): An optional JSON schema (as a dictionary) to guide
                                       the model when response_format is JSON.

        Returns:
            str: The generated text completion.

        Raises:
            ValueError: If the total message length exceeds the maximum allowed characters.
            Exception: For any errors encountered with the Gemini API.
        """
        generation_config = {"temperature": temperature}
        processed_messages_for_api = []
        system_instruction_content = ""

        # Separate system messages and extract their content
        user_and_assistant_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                # Why extract system messages?
                # Gemini's `generate_content` method does not directly support
                # a 'system' role in the messages list. System instructions
                # are typically handled by prepending them to the first user
                # turn or through a specific API parameter if available.
                system_instruction_content += msg.get("content", "") + "\n"
            else:
                user_and_assistant_messages.append(msg.copy()) # Copy to avoid modifying original

        # Apply formatting for response type, if not native JSON schema handling
        if response_format == ResponseFormat.JSON and json_schema:
            # When using native schema, the messages do not need prompt-based formatting for the schema.
            processed_messages_for_api = user_and_assistant_messages
            generation_config["response_mime_type"] = "application/json"
            generation_config["response_schema"] = json_schema
        else:
            # For other formats or generic JSON, use prompt engineering.
            # Use user_and_assistant_messages as base for formatting.
            processed_messages_for_api = self._format_messages_for_response(user_and_assistant_messages, response_format, json_schema)
            if response_format == ResponseFormat.JSON:
                generation_config["response_mime_type"] = "application/json"


        # Prepend system instruction to the first user message, if any.
        # Why prepend to the first user message?
        # This is a common workaround when the API does not support a dedicated
        # 'system' role. It contextualizes the user's intent with the system's
        # instructions.
        if system_instruction_content:
            found_first_user_message = False
            for msg in processed_messages_for_api:
                if msg.get("role") == "user":
                    msg["content"] = system_instruction_content.strip() + "\n\n" + msg.get("content", "")
                    found_first_user_message = True
                    break
            if not found_first_user_message:
                # If there are no user messages, but there was a system message,
                # create a user message with the system content. This case might be unusual
                # but ensures system content is not lost.
                processed_messages_for_api.insert(0, {"role": "user", "content": system_instruction_content.strip()})


        # Gemini API expects 'parts' key for content, not 'content'
        # Convert messages from {'role': ..., 'content': ...} to {'role': ..., 'parts': [{'text': ...}]}
        gemini_formatted_messages = []
        for msg in processed_messages_for_api:
            role = msg.get("role")
            content = msg.get("content")
            if role and content:
                # Map roles: 'user' -> 'user', 'assistant' -> 'model'
                mapped_role = "user" if role == "user" else "model"
                gemini_formatted_messages.append({"role": mapped_role, "parts": [{"text": content}]})

        self._validate_messages_length(gemini_formatted_messages) # Validate after Gemini formatting to get accurate length

        if not self.delay_between_requests:
            self.delay_between_requests = GeminiService._DEFAULT_DELAY
        time.sleep(self.delay_between_requests)

        try:
            response = self.client.generate_content(gemini_formatted_messages, generation_config=generation_config)
            return response.text
        except Exception as e:
            print(f"An error occurred with the Gemini API: {e}")
            raise

class OllamaService(GenerativeAIService):
    """
    Connects to a local Ollama instance.
    """
    # Ollama models are typically downloaded locally, so this list serves as examples.
    AVAILABLE_MODELS = [
        "llama3", "llama2", "mistral", "mixtral", "gemma", "codellama", "dolphin-mistral"
    ]
    # Ollama models vary greatly in context size. This is a general large limit.
    _MAX_INPUT_CHARS = 200_000

    def __init__(self, model_name: str | None = None, host: str | None = None,
                 delay_between_requests: float | None = None):

        # Ollama doesn't typically use an API key for local instances.
        super().__init__(model_name, self._MAX_INPUT_CHARS, delay_between_requests, api_key=None)
        self.host = host.rstrip('/')

    def generate_completion(self, messages: list[dict[str, str]], temperature: float = 0.7,
                            response_format: ResponseFormat = ResponseFormat.JSON,
                            json_schema: dict | None = None) -> str:
        """
        Generates a text completion using a local Ollama instance.

        Args:
            messages (list[dict[str, str]]): A list of message dictionaries.
            temperature (float): Controls the randomness of the output.
            response_format (ResponseFormat): The desired format for the model's output.
            json_schema (dict | None): An optional JSON schema (as a dictionary) to guide
                                       the model when response_format is JSON.

        Returns:
            str: The generated text completion.

        Raises:
            ValueError: If the total message length exceeds the maximum allowed characters.
            requests.exceptions.RequestException: For network-related errors (e.g., Ollama not running) or bad HTTP responses.
            KeyError: If the Ollama response format is unexpected.
        """
        formatted_messages = self._format_messages_for_response(messages, response_format, json_schema)
        self._validate_messages_length(formatted_messages) # Validate length AFTER formatting

        # Why add a delay?
        # To prevent overwhelming the local Ollama instance, especially if running
        # resource-intensive models or multiple concurrent requests.
        time.sleep(self.delay_between_requests)

        endpoint = f"{self.host}/api/chat" # Ollama uses /api/chat for message lists
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "stream": False,
            "options": {"temperature": temperature}
        }
        try:
            response = HTTP_SESSION.post(endpoint, json=payload, timeout=180, verify=False)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "").strip() # Ollama chat response format
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama at {self.host}. Is it running? Error: {e}")
            raise
        except KeyError as e:
            print(f"Unexpected response format from Ollama: {e}")
            raise

