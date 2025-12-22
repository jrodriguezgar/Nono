#standards for all modules/apps
print("-------------------------------------------------------------")
#app info
import os
import sys
import os, sys
def add_to_syspath(directory):
    def search(start):
        root = os.path.abspath(os.sep)
        while True:
            path = os.path.join(start, directory)
            if os.path.isdir(path):
                sys.path.insert(0, path) if path not in sys.path else None
                print(f"Se encontró y agregó a sys.path: {path}")
                return True
            if start.lower() == root.lower(): return False
            start = os.path.dirname(start)
    if search(os.path.abspath(os.getcwd())): return
    try:
        if search(os.path.dirname(os.path.abspath(__file__))): return
    except NameError: print("No se pudo obtener el directorio del módulo.")
    raise FileNotFoundError(f"No se encontró {directory}")
add_to_syspath("app_commons")
print("-------------------------------------------------------------")
import app_info
paths = app_info.get_caller_info()
global APPLICATION_PATH
APPLICATION_PATH = paths["application_path"]
global MODULE_PATH
MODULE_PATH = paths["module_path"]
print(f"Application Path:\n{APPLICATION_PATH}")
print(f"Module Path:\n{MODULE_PATH}")
print("-------------------------------------------------------------")
#environment
import env_manager as env_man
env_man.print_syspath()
print("-------------------------------------------------------------")


"""
This module orchestrates generative AI tasks with proper configuration and input handling.
"""
import json
from typing import Dict, Any, List

#my modules
add_to_syspath("lib")
from lib.fastetl.connectors import connector_json as conn_json
from lib.fastetl.connectors import connector_manager as conn_man
from lib.fastetl.connectors.connector_genai import (
    GeminiService,
    OpenAIService,
    PerplexityService,
    ResponseFormat
)


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Define the missing run_task function
def run_task_one_shot(ai_configuration, task_configuration, input_data, api_key):
    """
    Description:
        Runs a generative AI task in one shot using the GenAITaskRunner class.

    Args:
        ai_configuration (str): Path to the AI configuration JSON file.
        task_configuration (str): Path to the task definition JSON file.
        input_data (List[str]): List of input strings to process.
        api_key (str): API key for the selected AI service.

    Returns:
        str: The formatted output from the AI task.

    Raises:
        Exception: If task execution fails.

    Example Usage:
        output = run_task('./config.json', './task.json', ["John Doe"], "my_api_key")

    Cost:
        O(n), where n is the number of input items.
    """
    task_runner = GenAITaskRunner(ai_configuration, task_configuration, api_key)
    output_data = task_runner.run(input_data)
    return output_data


class GenAITaskRunner:
    """
    Description:
        Service class to initialize and run generative AI tasks using the GenerativeAITask class.
        Encapsulates the initialization and execution logic for easier reuse and integration.

    Args:
        genai_config_file (str): Path to the AI configuration JSON file.
        task_file (str): Path to the task definition JSON file.
        api_key (str): API key for the selected AI service.

    Example Usage:
        service = GenAITaskService(genai_config_file, task_file, api_key)
        results = service.run(input_data)

    Cost:
        O(n), where n is the number of input items processed by the AI task.
    """

    def __init__(self, genai_config_file: str, task_config_file: str, api_key: str):
        """
        Description:
            Initializes the GenAITaskService by creating a GenerativeAITask instance.

        Args:
            genai_config_file (str): Path to the AI configuration JSON file.
            task_file (str): Path to the task definition JSON file.
            api_key (str): API key for the selected AI service.

        Raises:
            Exception: If initialization fails.
        """
        try:
            # Load configurations directly in the constructor
            print("Initializing AI Task Service...")
            connection = conn_man.ConnectionManager()
            json_man = connection.create_connection(app_name="json", file_path=genai_config_file)
            self.genai_config = json_man.read()

            json_man = connection.create_connection(app_name="json", file_path=task_config_file)
            self.task_definition = json_man.read()

            service_name = self.genai_config.get("ai_service")
            if not service_name:
                raise ValueError("AI service is not specified in the configuration.")

            self.task_runner = GenerativeAITaskFactory.create_task(
                service_name, self.genai_config, self.task_definition, api_key
            )
            print(f"Tasker initialized. Using service: {service_name}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

    def run(self, input_data):
        """
        Description:
            Runs the AI task with the provided input data.

        Args:
            input_data (List[str]): List of input strings to process.

        Returns:
            str: The formatted output from the AI task.

        Raises:
            Exception: If task execution fails.

        Example Usage:
            results = service.run(["John Doe", "Jane Smith"])

        Cost:
            O(n), where n is the number of input items.
        """
        print("\n⏳ Running task... (This may take a moment)")
        output_data = self.task_runner.run(input_data)
        return output_data


class GenerativeAITask:
    def __init__(self, genai_config: Dict[str, Any], task_definition: Dict[str, Any], api_key: str, service):
        self.genai_config = genai_config
        self.task_definition = task_definition
        self.api_key = api_key
        self.service = service

        if not self.api_key:
            raise ValueError("API key is not specified.")

        # Initialize max_input_chars from the service
        self.max_input_chars = self.service.max_input_chars

    def generate_completion(self, data_input: List[str]) -> str:
        """
        Description:
            Generates a completion from the AI service for the provided data input without batching.
            Uses the system and user prompts defined in the task configuration.

        Args:
            data_input (List[str]): List of input strings to be processed by the AI.

        Returns:
            str: The AI-generated output as a JSON string.

        Raises:
            ValueError: If the API key is missing or the service is not properly initialized.
            Exception: For any errors during completion generation.

        Example Usage:
            result = self.generate_completion(["John Doe", "Jane Smith"])

        Cost:
            O(n), where n is the number of input items.

        """
        system_prompt = self.task_definition.get("system_prompt", "")
        user_prompt_template = self.task_definition.get("user_prompt", "")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_template.replace("{data_input_json}", json.dumps(data_input, indent=2))}
        ]

        # The service expects messages, temperature, response format, and optional schema.
        response_str = self.service.generate_completion(
            messages=messages,
            temperature=self.genai_config.get("temperature", 0.7),
            response_format=ResponseFormat.JSON,
            json_schema=self.task_definition.get("output_schema")
        )

        return response_str

    def generate_completion_batched(self, data_input: List[str]) -> str:
        system_prompt = self.task_definition.get("system_prompt", "")
        user_prompt_template = self.task_definition.get("user_prompt", "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_template}
        ]
        
        if not messages or not messages[-1].get("role") == "user":
            raise ValueError("The last message must be a 'user' message with content template.")

        base_prompt_length = sum(len(m.get("content", "")) for m in messages)
        
        all_results = []
        current_batch = []
        current_batch_length = 0

        for item in data_input:
            item_json_length = len(json.dumps([item], indent=2))
            
            if (current_batch_length + item_json_length + base_prompt_length) > self.max_input_chars and current_batch:
                batch_messages = messages.copy()
                data_as_json_string = json.dumps(current_batch, indent=2)
                batch_messages[-1]["content"] = user_prompt_template.replace("{data_input_json}", data_as_json_string)
                
                batch_response_str = self.service.generate_completion(
                    messages=batch_messages,
                    temperature=self.genai_config.get("temperature", 0.7),
                    response_format=ResponseFormat.JSON,
                    json_schema=self.task_definition.get("output_schema")
                )
                
                try:
                    batch_response_json = json.loads(batch_response_str)
                    if "results" in batch_response_json:
                        all_results.extend(batch_response_json["results"])
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse batch response: {batch_response_str}")

                current_batch = []
                current_batch_length = 0
            
            current_batch.append(item)
            current_batch_length += item_json_length

        if current_batch:
            batch_messages = messages.copy()
            data_as_json_string = json.dumps(current_batch, indent=2)
            batch_messages[-1]["content"] = user_prompt_template.replace("{data_input_json}", data_as_json_string)

            batch_response_str = self.service.generate_completion(
                messages=batch_messages,
                temperature=self.genai_config.get("temperature", 0.7),
                response_format=ResponseFormat.JSON,
                json_schema=self.task_definition.get("output_schema")
            )
            
            try:
                batch_response_json = json.loads(batch_response_str)
                if "results" in batch_response_json:
                    all_results.extend(batch_response_json["results"])
            except json.JSONDecodeError:
                print(f"Warning: Could not parse final batch response: {batch_response_str}")

        return json.dumps({"results": all_results}, indent=2, ensure_ascii=False)

    def _format_output(self, result_data: Dict[str, Any]) -> str:
        output_format = self.genai_config.get("output_format", "json")
        results_list = result_data.get("results", [])

        if not results_list:
            return "No results were returned from the AI."

        if output_format == "json":
            return json.dumps(result_data, indent=2, ensure_ascii=False)

        elif output_format == "csv":
            import io
            import csv
            output = io.StringIO()
            fieldnames = results_list[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_list)
            return output.getvalue()
        
        else:
            return json.dumps(result_data, indent=2, ensure_ascii=False)

    def run(self, data_input: List[str]) -> str:
        output_schema = self.task_definition.get("output_schema")

        try:
            #response_str = self.generate_completion_batched(data_input)
            response_str = self.generate_completion(data_input)
            result_data = json.loads(response_str)
            return self._format_output(result_data)

        except Exception as e:
            print(f"❌ An error occurred during task execution: {e}")
            raise


class GenerativeAITaskFactory:
    @staticmethod
    def create_task(service_name: str, genai_config: Dict[str, Any], task_definition: Dict[str, Any], api_key: str):
        """
        Description:
            Factory method to create a GenerativeAITask with the appropriate service.

        Args:
            service_name (str): Name of the AI service ("Gemini", "OpenAI", "Perplexity").
            genai_config (Dict[str, Any]): Configuration dictionary for the AI service.
            task_definition (Dict[str, Any]): Task definition dictionary.
            api_key (str): API key for authentication.

        Returns:
            GenerativeAITask: Initialized task with the appropriate service.

        Raises:
            ValueError: If the service name is not supported.

        Cost:
            O(1)
        """
        model_name = genai_config.get("model_name", "default_model")
        requests_per_second = genai_config.get("requests_per_second", 1.0)

        if service_name == "Gemini":
            service = GeminiService(
                model_name=model_name,
                api_key=api_key,
                requests_per_second=requests_per_second
            )
        elif service_name == "OpenAI":
            service = OpenAIService(
                model_name=model_name,
                api_key=api_key,
                requests_per_second=requests_per_second
            )
        elif service_name == "Perplexity":
            service = PerplexityService(
                model_name=model_name,
                api_key=api_key,
                requests_per_second=requests_per_second
            )
        else:
            raise ValueError(f"Service '{service_name}' is not supported.")

        return GenerativeAITask(genai_config, task_definition, api_key, service)


if __name__ == "__main__":
    #Demo
    input_data = [
        "John Doe",
        "Jane Smith",
        "ACME Corporation",
        "OpenAI",
        "Alice Johnson",
        "Sky Phoenix",
        "Dolores Fuertes",
        "Dolor Fuerte",        
        "Rosa de los Vientos",     
        "La Rosa de los Vientos"
    ]


    ai_configuration = './lib/genai_tasker/genai_config.json'
    task_configuration = './lib/genai_tasker/prompts/name_classifier.json'
    # msg_secrets = conn_man.load_secrets_from_file("./lib/connectors/secrets.json")
    # print(msg_secrets)    
    secrets = conn_man.SecretManager()
    api_key = secrets.get_secret('gemini_apikey', 'apikey')   
    run_task_one_shot(ai_configuration, task_configuration, input_data, api_key)
