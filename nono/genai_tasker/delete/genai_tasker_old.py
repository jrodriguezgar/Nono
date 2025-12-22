"""
This module is designed to facilitate the orchestration of generative AI tasks.
It provides a structured way to load configurations, initialize AI service clients,
and manage the execution of tasks with various AI models.
"""

import os
import json
import argparse
from typing import Dict, Any, List

from lib.genai_tasker.connector_genai import (
    GeminiService,
    OpenAIService,
    PerplexityService,
    ResponseFormat
)

# A mapper to select the correct AI service class based on the config file.
# Why use a mapper? It's a clean, scalable way to add new services
# without a messy if/elif/else chain.
SERVICE_MAPPER = {
    "Gemini": GeminiService,
    "OpenAI": OpenAIService,
    "Perplexity": PerplexityService,
    # Add other services from your connector here, e.g., "DeepSeek": DeepSeekService
}

print("-------------------------------------------------------------")
#module paths
py_module_path = __file__
print(f"Python Module Path: {py_module_path}")
print("-------------------------------------------------------------")

class GenerativeAITask:
    """
    Orchestrates a generative AI task by combining configuration, prompts,
    and data to produce a formatted output.
    """
    def __init__(self, config_path: str, task_path: str):
        """
        Initializes the task runner with configuration and task-specific prompts.

        Args:
            config_path (str): Path to the main JSON configuration file.
            task_path (str): Path to the task-specific JSON prompt file.
        """
        self.config = self._load_json_file(config_path)
        self.task_definition = self._load_json_file(task_path)
        self.service = self._initialize_service()

    @staticmethod
    def _load_json_file(file_path: str) -> Dict[str, Any]:
        """
        A static utility method to load a JSON file.
        
        Description:
            Loads a JSON file and returns its content as a dictionary.
            It's static because it doesn't depend on any instance state.
        Args:
            file_path (str): The path to the JSON file.
        Returns:
            Dict[str, Any]: The parsed JSON content.
        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        Example of use:
            config = GenerativeAITask._load_json_file("config.json")
        Cost:
            O(N) where N is the size of the file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError as e:
            # Print the current working directory to help debug missing file issues
            print(f"File not found: {file_path}")
            print(f"Current working directory: {os.getcwd()}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file: {file_path}")
            raise

    def _initialize_service(self) -> Any:
        """
        Initializes the AI service client based on the configuration.

        Description:
            Reads the service name from the config, finds the corresponding
            class in SERVICE_MAPPER, and instantiates it with the required
            parameters like API key and model name. This abstracts away the
            details of which service is being used.
        Returns:
            An instance of a class from lib.connector_genai (e.g., GeminiService).
        Raises:
            ValueError: If the service in config is not supported or API key is missing.
        """
        service_name = self.config.get("ai_service")
        if not service_name or service_name not in SERVICE_MAPPER:
            raise ValueError(f"Service '{service_name}' is not configured or supported.")

        # api_key_env_var = self.config.get("api_key_env")
        # if not api_key_env_var:
        #     raise ValueError("API key environment variable name is not specified in config.")

        api_key = 'AIzaSyA2ZZtkaCGu9Oib7gIinkQZ8odUaLzDmEw'

        # Why get the API key from an environment variable?
        # This is a critical security best practice. Hardcoding keys in the script
        # is dangerous as they can be accidentally exposed in version control.
        # api_key = os.environ.get(api_key_env_var)
        # if not api_key:
        #     raise ValueError(f"Environment variable '{api_key_env_var}' is not set.")

        service_class = SERVICE_MAPPER[service_name]
        
        return service_class(
            model_name=self.config.get("model_name"),
            api_key=api_key
        )


    def generate_completion_batched(self, data_input: list[str]) -> str:
        """
        Combina la preparación de mensajes y la generación por lotes para procesar
        una lista de entradas de datos.

        Args:
            data_input (list[str]): La lista de elementos de datos que se procesarán.

        Returns:
            str: El resultado consolidado de la tarea como una cadena JSON.
        
        Raises:
            ValueError: Si los mensajes de la plantilla no tienen el formato esperado.
        """
        # Lógica de _prepare_messages()
        system_prompt = self.task_definition.get("system_prompt", "")
        user_prompt_template = self.task_definition.get("user_prompt", "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_template}
        ]
        
        if not messages or not messages[-1].get("role") == "user":
            raise ValueError("El último mensaje debe ser un mensaje 'user' con la plantilla de contenido.")
        
        # Lógica de generate_completion_batched()
        base_prompt_length = sum(len(m.get("content", "")) for m in messages)
        max_prompt_length = self.max_input_chars
        
        all_results = []
        current_batch = []
        current_batch_length = 0

        for item in data_input:
            item_json_length = len(json.dumps([item], indent=2))
            
            if (current_batch_length + item_json_length + base_prompt_length) > max_prompt_length and current_batch:
                batch_messages = messages.copy()
                data_as_json_string = json.dumps(current_batch, indent=2)
                batch_messages[-1]["content"] = user_prompt_template.replace("{data_input_json}", data_as_json_string)
                
                batch_response_str = self.generate_completion(messages=batch_messages)
                
                try:
                    batch_response_json = json.loads(batch_response_str)
                    if "results" in batch_response_json:
                        all_results.extend(batch_response_json["results"])
                except json.JSONDecodeError:
                    print(f"Advertencia: No se pudo analizar la respuesta del lote: {batch_response_str}")

                current_batch = []
                current_batch_length = 0
            
            current_batch.append(item)
            current_batch_length += item_json_length

        if current_batch:
            batch_messages = messages.copy()
            data_as_json_string = json.dumps(current_batch, indent=2)
            batch_messages[-1]["content"] = user_prompt_template.replace("{data_input_json}", data_as_json_string)

            batch_response_str = self.generate_completion(messages=batch_messages)
            
            try:
                batch_response_json = json.loads(batch_response_str)
                if "results" in batch_response_json:
                    all_results.extend(batch_response_json["results"])
            except json.JSONDecodeError:
                print(f"Advertencia: No se pudo analizar la respuesta del lote final: {batch_response_str}")

        return json.dumps({"results": all_results}, indent=2, ensure_ascii=False)

    def _format_output(self, result_data: Dict[str, Any]) -> str:
        """
        Formats the structured AI result into the desired output string.

        Description:
            Converts the Python dictionary received from the AI into various
            formats like a text table, CSV, or plain JSON based on the configuration.
            This separates presentation logic from the AI interaction logic.
        Args:
            result_data (Dict[str, Any]): The parsed JSON response from the AI.
        Returns:
            str: The formatted output string.
        """
        output_format = self.config.get("output_format", "json")
        results_list = result_data.get("results", [])

        if not results_list:
            return "No results were returned from the AI."

        if output_format == "json":
            return json.dumps(result_data, indent=2, ensure_ascii=False)
        
        elif output_format == "table":
            # This logic benefits from having the 'tabulate' library installed.
            # pip install tabulate
            try:
                from tabulate import tabulate
                headers = results_list[0].keys()
                rows = [item.values() for item in results_list]
                return tabulate(rows, headers=headers, tablefmt="grid")
            except ImportError:
                return "The 'tabulate' library is required for table format. Please install it."
            except (IndexError, AttributeError):
                 return f"Could not format data into a table. Raw data: {results_list}"


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
            # Default to JSON if the format is unknown.
            return json.dumps(result_data, indent=2, ensure_ascii=False)

    def run(self, data_input: List[str]) -> str:
        """
        Executes the configured AI task with the given data.

        Description:
            This is the main public method. It takes the input data, prepares the
            API messages, calls the AI service, and returns the final, formatted output.
        Args:
            data_input (List[str]): A list of strings or items to be processed by the AI.
        Returns:
            str: The final output, formatted according to the configuration.
        Raises:
            Exception: Propagates exceptions from the AI service call.
        Example of use:
            task = GenerativeAITask("config.json", "task.json")
            names = ["John Doe", "ACME Corp", "Jane Smith"]
            formatted_result = task.run(names)
            print(formatted_result)
        Cost:
            Dependent on the AI service call, but the overhead of this method is O(M)
            where M is the size of the input data.
        """
        output_schema = self.task_definition.get("output_schema")

        try:
            # Why use the batched method? The original script had a batched method in GeminiService
            # which is great for handling large inputs that might exceed context windows.
            # We will call it if it exists, otherwise fall back to the standard one.
            #if hasattr(self.service, 'generate_completion_batched'):
            response_str = self.service.generate_completion_batched(data_input)
                # response_str = self.service.generate_completion_batched(
                #     messages=messages,
                #     temperature=self.config.get("temperature", 0.7),
                #     response_format=ResponseFormat.JSON,
                #     json_schema=output_schema,
                #     max_prompt_length=self.config.get("max_prompt_length")
                # )
            # else:
            #      response_str = self.service.generate_completion(
            #         messages=messages,
            #         temperature=self.config.get("temperature", 0.7),
            #         response_format=ResponseFormat.JSON,
            #         json_schema=output_schema
            #     )

            result_data = json.loads(response_str)
            return self._format_output(result_data)

        except Exception as e:
            # Why catch and re-raise with more context?
            # It helps in debugging by providing a clear message about where
            # the error occurred during the execution flow.
            print(f"❌ An error occurred during task execution: {e}")
            raise


def genai_init(genai_config_file, task_file):
    try:
        # Initialize and run the task
        print("Initializing AI Task Service...")
        task_runner = GenerativeAITask(config_path=genai_config_file, task_path=task_file)
        print(f"Task initialized. Using service: {task_runner.config['ai_service']}, Model: {task_runner.config['model_name']}")
        return task_runner
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def run_task_demo():
    #input_data examples
    input_data = """
    John Doe
    Jane Smith
    ACME Corporation
    OpenAI
    Alice Johnson
    """
        
    task_runner = genai_init('.//lib//genai_tasker//genai_tasker_config.json','.//lib//genai_tasker//prompts//name_classifier.json')

    print("\n⏳ Running task... (This may take a moment)")
    output_data = task_runner.run(input_data)
    
    print("\n--- AI Task Results ---")
    print(output_data)
    print("-----------------------")


if __name__ == "__main__":
    run_task_demo()