import os
import json
import argparse
import csv
from typing import List, Dict, Any

# Importamos las clases necesarias del módulo adjunto
from lib.connector_genai import GeminiService, ResponseFormat


def load_config(config_path: str) -> Dict[str, Any]:  
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_prompt(prompt_file_name) -> (str, str, dict):
    """
    Loads the prompt configuration from the prompts/name_classifier_config.json file.

    Returns:
        tuple: (system_prompt, user_prompt, output_schema)
    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the config file is not valid JSON.
    Example:
        system_prompt, user_prompt, output_schema = get_prompt("name_classifier_config.json")
    Cost:
        O(1) for file loading and parsing.
    """
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    config_path = os.path.join(prompts_dir, prompt_file_name)

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error al cargar el archivo de configuración: {e}")
        return

    return config["system_prompt"], config["user_prompt"], config["output_schema"]


def get_data_input(prompt_file_name) -> (str, str, dict):
    file_path = os.path.join(os.path.dirname(__file__), "lista_nombres.txt")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            names_to_check = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: El archivo 'lista_nombres.txt' no se encontró en la ruta: {file_path}")
        return
    

def config_task(prompt_file_name, data_input):
    api_key = "AIzaSyA2ZZtkaCGu9Oib7gIinkQZ8odUaLzDmEw"
    if not api_key:
        print("❌ Error: La variable de entorno GOOGLE_API_KEY no está configurada.")
        return

    try:
        global genai
        genai = GeminiService(model_name="gemini-1.5-flash", api_key=api_key)

def do_task(
    gemini_service: GeminiService,
    name_list: List[str],
    max_len: int,
    config: Dict[str, Any],
    output_format: str = "json"
) -> Any:
            
    except Exception as e:
        print(f"❌ Error al inicializar GeminiService: {e}")
        return
    

def config_task(prompt_file_name, data_input):
    temperature = 0.1
    max_len = 30000
    output_format = "json"

    output_schema, system_prompt, user_prompt_template = get_prompt(prompt_file_name) -> (str, str, dict)
    data_input = json.dumps(data_input, indent=2)
    user_prompt = user_prompt_template.replace("{names_json_string}", data_input)
    return output_schema, system_prompt, user_prompt


def do_task(
    genai,
    data_input,
    max_len: int,
    config: Dict[str, Any],
    output_format: str = "json"
) -> Any:



    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response_str = genai.generate_completion_batched(
            messages=messages,
            temperature=0.1,
            response_format=ResponseFormat.JSON,
            json_schema=output_schema,
            max_prompt_length=max_len
        )
        result = json.loads(response_str)
        if output_format == "json":
            return result
        elif output_format == "text":
            # Devuelve una cadena de texto legible
            lines = []
            for r in result["results"]:
                lines.append(f"{r['analyzed_string']}: {'Persona' if r['is_person'] else 'No persona'} - {r['reasoning']}")
            return "\n".join(lines)
        elif output_format == "csv":
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["analyzed_string", "is_person", "reasoning"])
            writer.writeheader()
            for r in result["results"]:
                writer.writerow(r)
            return output.getvalue()
        elif output_format == "xml":
            lines = ["<results>"]
            for r in result["results"]:
                lines.append(f"  <item><analyzed_string>{r['analyzed_string']}</analyzed_string><is_person>{str(r['is_person']).lower()}</is_person><reasoning>{r['reasoning']}</reasoning></item>")
            lines.append("</results>")
            return "\n".join(lines)
        elif output_format == "table":
            # Tabla simple en texto
            from tabulate import tabulate
            return tabulate(
                [ [r['analyzed_string'], r['is_person'], r['reasoning']] for r in result["results"] ],
                headers=["analyzed_string", "is_person", "reasoning"],
                tablefmt="grid"
            )
        else:
            return result
    except Exception as e:
        print(f"❌ Error al contactar con el API de Gemini: {e}")
        return {"error": str(e)}


def main():
    """
    Función principal para ejecutar el script desde la línea de comandos.
    """



    data_input = get_data_input("lista_nombres.txt")
    output_schema, system_prompt, user_prompt_template  = config_task("name_classifier.json", data_input)


    results = do_tack(
        gemini,
        data_input,
        args.max_prompt_length,
        config,
        .output_format
    )

    print("\n✅ Resultados de la clasificación:")
    if args.output_format == "json":
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(results)


if __name__ == "__main__":
    main()