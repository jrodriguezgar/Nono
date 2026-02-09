#!/usr/bin/env python3
"""
Nono GenAI Tasker - Main Entry Point

A unified framework for executing tasks using Generative AI.
Supports multiple LLM providers: Google Gemini, OpenAI, Perplexity, DeepSeek, Grok, and Ollama.

Usage:
    python main.py                          # Interactive mode
    python main.py --task prompts/name_classifier.json --data "María García,John Smith"
    python main.py --provider gemini --model gemini-3-flash-preview --prompt "Hello, world!"
    python main.py --list-providers         # List available providers
    python main.py --list-templates         # List available templates

Environment Variables:
    NONO_TEMPLATES_DIR: Custom templates directory
    NONO_PROMPTS_DIR: Custom prompts directory
    NONO_CONFIG_FILE: Custom config.toml path
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import Nono modules
from nono import __version__
from nono.config import NonoConfig, get_templates_dir, get_prompts_dir
from nono.tasker import (
    TaskExecutor,
    AIProvider,
    build_from_file_blocks,
    msg_log,
)


def list_providers() -> None:
    """List all available AI providers."""
    print("\n📦 Available AI Providers:")
    print("─" * 40)
    for provider in AIProvider:
        print(f"  • {provider.value}")
    print()


def list_templates() -> None:
    """List all available Jinja2 templates."""
    templates_path = get_templates_dir()
    
    print("\n📄 Available Templates:")
    print(f"   Path: {templates_path}")
    print("─" * 40)
    if templates_path.exists():
        templates = list(templates_path.glob("*.j2"))
        if templates:
            for template in sorted(templates):
                print(f"  • {template.name}")
        else:
            print("  (no templates found)")
    else:
        print(f"  Templates directory not found: {templates_path}")
    print()


def list_tasks() -> None:
    """List all available task definition files."""
    prompts_path = get_prompts_dir()
    
    print("\n📋 Available Task Definitions:")
    print(f"   Path: {prompts_path}")
    print("─" * 40)
    if prompts_path.exists():
        tasks = list(prompts_path.glob("*.json"))
        if tasks:
            for task in sorted(tasks):
                if task.name != "task_template.json":
                    print(f"  • {task.name}")
        else:
            print("  (no task definitions found)")
    else:
        print(f"  Prompts directory not found: {prompts_path}")
    print()


def show_config() -> None:
    """Show current configuration."""
    config = NonoConfig.get_all_config()
    
    print("\n⚙️  Current Configuration:")
    print("─" * 40)
    print(f"  Templates:   {config['templates_dir']}")
    print(f"  Prompts:     {config['prompts_dir']}")
    
    # Show environment variables if set
    if config['env_templates_dir']:
        print(f"\n  (NONO_TEMPLATES_DIR={config['env_templates_dir']})")
    if config['env_prompts_dir']:
        print(f"  (NONO_PROMPTS_DIR={config['env_prompts_dir']})")
    if config['env_config_file']:
        print(f"  (NONO_CONFIG_FILE={config['env_config_file']})")
    
    # Show provider defaults from config file
    file_config = config.get('config_file', {})
    if file_config:
        print("\n  Provider defaults:")
        for provider in ["google", "openai", "perplexity", "deepseek", "grok", "ollama"]:
            if provider in file_config:
                model = file_config[provider].get("default_model", "N/A")
                print(f"    • {provider}: {model}")
    print()


def run_simple_prompt(
    provider: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None
) -> None:
    """Run a simple prompt with the specified provider and model."""
    print(f"\n⏳ Initializing {provider.upper()} / {model}...")
    
    try:
        executor = TaskExecutor(provider=provider, model=model)
        print(f"   ✅ TaskExecutor created")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        print(f"\n📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"\n⏳ Sending request...")
        
        response = executor.execute(input_data=messages)
        
        print(f"\n{'─' * 60}")
        print("📤 Response:")
        print(f"{'─' * 60}")
        print(response)
        print(f"{'─' * 60}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def run_task(task_path: str, data: str) -> None:
    """Run a task from a JSON definition file."""
    task_file = Path(task_path)
    if not task_file.exists():
        # Try looking in configured prompts directory
        prompts_path = get_prompts_dir() / task_path
        if prompts_path.exists():
            task_file = prompts_path
        else:
            print(f"\n❌ Task file not found: {task_path}")
            print(f"   Searched in: {get_prompts_dir()}")
            sys.exit(1)
    
    print(f"\n📋 Loading task: {task_file.name}")
    
    with open(task_file, "r", encoding="utf-8") as f:
        task_config = json.load(f)
    
    task_name = task_config.get("task", {}).get("name", "unknown")
    provider = task_config.get("genai", {}).get("provider", "gemini")
    model = task_config.get("genai", {}).get("model", "gemini-3-flash-preview")
    
    print(f"   Task: {task_name}")
    print(f"   Provider: {provider.upper()} / {model}")
    
    # Parse input data
    if data.startswith("["):
        input_data = json.loads(data)
    else:
        input_data = [item.strip() for item in data.split(",")]
    
    print(f"   Input items: {len(input_data)}")
    
    try:
        executor = TaskExecutor(provider=provider, model=model)
        
        # Build prompts from task config
        system_prompt = task_config.get("prompts", {}).get("system", "")
        user_template = task_config.get("prompts", {}).get("user", "{data_input_json}")
        user_prompt = user_template.replace("{data_input_json}", json.dumps(input_data, ensure_ascii=False))
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"\n⏳ Executing task...")
        response = executor.execute(
            input_data=messages,
            config_overrides={"response_format": "json"}
        )
        
        print(f"\n{'─' * 60}")
        print("📤 Response:")
        print(f"{'─' * 60}")
        
        # Try to parse and pretty-print JSON
        try:
            result = json.loads(response.strip())
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(response)
        
        print(f"{'─' * 60}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def interactive_mode() -> None:
    """Run in interactive mode."""
    print(f"\n{'=' * 60}")
    print(f"  Nono GenAI Tasker v{__version__}")
    print(f"  Interactive Mode")
    print(f"{'=' * 60}")
    
    print("\nOptions:")
    print("  1. Run a simple prompt")
    print("  2. Run a task from file")
    print("  3. List providers")
    print("  4. List templates")
    print("  5. List tasks")
    print("  6. Show configuration")
    print("  q. Quit")
    
    while True:
        choice = input("\n> Select option (1-6, q): ").strip().lower()
        
        if choice == "q":
            print("\n👋 Goodbye!\n")
            break
        elif choice == "1":
            provider = input("  Provider (gemini/openai/perplexity/deepseek/grok/ollama): ").strip() or "gemini"
            model = input("  Model (leave empty for default): ").strip() or "gemini-3-flash-preview"
            prompt = input("  Prompt: ").strip()
            if prompt:
                run_simple_prompt(provider, model, prompt)
        elif choice == "2":
            list_tasks()
            task = input("  Task file: ").strip()
            data = input("  Data (comma-separated or JSON array): ").strip()
            if task and data:
                run_task(task, data)
        elif choice == "3":
            list_providers()
        elif choice == "4":
            list_templates()
        elif choice == "5":
            list_tasks()
        elif choice == "6":
            show_config()
        else:
            print("  Invalid option. Try again.")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Nono GenAI Tasker - Execute AI tasks with multiple providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive mode
  python main.py --prompt "Explain Python decorators"
  python main.py --task name_classifier.json --data "María,John,Apple Inc."
  python main.py --list-providers
        """
    )
    
    parser.add_argument("--version", "-v", action="version", version=f"Nono v{__version__}")
    parser.add_argument("--provider", "-p", default="gemini", help="AI provider (default: gemini)")
    parser.add_argument("--model", "-m", default="gemini-3-flash-preview", help="Model name")
    parser.add_argument("--prompt", help="Simple prompt to execute")
    parser.add_argument("--system", help="System prompt (optional)")
    parser.add_argument("--task", "-t", help="Task definition JSON file")
    parser.add_argument("--data", "-d", help="Input data (comma-separated or JSON array)")
    parser.add_argument("--list-providers", action="store_true", help="List available providers")
    parser.add_argument("--list-templates", action="store_true", help="List available templates")
    parser.add_argument("--list-tasks", action="store_true", help="List available task definitions")
    
    args = parser.parse_args()
    
    if args.list_providers:
        list_providers()
    elif args.list_templates:
        list_templates()
    elif args.list_tasks:
        list_tasks()
    elif args.prompt:
        run_simple_prompt(args.provider, args.model, args.prompt, args.system)
    elif args.task and args.data:
        run_task(args.task, args.data)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
