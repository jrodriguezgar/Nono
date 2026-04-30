"""
Example: Using Jinja2 Templates On-the-Fly

This example demonstrates how to:
1. Define Jinja2 templates inline (as strings, not files)
2. Use TaskPromptBuilder.from_template_string() to render prompts
3. Execute the rendered prompt with TaskExecutor

Use cases:
- Dynamic template generation
- Template customization at runtime
- Testing templates without creating files
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nono.tasker import TaskExecutor
from nono.tasker.jinja_prompt_builder import TaskPromptBuilder
import time

# Configuration
PROVIDER = "openrouter"
MODEL = "openrouter/auto"

# ============================================================================
# Example 1: Simple Template with Variables
# ============================================================================

SIMPLE_TEMPLATE = """
You are an expert {{ role }}.

Please {{ action }} the following:

{{ content }}

Provide a {{ output_style }} response.
"""


def run_simple_example():
    """Execute a simple template with variable substitution."""
    print(f"\n{'='*60}")
    print("  Example 1: Simple Template with Variables")
    print(f"{'='*60}")
    
    # Create prompt builder
    builder = TaskPromptBuilder()
    
    # Define variables
    variables = {
        "role": "translator",
        "action": "translate to Spanish",
        "content": "Hello, how are you today? I hope you are well.",
        "output_style": "brief"
    }
    
    # Render the template
    prompts = builder.from_template_string(
        template=SIMPLE_TEMPLATE,
        variables=variables
    )
    
    print("\n📝 TEMPLATE (raw):")
    print(SIMPLE_TEMPLATE)
    
    print("\n🔧 VARIABLES:")
    for k, v in variables.items():
        print(f"   {k}: {v}")
    
    print("\n✨ RENDERED PROMPT:")
    print(prompts[0])
    
    if input(f"\n Execute with AI? (y/n): ").lower() == 'y':
        executor = TaskExecutor(provider=PROVIDER, model=MODEL)
        response = executor.execute(input_data=prompts[0])
        print(f"\n🤖 RESPONSE:\n{response}")
    
    print("\n" + "─"*60)


# ============================================================================
# Example 2: Template with List Data and Filters
# ============================================================================

LIST_TEMPLATE = """
Classify the following items into categories.

Items to classify:
{{ items | bullet_list }}

For each item, provide:
- Category
- Confidence (high/medium/low)

Respond in JSON format.
"""


def run_list_example():
    """Execute a template with list data using custom filters."""
    print(f"\n{'='*60}")
    print("  Example 2: Template with List Data and Filters")
    print(f"{'='*60}")
    
    builder = TaskPromptBuilder()
    
    items = ["Apple", "Python", "JavaScript", "Banana", "C++", "Strawberry"]
    
    # Render with the 'items' variable
    prompts = builder.from_template_string(
        template=LIST_TEMPLATE,
        variables={"items": items}
    )
    
    print("\n📝 TEMPLATE (raw):")
    print(LIST_TEMPLATE)
    
    print("\n🔧 ITEMS:", items)
    
    print("\n✨ RENDERED PROMPT:")
    print(prompts[0])
    
    if input(f"\n Execute with AI? (y/n): ").lower() == 'y':
        executor = TaskExecutor(provider=PROVIDER, model=MODEL)
        response = executor.execute(
            input_data=prompts[0],
            config_overrides={"temperature": 0.3}
        )
        print(f"\n🤖 RESPONSE:\n{response}")
    
    print("\n" + "─"*60)


# ============================================================================
# Example 3: Template with Batching (for large datasets)
# ============================================================================

BATCH_TEMPLATE = """
Analyze the following {{ batch_type }} and rate each from 1-10:

{% for item in data %}
{{ loop.index }}. {{ item }}
{% endfor %}

Provide your ratings as a JSON object with names as keys.
"""


def run_batch_example():
    """Execute a template with automatic batching for large data."""
    print(f"\n{'='*60}")
    print("  Example 3: Template with Batching")
    print(f"{'='*60}")
    
    builder = TaskPromptBuilder(default_max_tokens=500)  # Low limit to force batching
    
    data = [
        "The Matrix",
        "Inception",
        "Interstellar",
        "The Dark Knight",
        "Pulp Fiction",
        "Fight Club",
        "Forrest Gump",
        "The Shawshank Redemption"
    ]
    
    # Render with batching - the 'data' key links to the data parameter
    prompts = builder.from_template_string(
        template=BATCH_TEMPLATE,
        variables={"batch_type": "movies"},
        data_key="data",
        data=data,
        max_tokens=500  # Force batching with low token limit
    )
    
    print("\n📝 TEMPLATE (raw):")
    print(BATCH_TEMPLATE)
    
    print(f"\n🔧 DATA ({len(data)} items):", data)
    
    print(f"\n📦 GENERATED {len(prompts)} BATCH(ES):")
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Batch {i} ---")
        print(prompt[:300] + ("..." if len(prompt) > 300 else ""))
    
    if input(f"\n Execute batch 1 with AI? (y/n): ").lower() == 'y':
        executor = TaskExecutor(provider=PROVIDER, model=MODEL)
        response = executor.execute(
            input_data=prompts[0],
            config_overrides={"temperature": 0.5}
        )
        print(f"\n🤖 RESPONSE (Batch 1):\n{response}")
    
    print("\n" + "─"*60)


# ============================================================================
# Example 4: Complex Multi-Section Template
# ============================================================================

COMPLEX_TEMPLATE = """
# Task: {{ task_name }}

## Context
{{ context }}

## Instructions
{% for instruction in instructions %}
{{ loop.index }}. {{ instruction }}
{% endfor %}

## Input Data
```{{ data_format }}
{{ input_data | to_pretty_json if data_format == 'json' else input_data }}
```

## Expected Output
{{ output_description }}

{% if constraints %}
## Constraints
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}
"""


def run_complex_example():
    """Execute a complex multi-section template."""
    print(f"\n{'='*60}")
    print("  Example 4: Complex Multi-Section Template")
    print(f"{'='*60}")
    
    builder = TaskPromptBuilder()
    
    variables = {
        "task_name": "Data Transformation",
        "context": "You are a data engineer processing user records.",
        "instructions": [
            "Parse the input data",
            "Validate all email addresses",
            "Normalize phone numbers to E.164 format",
            "Return the cleaned data"
        ],
        "data_format": "json",
        "input_data": {
            "users": [
                {"name": "John Doe", "email": "john@example.com", "phone": "555-1234"},
                {"name": "Jane Smith", "email": "jane@test.org", "phone": "(555) 567-8901"}
            ]
        },
        "output_description": "JSON array with cleaned user records",
        "constraints": [
            "Preserve original field names",
            "Use null for invalid values",
            "Include a 'valid' boolean field for each record"
        ]
    }
    
    prompts = builder.from_template_string(
        template=COMPLEX_TEMPLATE,
        variables=variables
    )
    
    print("\n📝 TEMPLATE (raw):")
    print(COMPLEX_TEMPLATE[:500] + "...")
    
    print("\n✨ RENDERED PROMPT:")
    print(prompts[0])
    
    if input(f"\n Execute with AI? (y/n): ").lower() == 'y':
        executor = TaskExecutor(provider=PROVIDER, model=MODEL)
        response = executor.execute(
            input_data=prompts[0],
            config_overrides={"temperature": 0.2}
        )
        print(f"\n🤖 RESPONSE:\n{response}")
    
    print("\n" + "─"*60)


# ============================================================================
# Example 5: Build and Execute in One Function
# ============================================================================

def execute_with_inline_template(
    template: str,
    variables: dict,
    provider: str = PROVIDER,
    model: str = MODEL,
    **overrides
) -> str:
    """
    Utility function to build and execute a prompt from an inline template.
    
    Args:
        template: Jinja2 template string.
        variables: Dictionary of template variables.
        provider: AI provider name.
        model: Model name.
        **overrides: Additional config overrides for execution.
    
    Returns:
        AI response as string.
    """
    builder = TaskPromptBuilder()
    prompts = builder.from_template_string(template=template, variables=variables)
    
    executor = TaskExecutor(provider=provider, model=model)
    return executor.execute(input_data=prompts[0], config_overrides=overrides)


def run_oneshot_example():
    """Demonstrate the one-shot utility function."""
    print(f"\n{'='*60}")
    print("  Example 5: One-Shot Execution Function")
    print(f"{'='*60}")
    
    template = """
    Generate {{ count }} creative names for a {{ product_type }}.
    Style: {{ style }}
    """
    
    variables = {
        "count": 5,
        "product_type": "coffee shop",
        "style": "modern and minimalist"
    }
    
    print("\n📝 TEMPLATE:", template.strip())
    print("\n🔧 VARIABLES:", variables)
    
    if input(f"\n Execute with AI? (y/n): ").lower() == 'y':
        response = execute_with_inline_template(
            template=template,
            variables=variables,
            temperature=0.8
        )
        print(f"\n🤖 RESPONSE:\n{response}")
    
    print("\n" + "─"*60)


# ============================================================================
# Main Menu
# ============================================================================

EXAMPLES = [
    {
        "name": "Simple Template with Variables",
        "description": "Basic variable substitution: {{ role }}, {{ content }}",
        "func": run_simple_example
    },
    {
        "name": "Template with List Data and Filters",
        "description": "Use custom filters like 'bullet_list' to format lists",
        "func": run_list_example
    },
    {
        "name": "Template with Batching",
        "description": "Automatic batch splitting for large datasets",
        "func": run_batch_example
    },
    {
        "name": "Complex Multi-Section Template",
        "description": "Multi-block template with conditionals and loops",
        "func": run_complex_example
    },
    {
        "name": "One-Shot Execution Function",
        "description": "Utility function: execute_with_inline_template()",
        "func": run_oneshot_example
    },
]


def show_menu():
    """Display the main menu."""
    print(f"\n{'─'*60}")
    print("📋 SELECT AN EXAMPLE TO RUN:")
    print(f"{'─'*60}")
    for i, ex in enumerate(EXAMPLES, 1):
        print(f"  {i}. {ex['name']}")
        print(f"     └─ {ex['description']}")
    print(f"{'─'*60}")
    print("  6. Run All Examples")
    print("  0. Exit")
    print(f"{'─'*60}")


def confirm_execution(example_name: str) -> bool:
    """Ask user to confirm before running an example."""
    print(f"\n🎯 Selected: {example_name}")
    response = input("   Proceed? (y/n): ").strip().lower()
    return response == 'y'


def main():
    """Run the interactive example menu."""
    print(f"\n{'#'*60}")
    print("  🧩 Jinja2 Templates On-the-Fly - Examples")
    print(f"{'#'*60}")
    print(f"\n  Provider: {PROVIDER}")
    print(f"  Model: {MODEL}")
    
    while True:
        show_menu()
        choice = input("\n  Choice: ").strip()
        
        if choice == '0':
            print("\n✅ Goodbye!")
            break
        
        elif choice == '6':
            if confirm_execution("Run All Examples"):
                for ex in EXAMPLES:
                    print(f"\n{'='*60}")
                    print(f"  ▶ Running: {ex['name']}")
                    print(f"{'='*60}")
                    ex['func']()
                print("\n✅ All examples completed!")
        
        elif choice in ['1', '2', '3', '4', '5']:
            idx = int(choice) - 1
            ex = EXAMPLES[idx]
            if confirm_execution(ex['name']):
                ex['func']()
        
        else:
            print("\n⚠️  Invalid choice. Please select 0-6.")


if __name__ == "__main__":
    main()
