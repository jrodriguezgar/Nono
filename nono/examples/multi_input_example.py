"""
Example: Multi-Input Product Categorization Task

This example demonstrates how to use the TaskExecutor with multiple data inputs.
Multiple placeholders in the prompt template allow passing different data sources
to the AI model for more contextual processing.

Features demonstrated:
    - Multiple data inputs with named placeholders
    - {data} for primary data (products list)
    - {categories} for available category options
    - {context} for additional context information
    - On-the-fly Jinja2 template (no .j2 file needed)

Usage: python multi_input_example.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nono.tasker import TaskExecutor
from nono.tasker.jinja_prompt_builder import TaskPromptBuilder
import json
import time

# Configuration
PROVIDER = "openrouter"
MODEL = "openrouter/auto"

# ============================================================================
# On-the-fly Jinja2 Templates (no physical .j2 file needed)
# ============================================================================

SYSTEM_TEMPLATE = """
You are a product classification expert. Categorize products accurately 
based on the available categories and additional context provided.

Always respond with a JSON array. Each element must have:
- "product": the original product name
- "category": the assigned category (must be from the available categories)
- "confidence": "high", "medium", or "low"

Return ONLY the JSON array, no explanations or additional text.
"""

USER_TEMPLATE = """
Categorize each of the following products:

PRODUCTS TO CATEGORIZE:
{% for item in data %}
{{ loop.index }}. "{{ item }}"
{% endfor %}

AVAILABLE CATEGORIES:
{% for cat in categories %}
- {{ cat }}
{% endfor %}

ADDITIONAL CONTEXT:
{{ context }}

Return ONLY the JSON array.
"""

# Sample data: products to categorize
PRODUCTS = [
    "iPhone 15 Pro Max",
    "Nike Air Jordan 1",
    "Samsung 65-inch OLED TV",
    "Adidas Running Shorts",
    "Sony WH-1000XM5 Headphones",
]

# Available categories
CATEGORIES = [
    "Electronics",
    "Footwear",
    "Apparel",
    "Home Entertainment",
    "Audio Equipment",
]

# Additional context for categorization
CONTEXT = """Prioritize the primary function of the product.
For wearable tech, consider if it's primarily worn as clothing or used as a device.
Headphones should be classified under Audio Equipment, not Electronics."""


def run_example():
    """Categorize products using multiple data inputs."""
    print(f"\n{'='*70}")
    print("  Multi-Input Product Categorization Example")
    print(f"{'='*70}")
    
    # Show input data
    print(f"\n{'─'*70}")
    print(f"📦 PRODUCTS TO CATEGORIZE ({len(PRODUCTS)} items):")
    print(f"{'─'*70}")
    for i, p in enumerate(PRODUCTS, 1):
        print(f"  {i:2}. {p}")
    
    print(f"\n📂 AVAILABLE CATEGORIES:")
    for cat in CATEGORIES:
        print(f"   • {cat}")
    
    print(f"\n📝 CONTEXT:")
    print(f"   {CONTEXT}")
    
    print(f"\n{'─'*70}")
    print(f"Provider: {PROVIDER.upper()} | Model: {MODEL}")
    print(f"API calls: 1")
    print(f"{'─'*70}")
    
    if input(f"\nCategorize all products with AI? (y/n): ").lower() != 'y':
        print("\n✅ Cancelled.")
        return
    
    print(f"\n⏳ Initializing TaskExecutor...")
    
    try:
        # Create TaskExecutor (API key resolved automatically)
        executor = TaskExecutor(provider=PROVIDER, model=MODEL)
        print(f"   ✅ TaskExecutor created ({PROVIDER.upper()} / {MODEL})")
        
        # Create prompt builder for on-the-fly templates
        builder = TaskPromptBuilder()
        
        # Define all variables for the templates
        variables = {
            "data": PRODUCTS,
            "categories": CATEGORIES,
            "context": CONTEXT
        }
        
        # Render user prompt from on-the-fly template with multiple inputs
        user_prompts = builder.from_template_string(
            template=USER_TEMPLATE,
            variables=variables
        )
        
        # System prompt is static (no variables needed)
        system_prompt = SYSTEM_TEMPLATE.strip()
        
        print(f"\n📋 GENERATED PROMPTS:")
        print(f"{'─'*70}")
        print(f"🔧 SYSTEM PROMPT:\n{system_prompt}")
        print(f"\n📝 USER PROMPT:\n{user_prompts[0]}")
        print(f"{'─'*70}")
        
        # Single API call with system + user messages
        print(f"\n⏳ Categorizing {len(PRODUCTS)} products...")
        
        start_time = time.time()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompts[0]}
        ]
        
        response = executor.execute(
            input_data=messages,
            config_overrides={"response_format": "json", "temperature": 0.1}
        )
        
        elapsed = time.time() - start_time
        print(f"   ✅ Response received in {elapsed:.2f}s")
        
        # Parse response
        response_text = response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            results = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"\n⚠️ JSON parse error: {e}")
            print(f"\n📄 Raw response:\n{response_text[:500]}...")
            import re
            match = re.search(r'\[[\s\S]*\]', response_text)
            if match:
                results = json.loads(match.group())
            else:
                raise
        
        # Show results
        print(f"\n{'='*70}")
        print("📊 CATEGORIZATION RESULTS:")
        print(f"{'='*70}")
        
        # Group by category
        by_category = {}
        
        for item in results:
            product = item.get("product", "?")
            category = item.get("category", "?")
            confidence = item.get("confidence", "?")
            
            # Emoji for confidence
            conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
            
            print(f"\n  📦 \"{product}\"")
            print(f"     Category: {category} {conf_emoji} ({confidence})")
            
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(product)
        
        # Summary
        print(f"\n{'─'*70}")
        print("📊 SUMMARY BY CATEGORY:")
        for cat, items in by_category.items():
            print(f"   📂 {cat}: {len(items)}")
            for item in items:
                print(f"      • {item}")
        print(f"{'─'*70}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    run_example()
