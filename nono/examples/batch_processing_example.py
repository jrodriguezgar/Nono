"""
Example: Batch Processing with Automatic Splitting (On-the-Fly Template)

Demonstrates automatic batching for large datasets using build_prompts()
with Jinja2 templates defined as strings (no .j2 file needed).

Features demonstrated:
    - On-the-fly Jinja2 template definition
    - Automatic splitting based on max_tokens
    - Batch generation without API calls
    - Summary statistics at the end

Usage: python batch_processing_example.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nono.tasker import build_prompts

# ============================================================================
# On-the-fly Jinja2 Template for Product Classification
# ============================================================================

TEMPLATE = """Classify the following products into categories.

Products to classify:
{% for item in data %}
- ID: {{ item.id }} | Name: {{ item.name }} | Description: {{ item.description }}
{% endfor %}

Available categories: Electronics, Fashion, Home, Sports, Books

Respond in JSON format:
{"classifications": [{"id": <product_id>, "category": "<category>"}]}"""


def generate_products(count: int) -> list:
    """Generate sample product data."""
    categories = ["Laptop", "Headphones", "Shoes", "Jacket", "Book", "Watch", "Chair", "Ball"]
    descriptions = [
        "High-performance device for professionals",
        "Premium noise-canceling audio",
        "Comfortable running footwear",
        "Waterproof outdoor apparel",
        "Bestselling novel 2025",
        "Smart fitness tracker",
        "Ergonomic office furniture",
        "Professional sports equipment",
    ]
    
    products = []
    for i in range(1, count + 1):
        idx = (i - 1) % len(categories)
        products.append({
            "id": i,
            "name": f"{categories[idx]} Pro {i}",
            "description": descriptions[idx]
        })
    
    return products


def main():
    """Batch processing example with automatic splitting."""
    print("\n" + "=" * 70)
    print("  Batch Processing with Automatic Splitting (On-the-Fly Template)")
    print("=" * 70)
    
    # Generate a large dataset
    num_products = 50
    products = generate_products(num_products)
    
    print(f"\n📦 Generated {num_products} products")
    print(f"\n📋 Sample products (first 5):")
    for p in products[:5]:
        print(f"   • ID {p['id']:2}: {p['name']} - {p['description'][:40]}...")
    
    # Use small max_tokens to force multiple batches
    max_tokens = 500
    
    print(f"\n{'─'*70}")
    print(f"⚙️  Configuration:")
    print(f"   • Total products: {num_products}")
    print(f"   • Max tokens per batch: {max_tokens}")
    print(f"{'─'*70}")
    
    # Build prompts with automatic batching (NO API call)
    prompts = build_prompts(
        template=TEMPLATE,
        data=products,
        data_key="data",
        max_tokens=max_tokens,
    )
    
    print(f"\n✨ Generated {len(prompts)} batch(es)")
    
    # Show each batch
    for i, prompt in enumerate(prompts, 1):
        # Count products in this batch (count "- ID:" occurrences)
        items_in_batch = prompt.count("- ID:")
        prompt_length = len(prompt)
        
        print(f"\n{'─'*70}")
        print(f"📄 BATCH {i}/{len(prompts)} ({items_in_batch} products, {prompt_length} chars)")
        print(f"{'─'*70}")
        
        # Show truncated prompt
        lines = prompt.split('\n')
        if len(lines) > 15:
            for line in lines[:6]:
                print(f"   {line}")
            print(f"   ... ({len(lines) - 12} lines omitted) ...")
            for line in lines[-6:]:
                print(f"   {line}")
        else:
            for line in lines:
                print(f"   {line}")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 BATCH SUMMARY")
    print(f"{'='*70}")
    print(f"   Total products:     {num_products}")
    print(f"   Max tokens/batch:   {max_tokens}")
    print(f"   Batches generated:  {len(prompts)}")
    
    # Detailed batch stats
    print(f"\n   {'Batch':<8} {'Products':<12} {'Characters':<12}")
    print(f"   {'-'*32}")
    
    total_chars = 0
    total_items = 0
    for i, prompt in enumerate(prompts, 1):
        items = prompt.count("- ID:")
        chars = len(prompt)
        total_chars += chars
        total_items += items
        print(f"   {i:<8} {items:<12} {chars:<12}")
    
    print(f"   {'-'*32}")
    print(f"   {'Total':<8} {total_items:<12} {total_chars:<12}")
    
    avg_per_batch = total_items / len(prompts) if prompts else 0
    print(f"\n   Avg products/batch: {avg_per_batch:.1f}")
    print(f"{'='*70}")
    print("\n✅ Done! (No API calls were made)")


if __name__ == "__main__":
    main()
