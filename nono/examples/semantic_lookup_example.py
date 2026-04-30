"""
Example: Semantic Lookup - Match input values to a reference list.
Usage: python semantic_lookup_example.py

This example demonstrates:
1. Using AI for semantic/fuzzy matching
2. Matching misspelled product names to a catalog
3. Handling exact, fuzzy, and no matches
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nono.tasker import TaskExecutor, build_from_file_blocks
import time

# Configuration
PROVIDER = "openrouter"
MODEL = "openrouter/auto"

# Reference list (product catalog)
REFERENCE_LIST = [
    "iPhone 15 Pro",
    "Samsung Galaxy S24",
    "MacBook Pro 14",
    "iPad Air",
    "AirPods Pro",
    "Apple Watch Series 9",
    "Sony WH-1000XM5",
    "Nintendo Switch OLED",
]

# Input data with typos and variations
INPUT_DATA = [
    "iphone 15 pro",          # exact (different case)
    "Samsun Galaxy S24",      # fuzzy (typo)
    "MacBook Pro",            # fuzzy (missing size)
    "iPod Air",               # fuzzy (wrong name)
    "AirPods",                # fuzzy (missing Pro)
    "Apple Wach",             # fuzzy (typo)
    "Sony Headphones",        # fuzzy (generic name)
    "PlayStation 5",          # none (not in catalog)
]


def run_example():
    """Match input values to reference list."""
    print(f"\n{'='*70}")
    print("  Semantic Lookup Example - Product Matching")
    print(f"{'='*70}")
    
    # Show reference list
    print(f"\n{'─'*70}")
    print("📦 REFERENCE CATALOG:")
    print(f"{'─'*70}")
    for item in REFERENCE_LIST:
        print(f"   • {item}")
    
    # Show input data
    print(f"\n{'─'*70}")
    print("🔍 INPUT DATA (with typos/variations):")
    print(f"{'─'*70}")
    for i, item in enumerate(INPUT_DATA):
        print(f"   {i}. {item}")
    
    print(f"\n{'─'*70}")
    print(f"Provider: {PROVIDER.upper()} | Model: {MODEL}")
    print(f"API calls: 1")
    print(f"{'─'*70}")
    
    if input(f"\nMatch input to catalog with AI? (y/n): ").lower() != 'y':
        print("\n✅ Cancelled.")
        return
    
    print(f"\n⏳ Initializing TaskExecutor...")
    
    try:
        # Create TaskExecutor (API key resolved automatically)
        executor = TaskExecutor(provider=PROVIDER, model=MODEL)
        print(f"   ✅ TaskExecutor created ({PROVIDER.upper()} / {MODEL})")
        
        # Build prompts using the semantic_lookup.j2 template
        prompts = build_from_file_blocks(
            "semantic_lookup.j2",
            reference_list=REFERENCE_LIST,
            data=INPUT_DATA
        )
        
        print(f"\n📋 GENERATED PROMPTS:")
        print(f"{'─'*70}")
        print(f"🔧 SYSTEM PROMPT:\n{prompts['system']}")
        print(f"\n📝 USER PROMPT:\n{prompts['user']}")
        print(f"{'─'*70}")
        
        # Single API call with system + user messages
        print(f"\n⏳ Matching {len(INPUT_DATA)} items...")
        
        start_time = time.time()
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]}
        ]
        
        response = executor.execute(
            input_data=messages,
            config_overrides={"temperature": 0.1}
        )
        
        elapsed = time.time() - start_time
        print(f"   ✅ Response received in {elapsed:.2f}s")
        
        # Parse TSV response
        print(f"\n{'='*70}")
        print("🔗 MATCHING RESULTS:")
        print(f"{'='*70}")
        
        exact_count = 0
        fuzzy_count = 0
        none_count = 0
        
        lines = response.strip().split('\n')
        for line in lines:
            if not line.strip():
                continue
            
            parts = line.split('\t')
            if len(parts) >= 4:
                key, original, matched, match_type = parts[0], parts[1], parts[2], parts[3]
            elif len(parts) == 3:
                key, original, match_type = parts[0], parts[1], parts[2]
                matched = ""
            else:
                continue
            
            # Emoji for match type
            match_type = match_type.lower().strip()
            if match_type == "exact":
                emoji = "✅"
                exact_count += 1
            elif match_type == "fuzzy":
                emoji = "🔶"
                fuzzy_count += 1
            else:
                emoji = "❌"
                none_count += 1
            
            print(f"\n  {emoji} [{key}] \"{original}\"")
            if matched:
                print(f"     → {matched} ({match_type})")
            else:
                print(f"     → No match found")
        
        # Summary
        print(f"\n{'─'*70}")
        print("📊 SUMMARY:")
        print(f"   ✅ Exact matches:  {exact_count}")
        print(f"   🔶 Fuzzy matches:  {fuzzy_count}")
        print(f"   ❌ No matches:     {none_count}")
        print(f"{'─'*70}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    run_example()
