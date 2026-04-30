"""
Example: Name Classifier - Identify person names in a list of strings.
Usage: python name_classifier_example.py

This example demonstrates:
1. Using AI to classify strings as person names or not
2. Batch processing multiple items in a single API call
3. Confidence scoring for classifications
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nono.tasker import TaskExecutor, build_from_file_blocks
import json
import time

# Configuration
PROVIDER = "openrouter"
MODEL = "openrouter/auto"

# Mixed list of strings to classify
STRINGS = [
    "María García",
    "Mesa de cocina",
    "John Smith",
    "Python 3.11",
    "Carlos Rodríguez López",
    "Silla ergonómica",
    "Dr. Ana Martínez",
    "iPhone 15 Pro",
    "José Luis",
    "Laptop Dell XPS",
    "李明",
    "New York City",
    "Mohammed Al-Rashid",
]


def run_example():
    """Classify strings as person names or not."""
    print(f"\n{'='*70}")
    print("  Name Classifier Example")
    print(f"{'='*70}")
    
    # Show input strings
    print(f"\n{'─'*70}")
    print(f"📋 STRINGS TO CLASSIFY ({len(STRINGS)} items):")
    print(f"{'─'*70}")
    for i, s in enumerate(STRINGS, 1):
        print(f"  {i:2}. {s}")
    
    print(f"\n{'─'*70}")
    print(f"Provider: {PROVIDER.upper()} | Model: {MODEL}")
    print(f"API calls: 1")
    print(f"{'─'*70}")
    
    if input(f"\nClassify all strings with AI? (y/n): ").lower() != 'y':
        print("\n✅ Cancelled.")
        return
    
    print(f"\n⏳ Initializing TaskExecutor...")
    
    try:
        # Create TaskExecutor (API key resolved automatically)
        executor = TaskExecutor(provider=PROVIDER, model=MODEL)
        print(f"   ✅ TaskExecutor created ({PROVIDER.upper()} / {MODEL})")
        
        # Build prompts using the name_classifier.j2 template
        prompts = build_from_file_blocks(
            "name_classifier.j2",
            data=STRINGS
        )
        
        print(f"\n📋 GENERATED PROMPTS:")
        print(f"{'─'*70}")
        print(f"🔧 SYSTEM PROMPT:\n{prompts['system']}")
        print(f"\n📝 USER PROMPT:\n{prompts['user']}")
        print(f"{'─'*70}")
        
        # Single API call with system + user messages
        print(f"\n⏳ Classifying {len(STRINGS)} strings...")
        
        start_time = time.time()
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]}
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
        print("🏷️  CLASSIFICATION RESULTS:")
        print(f"{'='*70}")
        
        names = []
        not_names = []
        
        for item in results:
            text = item.get("text", "?")
            is_name = item.get("is_name", False)
            confidence = item.get("confidence", "?")
            reason = item.get("reason", "")
            
            # Emoji for result
            emoji = "👤" if is_name else "📦"
            conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
            
            print(f"\n  {emoji} \"{text}\"")
            print(f"     Is name: {'Yes' if is_name else 'No'} {conf_emoji} ({confidence})")
            print(f"     Reason: {reason}")
            
            if is_name:
                names.append(text)
            else:
                not_names.append(text)
        
        # Summary
        print(f"\n{'─'*70}")
        print("📊 SUMMARY:")
        print(f"   👤 Names found: {len(names)}")
        for name in names:
            print(f"      • {name}")
        print(f"   📦 Not names: {len(not_names)}")
        print(f"{'─'*70}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    run_example()
