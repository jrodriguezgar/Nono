"""
Example: Logical Ordering - Order the steps to make a hamburger.
Usage: python logical_ordering_example.py

This example demonstrates:
1. Using TaskExecutor with J2 templates (logical_ordering.j2)
2. Identifying dependencies between tasks
3. Finding parallelizable operations
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

# Steps to make a hamburger (unordered)
STEPS = [
    "Put lettuce and tomato on the bun",
    "Toast the bun",
    "Form the meat patty",
    "Grill the patty for 3-4 minutes per side",
    "Add cheese on the patty (last 30 seconds)",
    "Slice tomato and lettuce",
    "Assemble the hamburger",
    "Season the patty with salt and pepper",
    "Prepare condiments (ketchup, mustard, mayo)",
    "Let the patty rest for 1 minute",
]


def run_example():
    """Order hamburger-making steps logically."""
    print(f"\n{'='*70}")
    print("  Logical Ordering Example - Making a Hamburger")
    print(f"{'='*70}")
    
    # Show unordered steps
    print(f"\n{'─'*70}")
    print("📋 UNORDERED STEPS:")
    print(f"{'─'*70}")
    for i, step in enumerate(STEPS, 1):
        print(f"  {i}. {step}")
    
    print(f"\n{'─'*70}")
    print(f"Provider: {PROVIDER.upper()} | Model: {MODEL}")
    print(f"API calls: 1")
    print(f"{'─'*70}")
    
    if input(f"\nOrder steps with AI? (y/n): ").lower() != 'y':
        print("\n✅ Cancelled.")
        return
    
    print(f"\n⏳ Initializing TaskExecutor...")
    
    try:
        # Build prompts using the logical_ordering.j2 template
        prompts = build_from_file_blocks(
            "logical_ordering.j2",
            items=STEPS,
            criteria="Logical cooking sequence, respecting dependencies",
            context="Making a gourmet hamburger from scratch"
        )
        
        # Show generated prompts
        print(f"\n{'─'*70}")
        print("📋 GENERATED PROMPTS FROM TEMPLATE:")
        print(f"{'─'*70}")
        print(f"\n🔧 SYSTEM PROMPT:\n{prompts['system'][:300]}...")
        print(f"\n📝 USER PROMPT:\n{prompts['user']}")
        print(f"{'─'*70}")
        
        # Create TaskExecutor (API key resolved automatically)
        executor = TaskExecutor(provider=PROVIDER, model=MODEL)
        print(f"   ✅ TaskExecutor created ({PROVIDER.upper()} / {MODEL})")
        
        # Single API call with system + user messages
        print(f"\n⏳ Ordering {len(STEPS)} steps...")
        
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
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"\n⚠️ JSON parse error: {e}")
            print(f"\n📄 Raw response:\n{response_text[:500]}...")
            import re
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                result = json.loads(match.group())
            else:
                raise
        
        # Show ordered steps
        print(f"\n{'='*70}")
        print("🍔 ORDERED STEPS:")
        print(f"{'='*70}")
        
        # Handle both template schema (ordered_items) and simple schema (ordered_steps)
        ordered_items = result.get("ordered_items", result.get("ordered_steps", []))
        for item in ordered_items:
            pos = item.get("position", "?")
            step = item.get("item", item.get("step", ""))
            reason = item.get("reason", "")
            print(f"\n  {pos}. {step}")
            print(f"     └─ {reason}")
        
        # Show parallel tasks/groups
        groups = result.get("groups", [])
        if groups:
            print(f"\n{'─'*70}")
            print("⚡ PARALLEL GROUPS (can be done simultaneously):")
            for group in groups:
                if group.get("can_be_parallel"):
                    print(f"   📦 {group.get('group_name', 'Group')}:")
                    for g_item in group.get("items", []):
                        print(f"      • {g_item}")
        
        # Fallback to old parallel_tasks format
        parallel = result.get("parallel_tasks", [])
        if parallel:
            print(f"\n{'─'*70}")
            print("⚡ PARALLEL TASKS:")
            for task in parallel:
                print(f"   • {task}")
        
        # Show critical path
        critical = result.get("critical_path", [])
        if critical:
            print(f"\n{'─'*70}")
            print("🔴 CRITICAL PATH:")
            print(f"   {' → '.join(str(c) for c in critical)}")
        
        # Show notes
        notes = result.get("notes", "")
        if notes:
            print(f"\n📝 Notes: {notes}")
        
        print(f"\n{'─'*70}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    run_example()
