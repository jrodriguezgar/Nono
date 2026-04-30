"""
Example: Decompose Tasks - Break down "Make a hamburger" into subtasks.
Usage: python decompose_tasks_example.py

This example demonstrates:
1. Using AI to decompose a complex task into subtasks
2. Identifying dependencies and priorities
3. Estimating effort for each subtask
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

# Main task to decompose
TASK = "Make a gourmet hamburger from scratch"
DESCRIPTION = """
Create a complete gourmet hamburger including:
- Homemade beef patty
- Fresh vegetables (lettuce, tomato, onion)
- Toasted brioche bun
- Special sauce
- Proper cooking and assembly
"""


def run_example():
    """Decompose hamburger-making task into subtasks."""
    print(f"\n{'='*70}")
    print("  Decompose Tasks Example - Gourmet Hamburger")
    print(f"{'='*70}")
    
    # Show the task
    print(f"\n{'─'*70}")
    print("📋 TASK TO DECOMPOSE:")
    print(f"{'─'*70}")
    print(f"  {TASK}")
    print(f"\n  Description:")
    for line in DESCRIPTION.strip().split('\n'):
        print(f"  {line}")
    
    print(f"\n{'─'*70}")
    print(f"Provider: {PROVIDER.upper()} | Model: {MODEL}")
    print(f"API calls: 1")
    print(f"{'─'*70}")
    
    if input(f"\nDecompose task with AI? (y/n): ").lower() != 'y':
        print("\n✅ Cancelled.")
        return
    
    print(f"\n⏳ Initializing TaskExecutor...")
    
    try:
        # Create TaskExecutor (API key resolved automatically)
        executor = TaskExecutor(provider=PROVIDER, model=MODEL)
        print(f"   ✅ TaskExecutor created ({PROVIDER.upper()} / {MODEL})")
        
        # Build prompts using the decompose_tasks.j2 template
        prompts = build_from_file_blocks(
            "decompose_tasks.j2",
            task=TASK,
            description=DESCRIPTION,
            granularity="Medium (5-10 subtasks)",
            max_subtasks=10,
            context="This is for a home kitchen with standard equipment"
        )
        
        print(f"\n📋 GENERATED PROMPTS:")
        print(f"{'─'*70}")
        print(f"🔧 SYSTEM PROMPT (preview):\n{prompts['system'][:300]}...")
        print(f"\n📝 USER PROMPT:\n{prompts['user']}")
        print(f"{'─'*70}")
        
        # Single API call with system + user messages
        print(f"\n⏳ Decomposing task...")
        
        start_time = time.time()
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]}
        ]
        
        response = executor.execute(
            input_data=messages,
            config_overrides={"response_format": "json", "temperature": 0.2, "max_tokens": 4096}
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
            
            # Try to repair truncated JSON
            import re
            result = None
            
            # First try: find complete JSON object
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                try:
                    result = json.loads(match.group())
                    print("   ✅ Extracted valid JSON from response")
                except json.JSONDecodeError:
                    pass
            
            # Second try: repair truncated JSON with improved algorithm
            if result is None:
                repaired = response_text
                
                # Step 1: Close any unterminated strings
                # Count quotes (ignoring escaped ones)
                quote_count = len(re.findall(r'(?<!\\)"', repaired))
                if quote_count % 2 != 0:
                    # Find last quote position and close the string
                    repaired = repaired.rstrip()
                    if not repaired.endswith('"'):
                        repaired += '"'
                
                # Step 2: Remove trailing incomplete elements
                # Remove incomplete key-value after last complete one
                repaired = re.sub(r',\s*"[^"]*"\s*:\s*"[^"]*$', '', repaired)  # incomplete string value
                repaired = re.sub(r',\s*"[^"]*"\s*:\s*\[[^\]]*$', '', repaired)  # incomplete array value
                repaired = re.sub(r',\s*"[^"]*"\s*:\s*\{[^}]*$', '', repaired)  # incomplete object value
                repaired = re.sub(r',\s*"[^"]*"\s*:\s*$', '', repaired)  # key with no value
                repaired = re.sub(r',\s*"[^"]*$', '', repaired)  # incomplete key
                repaired = re.sub(r',\s*$', '', repaired)  # trailing comma
                
                # Step 3: Track nesting and close in correct order
                stack = []
                in_string = False
                escape_next = False
                
                for char in repaired:
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"':
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if char == '{':
                        stack.append('}')
                    elif char == '[':
                        stack.append(']')
                    elif char in '}]' and stack and stack[-1] == char:
                        stack.pop()
                
                # Close in reverse order
                repaired += ''.join(reversed(stack))
                
                try:
                    result = json.loads(repaired)
                    print("   ✅ Repaired truncated JSON")
                except json.JSONDecodeError as repair_err:
                    print(f"   ❌ Could not repair JSON: {repair_err}")
                    raise
        
        # Show subtasks
        print(f"\n{'='*70}")
        print("📝 DECOMPOSED SUBTASKS:")
        print(f"{'='*70}")
        
        subtasks = result.get("subtasks", [])
        for task in subtasks:
            task_id = task.get("id", "?")
            title = task.get("title", "")
            description = task.get("description", "")
            effort = task.get("effort", "?")
            priority = task.get("priority", "medium")
            deps = task.get("dependencies", [])
            
            # Priority emoji
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
            
            print(f"\n  [{task_id}] {priority_emoji} {title}")
            print(f"      {description}")
            print(f"      ⏱️  Effort: {effort}")
            if deps:
                print(f"      🔗 Depends on: {', '.join(deps)}")
        
        # Show suggested order
        order = result.get("suggested_order", [])
        if order:
            print(f"\n{'─'*70}")
            print(f"📊 SUGGESTED ORDER: {' → '.join(order)}")
        
        # Show total effort
        total = result.get("total_effort", "Unknown")
        print(f"\n{'─'*70}")
        print(f"⏱️  TOTAL EFFORT: {total}")
        print(f"   SUBTASKS: {len(subtasks)}")
        print(f"{'─'*70}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    run_example()
