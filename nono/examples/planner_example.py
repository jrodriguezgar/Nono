"""
Example: Planner - Create a structured plan for a goal.
Usage: python planner_example.py

This example demonstrates:
1. Using AI to generate a detailed project plan
2. Breaking goals into phases with milestones
3. Identifying dependencies and risks
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

# Goal to plan
GOAL = "Learn Python programming from scratch"
CONTEXT = "Complete beginner with no prior programming experience"
TIMELINE = "3 months"
CONSTRAINTS = [
    "2 hours per day available",
    "Limited budget (prefer free resources)",
    "Learn by doing (practical projects)",
]


def run_example():
    """Create a plan for learning Python."""
    print(f"\n{'='*70}")
    print("  Planner Example - Learning Python")
    print(f"{'='*70}")
    
    # Show the goal
    print(f"\n{'─'*70}")
    print("🎯 GOAL:")
    print(f"   {GOAL}")
    print(f"\n📋 CONTEXT: {CONTEXT}")
    print(f"⏰ TIMELINE: {TIMELINE}")
    print(f"\n🚧 CONSTRAINTS:")
    for c in CONSTRAINTS:
        print(f"   • {c}")
    
    print(f"\n{'─'*70}")
    print(f"Provider: {PROVIDER.upper()} | Model: {MODEL}")
    print(f"API calls: 1")
    print(f"{'─'*70}")
    
    if input(f"\nGenerate plan with AI? (y/n): ").lower() != 'y':
        print("\n✅ Cancelled.")
        return
    
    print(f"\n⏳ Initializing TaskExecutor...")
    
    try:
        # Create TaskExecutor (API key resolved automatically)
        executor = TaskExecutor(provider=PROVIDER, model=MODEL)
        print(f"   ✅ TaskExecutor created ({PROVIDER.upper()} / {MODEL})")
        
        # Build prompts using the planner.j2 template
        prompts = build_from_file_blocks(
            "planner.j2",
            goal=GOAL,
            context=CONTEXT,
            timeline=TIMELINE,
            constraints=CONSTRAINTS
        )
        
        print(f"\n📋 GENERATED PROMPTS:")
        print(f"{'─'*70}")
        print(f"🔧 SYSTEM PROMPT (preview):\n{prompts['system'][:300]}...")
        print(f"\n📝 USER PROMPT:\n{prompts['user']}")
        print(f"{'─'*70}")
        
        # Single API call with system + user messages
        print(f"\n⏳ Generating plan...")
        
        start_time = time.time()
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]}
        ]
        
        response = executor.execute(
            input_data=messages,
            config_overrides={"response_format": "json", "temperature": 0.3}
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
            plan = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"\n⚠️ JSON parse error: {e}")
            print(f"\n📄 Raw response:\n{response_text[:500]}...")
            import re
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                plan = json.loads(match.group())
            else:
                raise
        
        # Show plan
        print(f"\n{'='*70}")
        print("📋 YOUR LEARNING PLAN:")
        print(f"{'='*70}")
        
        print(f"\n🎯 Goal: {plan.get('goal', GOAL)}")
        print(f"📝 Summary: {plan.get('summary', '')}")
        
        # Show phases
        phases = plan.get("phases", [])
        for phase in phases:
            num = phase.get("phase_number", "?")
            name = phase.get("name", "")
            desc = phase.get("description", "")
            duration = phase.get("duration", "?")
            milestones = phase.get("milestones", [])
            deliverables = phase.get("deliverables", [])
            
            print(f"\n{'─'*70}")
            print(f"📌 PHASE {num}: {name}")
            print(f"   ⏱️  Duration: {duration}")
            print(f"   📖 {desc}")
            
            if milestones:
                print(f"   🏁 Milestones:")
                for m in milestones:
                    print(f"      • {m}")
            
            if deliverables:
                print(f"   ✅ Deliverables:")
                for d in deliverables:
                    print(f"      • {d}")
        
        # Success criteria
        criteria = plan.get("success_criteria", [])
        if criteria:
            print(f"\n{'─'*70}")
            print("🏆 SUCCESS CRITERIA:")
            for c in criteria:
                print(f"   ✓ {c}")
        
        # Summary
        total = plan.get("total_duration", TIMELINE)
        print(f"\n{'─'*70}")
        print(f"⏱️  TOTAL DURATION: {total}")
        print(f"📊 PHASES: {len(phases)}")
        print(f"{'─'*70}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    run_example()
