"""
Example: Simple Prompt - Execute a prompt directly without templates.
Usage: python simple_prompt_example.py

This example demonstrates:
1. The simplest way to use the GenAI connector
2. Direct prompt execution without templates
3. Interactive prompt input from user
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nono.tasker import TaskExecutor
import time

# Configuration
PROVIDER = "openrouter"
MODEL = "openrouter/auto"

# Default prompt examples
DEFAULT_PROMPTS = [
    "Explain what Python is in 2 sentences.",
    "Give me 3 ideas for a weekend project.",
    "Translate 'Hello, how are you?' to Spanish, French, and German.",
]


def run_example():
    """Execute a simple prompt directly."""
    print(f"\n{'='*60}")
    print("  Simple Prompt Example")
    print(f"{'='*60}")
    
    # Show default prompts
    print(f"\n📝 EXAMPLE PROMPTS:")
    for i, p in enumerate(DEFAULT_PROMPTS, 1):
        print(f"   {i}. {p}")
    
    # Get user prompt
    print(f"\n{'─'*60}")
    choice = input("Enter a number (1-3) or type your own prompt: ").strip()
    
    if choice in ['1', '2', '3']:
        prompt = DEFAULT_PROMPTS[int(choice) - 1]
    elif choice:
        prompt = choice
    else:
        prompt = DEFAULT_PROMPTS[0]
    
    print(f"\n🎯 PROMPT: {prompt}")
    print(f"{'─'*60}")
    
    if input(f"\nExecute with AI? (y/n): ").lower() != 'y':
        print("\n✅ Cancelled.")
        return
    
    print(f"\n⏳ Initializing TaskExecutor...")
    
    try:
        # Create TaskExecutor (API key resolved automatically)
        executor = TaskExecutor(provider=PROVIDER, model=MODEL)
        
        print(f"⏳ Generating response...")
        start_time = time.time()
        
        response = executor.execute(
            input_data=prompt,
            config_overrides={"temperature": 0.7}
        )
        
        elapsed = time.time() - start_time
        
        # Show response
        print(f"\n{'='*60}")
        print("🤖 RESPONSE:")
        print(f"{'='*60}")
        print(response)
        print(f"{'─'*60}")
        print(f"⏱️  Time: {elapsed:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    run_example()
