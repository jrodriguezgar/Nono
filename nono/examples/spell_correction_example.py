"""
Example: Spell Correction - Fix spelling errors in text.
Usage: python spell_correction_example.py

This example demonstrates:
1. Using AI to correct spelling errors with TaskExecutor (genai_tasker)
2. Using Jinja2 templates (spell_correction.j2) for prompts
3. Batch processing multiple texts
4. Tracking which texts were corrected
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nono.tasker import TaskExecutor, build_from_file_blocks
import time

# Configuration
PROVIDER = "openrouter"
MODEL = "openrouter/auto"

# Texts with spelling errors (Spanish and English mixed)
TEXTS_WITH_ERRORS = [
    "Hola, ¿cómo estás?",              # Correct
    "Tengo una pregunta mui importante",  # mui → muy
    "The wether is very nice today",      # wether → weather
    "Necesito comprar leche y pan",       # Correct
    "I have recieved your mesage",        # recieved → received, mesage → message
    "El perro esta jugando en el parqe",  # esta → está, parqe → parque
    "Thankyou for your help",             # Thankyou → Thank you
    "Buenos días, ¿qué hora es?",         # Correct
]


def run_example():
    """Correct spelling errors in texts."""
    print(f"\n{'='*70}")
    print("  Spell Correction Example")
    print(f"{'='*70}")
    
    # Show input texts
    print(f"\n{'─'*70}")
    print(f"📝 INPUT TEXTS ({len(TEXTS_WITH_ERRORS)} items):")
    print(f"{'─'*70}")
    for i, text in enumerate(TEXTS_WITH_ERRORS):
        print(f"   {i}. {text}")
    
    print(f"\n{'─'*70}")
    print(f"Provider: {PROVIDER.upper()} | Model: {MODEL}")
    print(f"API calls: 1")
    print(f"{'─'*70}")
    
    # Build prompts using the spell_correction.j2 template (before confirmation)
    prompts = build_from_file_blocks(
        "spell_correction.j2",
        data=TEXTS_WITH_ERRORS,
        language="auto"
    )
    
    # Show generated prompts
    print(f"\n{'─'*70}")
    print("📋 GENERATED PROMPTS:")
    print(f"{'─'*70}")
    print(f"\n🔧 SYSTEM PROMPT:\n{prompts['system']}")
    print(f"\n📝 USER PROMPT:\n{prompts['user']}")
    print(f"{'─'*70}")
    
    if input(f"\nCorrect spelling with AI? (y/n): ").lower() != 'y':
        print("\n✅ Cancelled.")
        return
    
    print(f"\n⏳ Initializing TaskExecutor...")
    
    try:
        # Create TaskExecutor (API key resolved automatically)
        executor = TaskExecutor(provider=PROVIDER, model=MODEL)
        print(f"   ✅ TaskExecutor created ({PROVIDER.upper()} / {MODEL})")
        print(f"   ✅ Prompt built from template: spell_correction.j2")
        
        # Single API call with system + user messages
        print(f"\n⏳ Correcting {len(TEXTS_WITH_ERRORS)} texts...")
        
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
        print("✏️  CORRECTION RESULTS:")
        print(f"{'='*70}")
        
        corrected_count = 0
        unchanged_count = 0
        
        lines = response.strip().split('\n')
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            parts = line.split('\t')
            if len(parts) >= 3:
                key, corrected, was_changed = parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                key, corrected = parts[0], parts[1]
                was_changed = "0"
            else:
                continue
            
            # Get original text
            try:
                idx = int(key)
                original = TEXTS_WITH_ERRORS[idx] if idx < len(TEXTS_WITH_ERRORS) else "?"
            except:
                original = "?"
            
            # Check if changed
            changed = was_changed.strip() == "1"
            if changed:
                emoji = "✏️"
                corrected_count += 1
            else:
                emoji = "✅"
                unchanged_count += 1
            
            print(f"\n  {emoji} [{key}]")
            print(f"     Original:  {original}")
            print(f"     Corrected: {corrected}")
            if changed and original != corrected:
                print(f"     Status: CORRECTED")
            else:
                print(f"     Status: No changes needed")
        
        # Summary
        print(f"\n{'─'*70}")
        print("📊 SUMMARY:")
        print(f"   ✏️  Texts corrected: {corrected_count}")
        print(f"   ✅ Already correct: {unchanged_count}")
        print(f"{'─'*70}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    run_example()
