"""
Example: Data Loss Prevention - Anonymize sensitive data in text.
Usage: python data_loss_prevention_example.py

This example demonstrates:
1. Using AI to detect and anonymize PII (Personally Identifiable Information)
2. Processing multiple text samples in a single API call
3. GDPR/CCPA compliance through data redaction
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

# Sample texts with sensitive data
SAMPLE_TEXTS = [
    "Contact John Smith at john.smith@email.com or call 555-123-4567.",
    "Patient Maria García, DOB: 15/03/1985, SSN: 123-45-6789, diagnosed with diabetes.",
    "Send payment to account ES91 2100 0418 4502 0005 1332. Card: 4532-1234-5678-9012.",
]


def run_example():
    """Anonymize sensitive data in sample texts."""
    print(f"\n{'='*70}")
    print("  Data Loss Prevention Example - PII Anonymization")
    print(f"{'='*70}")
    
    # Show input texts
    print(f"\n{'─'*70}")
    print("📋 INPUT TEXTS (with sensitive data):")
    print(f"{'─'*70}")
    for i, text in enumerate(SAMPLE_TEXTS, 1):
        print(f"  {i}. {text}")
    
    print(f"\n{'─'*70}")
    print(f"Provider: {PROVIDER.upper()} | Model: {MODEL}")
    print(f"API calls: 1 (batch processing)")
    print(f"{'─'*70}")
    
    # Build prompts using the data_loss_prevention.j2 template
    # Format texts as a numbered batch for processing
    texts_batch = "\n".join([f"{i+1}. {t}" for i, t in enumerate(SAMPLE_TEXTS)])
    
    prompts = build_from_file_blocks(
        "data_loss_prevention.j2",
        text=texts_batch,
        data_types=["Names", "Emails", "Phone numbers", "Dates of birth", "SSN/IDs", "Bank accounts", "Credit cards", "Medical information"],
        replacement_style="brackets (e.g., [NAME], [EMAIL], [PHONE])",
        context="Process as a batch of 3 texts. Return a JSON array with one object per text containing: text_id, original, anonymized, redactions.",
        compliance="GDPR, CCPA"
    )
    
    # Show generated prompts
    print(f"\n📋 GENERATED PROMPTS FROM TEMPLATE:")
    print(f"{'─'*70}")
    print(f"\n🔧 SYSTEM PROMPT (preview):\n{prompts['system'][:400]}...")
    print(f"\n📝 USER PROMPT:\n{prompts['user']}")
    print(f"{'─'*70}")
    
    if input(f"\nAnonymize all texts with 1 API call? (y/n): ").lower() != 'y':
        print("\n✅ Cancelled.")
        return
    
    print(f"\n⏳ Initializing TaskExecutor...")
    
    try:
        # Create TaskExecutor (API key resolved automatically)
        executor = TaskExecutor(provider=PROVIDER, model=MODEL)
        print(f"   ✅ TaskExecutor created ({PROVIDER.upper()} / {MODEL})")
        
        # Single API call with system + user messages
        print(f"\n⏳ Anonymizing {len(SAMPLE_TEXTS)} texts...")
        
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
        
        # Parse response - clean markdown wrapping
        response_text = response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Try to parse JSON
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
        print("🔒 ANONYMIZATION RESULTS:")
        print(f"{'='*70}")
        
        total_redactions = 0
        for result in results:
            text_id = result.get("text_id", "?")
            original = result.get("original", "")
            anonymized = result.get("anonymized", "")
            redactions = result.get("redactions", [])
            
            print(f"\n📄 Text {text_id}:")
            print(f"   Original:   {original}")
            print(f"   Anonymized: {anonymized}")
            
            if redactions:
                print(f"   Redactions ({len(redactions)}):")
                for r in redactions:
                    rtype = r.get("type", "unknown")
                    orig = r.get("original", "")
                    repl = r.get("replacement", "")
                    print(f"      • {rtype}: '{orig}' → '{repl}'")
                total_redactions += len(redactions)
        
        # Summary
        print(f"\n{'─'*70}")
        print("📈 SUMMARY:")
        print(f"   Texts processed: {len(results)}")
        print(f"   Total redactions: {total_redactions}")
        print(f"   API calls: 1")
        print(f"   Time: {elapsed:.2f}s")
        print(f"{'─'*70}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    run_example()
