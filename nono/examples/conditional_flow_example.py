"""
Example: Conditional Flow - Route support tickets to different queues.
Usage: python conditional_flow_example.py

This example demonstrates:
1. Using TaskExecutor with J2 templates
2. Processing 7 support tickets through 3 conditional nodes
3. Expected distribution: 3 → Urgent, 2 → Normal, 2 → Low priority

Flow diagram:
                    ┌─────────────┐
                    │   INPUT     │
                    │  7 tickets  │
                    └─────┬───────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
      ┌──────────┐  ┌──────────┐  ┌──────────┐
      │  URGENT  │  │  NORMAL  │  │   LOW    │
      │ 3 items  │  │ 2 items  │  │ 2 items  │
      └──────────┘  └──────────┘  └──────────┘
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

# 7 Support tickets with expected routing:
# - 3 should go to URGENT (system down, critical, payment failed)
# - 2 should go to NORMAL (feature questions)
# - 2 should go to LOW (documentation, typos)
TICKETS = [
    {"id": "T001", "subject": "CRITICAL: Production system is completely down!"},
    {"id": "T002", "subject": "How do I export my data to CSV?"},
    {"id": "T003", "subject": "Payment processing failed - customer charged twice!"},
    {"id": "T004", "subject": "Found a typo in the documentation page"},
    {"id": "T005", "subject": "URGENT: Security breach detected in our account"},
    {"id": "T006", "subject": "Where can I find the API documentation?"},
    {"id": "T007", "subject": "Can you explain how the billing cycle works?"},
]

# Route definitions
ROUTES = [
    {"id": "urgent", "name": "URGENT", "description": "Critical: system down, security, payment failures"},
    {"id": "normal", "name": "NORMAL", "description": "Feature questions, how-to, billing inquiries"},
    {"id": "low", "name": "LOW", "description": "Documentation, typos, minor issues"}
]


def run_example():
    """Process all 7 tickets in a single API call."""
    print(f"\n{'='*70}")
    print("  Conditional Flow Example - Support Ticket Router")
    print(f"{'='*70}")
    
    # Show tickets first
    print(f"\n{'─'*70}")
    print("📋 INPUT TICKETS (7 total):")
    print(f"{'─'*70}")
    for i, ticket in enumerate(TICKETS, 1):
        print(f"  {i}. [{ticket['id']}] {ticket['subject']}")
    
    print(f"\n{'─'*70}")
    print("🎯 AVAILABLE ROUTES:")
    print(f"{'─'*70}")
    for route in ROUTES:
        emoji = {"urgent": "🔴", "normal": "🟡", "low": "🟢"}[route['id']]
        print(f"  {emoji} {route['name']}: {route['description']}")
    
    print(f"\n{'─'*70}")
    print(f"Expected distribution: 3 URGENT, 2 NORMAL, 2 LOW")
    print(f"Provider: {PROVIDER.upper()} | Model: {MODEL}")
    print(f"API calls: 1 (batch processing)")
    print(f"{'─'*70}")
    
    # Build prompts using the conditional_flow.j2 template
    # Format tickets as input for batch processing
    tickets_input = "\n".join([f"[{t['id']}] {t['subject']}" for t in TICKETS])
    
    prompts = build_from_file_blocks(
        "conditional_flow.j2",
        input=tickets_input,
        input_type="Support tickets (batch of 7)",
        routes=ROUTES,
        default_route="low",
        context="Route each ticket to the appropriate queue. Return a JSON array with one object per ticket containing: ticket_id, route_id, reasoning.",
        multi_match=True
    )
    
    # Show generated prompts
    print(f"\n📋 GENERATED PROMPTS FROM TEMPLATE:")
    print(f"{'─'*70}")
    print(f"\n🔧 SYSTEM PROMPT (preview):\n{prompts['system'][:400]}...")
    print(f"\n📝 USER PROMPT:\n{prompts['user']}")
    print(f"{'─'*70}")
    
    if input(f"\nRoute all tickets with 1 API call? (y/n): ").lower() != 'y':
        print("\n✅ Cancelled.")
        return
    
    print(f"\n⏳ Initializing TaskExecutor...")
    
    try:
        # Create TaskExecutor (API key resolved automatically)
        executor = TaskExecutor(provider=PROVIDER, model=MODEL)
        print(f"   ✅ TaskExecutor created ({PROVIDER.upper()} / {MODEL})")
        
        # Single API call for all tickets with system + user messages
        print(f"\n⏳ Processing all 7 tickets in 1 API call...")
        
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
            decisions = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"\n⚠️ JSON parse error: {e}")
            print(f"\n📄 Raw response:\n{response_text[:500]}...")
            # Fallback: try to extract JSON array from response
            import re
            match = re.search(r'\[[\s\S]*\]', response_text)
            if match:
                decisions = json.loads(match.group())
            else:
                raise
        
        # Organize results
        results = {"urgent": [], "normal": [], "low": []}
        ticket_map = {t['id']: t for t in TICKETS}
        
        print(f"\n{'─'*70}")
        print("🔄 ROUTING DECISIONS:")
        print(f"{'─'*70}")
        
        for decision in decisions:
            ticket_id = decision.get("ticket_id", "")
            route_id = decision.get("route_id", "low").lower()
            reasoning = decision.get("reasoning", "")
            
            if ticket_id in ticket_map:
                ticket = ticket_map[ticket_id]
                if route_id in results:
                    results[route_id].append(ticket)
                else:
                    results["low"].append(ticket)
                
                emoji = {"urgent": "🔴", "normal": "🟡", "low": "🟢"}.get(route_id, "⚪")
                print(f"  {emoji} [{ticket_id}] → {route_id.upper()}: {reasoning}")
        
        # Show final distribution
        print(f"\n{'='*70}")
        print("📊 FINAL DISTRIBUTION:")
        print(f"{'='*70}")
        
        for route_id, route_name, emoji in [("urgent", "URGENT", "🔴"), ("normal", "NORMAL", "🟡"), ("low", "LOW", "🟢")]:
            tickets_in_route = results[route_id]
            print(f"\n{emoji} {route_name} ({len(tickets_in_route)} tickets):")
            if tickets_in_route:
                for t in tickets_in_route:
                    print(f"   - [{t['id']}] {t['subject'][:50]}")
            else:
                print("   (empty)")
        
        # Summary
        print(f"\n{'─'*70}")
        print("📈 SUMMARY:")
        print(f"   🔴 URGENT: {len(results['urgent'])} tickets (expected: 3)")
        print(f"   🟡 NORMAL: {len(results['normal'])} tickets (expected: 2)")
        print(f"   🟢 LOW:    {len(results['low'])} tickets (expected: 2)")
        print(f"   ⚡ API calls: 1 (saved 6 calls!)")
        print(f"{'─'*70}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    run_example()
