"""
Example: Calculator - Generate and execute a calculator using CodeExecuter.
Usage: python calculator_executer.py

This example demonstrates:
1. Using CodeExecuter to generate Python code from natural language
2. Automatic execution in a secure sandbox (SAFE mode)
3. Retry on error capability
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nono.executer import CodeExecuter, SecurityMode


def run_example():
    """Generate and execute calculator code using CodeExecuter."""
    print(f"\n{'='*60}")
    print("  Calculator Example (CodeExecuter)")
    print(f"{'='*60}")
    
    # Initialize CodeExecuter with Google Gemini
    executer = CodeExecuter(
        provider="google",
        model_name="gemini-3-flash-preview",
        temperature="math"
    )
    
    # Instruction for the calculator
    instruction = """simple calculator that:
- Asks for two numbers and an operator (+, -, *, /)
- Performs the calculation
- Handles division by zero
- Prints the result"""

    print(f"\n[Topic]\n{'─'*60}")
    print(instruction)
    print(f"{'─'*60}\n")
    
    # Generate and execute
    print("⏳ Generating and executing code...")
    result = executer.run(
        instruction=instruction,
        retry_on_error=True,
        max_retries=2,
        save_execution=True
    )
    
    # Show results
    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"Success: {'✅' if result.success else '❌'} {result.success}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    print(f"Execution ID: {result.execution_id}")
    
    print(f"\n[Generated Code]\n{'─'*60}")
    print(result.code)
    print(f"{'─'*60}")
    
    if result.output:
        print(f"\n[Output]\n{'─'*60}")
        print(result.output)
        print(f"{'─'*60}")
    
    if result.error:
        print(f"\n[Error]\n{'─'*60}")
        print(result.error)
        print(f"{'─'*60}")


if __name__ == "__main__":
    run_example()
