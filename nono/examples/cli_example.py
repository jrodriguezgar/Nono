"""
Example usage of the Nono CLI module.

Demonstrates:
    - Factory function for quick CLI creation
    - Manual CLI construction with custom groups
    - All output utilities (colored prints, tables, progress, etc.)
    - Confirmation prompts
    - Statistics tracking

Run: python -m nono.examples.cli_example --help
Run: python -m nono.examples.cli_example --provider gemini --prompt "Hello world"
Run: python -m nono.examples.cli_example --dry-run -v
"""

import time
from nono.cli import (
    CLIBase,
    create_cli,
    Colors,
    OutputFormat,
    AIProvider,
    # Output utilities
    cprint,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_debug,
    print_header,
    print_subheader,
    print_table,
    print_key_value,
    print_summary,
    print_progress,
    print_spinner,
    confirm_action,
    format_duration,
)


def demo_colored_output():
    """Demonstrate colored output utilities."""
    print_header("COLORED OUTPUT DEMO")
    
    # Status messages
    print_success("This is a success message")
    print_error("This is an error message")
    print_warning("This is a warning message")
    print_info("This is an info message")
    print_debug("This is a debug message (dim)")
    
    print()
    
    # Custom colored print
    cprint("Bold cyan text", Colors.CYAN, bold=True)
    cprint("Dim gray text", Colors.GRAY, dim=True)
    cprint("Magenta highlight", Colors.MAGENTA)


def demo_tables():
    """Demonstrate table output."""
    print_header("TABLE OUTPUT DEMO")
    
    headers = ["Provider", "Model", "Status", "Requests"]
    rows = [
        ["Gemini", "gemini-3-flash-preview", "Active", 150],
        ["OpenAI", "gpt-4o-mini", "Active", 89],
        ["Perplexity", "sonar", "Limited", 23],
        ["Ollama", "llama3", "Offline", 0],
    ]
    
    print_table(headers, rows)
    
    print_subheader("With Index Column")
    print_table(headers, rows, show_index=True)


def demo_key_value():
    """Demonstrate key-value output."""
    print_header("KEY-VALUE OUTPUT DEMO")
    
    print_key_value("provider", "gemini")
    print_key_value("model", "gemini-3-flash-preview")
    print_key_value("temperature", 0.7)
    print_key_value("max_tokens", 4096)
    print_key_value("success_count", 95)
    print_key_value("error_count", 5)
    print_key_value("is_active", True)
    print_key_value("is_deprecated", False)


def demo_progress():
    """Demonstrate progress bar."""
    print_header("PROGRESS BAR DEMO")
    
    total = 50
    print_info("Processing items...")
    for i in range(total + 1):
        print_progress(i, total, prefix="Progress", suffix="Complete")
        time.sleep(0.02)
    
    print()
    print_info("Custom progress bar:")
    for i in range(total + 1):
        print_progress(i, total, prefix="Download", suffix="", fill="▓", empty="░")
        time.sleep(0.02)


def demo_spinner():
    """Demonstrate spinner animation."""
    print_header("SPINNER DEMO")
    
    update = print_spinner("Loading data...")
    for _ in range(30):
        update()
        time.sleep(0.05)
    print("\r" + " " * 50 + "\r", end="")  # Clear line
    print_success("Data loaded!")


def demo_summary():
    """Demonstrate summary statistics output."""
    print_header("SUMMARY OUTPUT DEMO")
    
    stats = {
        'total_requests': 253,
        'success': 240,
        'failed': 8,
        'skipped': 5,
        'tokens_used': 125000,
        'elapsed_time': '2m 34s',
    }
    
    print_summary(stats, title="EXECUTION SUMMARY")


def demo_confirmation():
    """Demonstrate confirmation prompt."""
    print_header("CONFIRMATION PROMPT DEMO")
    
    print_info("Confirmation prompts wait for user input.")
    print_info("Skipping in this demo to avoid blocking.\n")
    
    # Uncomment to test interactive confirmation:
    # if confirm_action("Do you want to proceed with this operation?", default=False):
    #     print_success("User confirmed!")
    # else:
    #     print_warning("User cancelled.")


def demo_format_duration():
    """Demonstrate duration formatting."""
    print_header("DURATION FORMATTING DEMO")
    
    durations = [0.05, 0.5, 1.5, 30, 90, 3600, 7380]
    
    for seconds in durations:
        formatted = format_duration(seconds)
        print(f"  {seconds:>8} seconds → {formatted}")


def demo_cli_factory():
    """Demonstrate CLI creation with factory function."""
    print_header("CLI FACTORY DEMO")
    
    cli = create_cli(
        prog="demo_tool",
        description="Demo tool for CLI features",
        version="1.0.0",
        with_provider=True,
        with_task=True,
        with_io=True,
        with_batch=True
    )
    
    cli.add_examples([
        "%(prog)s --provider gemini --prompt 'Hello' -o output.txt",
        "%(prog)s --batch --batch-file inputs.txt",
    ])
    
    print_info("CLI created with factory function")
    print_info(f"  Program: {cli.config.prog_name}")
    print_info(f"  Version: {cli.config.version}")
    print_info(f"  Argument groups: {list(cli._groups.keys())}")


def demo_cli_manual():
    """Demonstrate manual CLI construction."""
    print_header("MANUAL CLI CONSTRUCTION DEMO")
    
    cli = CLIBase(
        prog="custom_tool",
        description="Custom tool with specific needs",
        version="2.0.0"
    )
    
    # Add only the groups we need
    cli.add_ai_provider_group()
    
    # Add custom group
    custom_group = cli.add_group("custom", "Custom Options")
    custom_group.add_argument('--custom-flag', action='store_true', help="Custom flag")
    custom_group.add_argument('--custom-value', type=int, default=42, help="Custom value")
    
    print_info("CLI created manually with custom group")
    print_info(f"  Groups: {list(cli._groups.keys())}")


def demo_statistics_tracking():
    """Demonstrate statistics tracking."""
    print_header("STATISTICS TRACKING DEMO")
    
    cli = CLIBase(prog="stats_demo", version="1.0.0")
    
    # Simulate work with stats tracking
    cli.start_time = __import__('datetime').datetime.now()
    
    cli.increment_stat('requests', 100)
    cli.increment_stat('tokens', 5000)
    cli.increment_stat('tokens', 3000)  # Accumulates
    cli.set_stat('provider', 'gemini')
    cli.set_stat('model', 'gemini-3-flash-preview')
    cli.increment_stat('errors', 3)
    cli.increment_stat('success', 97)
    
    # Simulate some elapsed time
    time.sleep(0.1)
    
    print_info(f"Elapsed time: {cli.get_elapsed_time()}")
    print_info(f"Stats collected: {cli.stats}")
    
    cli.print_final_summary(title="DEMO STATISTICS")


def main():
    """Run all demos."""
    print()
    cprint("=" * 70, Colors.MAGENTA, bold=True)
    cprint(" NONO CLI MODULE DEMONSTRATION", Colors.MAGENTA, bold=True)
    cprint("=" * 70, Colors.MAGENTA, bold=True)
    
    demo_colored_output()
    demo_tables()
    demo_key_value()
    demo_progress()
    demo_spinner()
    demo_summary()
    demo_confirmation()
    demo_format_duration()
    demo_cli_factory()
    demo_cli_manual()
    demo_statistics_tracking()
    
    print()
    cprint("=" * 70, Colors.MAGENTA, bold=True)
    cprint(" ALL DEMOS COMPLETED SUCCESSFULLY!", Colors.GREEN, bold=True)
    cprint("=" * 70, Colors.MAGENTA, bold=True)
    print()


if __name__ == "__main__":
    main()
