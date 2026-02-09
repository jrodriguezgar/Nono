"""
Nono CLI - Command Line Interface for GenAI Tasker

Execute generative AI tasks from the command line with multi-provider support.

Usage:
    python -m nono.cli --help
    python -m nono.cli @params.txt
    python -m nono.cli --provider gemini --task summarize --input doc.txt --output result.json

Supported Providers:
    - gemini (Google Gemini) - Default
    - openai (OpenAI GPT)
    - perplexity (Perplexity AI)
    - deepseek (DeepSeek)
    - grok (xAI Grok)
    - ollama (Local Ollama)

Author: DatamanEdge
Version: 0.1.0
"""

from __future__ import annotations

import sys
import os
import argparse
import json
import logging
import time
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

__all__ = [
    "CLIBase",
    "OutputFormat",
    "LogLevel",
    "Colors",
    "CLIConfig",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "print_header",
    "print_table",
    "print_summary",
    "print_progress",
    "confirm_action",
    "cprint",
    "create_cli",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================================
# CONSTANTS AND ENUMS
# ============================================================================

class OutputFormat(Enum):
    """Supported output formats for CLI display."""
    TABLE = "table"
    JSON = "json"
    CSV = "csv"
    TEXT = "text"
    MARKDOWN = "markdown"
    SUMMARY = "summary"
    QUIET = "quiet"


class LogLevel(Enum):
    """Logging verbosity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    QUIET = "quiet"


class AIProvider(Enum):
    """Supported AI providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    PERPLEXITY = "perplexity"
    DEEPSEEK = "deepseek"
    GROK = "grok"
    OLLAMA = "ollama"


class Colors:
    """ANSI color codes with Windows compatibility."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    
    # Semantic colors
    SUCCESS = GREEN
    ERROR = RED
    WARNING = YELLOW
    INFO = CYAN
    HIGHLIGHT = MAGENTA
    
    _enabled: bool = True
    
    @classmethod
    def disable(cls) -> None:
        """Disable colors for non-TTY or unsupported terminals."""
        cls._enabled = False
        for attr in dir(cls):
            if not attr.startswith('_') and isinstance(getattr(cls, attr), str):
                setattr(cls, attr, '')
    
    @classmethod
    def enable(cls) -> None:
        """Re-enable colors (reinitialize)."""
        cls._enabled = True
        cls.init()
    
    @classmethod
    def init(cls) -> None:
        """Initialize colors with Windows ANSI support."""
        if not cls._enabled:
            return
            
        if sys.platform == 'win32':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                # Enable ANSI escape sequences in Windows terminal
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                try:
                    import colorama
                    colorama.init()
                except ImportError:
                    cls.disable()
        
        if not sys.stdout.isatty():
            cls.disable()


# Initialize colors on module import
Colors.init()


# ============================================================================
# OUTPUT UTILITIES
# ============================================================================

def cprint(
    message: str, 
    color: str = "", 
    bold: bool = False, 
    dim: bool = False,
    file=sys.stdout
) -> None:
    """Print colored message to terminal."""
    prefix = ""
    if bold:
        prefix += Colors.BOLD
    if dim:
        prefix += Colors.DIM
    if color:
        prefix += color
    suffix = Colors.RESET if (bold or dim or color) else ""
    print(f"{prefix}{message}{suffix}", file=file)


def print_success(message: str) -> None:
    """Print success message with checkmark."""
    cprint(f"✓ {message}", Colors.SUCCESS)


def print_error(message: str, file=sys.stderr) -> None:
    """Print error message with X mark."""
    cprint(f"✗ {message}", Colors.ERROR, file=file)


def print_warning(message: str) -> None:
    """Print warning message with warning sign."""
    cprint(f"⚠ {message}", Colors.WARNING)


def print_info(message: str) -> None:
    """Print info message with info symbol."""
    cprint(f"ℹ {message}", Colors.INFO)


def print_debug(message: str) -> None:
    """Print debug message with dim formatting."""
    cprint(f"  {message}", Colors.GRAY, dim=True)


def print_header(title: str, width: int = 70, char: str = "=") -> None:
    """Print formatted section header."""
    print()
    cprint(char * width, Colors.CYAN, bold=True)
    cprint(f" {title}", Colors.CYAN, bold=True)
    cprint(char * width, Colors.CYAN, bold=True)


def print_subheader(title: str, char: str = "-") -> None:
    """Print formatted subsection header."""
    print()
    cprint(f"{char * 3} {title} {char * 3}", Colors.CYAN)


def print_table(
    headers: List[str],
    rows: List[List[Any]],
    max_col_width: int = 40,
    show_index: bool = False
) -> None:
    """Print formatted ASCII table."""
    if not headers or not rows:
        return
    
    # Add index column if requested
    if show_index:
        headers = ["#"] + headers
        rows = [[i + 1] + list(row) for i, row in enumerate(rows)]
    
    # Calculate column widths
    col_widths = []
    for i, header in enumerate(headers):
        max_width = len(str(header))
        for row in rows:
            if i < len(row):
                max_width = max(max_width, len(str(row[i])))
        col_widths.append(min(max_width, max_col_width))
    
    def truncate(value: Any, width: int) -> str:
        s = str(value)
        return s[:width - 3] + "..." if len(s) > width else s
    
    # Print header
    header_row = " │ ".join(
        truncate(h, w).ljust(w) for h, w in zip(headers, col_widths)
    )
    separator = "─┼─".join("─" * w for w in col_widths)
    
    print()
    cprint(header_row, Colors.CYAN, bold=True)
    print(separator)
    
    # Print rows
    for row in rows:
        row_str = " │ ".join(
            truncate(row[i] if i < len(row) else "", w).ljust(w)
            for i, w in enumerate(col_widths)
        )
        print(row_str)
    print()


def print_key_value(key: str, value: Any, indent: int = 2) -> None:
    """Print key-value pair with formatting."""
    spaces = " " * indent
    key_display = key.replace('_', ' ').title()
    print(f"{spaces}{key_display}: ", end="")
    
    # Color based on content
    if isinstance(value, bool):
        color = Colors.SUCCESS if value else Colors.WARNING
    elif isinstance(value, (int, float)) and value == 0:
        color = Colors.GRAY
    elif 'error' in key.lower():
        color = Colors.ERROR if value else Colors.GRAY
    elif 'success' in key.lower() or 'completed' in key.lower():
        color = Colors.SUCCESS
    elif 'warning' in key.lower() or 'skipped' in key.lower():
        color = Colors.WARNING
    else:
        color = Colors.WHITE
    
    cprint(str(value), color)


def print_summary(
    stats: Dict[str, Any],
    title: str = "SUMMARY",
    width: int = 70
) -> None:
    """Print formatted summary statistics."""
    print()
    cprint("=" * width, Colors.CYAN, bold=True)
    cprint(f" {title}", Colors.CYAN, bold=True)
    cprint("=" * width, Colors.CYAN, bold=True)
    
    for key, value in stats.items():
        print_key_value(key, value)
    
    cprint("=" * width, Colors.CYAN, bold=True)


def print_progress(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    width: int = 40,
    fill: str = "█",
    empty: str = "░"
) -> None:
    """Print progress bar."""
    if total == 0:
        percent, filled = 100, width
    else:
        percent = (current / total) * 100
        filled = int(width * current // total)
    
    bar = fill * filled + empty * (width - filled)
    
    # Color based on progress
    if percent < 33:
        color = Colors.RED
    elif percent < 66:
        color = Colors.YELLOW
    else:
        color = Colors.GREEN
    
    print(f"\r{prefix} {color}|{bar}|{Colors.RESET} {percent:5.1f}% {suffix}", end="", flush=True)
    
    if current >= total:
        print()


def print_spinner(message: str, frames: Optional[List[str]] = None) -> Callable[[], None]:
    """Create a spinner context for long operations.
    
    Returns a function to call for each frame update.
    
    Usage:
        update = print_spinner("Loading...")
        for _ in range(100):
            update()
            time.sleep(0.1)
    """
    frames = frames or ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    frame_idx = [0]
    
    def update():
        frame = frames[frame_idx[0] % len(frames)]
        print(f"\r{Colors.CYAN}{frame}{Colors.RESET} {message}", end="", flush=True)
        frame_idx[0] += 1
    
    return update


def confirm_action(message: str, default: bool = False) -> bool:
    """Prompt user for confirmation."""
    suffix = " [Y/n]" if default else " [y/N]"
    try:
        response = input(f"{Colors.WARNING}?{Colors.RESET} {message}{suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    
    if not response:
        return default
    return response in ('y', 'yes', 'si', 's', '1', 'true')


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


# ============================================================================
# CLI CONFIGURATION
# ============================================================================

@dataclass
class CLIConfig:
    """Configuration for CLI behavior and appearance."""
    prog_name: str = "nono"
    version: str = "0.1.0"
    description: str = "GenAI task execution framework with multi-provider support"
    epilog: str = ""
    
    colors_enabled: bool = True
    default_output_format: OutputFormat = OutputFormat.SUMMARY
    default_log_level: LogLevel = LogLevel.INFO
    default_provider: AIProvider = AIProvider.GEMINI
    
    allow_parameter_files: bool = True
    require_confirmation: bool = False
    dry_run_by_default: bool = False
    
    default_timeout: int = 60
    default_max_tokens: int = 4096


# ============================================================================
# CLI BASE CLASS
# ============================================================================

class CLIBase:
    """Base class for CLI applications with consistent behavior.
    
    Usage:
        cli = CLIBase(prog="nono", description="GenAI Tasker", version="0.1.0")
        cli.add_ai_provider_group()
        cli.add_task_group()
        args = cli.parse_args()
        cli.print_final_summary()
    """
    
    def __init__(
        self,
        prog: Optional[str] = None,
        description: str = "",
        version: str = "0.1.0",
        epilog: Optional[str] = None,
        config: Optional[CLIConfig] = None
    ):
        """Initialize CLI.
        
        Args:
            prog: Program name
            description: Program description for help
            version: Version string
            epilog: Text to display after help
            config: CLIConfig instance
        """
        self.config = config or CLIConfig(
            prog_name=prog or os.path.basename(sys.argv[0]),
            version=version,
            description=description,
            epilog=epilog or ""
        )
        
        # Build epilog with examples if provided
        examples_text = ""
        self._examples: List[str] = []
        
        self.parser = argparse.ArgumentParser(
            prog=prog,
            description=description,
            epilog=epilog,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            fromfile_prefix_chars='@' if self.config.allow_parameter_files else None
        )
        
        self.parser.add_argument(
            '--version', '-V',
            action='version',
            version=f'%(prog)s {version}'
        )
        
        self._add_global_arguments()
        
        self._groups: Dict[str, argparse._ArgumentGroup] = {}
        self.args: Optional[argparse.Namespace] = None
        self.stats: Dict[str, Any] = {}
        self.start_time: Optional[datetime] = None
    
    def _add_global_arguments(self) -> None:
        """Add global arguments available to all CLI tools."""
        grp = self.parser.add_argument_group("Global Options")
        
        grp.add_argument(
            '--verbose', '-v',
            action='count',
            default=0,
            help="Increase verbosity (-v=INFO, -vv=DEBUG)"
        )
        grp.add_argument(
            '--quiet', '-q',
            action='store_true',
            help="Suppress non-error output"
        )
        grp.add_argument(
            '--no-color',
            action='store_true',
            help="Disable colored output"
        )
        grp.add_argument(
            '--dry-run',
            action='store_true',
            default=self.config.dry_run_by_default,
            help="Simulate without making API calls"
        )
        grp.add_argument(
            '--output-format', '-F',
            choices=[f.value for f in OutputFormat],
            default=self.config.default_output_format.value,
            help="Output display format (default: %(default)s)"
        )
        grp.add_argument(
            '--config-file',
            type=str,
            metavar="FILE",
            help="Load configuration from TOML/JSON file"
        )
        grp.add_argument(
            '--log-file',
            type=str,
            metavar="FILE",
            help="Write logs to file"
        )
    
    def add_group(
        self,
        name: str,
        title: Optional[str] = None,
        description: Optional[str] = None
    ) -> argparse._ArgumentGroup:
        """Add a custom argument group."""
        group = self.parser.add_argument_group(title or name.title(), description)
        self._groups[name] = group
        return group
    
    def add_example(self, example: str) -> None:
        """Add a usage example to help text."""
        self._examples.append(example)
    
    def add_examples(self, examples: List[str]) -> None:
        """Add multiple usage examples."""
        self._examples.extend(examples)
    
    def _build_epilog(self) -> str:
        """Build epilog with examples."""
        if not self._examples:
            return self.config.epilog
        
        examples_text = "\n\nExamples:\n"
        for example in self._examples:
            examples_text += f"  {example}\n"
        
        return examples_text + (f"\n{self.config.epilog}" if self.config.epilog else "")
    
    # -------------------------------------------------------------------
    # AI PROVIDER ARGUMENT GROUP
    # -------------------------------------------------------------------
    
    def add_ai_provider_group(self) -> argparse._ArgumentGroup:
        """Add AI provider connection arguments."""
        group = self.add_group("ai_provider", "AI Provider Configuration")
        
        group.add_argument(
            '--provider', '-p',
            choices=[p.value for p in AIProvider],
            default=self.config.default_provider.value,
            help="AI provider (default: %(default)s)"
        )
        group.add_argument(
            '--model', '-m',
            type=str,
            help="Model name (uses provider default if not specified)"
        )
        group.add_argument(
            '--api-key',
            type=str,
            help="API key (or use environment variable)"
        )
        group.add_argument(
            '--api-key-file',
            type=str,
            metavar="FILE",
            help="Read API key from file"
        )
        group.add_argument(
            '--temperature',
            type=float,
            default=0.7,
            help="Generation temperature 0.0-2.0 (default: %(default)s)"
        )
        group.add_argument(
            '--max-tokens',
            type=int,
            default=self.config.default_max_tokens,
            help="Maximum tokens in response (default: %(default)s)"
        )
        group.add_argument(
            '--timeout',
            type=int,
            default=self.config.default_timeout,
            help="Request timeout in seconds (default: %(default)s)"
        )
        
        # Ollama-specific
        group.add_argument(
            '--ollama-host',
            type=str,
            default="http://localhost:11434",
            help="Ollama server URL (default: %(default)s)"
        )
        
        return group
    
    # -------------------------------------------------------------------
    # TASK ARGUMENT GROUP
    # -------------------------------------------------------------------
    
    def add_task_group(self) -> argparse._ArgumentGroup:
        """Add task execution arguments."""
        group = self.add_group("task", "Task Configuration")
        
        group.add_argument(
            '--task', '-t',
            type=str,
            help="Task name or path to task JSON file"
        )
        group.add_argument(
            '--prompt',
            type=str,
            help="Direct prompt text (alternative to task file)"
        )
        group.add_argument(
            '--system-prompt',
            type=str,
            help="System prompt/instructions"
        )
        group.add_argument(
            '--template',
            type=str,
            help="Jinja2 template file for prompt"
        )
        group.add_argument(
            '--variables', '--vars',
            type=str,
            help="JSON string or file with template variables"
        )
        
        return group
    
    # -------------------------------------------------------------------
    # INPUT/OUTPUT ARGUMENT GROUP
    # -------------------------------------------------------------------
    
    def add_io_group(
        self,
        input_formats: Optional[List[str]] = None,
        output_formats: Optional[List[str]] = None
    ) -> argparse._ArgumentGroup:
        """Add input/output arguments."""
        group = self.add_group("io", "Input/Output")
        
        input_formats = input_formats or ['text', 'json', 'csv', 'file']
        output_formats = output_formats or ['text', 'json', 'csv', 'markdown']
        
        group.add_argument(
            '--input', '-i',
            type=str,
            help="Input file or data"
        )
        group.add_argument(
            '--input-format',
            choices=input_formats,
            default='text',
            help="Input format (default: %(default)s)"
        )
        group.add_argument(
            '--output', '-o',
            type=str,
            help="Output file (stdout if not specified)"
        )
        group.add_argument(
            '--output-type',
            choices=output_formats,
            default='text',
            help="Output format for file (default: %(default)s)"
        )
        group.add_argument(
            '--append',
            action='store_true',
            help="Append to output file instead of overwriting"
        )
        
        return group
    
    # -------------------------------------------------------------------
    # BATCH PROCESSING ARGUMENT GROUP
    # -------------------------------------------------------------------
    
    def add_batch_group(self) -> argparse._ArgumentGroup:
        """Add batch processing arguments."""
        group = self.add_group("batch", "Batch Processing")
        
        group.add_argument(
            '--batch',
            action='store_true',
            help="Enable batch processing mode"
        )
        group.add_argument(
            '--batch-file',
            type=str,
            help="File with batch inputs (one per line or JSON array)"
        )
        group.add_argument(
            '--batch-size',
            type=int,
            default=10,
            help="Number of items per batch (default: %(default)s)"
        )
        group.add_argument(
            '--delay',
            type=float,
            default=0.5,
            help="Delay between requests in seconds (default: %(default)s)"
        )
        group.add_argument(
            '--retry',
            type=int,
            default=3,
            help="Number of retries on failure (default: %(default)s)"
        )
        group.add_argument(
            '--continue-on-error',
            action='store_true',
            help="Continue batch processing on individual failures"
        )
        
        return group
    
    # -------------------------------------------------------------------
    # ARGUMENT PARSING AND EXECUTION
    # -------------------------------------------------------------------
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command-line arguments."""
        # Update epilog with examples before parsing
        if self._examples:
            self.parser.epilog = self._build_epilog()
        
        # Show help if no arguments
        if args is None and len(sys.argv) == 1:
            self._print_usage_hint()
            sys.exit(1)
        
        self.args = self.parser.parse_args(args)
        self._post_process_args()
        self._configure_logging()
        self.start_time = datetime.now()
        
        return self.args
    
    def _print_usage_hint(self) -> None:
        """Print brief usage hint when no arguments provided."""
        cprint(f"{self.config.prog_name} v{self.config.version}", Colors.CYAN, bold=True)
        print(f"\n{self.config.description}\n")
        print(f"Usage: {self.config.prog_name} [options]")
        print(f"Try '{self.config.prog_name} --help' for more information.")
        
        if self._examples:
            print("\nQuick examples:")
            for example in self._examples[:3]:
                print(f"  {example}")
    
    def _post_process_args(self) -> None:
        """Post-process parsed arguments."""
        if getattr(self.args, 'no_color', False):
            Colors.disable()
        
        # Load secrets from files
        secret_mappings = [
            ('api_key_file', 'api_key'),
        ]
        
        for file_arg, target_arg in secret_mappings:
            file_path = getattr(self.args, file_arg, None)
            if file_path and os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    setattr(self.args, target_arg, f.readline().strip())
        
        # Load variables from file if JSON file path provided
        variables = getattr(self.args, 'variables', None)
        if variables and os.path.isfile(variables):
            with open(variables, 'r', encoding='utf-8') as f:
                setattr(self.args, 'variables', json.load(f))
        elif variables:
            try:
                setattr(self.args, 'variables', json.loads(variables))
            except json.JSONDecodeError:
                pass  # Keep as string
    
    def _configure_logging(self) -> None:
        """Configure logging based on verbosity."""
        if getattr(self.args, 'quiet', False):
            level = logging.ERROR
        elif getattr(self.args, 'verbose', 0) >= 2:
            level = logging.DEBUG
        elif getattr(self.args, 'verbose', 0) >= 1:
            level = logging.INFO
        else:
            level = logging.WARNING
        
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        handlers: List[logging.Handler] = [logging.StreamHandler()]
        
        log_file = getattr(self.args, 'log_file', None)
        if log_file:
            handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
        
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=handlers
        )
    
    # -------------------------------------------------------------------
    # STATISTICS AND OUTPUT
    # -------------------------------------------------------------------
    
    def increment_stat(self, key: str, amount: Union[int, float] = 1) -> None:
        """Increment a statistic counter."""
        self.stats[key] = self.stats.get(key, 0) + amount
    
    def set_stat(self, key: str, value: Any) -> None:
        """Set a statistic value."""
        self.stats[key] = value
    
    def get_elapsed_time(self) -> str:
        """Get formatted elapsed time since start."""
        if not self.start_time:
            return "0s"
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return format_duration(elapsed)
    
    def get_elapsed_seconds(self) -> float:
        """Get elapsed seconds since start."""
        if not self.start_time:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()
    
    def print_final_summary(self, title: Optional[str] = None) -> None:
        """Print final execution summary with elapsed time."""
        self.stats['elapsed_time'] = self.get_elapsed_time()
        
        title = title or f"{self.config.prog_name.upper()} RESULTS"
        print_summary(self.stats, title=title)
    
    def exit_with_error(self, message: str, code: int = 1) -> None:
        """Print error and exit with code."""
        print_error(message)
        sys.exit(code)
    
    def exit_success(self, message: Optional[str] = None) -> None:
        """Print success message and exit."""
        if message:
            print_success(message)
        sys.exit(0)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_cli(
    prog: str = "nono",
    description: str = "GenAI task execution framework",
    version: str = "0.1.0",
    with_provider: bool = True,
    with_task: bool = True,
    with_io: bool = True,
    with_batch: bool = False
) -> CLIBase:
    """Factory function to create configured CLI instance.
    
    Args:
        prog: Program name
        description: Program description
        version: Version string
        with_provider: Add AI provider arguments
        with_task: Add task configuration arguments
        with_io: Add input/output arguments
        with_batch: Add batch processing arguments
    
    Returns:
        Configured CLIBase instance
    """
    cli = CLIBase(prog=prog, description=description, version=version)
    
    if with_provider:
        cli.add_ai_provider_group()
    
    if with_task:
        cli.add_task_group()
    
    if with_io:
        cli.add_io_group()
    
    if with_batch:
        cli.add_batch_group()
    
    return cli


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> int:
    """Main CLI entry point for nono."""
    cli = create_cli(
        prog="nono",
        description="Nono GenAI Tasker - Execute AI tasks with multi-provider support",
        version="0.1.0",
        with_provider=True,
        with_task=True,
        with_io=True,
        with_batch=True
    )
    
    cli.add_examples([
        "%(prog)s --provider gemini --prompt 'Summarize this text' -i document.txt",
        "%(prog)s --provider openai --task summarize --input data.json -o result.json",
        "%(prog)s --provider ollama --model llama3 --template prompt.j2 --vars '{\"topic\": \"AI\"}'",
        "%(prog)s @params.txt",
        "%(prog)s --batch --batch-file inputs.txt --provider gemini -o outputs/",
    ])
    
    try:
        args = cli.parse_args()
        
        # Show configuration in verbose mode
        if args.verbose >= 1:
            print_info(f"Provider: {args.provider}")
            if args.model:
                print_info(f"Model: {args.model}")
            if args.dry_run:
                print_warning("DRY RUN - No API calls will be made")
        
        # Dry run just shows what would happen
        if args.dry_run:
            print_header("DRY RUN SIMULATION")
            print_key_value("provider", args.provider)
            print_key_value("model", args.model or "(default)")
            print_key_value("task", args.task or args.prompt or "(none)")
            print_key_value("input", args.input or "(stdin)")
            print_key_value("output", args.output or "(stdout)")
            
            cli.set_stat('mode', 'dry-run')
            cli.set_stat('provider', args.provider)
            cli.print_final_summary()
            return 0
        
        # TODO: Actual task execution would go here
        # This is a placeholder that demonstrates the CLI structure
        
        if not args.task and not args.prompt:
            print_warning("No task or prompt specified. Use --task or --prompt.")
            print_info("Try 'nono --help' for usage information.")
            return 1
        
        # Placeholder execution
        print_info("Executing task...")
        cli.increment_stat('tasks_executed', 1)
        cli.increment_stat('tokens_used', 0)
        cli.set_stat('provider', args.provider)
        
        print_success("Task completed!")
        cli.print_final_summary()
        return 0
        
    except KeyboardInterrupt:
        print()
        print_warning("Interrupted by user")
        return 130
    except Exception as e:
        cli.exit_with_error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
