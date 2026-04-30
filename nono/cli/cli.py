"""
Nono CLI - Command Line Interface for GenAI Tasker

Execute generative AI tasks from the command line with multi-provider support.

Usage:
    python -m nono.cli --help
    python -m nono.cli <command> --help
    python -m nono.cli @params.txt
    python -m nono.cli run --provider google --prompt "Explain what Python is"

Core Commands:
    run        Execute a prompt or task
    agent      Run a named agent template
    workflow   Run a named workflow template
    info       List available resources
    providers  Show all supported AI providers
    config     Configuration management (init, show)

Use 'nono providers' to see the full list of supported AI providers.
Default provider: google (Google Gemini)

Author: DatamanEdge
Version: 0.2.0
"""

from __future__ import annotations

import sys
import os
import argparse
import json
import logging
import shutil
import time
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

from nono import __version__

__all__ = [
    "CLIBase",
    "CLIResult",
    "Subcommand",
    "OutputFormat",
    "LogLevel",
    "AIProvider",
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
    "run_api",
    "main",
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
    GOOGLE = "google"
    OPENAI = "openai"
    PERPLEXITY = "perplexity"
    DEEPSEEK = "deepseek"
    XAI = "xai"
    GROQ = "groq"
    CEREBRAS = "cerebras"
    NVIDIA = "nvidia"
    GITHUB = "github"
    OPENROUTER = "openrouter"
    HUGGINGFACE = "huggingface"
    FOUNDRY = "foundry"
    VERCEL = "vercel"
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
    _original_values: dict[str, str] = {}
    
    @classmethod
    def disable(cls) -> None:
        """Disable colors for non-TTY or unsupported terminals."""
        cls._enabled = False
        if not cls._original_values:
            cls._original_values = {
                attr: getattr(cls, attr)
                for attr in dir(cls)
                if not attr.startswith('_') and isinstance(getattr(cls, attr), str)
            }
        for attr in cls._original_values:
            setattr(cls, attr, '')
    
    @classmethod
    def enable(cls) -> None:
        """Re-enable colors (restore original values)."""
        cls._enabled = True
        for attr, value in cls._original_values.items():
            setattr(cls, attr, value)
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
    """Configuration for CLI behavior and appearance.

    Values are loaded from ``[cli]`` in ``config.toml`` when using
    :meth:`from_config_toml`.  Hard-coded defaults remain as fallback.
    """
    prog_name: str = "nono"
    version: str = __version__
    description: str = "GenAI task execution framework with multi-provider support"
    epilog: str = ""
    
    colors_enabled: bool = True
    default_output_format: OutputFormat = OutputFormat.SUMMARY
    default_log_level: LogLevel = LogLevel.INFO
    default_provider: AIProvider = AIProvider.GOOGLE
    
    allow_parameter_files: bool = True
    require_confirmation: bool = False
    dry_run_by_default: bool = False
    
    default_timeout: int = 60
    default_max_tokens: int = 4096

    @classmethod
    def from_config_toml(cls, **overrides: Any) -> "CLIConfig":
        """Create a ``CLIConfig`` merging ``config.toml`` values.

        ``overrides`` take precedence over TOML values, which take
        precedence over dataclass defaults.

        Returns:
            Populated ``CLIConfig`` instance.
        """
        kwargs: Dict[str, Any] = {}
        try:
            from nono.config import load_config as _load_config
            cfg = _load_config()

            _MAP: Dict[str, tuple] = {
                "colors_enabled": ("cli.colors_enabled", bool),
                "default_output_format": ("cli.default_output_format", str),
                "default_log_level": ("cli.default_log_level", str),
                "default_provider": ("cli.default_provider", str),
                "allow_parameter_files": ("cli.allow_parameter_files", bool),
                "require_confirmation": ("cli.require_confirmation", bool),
                "dry_run_by_default": ("cli.dry_run_by_default", bool),
                "default_timeout": ("cli.default_timeout", int),
                "default_max_tokens": ("cli.default_max_tokens", int),
            }
            for field_name, (key, typ) in _MAP.items():
                val = cfg.get(key)
                if val is not None:
                    if typ is bool and isinstance(val, str):
                        val = val.lower() in ("true", "1", "yes")
                    else:
                        val = typ(val)
                    # Convert string enums
                    if field_name == "default_output_format":
                        val = OutputFormat(val)
                    elif field_name == "default_log_level":
                        val = LogLevel(val.upper())
                    elif field_name == "default_provider":
                        val = AIProvider(val)
                    kwargs[field_name] = val
        except Exception:
            pass  # Graceful fallback to defaults

        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass
class Subcommand:
    """Definition of a CLI subcommand."""
    name: str
    help: str
    handler: Optional[Callable[[argparse.Namespace, 'CLIBase'], None]] = None
    aliases: List[str] = field(default_factory=list)
    parser: Optional[argparse.ArgumentParser] = None


@dataclass
class CLIResult:
    """Structured result for programmatic API usage.
    
    Returned by ``run_api()`` so callers (REST endpoints, test harnesses,
    orchestrators) can inspect the outcome without parsing stdout.
    
    Attributes:
        ok:     True when the command succeeded.
        data:   Arbitrary payload — dict, list, str, or None.
        stats:  Counters collected via ``cli.increment_stat()``.
        error:  Error message when ``ok`` is False, else None.
    """
    ok: bool = True
    data: Any = None
    stats: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None


# ============================================================================
# CLI BASE CLASS
# ============================================================================

class CLIBase:
    """Base class for CLI applications with consistent behavior and subcommand support.
    
    Usage without subcommands:
        cli = CLIBase(prog="nono", description="GenAI Tasker", version="0.2.0")
        cli.add_ai_provider_group()
        args = cli.parse_args()
        cli.print_final_summary()
    
    Usage with subcommands:
        cli = CLIBase(prog="nono", description="GenAI Tasker", version="0.2.0")
        cli.init_subcommands()
        run_parser = cli.add_subcommand("run", "Execute a task", handler=run_task)
        run_parser.add_argument('--prompt', required=True)
        args = cli.parse_args()
        cli.run()
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
        self.config = config or CLIConfig.from_config_toml(
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
        self._subparsers: Optional[argparse._SubParsersAction] = None
        self._subcommands: Dict[str, Subcommand] = {}
        self._handlers: Dict[str, Callable] = {}
        self.args: Optional[argparse.Namespace] = None
        self.stats: Dict[str, Any] = {}
        self.last_result: Any = None  # Set by handlers for run_api()
        self.start_time: Optional[datetime] = None
    
    # -------------------------------------------------------------------
    # SUBCOMMAND SUPPORT
    # -------------------------------------------------------------------
    
    def init_subcommands(
        self, title: str = "Commands", dest: str = "command"
    ) -> argparse._SubParsersAction:
        """Initialize subcommand support. Must be called before add_subcommand().
        
        Args:
            title: Title for the subcommands section in help
            dest: Attribute name where the subcommand name will be stored
        
        Returns:
            The subparsers action object
        """
        self._subparsers = self.parser.add_subparsers(
            title=title,
            dest=dest,
            help="Available commands (use '<command> --help' for details)"
        )
        return self._subparsers
    
    def add_subcommand(
        self,
        name: str,
        help: str,
        handler: Optional[Callable[[argparse.Namespace, 'CLIBase'], None]] = None,
        aliases: Optional[List[str]] = None
    ) -> argparse.ArgumentParser:
        """Add a subcommand to the CLI.
        
        Args:
            name: Subcommand name (e.g., 'run', 'list')
            help: Help text for the subcommand
            handler: Function(args, cli) to handle this command
            aliases: Optional list of aliases for the subcommand
        
        Returns:
            The subcommand's ArgumentParser for adding arguments
        
        Example:
            run_parser = cli.add_subcommand("run", "Execute task", handler=run_task)
            run_parser.add_argument('--prompt', required=True)
        """
        if not self._subparsers:
            self.init_subcommands()
        
        assert self._subparsers is not None  # guaranteed by init_subcommands
        aliases = aliases or []
        subparser = self._subparsers.add_parser(
            name,
            help=help,
            aliases=aliases,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Add global options to subcommand
        self._add_global_arguments_to_subparser(subparser)
        
        subcommand = Subcommand(
            name=name,
            help=help,
            handler=handler,
            aliases=aliases,
            parser=subparser
        )
        self._subcommands[name] = subcommand
        for alias in aliases:
            self._subcommands[alias] = subcommand
        
        if handler:
            self._handlers[name] = handler
            for alias in aliases:
                self._handlers[alias] = handler
        
        return subparser
    
    def _add_global_arguments_to_subparser(
        self, subparser: argparse.ArgumentParser
    ) -> None:
        """Add global options to a subparser."""
        grp = subparser.add_argument_group("Global Options")
        grp.add_argument('--verbose', '-v', action='count', default=0,
                         help="Increase verbosity (-v=INFO, -vv=DEBUG)")
        grp.add_argument('--quiet', '-q', action='store_true',
                         help="Suppress non-error output")
        grp.add_argument('--no-color', action='store_true',
                         help="Disable colored output")
        grp.add_argument('--ci', action='store_true',
                         help="CI mode: implies --no-color --quiet --output-format json")
        grp.add_argument('--dry-run', action='store_true',
                         default=self.config.dry_run_by_default,
                         help="Simulate without making API calls")
        grp.add_argument('--log-file', type=str, metavar="FILE",
                         help="Write logs to file")
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
    
    def set_handler(
        self, command: str,
        handler: Callable[[argparse.Namespace, 'CLIBase'], None]
    ) -> None:
        """Set or update the handler for a subcommand."""
        self._handlers[command] = handler
        if command in self._subcommands:
            self._subcommands[command].handler = handler
    
    def run(self) -> None:
        """Execute the handler for the parsed subcommand. Must be called after parse_args()."""
        if not self.args:
            raise RuntimeError("parse_args() must be called before run()")
        
        command = getattr(self.args, 'command', None)
        if not command:
            self.parser.print_help()
            sys.exit(1)
        
        handler = self._handlers.get(command)
        if handler:
            handler(self.args, self)
        else:
            print_error(f"No handler registered for command: {command}")
            sys.exit(1)
    
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
            '--ci',
            action='store_true',
            help="CI mode: implies --no-color --quiet --output-format json"
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
    # AGENT ARGUMENT GROUP
    # -------------------------------------------------------------------

    def add_agent_group(self) -> argparse._ArgumentGroup:
        """Add agent orchestration arguments."""
        group = self.add_group("agent", "Agent Configuration")

        group.add_argument(
            '--agent-provider',
            choices=[p.value for p in AIProvider],
            default=None,
            help="AI provider for agents (overrides --provider)"
        )
        group.add_argument(
            '--agent-model',
            type=str,
            help="Model for agent LLM calls (overrides --model)"
        )
        group.add_argument(
            '--agent-temperature',
            type=float,
            default=None,
            help="Temperature for agent calls (default: from config)"
        )
        group.add_argument(
            '--agent-max-tokens',
            type=int,
            default=None,
            help="Max tokens for agent responses (default: from config)"
        )
        group.add_argument(
            '--router-mode',
            choices=['single', 'sequential', 'parallel', 'loop'],
            default=None,
            help="Force RouterAgent execution mode (default: LLM decides)"
        )
        group.add_argument(
            '--router-max-iterations',
            type=int,
            default=None,
            help="Max iterations for RouterAgent loop mode (default: 3)"
        )
        group.add_argument(
            '--agents',
            type=str,
            nargs='+',
            metavar="NAME",
            help="Agent names to include in orchestration"
        )

        return group

    # -------------------------------------------------------------------
    # WORKFLOW ARGUMENT GROUP
    # -------------------------------------------------------------------

    def add_workflow_group(self) -> argparse._ArgumentGroup:
        """Add workflow pipeline arguments."""
        group = self.add_group("workflow", "Workflow Configuration")

        group.add_argument(
            '--workflow',
            type=str,
            metavar="NAME",
            help="Workflow name or path to workflow definition"
        )
        group.add_argument(
            '--steps',
            type=str,
            nargs='+',
            metavar="STEP",
            help="Workflow steps to execute (default: all)"
        )
        group.add_argument(
            '--input-key',
            type=str,
            default=None,
            help="State key for workflow input (default: from config)"
        )
        group.add_argument(
            '--output-key',
            type=str,
            default=None,
            help="State key for workflow output (default: from config)"
        )
        group.add_argument(
            '--state',
            type=str,
            help="Initial state as JSON string or path to JSON file"
        )
        group.add_argument(
            '--stream',
            action='store_true',
            help="Stream step-by-step output during workflow execution"
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
        assert self.args is not None  # called after parse_args
        
        # CI mode implies no-color, quiet, json output
        if getattr(self.args, 'ci', False):
            self.args.no_color = True
            self.args.quiet = True
            if hasattr(self.args, 'output_format'):
                self.args.output_format = OutputFormat.JSON.value
        
        if getattr(self.args, 'no_color', False):
            Colors.disable()
        
        # Load secrets from files
        secret_mappings = [
            ('api_key_file', 'api_key'),
        ]
        
        cwd = os.path.realpath(os.getcwd())
        
        for file_arg, target_arg in secret_mappings:
            file_path = getattr(self.args, file_arg, None)
            if file_path and os.path.isfile(file_path):
                real = os.path.realpath(file_path)
                if not real.startswith(cwd + os.sep) and real != cwd:
                    print_error(f"Refusing to read '{file_path}': outside working directory.")
                    continue
                with open(file_path, 'r', encoding='utf-8') as f:
                    setattr(self.args, target_arg, f.readline().strip())
        
        # Load variables from file if JSON file path provided
        variables = getattr(self.args, 'variables', None)
        if variables and os.path.isfile(variables):
            real = os.path.realpath(variables)
            if not real.startswith(cwd + os.sep) and real != cwd:
                print_error(f"Refusing to read '{variables}': outside working directory.")
            else:
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
        
        if not logging.root.handlers:
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
        """Print final execution summary with elapsed time.

        Suppressed when ``--quiet`` or ``--ci`` is active.
        """
        if getattr(self.args, 'quiet', False) or getattr(self.args, 'ci', False):
            return

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
    version: str = __version__,
    with_provider: bool = True,
    with_task: bool = True,
    with_io: bool = True,
    with_batch: bool = False,
    with_agent: bool = False,
    with_workflow: bool = False,
) -> CLIBase:
    """Factory function to create configured CLI instance.

    Args:
        prog: Program name.
        description: Program description.
        version: Version string.
        with_provider: Add AI provider arguments.
        with_task: Add task configuration arguments.
        with_io: Add input/output arguments.
        with_batch: Add batch processing arguments.
        with_agent: Add agent orchestration arguments.
        with_workflow: Add workflow pipeline arguments.

    Returns:
        Configured CLIBase instance.
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

    if with_agent:
        cli.add_agent_group()

    if with_workflow:
        cli.add_workflow_group()

    return cli


# ============================================================================
# SUBCOMMAND HANDLERS
# ============================================================================


def _handle_run(args: argparse.Namespace, cli: CLIBase) -> None:
    """Handle the ``run`` subcommand — execute a prompt or task."""
    from nono.tasker import TaskExecutor

    if args.verbose >= 1:
        print_info(f"Provider: {args.provider}")
        if args.model:
            print_info(f"Model: {args.model}")

    if args.dry_run:
        print_header("DRY RUN SIMULATION")
        print_key_value("provider", args.provider)
        print_key_value("model", args.model or "(default)")
        print_key_value("prompt", getattr(args, 'prompt', None) or "(none)")
        print_key_value("task", getattr(args, 'task', None) or "(none)")
        print_key_value("input", getattr(args, 'input', None) or "(stdin)")
        print_key_value("output", getattr(args, 'output', None) or "(stdout)")
        cli.set_stat("mode", "dry-run")
        cli.set_stat("provider", args.provider)
        cli.print_final_summary()
        return

    prompt = getattr(args, 'prompt', None)
    task = getattr(args, 'task', None)
    input_file = getattr(args, 'input', None)

    # Read input from file when provided
    input_text: Optional[str] = None
    if input_file and os.path.isfile(input_file):
        with open(input_file, "r", encoding="utf-8") as fh:
            input_text = fh.read()

    if not task and not prompt and not input_text:
        print_warning("No task, prompt, or input specified.")
        print_info("Try 'nono run --help' for usage information.")
        raise SystemExit(1)

    executor = TaskExecutor(
        provider=args.provider,
        model=args.model or "",
    )

    if task:
        result = executor.run_json_task(task_file=task, data_input=input_text)
    else:
        full_prompt = prompt or ""
        if input_text:
            full_prompt = f"{full_prompt}\n\n{input_text}" if full_prompt else input_text
        system_prompt = getattr(args, 'system_prompt', None)
        result = executor.execute(full_prompt, system_prompt=system_prompt)

    cli.last_result = result
    cli.increment_stat("tasks_executed", 1)
    cli.set_stat("provider", args.provider)

    # Write output
    output_file = getattr(args, 'output', None)
    if output_file:
        with open(output_file, "a" if getattr(args, 'append', False) else "w",
                   encoding="utf-8") as fh:
            fh.write(str(result))
        print_success(f"Output written to {output_file}")
    else:
        print(result)

    cli.print_final_summary()


def _handle_task(args: argparse.Namespace, cli: CLIBase) -> None:
    """Handle the ``task`` subcommand — run a named JSON task."""
    from nono.tasker import TaskExecutor

    task_name: str = args.task_name

    if args.dry_run:
        print_header("DRY RUN SIMULATION")
        print_key_value("task", task_name)
        print_key_value("provider", getattr(args, 'provider', 'google'))
        cli.set_stat("mode", "dry-run")
        cli.print_final_summary()
        return

    # Resolve task file
    tasks_dir = os.path.join(os.path.dirname(__file__), "tasker", "prompts")
    task_path = os.path.join(tasks_dir, f"{task_name}.json")

    real_tasks_dir = os.path.realpath(tasks_dir) + os.sep
    real_task_path = os.path.realpath(task_path)
    if not real_task_path.startswith(real_tasks_dir):
        print_error("Invalid task name.")
        raise SystemExit(1)

    if not os.path.isfile(task_path):
        available = [
            f.removesuffix(".json")
            for f in sorted(os.listdir(tasks_dir))
            if f.endswith(".json")
        ] if os.path.isdir(tasks_dir) else []
        print_error(f"Task '{task_name}' not found. Available: {available}")
        raise SystemExit(1)

    with open(task_path, "r", encoding="utf-8") as fh:
        task_def = json.load(fh)

    genai_cfg = task_def.get("genai", {})
    provider = genai_cfg.get("provider", "google")
    model = genai_cfg.get("model", "")

    executor = TaskExecutor(provider=provider, model=model)

    data_input = getattr(args, 'data', None)
    result = executor.run_json_task(task_file=task_path, data_input=data_input)

    cli.last_result = result
    cli.increment_stat("tasks_executed", 1)
    cli.set_stat("provider", provider)
    print(result)
    cli.print_final_summary()


def _handle_agent(args: argparse.Namespace, cli: CLIBase) -> None:
    """Handle the ``agent`` subcommand — run a named agent template."""
    from nono.agent import Runner
    from nono.agent.templates import (
        planner_agent, decomposer_agent, summarizer_agent, reviewer_agent,
        coder_agent, classifier_agent, extractor_agent, writer_agent,
        guardrail_agent, plan_and_execute, research_and_write,
        draft_review_loop, classify_and_route,
    )

    builders: Dict[str, Callable] = {
        "planner": planner_agent,
        "decomposer": decomposer_agent,
        "summarizer": summarizer_agent,
        "reviewer": reviewer_agent,
        "coder": coder_agent,
        "classifier": classifier_agent,
        "extractor": extractor_agent,
        "writer": writer_agent,
        "guardrail": guardrail_agent,
        "plan_and_execute": plan_and_execute,
        "research_and_write": research_and_write,
        "draft_review_loop": draft_review_loop,
        "classify_and_route": classify_and_route,
    }

    agent_name: str = args.agent_name
    builder = builders.get(agent_name)
    if not builder:
        print_error(
            f"Agent '{agent_name}' not found. "
            f"Available: {sorted(builders.keys())}"
        )
        raise SystemExit(1)

    if args.dry_run:
        print_header("DRY RUN SIMULATION")
        print_key_value("agent", agent_name)
        print_key_value("message", args.message)
        cli.set_stat("mode", "dry-run")
        cli.print_final_summary()
        return

    agent = builder()
    runner = Runner(agent=agent)

    start = time.perf_counter()
    result = runner.run(args.message)
    elapsed = (time.perf_counter() - start) * 1000

    cli.last_result = result
    cli.set_stat("agent", agent_name)
    cli.set_stat("duration_ms", round(elapsed, 1))
    print(result)
    cli.print_final_summary()


def _handle_workflow(args: argparse.Namespace, cli: CLIBase) -> None:
    """Handle the ``workflow`` subcommand — run a named workflow template."""
    from nono.workflows.templates import (
        build_sentiment_pipeline,
        build_content_pipeline,
        build_data_enrichment,
        build_content_review_pipeline,
    )
    from nono.hitl import console_handler, make_auto_handler

    builders: Dict[str, Callable] = {
        "sentiment_pipeline": build_sentiment_pipeline,
        "content_pipeline": build_content_pipeline,
        "data_enrichment": build_data_enrichment,
        "content_review_pipeline": build_content_review_pipeline,
    }

    # Workflow names that accept a handler kwarg for HITL.
    hitl_workflows = {"content_review_pipeline"}

    wf_name: str = args.workflow_name
    builder = builders.get(wf_name)
    if not builder:
        print_error(
            f"Workflow '{wf_name}' not found. "
            f"Available: {sorted(builders.keys())}"
        )
        raise SystemExit(1)

    state_raw = getattr(args, 'state', None) or "{}"
    try:
        state = json.loads(state_raw) if isinstance(state_raw, str) else state_raw
    except json.JSONDecodeError:
        print_error("--state must be valid JSON.")
        raise SystemExit(1)

    if args.dry_run:
        print_header("DRY RUN SIMULATION")
        print_key_value("workflow", wf_name)
        print_key_value("state", str(state))
        print_key_value("interactive", getattr(args, 'interactive', False))
        cli.set_stat("mode", "dry-run")
        cli.print_final_summary()
        return

    # Resolve HITL handler
    interactive = getattr(args, 'interactive', False)
    hitl_raw = getattr(args, 'hitl_responses', None)

    if wf_name in hitl_workflows:
        if interactive:
            flow = builder(handler=console_handler)
        elif hitl_raw:
            try:
                hitl_cfg = json.loads(hitl_raw)
            except json.JSONDecodeError:
                print_error("--hitl-responses must be valid JSON.")
                raise SystemExit(1)
            flow = builder(handler=make_auto_handler(hitl_cfg))
        else:
            flow = builder()
    else:
        flow = builder()

    start = time.perf_counter()
    result = flow.run(**state)
    elapsed = (time.perf_counter() - start) * 1000

    cli.last_result = result
    cli.set_stat("workflow", wf_name)
    cli.set_stat("duration_ms", round(elapsed, 1))
    print(json.dumps(result, indent=2, default=str))
    cli.print_final_summary()


def _handle_skill(args: argparse.Namespace, cli: CLIBase) -> None:
    """Handle the ``skill`` subcommand — run a registered skill."""
    from nono.agent.skill import registry
    # Ensure built-in skills are loaded into the registry
    import nono.agent.skills  # noqa: F401

    skill_name: str = args.skill_name
    skill = registry.get(skill_name)
    if not skill:
        print_error(
            f"Skill '{skill_name}' not found. "
            f"Available: {registry.names}"
        )
        raise SystemExit(1)

    if args.dry_run:
        desc = skill.descriptor
        print_header("DRY RUN SIMULATION")
        print_key_value("skill", desc.name)
        print_key_value("description", desc.description)
        print_key_value("tags", ", ".join(desc.tags))
        print_key_value("input_keys", ", ".join(desc.input_keys))
        print_key_value("output_keys", ", ".join(desc.output_keys))
        cli.set_stat("mode", "dry-run")
        cli.print_final_summary()
        return

    overrides: Dict[str, Any] = {}
    provider = getattr(args, 'provider', None)
    if provider:
        overrides["provider"] = provider

    start = time.perf_counter()
    result = skill.run(args.message, **overrides)
    elapsed = (time.perf_counter() - start) * 1000

    cli.last_result = result
    cli.set_stat("skill", skill_name)
    cli.set_stat("duration_ms", round(elapsed, 1))
    print(result)
    cli.print_final_summary()


def _handle_mcp(args: argparse.Namespace, cli: CLIBase) -> None:
    """Handle the ``mcp`` subcommand — manage MCP servers."""
    from nono.connector.mcp_client import MCPManager

    action = getattr(args, "mcp_action", "list") or "list"
    mgr = MCPManager.from_config()

    # ── list ──────────────────────────────────────────────────────────
    if action == "list":
        servers = mgr.list_servers()
        output_format = getattr(args, 'output_format', OutputFormat.SUMMARY.value)

        if output_format == OutputFormat.JSON.value:
            cli.last_result = servers
            print(json.dumps(servers, indent=2))
            return

        if not servers:
            print_info("No MCP servers configured.")
            print_info("Use 'nono mcp add <name> --command <cmd>' to add one.")
            return

        print_header("MCP Servers")
        headers = ["Name", "Transport", "Target", "Enabled"]
        rows = []
        for srv in servers:
            target = srv.get("command", "") or srv.get("url", "")
            if srv.get("args"):
                target += " " + " ".join(srv["args"])
            enabled = "yes" if srv.get("enabled", True) else "no"
            rows.append([srv["name"], srv.get("transport", "stdio"), target, enabled])
        print_table(headers, rows)
        cli.last_result = servers

    # ── add ───────────────────────────────────────────────────────────
    elif action == "add":
        name: str = args.server_name
        transport: str = args.transport
        command: str = getattr(args, "command", "") or ""
        url: str = getattr(args, "url", "") or ""
        mcp_args: list[str] = getattr(args, "args", []) or []
        env_pairs: list[str] = getattr(args, "env", []) or []
        header_pairs: list[str] = getattr(args, "header", []) or []
        timeout: float = getattr(args, "timeout", 30.0)

        # Parse key=value pairs
        env = dict(pair.split("=", 1) for pair in env_pairs) if env_pairs else None
        hdrs = dict(pair.split("=", 1) for pair in header_pairs) if header_pairs else None

        # Validate transport requirements
        if transport == "stdio" and not command:
            print_error("--command is required for stdio transport.")
            raise SystemExit(1)
        if transport in ("http", "sse") and not url:
            print_error("--url is required for http/sse transport.")
            raise SystemExit(1)

        existed = name in mgr
        mgr.add(
            name,
            transport=transport,
            command=command,
            args=mcp_args if mcp_args else None,
            url=url,
            headers=hdrs,
            env=env,
            timeout=timeout,
        )
        mgr.save()

        verb = "Updated" if existed else "Added"
        print_success(f"{verb} MCP server '{name}' ({transport}).")
        cli.last_result = {"action": "add", "name": name, "transport": transport}

    # ── remove ────────────────────────────────────────────────────────
    elif action == "remove":
        name = args.server_name
        if mgr.remove(name):
            mgr.save()
            print_success(f"Removed MCP server '{name}'.")
            cli.last_result = {"action": "remove", "name": name}
        else:
            print_error(f"MCP server '{name}' not found.")
            raise SystemExit(1)

    # ── enable / disable ──────────────────────────────────────────────
    elif action == "enable":
        name = args.server_name
        if mgr.enable(name):
            mgr.save()
            print_success(f"Enabled MCP server '{name}'.")
        else:
            print_error(f"MCP server '{name}' not found.")
            raise SystemExit(1)

    elif action == "disable":
        name = args.server_name
        if mgr.disable(name):
            mgr.save()
            print_success(f"Disabled MCP server '{name}'.")
        else:
            print_error(f"MCP server '{name}' not found.")
            raise SystemExit(1)

    # ── tools ─────────────────────────────────────────────────────────
    elif action == "tools":
        name = getattr(args, "server_name", None)
        try:
            if name:
                tools = mgr.get_tools(name)
            else:
                tools = mgr.get_all_tools()
        except Exception as exc:
            print_error(f"Failed to get MCP tools: {exc}")
            raise SystemExit(1)

        output_format = getattr(args, 'output_format', OutputFormat.SUMMARY.value)
        tool_data = [{"name": t.name, "description": t.description} for t in tools]

        if output_format == OutputFormat.JSON.value:
            cli.last_result = tool_data
            print(json.dumps(tool_data, indent=2))
            return

        if not tools:
            print_info("No tools discovered from MCP server(s).")
            return

        src = f"server '{name}'" if name else "all servers"
        print_header(f"MCP Tools ({src})")
        headers_list = ["Name", "Description"]
        rows = [[t.name, t.description[:70]] for t in tools]
        print_table(headers_list, rows)
        cli.last_result = tool_data


def _handle_info(args: argparse.Namespace, cli: CLIBase) -> None:
    """Handle the ``info`` subcommand — list available resources."""
    tasks_dir = os.path.join(os.path.dirname(__file__), "tasker", "prompts")
    templates_dir = os.path.join(os.path.dirname(__file__), "tasker", "templates")

    tasks = [
        f.removesuffix(".json")
        for f in sorted(os.listdir(tasks_dir))
        if f.endswith(".json")
    ] if os.path.isdir(tasks_dir) else []

    templates = [
        f.removesuffix(".j2")
        for f in sorted(os.listdir(templates_dir))
        if f.endswith(".j2")
    ] if os.path.isdir(templates_dir) else []

    agents = [
        "planner", "decomposer", "summarizer", "reviewer", "coder",
        "classifier", "extractor", "writer", "guardrail",
        "plan_and_execute", "research_and_write", "draft_review_loop",
        "classify_and_route",
    ]

    workflows = ["sentiment_pipeline", "content_pipeline", "data_enrichment", "content_review_pipeline"]

    # Skills
    from nono.agent.skill import registry as skill_registry
    import nono.agent.skills  # noqa: F401 — ensure built-ins are loaded
    skill_names = skill_registry.names

    # MCP servers
    from nono.connector.mcp_client import load_mcp_config
    mcp_servers = load_mcp_config()
    mcp_names = [s.get("name", "") for s in mcp_servers]

    providers = [p.value for p in AIProvider]

    output_format = getattr(args, 'output_format', OutputFormat.SUMMARY.value)

    if output_format == OutputFormat.JSON.value:
        data = {
            "version": __version__,
            "providers": providers,
            "tasks": tasks,
            "templates": templates,
            "agents": agents,
            "workflows": workflows,
            "skills": skill_names,
            "mcp_servers": mcp_names,
        }
        cli.last_result = data
        print(json.dumps(data, indent=2))
        return

    print_header(f"Nono v{__version__} — Available Resources")

    print_subheader("AI Providers")
    print(", ".join(providers))

    if tasks:
        print_subheader("Tasks (JSON)")
        for name in tasks:
            print(f"  - {name}")

    if templates:
        print_subheader("Templates (Jinja2)")
        for name in templates:
            print(f"  - {name}")

    print_subheader("Agent Templates")
    for name in agents:
        print(f"  - {name}")

    print_subheader("Workflow Templates")
    for name in workflows:
        print(f"  - {name}")

    if skill_names:
        print_subheader("Skills")
        for name in skill_names:
            print(f"  - {name}")

    if mcp_names:
        print_subheader("MCP Servers")
        for name in mcp_names:
            print(f"  - {name}")


# Provider descriptions for the ``providers`` subcommand.
_PROVIDER_DETAILS: Dict[str, tuple[str, str]] = {
    "google":     ("Google Gemini",        "gemini-3-flash-preview"),
    "openai":     ("OpenAI GPT",           "gpt-4o-mini"),
    "perplexity": ("Perplexity AI",        "sonar"),
    "deepseek":   ("DeepSeek",             "deepseek-chat"),
    "xai":        ("xAI Grok",             "grok-3"),
    "groq":       ("Groq",                 "llama-3.3-70b-versatile"),
    "cerebras":   ("Cerebras",             "llama-3.3-70b"),
    "nvidia":     ("NVIDIA",               "meta/llama-3.3-70b-instruct"),
    "github":     ("GitHub Models",        "openai/gpt-5"),
    "openrouter": ("OpenRouter",           "openrouter/auto"),
    "huggingface":("Hugging Face",          "meta-llama/Llama-3.3-70B-Instruct"),
    "foundry":    ("Azure AI Foundry",     "openai/gpt-4o"),
    "vercel":     ("Vercel AI SDK",        "anthropic/claude-opus-4.5"),
    "ollama":     ("Local Ollama",         "llama3"),
}


def _handle_providers(args: argparse.Namespace, cli: CLIBase) -> None:
    """Handle the ``providers`` subcommand — list all supported AI providers."""
    output_format = getattr(args, 'output_format', OutputFormat.SUMMARY.value)

    if output_format == OutputFormat.JSON.value:
        data = {
            p.value: {
                "name": _PROVIDER_DETAILS.get(p.value, (p.value, ""))[0],
                "default_model": _PROVIDER_DETAILS.get(p.value, ("", ""))[1],
            }
            for p in AIProvider
        }
        cli.last_result = data
        print(json.dumps(data, indent=2))
        return

    print_header("Supported AI Providers")
    headers = ["Provider", "Name", "Default Model"]
    rows = [
        [p.value, *_PROVIDER_DETAILS.get(p.value, (p.value, ""))]
        for p in AIProvider
    ]
    print_table(headers, rows)
    print_info(
        "Use --provider <id> with any command. "
        "Default: google (Google Gemini)"
    )


def _handle_config(args: argparse.Namespace, cli: CLIBase) -> None:
    """Handle the ``config`` subcommand — init or show configuration."""
    from nono.config import load_config, create_sample_config

    action = getattr(args, 'config_action', 'show')

    if action == 'init':
        dest = getattr(args, 'dest', 'config.toml')
        fmt = getattr(args, 'full', False)

        if os.path.exists(dest):
            if not confirm_action(f"'{dest}' already exists. Overwrite?"):
                print_info("Aborted.")
                return

        if dest.endswith('.toml'):
            _write_toml_config(dest, full=fmt)
        else:
            ok = create_sample_config(dest)
            if not ok:
                print_error(f"Failed to write {dest}")
                raise SystemExit(1)

        print_success(f"Configuration written to {dest}")
        if not fmt:
            print_info(
                "This is a minimal config. "
                "Use 'nono config init --full' for all options."
            )
        cli.last_result = {"file": dest, "full": fmt}
        return

    # --- action == 'show' (default) ---
    config_file = getattr(args, 'config_file', None)
    cfg = load_config(filepath=config_file, env_prefix='NONO_')
    all_values = cfg.all()

    output_format = getattr(args, 'output_format', OutputFormat.SUMMARY.value)
    if output_format == OutputFormat.JSON.value:
        cli.last_result = all_values
        print(json.dumps(all_values, indent=2))
        return

    print_header("Current Configuration")
    for key in sorted(all_values):
        source = cfg.get_source(key)
        source_label = f" ({source.value})" if source else ""
        print_key_value(key, f"{all_values[key]}{source_label}")


def _write_toml_config(dest: str, *, full: bool) -> None:
    """Write a TOML config file — minimal or full.

    Args:
        dest: Destination file path.
        full: If ``True`` write all provider sections and options.
    """
    if full:
        # Copy the built-in config.toml as a full template
        src = Path(__file__).resolve().parent.parent / "config" / "config.toml"
        if src.exists():
            shutil.copy2(src, dest)
            return

    lines = [
        "# Nono GenAI Framework — Minimal Configuration",
        "# Run 'nono config init --full' to generate all options.",
        "",
        "[google]",
        'default_model = "gemini-3-flash-preview"',
        "",
        "# Uncomment and set your API key (or use GOOGLE_API_KEY env var)",
        '# api_key = ""',
        "",
        "[rate_limits]",
        "delay_between_requests = 0.5",
        "",
        "[paths]",
        'templates_dir = ""',
        'prompts_dir = ""',
    ]
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _handle_init(args: argparse.Namespace, cli: CLIBase) -> None:
    """Handle the ``init`` subcommand — scaffold a new Nono project."""
    from nono.project import init_project

    try:
        project = init_project(
            args.path,
            name=args.name,
            description=args.description,
            default_provider=args.provider,
        )
        print_success(f"Project '{project.name}' created at {project.root}")

        print()
        print_info("Structure:")
        for child in sorted(project.root.iterdir()):
            if child.name.startswith("."):
                continue
            suffix = "/" if child.is_dir() else ""
            print(f"  {child.name}{suffix}")

        print()
        print_info("Next steps:")
        print(f"  cd {project.root.name}")
        print("  nono info           # list available resources")
        print("  nono skill ...      # run a skill")

        cli.last_result = {
            "name": project.name,
            "root": str(project.root),
            "version": project.version,
        }
    except FileExistsError as exc:
        print_error(str(exc))
        raise SystemExit(1) from exc


def _handle_project(args: argparse.Namespace, cli: CLIBase) -> None:
    """Handle the ``project`` subcommand — show project information."""
    from nono.project import load_project

    try:
        project = load_project(args.path or ".")

        output_format = getattr(
            args, 'output_format', OutputFormat.SUMMARY.value
        )

        if output_format == OutputFormat.JSON.value:
            data = {
                "name": project.name,
                "description": project.description,
                "version": project.version,
                "root": str(project.root),
                "skills_dir": str(project.skills_dir),
                "prompts_dir": str(project.prompts_dir),
                "templates_dir": str(project.templates_dir),
                "workflows_dir": str(project.workflows_dir),
                "data_dir": str(project.data_dir),
                "skills": len(project.load_skills()),
                "prompts": len(project.list_prompts()),
                "templates": len(project.list_templates()),
                "workflows": len(project.list_workflows()),
                "data_files": len(project.list_data()),
            }
            cli.last_result = data
            print(json.dumps(data, indent=2))
            return

        print_header(f"Project: {project.name}")

        if project.description:
            print(f"  {project.description}")
            print()

        print(f"  Version:   {project.version}")
        print(f"  Root:      {project.root}")
        print(f"  Provider:  {project.manifest.default_provider or '(global default)'}")

        if project.manifest.default_model:
            print(f"  Model:     {project.manifest.default_model}")

        # Resource counts
        skills = project.load_skills()
        prompts = project.list_prompts()
        templates = project.list_templates()
        workflows = project.list_workflows()
        data_files = project.list_data()

        print()
        print_subheader("Resources")
        print(f"  Skills:     {len(skills)}")
        print(f"  Prompts:    {len(prompts)}")
        print(f"  Templates:  {len(templates)}")
        print(f"  Workflows:  {len(workflows)}")
        print(f"  Data files: {len(data_files)}")

        if skills:
            print()
            print_subheader("Skills")
            for s in skills:
                print(f"  - {s.descriptor.name}: {s.descriptor.description[:60]}")

        if prompts:
            print()
            print_subheader("Prompts")
            for p in prompts:
                print(f"  - {p.stem}")

        cli.last_result = {"name": project.name, "root": str(project.root)}

    except FileNotFoundError as exc:
        print_error(str(exc))
        raise SystemExit(1) from exc


# ============================================================================
# SHARED SUBCOMMAND SETUP
# ============================================================================


def _setup_subcommands(cli: CLIBase) -> None:
    """Register all subcommands on a CLIBase instance.

    Shared by ``main()`` and ``run_api()`` so both use the same
    command structure and handlers.
    """
    cli.init_subcommands()

    # --- run: execute a prompt or task file ---
    run_parser = cli.add_subcommand(
        "run", "Execute a prompt or task", handler=_handle_run, aliases=["r"],
    )
    run_parser.add_argument(
        "--prompt", type=str, help="Direct prompt text",
    )
    run_parser.add_argument(
        "--task", "-t", type=str, help="Path to task JSON file",
    )
    run_parser.add_argument(
        "--system-prompt", type=str, help="System prompt / instructions",
    )
    run_parser.add_argument(
        "--input", "-i", type=str, help="Input file",
    )
    run_parser.add_argument(
        "--output", "-o", type=str, help="Output file (stdout if omitted)",
    )
    run_parser.add_argument(
        "--append", action="store_true", help="Append to output file",
    )
    run_parser.add_argument(
        "--provider", "-p",
        choices=[p.value for p in AIProvider],
        default=AIProvider.GOOGLE.value,
        help="AI provider (default: %(default)s)",
    )
    run_parser.add_argument(
        "--model", "-m", type=str, help="Model name (provider default if omitted)",
    )

    # --- task: run a named JSON task from nono/tasker/prompts/ ---
    task_parser = cli.add_subcommand(
        "task", "Run a named JSON task", handler=_handle_task, aliases=["t"],
    )
    task_parser.add_argument(
        "task_name", type=str, help="Task name (from 'nono info')",
    )
    task_parser.add_argument(
        "--data", "-d", type=str, help="Data input (string or JSON)",
    )
    task_parser.add_argument(
        "--provider", "-p",
        choices=[p.value for p in AIProvider],
        default=AIProvider.GOOGLE.value,
        help="AI provider override (default: from task JSON)",
    )

    # --- agent: run a named agent template ---
    agent_parser = cli.add_subcommand(
        "agent", "Run a named agent template", handler=_handle_agent, aliases=["a"],
    )
    agent_parser.add_argument(
        "agent_name", type=str, help="Agent template name (from 'nono info')",
    )
    agent_parser.add_argument(
        "message", type=str, help="Message to send to the agent",
    )

    # --- workflow: run a named workflow template ---
    wf_parser = cli.add_subcommand(
        "workflow", "Run a named workflow template", handler=_handle_workflow,
        aliases=["wf"],
    )
    wf_parser.add_argument(
        "workflow_name", type=str, help="Workflow template name (from 'nono info')",
    )
    wf_parser.add_argument(
        "--state", "-s", type=str, default="{}",
        help='Initial state as JSON string (default: "{}")',
    )
    wf_parser.add_argument(
        "--interactive", action="store_true",
        help="Enable interactive human-in-the-loop prompts on stdin",
    )
    wf_parser.add_argument(
        "--hitl-responses", type=str, default=None,
        help='Pre-configured HITL responses as JSON: {"step": {"approved": true}}',
    )

    # --- skill: run a registered skill ---
    skill_parser = cli.add_subcommand(
        "skill", "Run a registered skill", handler=_handle_skill, aliases=["sk"],
    )
    skill_parser.add_argument(
        "skill_name", type=str, help="Skill name (from 'nono info')",
    )
    skill_parser.add_argument(
        "message", type=str, help="Input message for the skill",
    )
    skill_parser.add_argument(
        "--provider", "-p",
        choices=[p.value for p in AIProvider],
        default=None,
        help="AI provider override",
    )

    # --- mcp: manage MCP servers ---
    mcp_parser = cli.add_subcommand(
        "mcp", "Manage Model Context Protocol (MCP) servers",
        handler=_handle_mcp,
    )
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_action")
    mcp_sub.default = "list"

    # mcp list
    mcp_sub.add_parser("list", help="List configured MCP servers")

    # mcp add
    mcp_add = mcp_sub.add_parser("add", help="Add or update an MCP server")
    mcp_add.add_argument("server_name", type=str, help="Unique server name")
    mcp_add.add_argument(
        "--transport", "-t", choices=["stdio", "http", "sse"],
        default="stdio", help="Transport type (default: stdio)",
    )
    mcp_add.add_argument("--command", "-c", type=str, help="Executable (stdio)")
    mcp_add.add_argument(
        "--args", "-a", nargs="*", default=[], help="Command arguments (stdio)",
    )
    mcp_add.add_argument("--url", "-u", type=str, help="Server URL (http/sse)")
    mcp_add.add_argument(
        "--env", "-e", nargs="*", default=[],
        help="Environment vars as KEY=VALUE (stdio)",
    )
    mcp_add.add_argument(
        "--header", nargs="*", default=[],
        help="HTTP headers as KEY=VALUE (http/sse)",
    )
    mcp_add.add_argument(
        "--timeout", type=float, default=30.0,
        help="Connection timeout in seconds (default: 30)",
    )

    # mcp remove
    mcp_rm = mcp_sub.add_parser("remove", help="Remove an MCP server", aliases=["rm"])
    mcp_rm.add_argument("server_name", type=str, help="Server name to remove")

    # mcp enable
    mcp_en = mcp_sub.add_parser("enable", help="Enable a disabled MCP server")
    mcp_en.add_argument("server_name", type=str, help="Server name to enable")

    # mcp disable
    mcp_dis = mcp_sub.add_parser("disable", help="Disable an MCP server")
    mcp_dis.add_argument("server_name", type=str, help="Server name to disable")

    # mcp tools
    mcp_tools_p = mcp_sub.add_parser("tools", help="List tools from MCP server(s)")
    mcp_tools_p.add_argument(
        "server_name", nargs="?", default=None,
        help="Server name (omit for all servers)",
    )

    # --- info: list available resources ---
    cli.add_subcommand(
        "info", "List available tasks, agents, workflows, and providers",
        handler=_handle_info, aliases=["ls"],
    )

    # --- init: scaffold a new project ---
    init_parser = cli.add_subcommand(
        "init", "Create a new Nono project", handler=_handle_init,
    )
    init_parser.add_argument(
        "path", type=str, nargs="?", default=".",
        help="Directory for the new project (default: current directory)",
    )
    init_parser.add_argument(
        "--name", "-n", type=str, default="",
        help="Project name (default: directory name)",
    )
    init_parser.add_argument(
        "--description", "-d", type=str, default="",
        help="One-line project description",
    )
    init_parser.add_argument(
        "--provider", "-p",
        choices=[p.value for p in AIProvider],
        default=AIProvider.GOOGLE.value,
        help="Default AI provider (default: %(default)s)",
    )

    # --- project: show project info ---
    project_parser = cli.add_subcommand(
        "project", "Show current project information",
        handler=_handle_project, aliases=["proj"],
    )
    project_parser.add_argument(
        "--path", type=str, default=None,
        help="Project directory (default: auto-detect from cwd)",
    )

    # --- providers: show all supported AI providers ---
    cli.add_subcommand(
        "providers", "Show all supported AI providers with default models",
        handler=_handle_providers,
    )

    # --- config: configuration management (init, show) ---
    config_parser = cli.add_subcommand(
        "config", "Configuration management (init, show)",
        handler=_handle_config,
    )
    config_sub = config_parser.add_subparsers(dest="config_action")
    config_sub.default = "show"

    cfg_init_parser = config_sub.add_parser(
        "init", help="Generate a new config file",
    )
    cfg_init_parser.add_argument(
        "--dest", type=str, default="config.toml",
        help="Destination file path (default: config.toml)",
    )
    cfg_init_parser.add_argument(
        "--full", action="store_true",
        help="Generate a full config with all providers and options",
    )

    config_sub.add_parser(
        "show", help="Display current resolved configuration",
    )

    # --- wizard ---------------------------------------------------------------
    wizard_parser = cli.add_subcommand(
        "wizard", help="Interactive decision wizard — choose the right pattern",
        handler=_handle_wizard,
    )
    wizard_parser.add_argument(
        "--json", dest="wizard_json", action="store_true",
        help="Output recommendation as JSON",
    )


# ============================================================================
# WIZARD HANDLER
# ============================================================================


def _handle_wizard(args: "argparse.Namespace") -> str:
    """Run the interactive decision wizard."""
    from nono.wizard import recommend_interactive

    rec = recommend_interactive()

    if getattr(args, "wizard_json", False):
        return rec.as_json()

    return ""


# ============================================================================
# PROGRAMMATIC API
# ============================================================================


def run_api(argv: List[str]) -> CLIResult:
    """Programmatic entry point — run a CLI command and return structured results.

    Designed for embedding in REST APIs, test harnesses, orchestrators, or
    any Python code that needs to invoke CLI logic without ``subprocess``.

    Args:
        argv: Argument list, same as ``sys.argv[1:]``.

    Returns:
        CLIResult with ok, data, stats, and optional error.

    Example::

        result = run_api(["run", "--provider", "google", "--prompt", "Hello"])
        if result.ok:
            print(result.data)
    """
    cli = CLIBase(
        prog="nono",
        description="Nono GenAI Tasker",
        version=__version__,
    )
    _setup_subcommands(cli)

    try:
        # Force CI mode for programmatic usage
        if '--ci' not in argv:
            argv = list(argv) + ['--ci']

        cli.parse_args(argv)
        cli.run()

        return CLIResult(
            ok=True,
            data=cli.last_result,
            stats=dict(cli.stats),
        )
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code == 0:
            return CLIResult(ok=True, data=cli.last_result, stats=dict(cli.stats))
        return CLIResult(
            ok=False,
            error=f"CLI exited with code {code}",
            stats=dict(cli.stats),
        )
    except Exception as exc:
        return CLIResult(
            ok=False,
            error=str(exc),
            stats=dict(cli.stats),
        )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point returning an exit code.

    Exit codes:
        0 — success
        1 — runtime error
        2 — usage / argument error (argparse default)
        130 — interrupted by user (Ctrl+C)

    Args:
        argv: Override ``sys.argv[1:]`` for testing. Pass ``None`` for
              normal CLI usage.

    Returns:
        Integer exit code suitable for ``sys.exit()``.
    """
    Colors.init()

    cli = CLIBase(
        prog="nono",
        description="Nono GenAI Tasker — Execute AI tasks with multi-provider support",
        version=__version__,
    )
    _setup_subcommands(cli)

    cli.add_examples([
        "%(prog)s run --provider google --prompt 'Summarize this text'",
        "%(prog)s run --provider openai --prompt 'Translate to Spanish' -i doc.txt -o out.txt",
        "%(prog)s task name_classifier --data '[\"John Smith\", \"Tokyo\"]'",
        "%(prog)s agent summarizer 'Summarize the key features of Python 3.13'",
        "%(prog)s workflow sentiment_pipeline --state '{\"input\": \"I love this!\"}'",
        "%(prog)s info",
        "%(prog)s @params.txt",
    ])

    try:
        cli.parse_args(argv)
        cli.run()
        return 0

    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except KeyboardInterrupt:
        print()
        print_warning("Interrupted by user")
        return 130
    except Exception as exc:
        print_error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
