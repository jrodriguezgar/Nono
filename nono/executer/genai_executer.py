"""
GenAI Executer - Generative AI Code Generation and Execution Module

This module provides a unified interface for generating Python code from natural
language instructions using AI and executing it in a controlled environment.
It follows S.O.L.I.D principles and integrates with the genai_tasker connector.

Author: DatamanEdge
License: MIT
Version: 1.0.0
"""

import io
import os
import re
import sys
import json
import logging
import functools
import subprocess
import tempfile
import threading
import traceback
from abc import ABC, abstractmethod
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# Configure Logging
logger = logging.getLogger("GenAICodeExecuter")
logger.addHandler(logging.NullHandler())


def msg_log(message: str, level: int = logging.INFO) -> None:
    """
    Log a message and print it to the console with timestamp.
    
    Args:
        message: The message to log and print.
        level: The logging level (default: INFO).
    
    Returns:
        None
    """
    logger.log(level, message)


def event_log(message: Optional[str] = None, level: int = logging.INFO):
    """
    Decorator for managing log messages at function entry and exit.
    
    Automatically logs when a function starts, completes, or raises an error.
    
    Args:
        message: Custom message to display. Defaults to function name.
        level: The logging level (default: INFO).
    
    Returns:
        Callable: Decorated function wrapper.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            act_msg = message if message else f"Execution of {func.__name__}"
            
            msg_log(f"Starting: {act_msg}", level)
            try:
                result = func(*args, **kwargs)
                msg_log(f"Completed: {act_msg}", level)
                return result
            except Exception as e:
                msg_log(f"Error in {act_msg}: {str(e)}", logging.ERROR)
                raise
        return wrapper
    return decorator


class ExecutionMode(Enum):
    """Enumeration of code execution modes."""
    
    SUBPROCESS = "subprocess"  # Execute in separate process (safer)
    EXEC = "exec"              # Execute with exec() (faster but less isolated)


class SecurityMode(Enum):
    """
    Enumeration of security modes for code execution.
    
    SAFE: Default mode. Blocks dangerous operations like:
        - File system modifications (write, delete)
        - Network operations (requests, sockets)
        - System commands (os.system, subprocess calls)
        - Dynamic code execution (eval, exec, compile)
        - Module manipulation (__import__, importlib)
    
    PERMISSIVE: Allows all operations. Use with caution.
        Only enable when you trust the generated code completely.
    """
    
    SAFE = "safe"            # Default: blocks dangerous operations
    PERMISSIVE = "permissive"  # Allows all operations (use with caution)


@dataclass
class ExecutionResult:
    """
    Result of code execution.
    
    Attributes:
        success: Whether execution completed without errors.
        output: Standard output from execution.
        error: Error message if execution failed.
        code: The generated Python code.
        execution_time: Time taken to execute in seconds.
        return_value: Return value if code returned something.
        execution_id: Unique identifier for the execution.
        instruction: The original instruction that generated the code.
        timestamp: When the execution occurred.
    """
    
    success: bool
    output: str
    error: str = ""
    code: str = ""
    execution_time: float = 0.0
    return_value: Any = None
    execution_id: str = ""
    instruction: str = ""
    timestamp: str = ""


class CodeExecutionError(Exception):
    """Exception raised when code execution fails."""
    
    def __init__(self, message: str, code: str = "", original_error: Optional[Exception] = None):
        super().__init__(message)
        self.code = code
        self.original_error = original_error


# Import tasker for AI operations and prompt building
try:
    from ..tasker import TaskExecutor, TaskPromptBuilder, build_prompt
except ImportError:
    # Fallback for when running as script - add parent dir to path
    _nono_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _nono_dir not in sys.path:
        sys.path.insert(0, _nono_dir)
    try:
        from tasker import TaskExecutor, TaskPromptBuilder, build_prompt
    except ImportError:
        from nono.tasker import TaskExecutor, TaskPromptBuilder, build_prompt


# Code generation template file (located in tasker/templates/)
CODE_GENERATION_TEMPLATE = "python_programming.j2"

# Restricted builtins for sandboxed exec() — no open/eval/exec/compile/__import__
_SAFE_BUILTINS: dict[str, Any] = {
    "print": print, "range": range, "len": len, "int": int,
    "float": float, "str": str, "bool": bool, "list": list,
    "dict": dict, "tuple": tuple, "set": set, "frozenset": frozenset,
    "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
    "sorted": sorted, "reversed": reversed, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter, "any": any, "all": all,
    "isinstance": isinstance, "issubclass": issubclass, "type": type,
    "hasattr": hasattr,
    "repr": repr, "hash": hash, "id": id, "chr": chr, "ord": ord,
    "hex": hex, "oct": oct, "bin": bin, "format": format,
    "True": True, "False": False, "None": None,
    "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError,
    "RuntimeError": RuntimeError, "StopIteration": StopIteration,
    "Exception": Exception,
}


class CodeExecuter:
    """
    Main class for generating and executing Python code using AI.
    
    This class wraps TaskExecutor from the tasker module to generate Python 
    code from natural language instructions using J2 templates, and provides
    safe execution environments for the generated code.
    
    Attributes:
        tasker: TaskExecutor instance for AI operations.
        prompt_builder: TaskPromptBuilder for J2 template rendering.
        execution_mode: How to execute generated code (subprocess/exec).
        security_mode: Security level for code execution (safe/permissive).
        allowed_modules: List of allowed modules for sandboxed execution.
    """
    
    # Default allowed modules for sandboxed execution (SAFE mode)
    DEFAULT_ALLOWED_MODULES = [
        "math", "datetime", "json", "re", "pathlib",
        "collections", "itertools", "functools", "operator",
        "random", "statistics", "decimal", "fractions",
        "string", "textwrap", "unicodedata", "typing"
    ]
    
    # Dangerous patterns blocked in SAFE mode (pre-compiled)
    DANGEROUS_PATTERNS = [
        # File write/delete operations
        re.compile(r"\bopen\s*\([^)]*['\"][wa]", re.IGNORECASE),
        re.compile(r"\bos\.remove\b", re.IGNORECASE),
        re.compile(r"\bos\.unlink\b", re.IGNORECASE),
        re.compile(r"\bos\.rmdir\b", re.IGNORECASE),
        re.compile(r"\bshutil\.rmtree\b", re.IGNORECASE),
        re.compile(r"\bshutil\.move\b", re.IGNORECASE),
        re.compile(r"\bshutil\.copy\b", re.IGNORECASE),
        re.compile(r"\bpathlib\.Path\([^)]*\)\.unlink\b", re.IGNORECASE),
        re.compile(r"\bpathlib\.Path\([^)]*\)\.rmdir\b", re.IGNORECASE),
        re.compile(r"\.write\s*\(", re.IGNORECASE),
        re.compile(r"\.writelines\s*\(", re.IGNORECASE),
        # Network operations
        re.compile(r"\brequests\.", re.IGNORECASE),
        re.compile(r"\burllib\.", re.IGNORECASE),
        re.compile(r"\bhttpx\.", re.IGNORECASE),
        re.compile(r"\bsocket\.", re.IGNORECASE),
        re.compile(r"\bhttp\.client\.", re.IGNORECASE),
        re.compile(r"\baiohttp\.", re.IGNORECASE),
        re.compile(r"\bftplib\.", re.IGNORECASE),
        # System command execution
        re.compile(r"\bos\.system\b", re.IGNORECASE),
        re.compile(r"\bos\.popen\b", re.IGNORECASE),
        re.compile(r"\bos\.spawn", re.IGNORECASE),
        re.compile(r"\bsubprocess\.", re.IGNORECASE),
        re.compile(r"\bcommands\.", re.IGNORECASE),
        # Dynamic code execution
        re.compile(r"\beval\s*\(", re.IGNORECASE),
        re.compile(r"\bexec\s*\(", re.IGNORECASE),
        re.compile(r"\bcompile\s*\(", re.IGNORECASE),
        # Module manipulation
        re.compile(r"\b__import__\s*\(", re.IGNORECASE),
        re.compile(r"\bimportlib\.", re.IGNORECASE),
        # Environment manipulation
        re.compile(r"\bos\.environ\[", re.IGNORECASE),
        re.compile(r"\bos\.putenv\b", re.IGNORECASE),
        re.compile(r"\bos\.setenv\b", re.IGNORECASE),
        # Process manipulation
        re.compile(r"\bos\.kill\b", re.IGNORECASE),
        re.compile(r"\bos\.fork\b", re.IGNORECASE),
        re.compile(r"\bos\.exec", re.IGNORECASE),
        re.compile(r"\bmultiprocessing\.", re.IGNORECASE),
        re.compile(r"\bthreading\.", re.IGNORECASE),
        # Dunder attribute access — blocks type-hierarchy traversal
        # (e.g. ().__class__.__base__.__subclasses__() → os._wrap_close → os.system)
        re.compile(r"__subclasses__"),
        re.compile(r"__mro__"),
        re.compile(r"__base__"),
        re.compile(r"__bases__"),
        re.compile(r"__globals__"),
        re.compile(r"__builtins__"),
        re.compile(r"__code__"),
        re.compile(r"__reduce__"),
        re.compile(r"__reduce_ex__"),
        re.compile(r"__getattr__"),
        re.compile(r"__init_subclass__"),
        # Block globals()/locals() — prevents sandbox code from mutating __builtins__
        re.compile(r"\bglobals\s*\("),
        re.compile(r"\blocals\s*\("),
    ]
    
    # Warning patterns (allowed but logged in SAFE mode, pre-compiled)
    WARNING_PATTERNS = [
        re.compile(r"\bos\.listdir\b", re.IGNORECASE),
        re.compile(r"\bos\.walk\b", re.IGNORECASE),
        re.compile(r"\bos\.path\.", re.IGNORECASE),
        re.compile(r"\bglob\.", re.IGNORECASE),
        re.compile(r"\bpathlib\.Path\(", re.IGNORECASE),
    ]
    
    def __init__(
        self,
        provider: str,
        model_name: str,
        config_file: Optional[str] = None,
        execution_mode: Optional[ExecutionMode] = None,
        security_mode: Optional[SecurityMode] = None,
        allowed_modules: Optional[List[str]] = None,
        temperature: Union[float, str] = "coding"
    ):
        """
        Initialize the CodeExecuter.
        
        This class wraps TaskExecutor from tasker to handle AI operations,
        using J2 templates for prompt building. Provider, model and credentials
        are handled by TaskExecutor/connector_genai (same as tasker).
        
        Args:
            provider: AI provider name (google, openai, perplexity, groq, etc.). Required.
            model_name: Model name to use. Required.
            config_file: Path to configuration file (JSON). If None, looks for config.json in module dir.
            execution_mode: How to execute code (subprocess/exec).
            security_mode: Security level (SAFE blocks dangerous ops, PERMISSIVE allows all).
            allowed_modules: List of allowed modules for execution.
            temperature: Sampling temperature for code generation (default: 0.2 for deterministic code).
        """
        # Try to load default config file if none specified
        default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        effective_config_file = config_file or (default_config if os.path.exists(default_config) else None)
        
        if effective_config_file:
            logger.info(f"Loading config from: {effective_config_file}")
        
        # Load config file to get execution settings only
        config_data = self._load_config_file(effective_config_file)
        
        # Resolve execution settings (param > config > default)
        exec_config = config_data.get("execution", {})
        
        if execution_mode is not None:
            self.execution_mode = execution_mode
        else:
            mode_str = exec_config.get("mode", "subprocess")
            self.execution_mode = ExecutionMode(mode_str.lower())
        
        if security_mode is not None:
            self.security_mode = security_mode
        else:
            sec_str = exec_config.get("security", "safe")
            self.security_mode = SecurityMode(sec_str.lower())
        
        self.default_timeout = exec_config.get("timeout", 30)
        self.default_max_retries = exec_config.get("max_retries", 2)
        self.default_save_executions = exec_config.get("save_executions", True)
        
        self.allowed_modules = allowed_modules or self.DEFAULT_ALLOWED_MODULES
        
        # Initialize TaskExecutor from tasker (handles provider, model, and credentials)
        self.tasker = TaskExecutor(
            provider=provider,
            model=model_name,
            temperature=temperature,
            max_tokens=4096
        )
        
        # Initialize prompt builder for J2 templates
        self.prompt_builder = TaskPromptBuilder()
        
        # Setup executions directory
        self.executions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "executions")
        os.makedirs(self.executions_dir, exist_ok=True)
        
        security_label = "🔒 SAFE" if self.security_mode == SecurityMode.SAFE else "⚠️ PERMISSIVE"
        msg_log(f"CodeExecuter initialized - Provider: {provider}, Model: {model_name}, Security: {security_label}", logging.INFO)
    
    def _load_config_file(self, config_file: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_file: Path to config file.
        
        Returns:
            Configuration dictionary or empty dict if not found.
        """
        if not config_file or not os.path.exists(config_file):
            return {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading config file {config_file}: {e}")
            return {}
    
    @event_log(message="Code Generation")
    def generate_code(self, instruction: str, context: Optional[str] = None, previous_error: Optional[str] = None) -> str:
        """
        Generate Python code from natural language instruction.
        
        Uses J2 template (code_executer.j2) to build prompts and
        TaskExecutor to generate the code.
        
        Args:
            instruction: Natural language description of desired code.
            context: Optional additional context or constraints.
            previous_error: Optional error from previous attempt (for retry).
        
        Returns:
            Generated Python code as string.
        """
        # Build prompts using J2 template (gets both system and user blocks)
        # Map: instruction -> topic, context -> constraints, previous_error -> requirements
        requirements = [f"Fix error: {previous_error}"] if previous_error else None
        prompts = self.prompt_builder.from_template_file_blocks(
            CODE_GENERATION_TEMPLATE,
            topic=instruction,
            constraints=context,
            requirements=requirements
        )
        
        # Prepare messages
        messages = []
        if prompts.get("system"):
            messages.append({"role": "system", "content": prompts["system"]})
        if prompts.get("user"):
            messages.append({"role": "user", "content": prompts["user"]})
        
        # Generate code using TaskExecutor
        code = self.tasker.execute(messages)
        
        # Clean up the response - remove any markdown formatting
        code = self._clean_code_response(code)
        
        return code
    
    def _clean_code_response(self, code: str) -> str:
        """
        Clean up AI response to extract pure Python code.
        
        Args:
            code: Raw response from AI.
        
        Returns:
            Clean Python code.
        """
        code = code.strip()
        
        # Remove markdown code blocks if present
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()
    
    def _validate_code_security(self, code: str) -> tuple[bool, List[str]]:
        """
        Validate code against security patterns in SAFE mode.
        
        Args:
            code: Python code to validate.
        
        Returns:
            Tuple of (is_safe, list_of_violations).
        """
        if self.security_mode == SecurityMode.PERMISSIVE:
            return True, []
        
        violations = []
        warnings = []
        
        # Check for dangerous patterns (pre-compiled)
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(code):
                violations.append(f"Blocked pattern detected: {pattern.pattern}")
        
        # Check for warning patterns (log but don't block, pre-compiled)
        for pattern in self.WARNING_PATTERNS:
            if pattern.search(code):
                warnings.append(f"Warning pattern: {pattern.pattern}")
        
        # Log warnings
        for warning in warnings:
            logger.warning(f"Security warning in generated code: {warning}")
        
        return len(violations) == 0, violations
    
    def set_security_mode(self, mode: SecurityMode) -> None:
        """
        Change the security mode at runtime.
        
        Args:
            mode: New security mode (SAFE or PERMISSIVE).
        
        Warning:
            Switching to PERMISSIVE mode allows potentially dangerous operations.
        """
        old_mode = self.security_mode
        self.security_mode = mode
        
        if mode == SecurityMode.PERMISSIVE:
            msg_log("⚠️ WARNING: Security mode changed to PERMISSIVE. Dangerous operations are now allowed!", logging.WARNING)
        else:
            msg_log("🔒 Security mode changed to SAFE. Dangerous operations are now blocked.", logging.INFO)
    
    @event_log(message="Code Execution (Subprocess)")
    def _execute_subprocess(self, code: str, timeout: int = 30) -> ExecutionResult:
        """
        Execute code in a separate subprocess.
        
        Args:
            code: Python code to execute.
            timeout: Maximum execution time in seconds.
        
        Returns:
            ExecutionResult with output and status.
        """
        start_time = datetime.now()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute in subprocess
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result.returncode == 0:
                return ExecutionResult(
                    success=True,
                    output=result.stdout,
                    error=result.stderr if result.stderr else "",
                    code=code,
                    execution_time=execution_time
                )
            else:
                return ExecutionResult(
                    success=False,
                    output=result.stdout,
                    error=result.stderr,
                    code=code,
                    execution_time=execution_time
                )
        
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {timeout} seconds",
                code=code,
                execution_time=timeout
            )
        
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                code=code,
                execution_time=(datetime.now() - start_time).total_seconds()
            )
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except OSError:
                pass
    
    @event_log(message="Code Execution (Exec)")
    def _execute_exec(self, code: str, timeout: int = 30) -> ExecutionResult:
        """
        Execute code using Python's exec() function.
        
        Args:
            code: Python code to execute.
            timeout: Maximum execution time in seconds.
        
        Returns:
            ExecutionResult with output and status.
        """
        start_time = datetime.now()
        
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Create execution namespace with restricted builtins
        exec_globals = {
            "__builtins__": _SAFE_BUILTINS,
            "__name__": "__main__",
        }
        
        _thread_exc: list[BaseException] = []

        def _run() -> None:
            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, exec_globals)  # noqa: S102
            except BaseException as exc:
                _thread_exc.append(exc)

        # Guard against unbounded leaked-thread accumulation
        _MAX_LEAKED_EXEC_THREADS = 8
        leaked = sum(
            1 for t in threading.enumerate()
            if t.name.startswith("nono-exec-") and t.is_alive()
        )
        if leaked >= _MAX_LEAKED_EXEC_THREADS:
            return ExecutionResult(
                success=False,
                output="",
                error=(
                    f"Too many timed-out exec threads ({leaked}). "
                    "Refusing new execution to prevent resource exhaustion."
                ),
                code=code,
                execution_time=0.0,
            )

        runner = threading.Thread(target=_run, daemon=True, name="nono-exec-sandbox")
        runner.start()
        runner.join(timeout=timeout)

        if runner.is_alive():
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.warning(
                "Exec thread leaked (timed out after %ds). "
                "Active leaked threads: %d",
                timeout, leaked + 1,
            )
            return ExecutionResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=f"Execution timed out after {timeout}s",
                code=code,
                execution_time=execution_time,
            )
        
        try:
            execution_time = (datetime.now() - start_time).total_seconds()

            if _thread_exc:
                exc = _thread_exc[0]
                error_msg = f"{type(exc).__name__}: {exc}"
                return ExecutionResult(
                    success=False,
                    output=stdout_capture.getvalue(),
                    error=error_msg,
                    code=code,
                    execution_time=execution_time,
                )

            return ExecutionResult(
                success=True,
                output=stdout_capture.getvalue(),
                error=stderr_capture.getvalue(),
                code=code,
                execution_time=execution_time,
                return_value=exec_globals.get("result", None)
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            
            return ExecutionResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=error_msg,
                code=code,
                execution_time=execution_time
            )
        finally:
            stdout_capture.close()
            stderr_capture.close()
    
    def execute_code(self, code: str, timeout: int = 30) -> ExecutionResult:
        """
        Execute the given Python code.
        
        Args:
            code: Python code to execute.
            timeout: Maximum execution time (for subprocess mode).
        
        Returns:
            ExecutionResult with output and status.
        """
        # Validate code security in SAFE mode
        is_safe, violations = self._validate_code_security(code)
        
        if not is_safe:
            violation_msg = "\n".join(violations)
            error_msg = (
                f"🔒 SECURITY BLOCK: Code contains potentially dangerous operations.\n"
                f"Violations:\n{violation_msg}\n\n"
                f"To execute this code, either:\n"
                f"1. Remove the dangerous operations from your instruction\n"
                f"2. Use security_mode=SecurityMode.PERMISSIVE (not recommended)"
            )
            logger.warning(f"Code blocked by security check: {len(violations)} violations found")
            return ExecutionResult(
                success=False,
                output="",
                error=error_msg,
                code=code,
                execution_time=0.0
            )
        
        if self.execution_mode == ExecutionMode.SUBPROCESS:
            return self._execute_subprocess(code, timeout)
        else:
            return self._execute_exec(code, timeout)
    
    @event_log(message="Generate and Execute")
    def run(
        self,
        instruction: str,
        context: Optional[str] = None,
        timeout: Optional[int] = None,
        retry_on_error: bool = True,
        max_retries: Optional[int] = None,
        save_execution: Optional[bool] = None
    ) -> ExecutionResult:
        """
        Generate code from instruction and execute it.
        
        This is the main entry point that combines code generation
        and execution in a single call. Saves execution to the executions directory.
        
        Args:
            instruction: Natural language description of desired code.
            context: Optional additional context.
            timeout: Maximum execution time in seconds (default from config).
            retry_on_error: Whether to retry if execution fails.
            max_retries: Maximum number of retry attempts (default from config).
            save_execution: Whether to save execution to executions directory (default from config).
        
        Returns:
            ExecutionResult with generated code and output.
        """
        # Use config defaults if not specified (ensure int types for type checker)
        effective_timeout: int = timeout if timeout is not None else self.default_timeout
        effective_max_retries: int = max_retries if max_retries is not None else self.default_max_retries
        effective_save: bool = save_execution if save_execution is not None else self.default_save_executions
        
        # Generate unique execution ID
        execution_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        timestamp = datetime.now().isoformat()
        
        attempts = 0
        last_result = None
        last_error = ""
        
        while attempts <= effective_max_retries:
            attempts += 1
            
            # Generate code (pass previous error for retry, not merged with context)
            try:
                code = self.generate_code(
                    instruction, 
                    context=context,
                    previous_error=last_error if retry_on_error else None
                )
            except Exception as e:
                logger.error(f"Code generation failed: {e}")
                final_result = ExecutionResult(
                    success=False,
                    output="",
                    error=f"Code generation failed: {str(e)}",
                    code="",
                    execution_id=execution_id,
                    instruction=instruction,
                    timestamp=timestamp
                )
                if effective_save:
                    self._save_execution(final_result)
                return final_result
            
            # Execute code
            result = self.execute_code(code, effective_timeout)
            
            # Enrich result with metadata
            result.execution_id = execution_id
            result.instruction = instruction
            result.timestamp = timestamp
            
            last_result = result
            
            if result.success:
                if effective_save:
                    self._save_execution(result)
                return result
            
            # Prepare error for retry
            last_error = result.error
            
            if not retry_on_error or attempts > effective_max_retries:
                break
            
            logger.warning(f"Execution failed, retrying ({attempts}/{effective_max_retries})...")
        
        final_result = last_result if last_result else ExecutionResult(
            success=False,
            output="",
            error="Failed to generate and execute code",
            code="",
            execution_id=execution_id,
            instruction=instruction,
            timestamp=timestamp
        )
        
        if effective_save:
            self._save_execution(final_result)
        
        return final_result
    
    def _save_execution(self, result: ExecutionResult) -> str:
        """
        Save execution result to the executions directory.
        
        Creates a single JSON file containing execution metadata and generated code.
        
        Args:
            result: ExecutionResult to save.
        
        Returns:
            Path to the saved JSON file.
        """
        # Ensure executions directory exists
        os.makedirs(self.executions_dir, exist_ok=True)
        
        # Create filename with status
        base_name = result.execution_id
        status_suffix = "_OK" if result.success else "_FAIL"
        json_file = os.path.join(self.executions_dir, f"{base_name}{status_suffix}.json")
        
        # Save all data in single JSON file
        execution_data = {
            "execution_id": result.execution_id,
            "timestamp": result.timestamp,
            "instruction": result.instruction,
            "success": result.success,
            "execution_time": result.execution_time,
            "output": result.output,
            "error": result.error,
            "code": result.code,
            "provider": self.tasker.config.provider.value,
            "model": self.tasker.config.model_name,
            "security_mode": self.security_mode.value,
            "execution_mode": self.execution_mode.value
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(execution_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Execution saved: {json_file}")
        return json_file
    
    def list_executions(self, limit: int = 20, success_only: bool = False) -> List[Dict[str, Any]]:
        """
        List recent executions from the executions directory.
        
        Args:
            limit: Maximum number of executions to return.
            success_only: If True, only return successful executions.
        
        Returns:
            List of execution metadata dictionaries.
        """
        executions = []
        
        if not os.path.exists(self.executions_dir):
            return executions
        
        # Find all JSON files in executions directory
        json_files = sorted(
            [f for f in os.listdir(self.executions_dir) if f.endswith('.json')],
            reverse=True  # Most recent first
        )
        
        for json_file in json_files[:limit * 2]:  # Get more in case of filtering
            try:
                with open(os.path.join(self.executions_dir, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if success_only and not data.get('success', False):
                    continue
                    
                executions.append(data)
                
                if len(executions) >= limit:
                    break
            except Exception as e:
                logger.warning(f"Error reading execution file {json_file}: {e}")
        
        return executions
    
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific execution by ID.
        
        Args:
            execution_id: The execution ID to retrieve.
        
        Returns:
            Execution metadata dictionary or None if not found.
        """
        # Try both success and fail suffixes
        for suffix in ["_OK", "_FAIL"]:
            json_file = os.path.join(self.executions_dir, f"{execution_id}{suffix}.json")
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error reading execution {execution_id}: {e}")
                    return None
        
        return None
    
    def clear_executions(self, keep_recent: int = 0) -> int:
        """
        Clear execution history.
        
        Args:
            keep_recent: Number of most recent executions to keep. If 0, delete all.
        
        Returns:
            Number of executions deleted.
        """
        if not os.path.exists(self.executions_dir):
            return 0
        
        # Get all JSON files sorted by modification time
        all_files = []
        for f in os.listdir(self.executions_dir):
            if f.endswith('.json'):
                filepath = os.path.join(self.executions_dir, f)
                all_files.append((filepath, os.path.getmtime(filepath)))
        
        all_files.sort(key=lambda x: x[1], reverse=True)
        
        # Files to delete (skip the most recent ones if keep_recent > 0)
        files_to_delete = all_files[keep_recent:] if keep_recent > 0 else all_files
        
        deleted_count = 0
        for filepath, _ in files_to_delete:
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Error deleting {filepath}: {e}")
        
        logger.info(f"Cleared {deleted_count} execution files")
        return deleted_count
