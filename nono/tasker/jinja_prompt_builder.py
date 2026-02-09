"""
JinjaPromptPy Prompt Builder for GenAI Tasker

This module provides the exclusive prompt building mechanism using jinjapromptpy.
All prompts in the GenAI Tasker system are built using Jinja2 templates.

Requires: jinjapromptpy (pip install jinja-prompt-py)

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

from jinjapromptpy import PromptGenerator, PromptTemplate, BatchResult, RenderResult

# Import central configuration
from ..config import NonoConfig

logger = logging.getLogger("GenAITaskExecutor.PromptBuilder")


# ============================================================================
# Default Templates Directory (now uses central config)
# ============================================================================

def get_templates_dir() -> Path:
    """Get templates directory from central configuration."""
    return NonoConfig.get_templates_dir()


# For backward compatibility
TEMPLATES_DIR = Path(__file__).parent / "templates"


# ============================================================================
# Custom Jinja2 Filters for GenAI Tasks
# ============================================================================

def to_compact_json(value: Any) -> str:
    """Convert value to compact JSON string (no whitespace)."""
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def to_pretty_json(value: Any, indent: int = 2) -> str:
    """Convert value to pretty-printed JSON string."""
    return json.dumps(value, ensure_ascii=False, indent=indent)


def truncate(value: str, length: int = 100, suffix: str = "...") -> str:
    """Truncate string to specified length."""
    if len(value) <= length:
        return value
    return value[:length - len(suffix)] + suffix


def escape_quotes(value: str) -> str:
    """Escape double quotes in string."""
    return value.replace('"', '\\"')


def numbered_list(items: List[Any], start: int = 1) -> str:
    """Format items as a numbered list."""
    lines = []
    for i, item in enumerate(items, start=start):
        lines.append(f"{i}. {item}")
    return "\n".join(lines)


def bullet_list(items: List[Any], bullet: str = "-") -> str:
    """Format items as a bullet list."""
    lines = []
    for item in items:
        lines.append(f"{bullet} {item}")
    return "\n".join(lines)


DEFAULT_FILTERS = {
    "to_compact_json": to_compact_json,
    "to_pretty_json": to_pretty_json,
    "truncate": truncate,
    "escape_quotes": escape_quotes,
    "numbered_list": numbered_list,
    "bullet_list": bullet_list,
}


# ============================================================================
# Prompt Builder Class
# ============================================================================

class TaskPromptBuilder:
    """
    Builds prompts for GenAI tasks using Jinja2 templates via jinjapromptpy.
    
    This class provides a unified interface for building prompts from:
    - Jinja2 template files (.j2)
    - Jinja2 template strings
    - Legacy JSON task definitions (with automatic template conversion)
    
    Features:
    - Automatic batching for large datasets
    - Custom Jinja2 filters
    - Template inheritance support
    - Token-aware splitting
    
    Example:
        >>> builder = TaskPromptBuilder()
        >>> prompts = builder.from_template_file(
        ...     "name_classifier.j2",
        ...     data=["Alice", "Bob", "Table", "Chair"],
        ...     max_tokens=4096
        ... )
        >>> for prompt in prompts:
        ...     response = ai_client.generate(prompt)
    """
    
    def __init__(
        self,
        templates_dir: Optional[Union[str, Path]] = None,
        custom_filters: Optional[Dict[str, Callable]] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        default_max_tokens: int = 8192,
        overlap_items: int = 0
    ):
        """
        Initialize the TaskPromptBuilder.
        
        Args:
            templates_dir: Directory containing template files. 
                          If None, uses central config (NonoConfig.get_templates_dir()).
                          Can be set via NONO_TEMPLATES_DIR env var or programmatically.
            custom_filters: Additional custom Jinja2 filters.
            token_counter: Custom function to count tokens in a string.
            default_max_tokens: Default maximum tokens per prompt.
            overlap_items: Number of items to overlap between batches.
        """
        # Use provided path, or resolve from central config
        if templates_dir:
            self._templates_dir = Path(templates_dir)
        else:
            self._templates_dir = get_templates_dir()
        
        self._custom_filters = {**DEFAULT_FILTERS, **(custom_filters or {})}
        self._token_counter = token_counter
        self._default_max_tokens = default_max_tokens
        self._overlap_items = overlap_items
        
        logger.info(f"TaskPromptBuilder initialized with templates from: {self._templates_dir}")
    
    def _create_generator(self) -> PromptGenerator:
        """Create a new PromptGenerator with configured filters."""
        generator = PromptGenerator()
        
        # Add all custom filters
        for name, func in self._custom_filters.items():
            generator.with_filter(name, func)
        
        # Set token counter if provided
        if self._token_counter:
            generator.with_token_counter(self._token_counter)
        
        return generator
    
    def from_template_string(
        self,
        template: str,
        variables: Optional[Dict[str, Any]] = None,
        data_key: Optional[str] = None,
        data: Optional[List[Any]] = None,
        max_tokens: Optional[int] = None,
        **extra_vars: Any
    ) -> List[str]:
        """
        Build prompts from a Jinja2 template string.
        
        Args:
            template: Jinja2 template string.
            variables: Dictionary of template variables.
            data_key: Variable name for batch data in template.
            data: List of data items for batching.
            max_tokens: Maximum tokens per prompt.
            **extra_vars: Additional template variables.
        
        Returns:
            List of rendered prompt strings.
        """
        generator = self._create_generator()
        generator.with_template_string(template)
        
        # Set max tokens
        effective_max_tokens = max_tokens or self._default_max_tokens
        generator.with_max_tokens(effective_max_tokens)
        
        # Add variables
        all_vars = variables or {}
        all_vars.update(extra_vars)
        generator.with_variables(**all_vars)
        
        # Handle batch data
        if data_key and data is not None:
            generator.with_data(data_key, data, overlap=self._overlap_items)
        
        # Generate prompts
        result = generator.generate()
        return [prompt.content for prompt in result.prompts]
    
    def from_template_file(
        self,
        template_name: str,
        variables: Optional[Dict[str, Any]] = None,
        data_key: Optional[str] = None,
        data: Optional[List[Any]] = None,
        max_tokens: Optional[int] = None,
        search_paths: Optional[List[Union[str, Path]]] = None,
        **extra_vars: Any
    ) -> List[str]:
        """
        Build prompts from a Jinja2 template file.
        
        Args:
            template_name: Template filename (with or without .j2 extension).
            variables: Dictionary of template variables.
            data_key: Variable name for batch data in template.
            data: List of data items for batching.
            max_tokens: Maximum tokens per prompt.
            search_paths: Additional paths to search for templates.
            **extra_vars: Additional template variables.
        
        Returns:
            List of rendered prompt strings.
        """
        # Ensure .j2 extension
        if not template_name.endswith('.j2'):
            template_name += '.j2'
        
        # Build search paths
        paths = [self._templates_dir]
        if search_paths:
            paths.extend([Path(p) for p in search_paths])
        
        # Find template file
        template_path = None
        for search_path in paths:
            candidate = Path(search_path) / template_name
            if candidate.exists():
                template_path = candidate
                break
        
        if template_path is None:
            raise FileNotFoundError(
                f"Template '{template_name}' not found in: {[str(p) for p in paths]}"
            )
        
        generator = self._create_generator()
        generator.with_template_file(template_path, search_paths=[str(p) for p in paths])
        
        # Set max tokens
        effective_max_tokens = max_tokens or self._default_max_tokens
        generator.with_max_tokens(effective_max_tokens)
        
        # Add variables
        all_vars = variables or {}
        all_vars.update(extra_vars)
        generator.with_variables(**all_vars)
        
        # Handle batch data
        if data_key and data is not None:
            generator.with_data(data_key, data, overlap=self._overlap_items)
        
        # Generate prompts
        result = generator.generate()
        return [prompt.content for prompt in result.prompts]
    
    def from_template_file_blocks(
        self,
        template_name: str,
        variables: Optional[Dict[str, Any]] = None,
        search_paths: Optional[List[Union[str, Path]]] = None,
        **extra_vars: Any
    ) -> Dict[str, str]:
        """
        Extract and render individual blocks from a Jinja2 template file.
        
        This method parses templates with {% block system %} and {% block user %}
        blocks and returns them as separate rendered strings.
        
        Args:
            template_name: Template filename (with or without .j2 extension).
            variables: Dictionary of template variables.
            search_paths: Additional paths to search for templates.
            **extra_vars: Additional template variables.
        
        Returns:
            Dictionary with 'system' and 'user' keys containing rendered prompts.
        
        Example:
            >>> prompts = builder.from_template_file_blocks("planner.j2", goal="...")
            >>> print(prompts["system"])  # System prompt
            >>> print(prompts["user"])    # User prompt
        """
        import re
        
        # Ensure .j2 extension
        if not template_name.endswith('.j2'):
            template_name += '.j2'
        
        # Build search paths
        paths = [self._templates_dir]
        if search_paths:
            paths.extend([Path(p) for p in search_paths])
        
        # Find template file
        template_path = None
        for search_path in paths:
            candidate = Path(search_path) / template_name
            if candidate.exists():
                template_path = candidate
                break
        
        if template_path is None:
            raise FileNotFoundError(
                f"Template '{template_name}' not found in: {[str(p) for p in paths]}"
            )
        
        # Read template content
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Prepare variables
        all_vars = variables or {}
        all_vars.update(extra_vars)
        
        # Extract and render blocks
        result = {}
        for block_name in ['system', 'user']:
            # Pattern to match block content
            pattern = rf'{{% block {block_name} %}}(.*?){{% endblock %}}'
            match = re.search(pattern, template_content, re.DOTALL)
            
            if match:
                block_content = match.group(1).strip()
                # Render the block content as a standalone template
                rendered = self.from_template_string(
                    template=block_content,
                    variables=all_vars
                )
                result[block_name] = rendered[0] if rendered else ""
            else:
                result[block_name] = ""
        
        return result

    def from_task_definition(
        self,
        task_def: Dict[str, Any],
        data: Any,
        max_tokens: Optional[int] = None,
        **extra_vars: Any
    ) -> List[str]:
        """
        Build prompts from a JSON task definition (legacy format).
        
        This method provides backward compatibility with existing JSON task files
        by converting the user prompt template to a Jinja2 template.
        
        Args:
            task_def: Task definition dictionary (from JSON file).
            data: The main data to process.
            max_tokens: Maximum tokens per prompt.
            **extra_vars: Additional named variables (categories, context, etc.).
        
        Returns:
            List of rendered user prompt strings.
        """
        prompts_config = task_def.get("prompts", {})
        user_prompt_template = prompts_config.get("user", "")
        
        # Convert legacy {placeholder} syntax to Jinja2 {{ placeholder }}
        # Also handle the special {data_input_json} placeholder
        jinja_template = self._convert_legacy_template(user_prompt_template)
        
        # Prepare variables
        variables = {"data": data, **extra_vars}
        
        # Determine if we should batch
        data_key = None
        batch_data = None
        if isinstance(data, list) and len(data) > 0:
            data_key = "data"
            batch_data = data
        
        return self.from_template_string(
            template=jinja_template,
            variables=variables,
            data_key=data_key,
            data=batch_data,
            max_tokens=max_tokens
        )
    
    def _convert_legacy_template(self, template: str) -> str:
        """
        Convert legacy placeholder syntax to Jinja2 syntax.
        
        Converts:
        - {data_input_json} -> {{ data | to_json }}
        - {variable} -> {{ variable }}
        
        Args:
            template: Legacy template string.
        
        Returns:
            Jinja2-compatible template string.
        """
        import re
        
        # First, handle the special data_input_json placeholder
        result = template.replace("{data_input_json}", "{{ data | to_json }}")
        
        # Then convert remaining {placeholder} to {{ placeholder }}
        # But don't convert if it's already Jinja2 syntax {{ }}
        def replace_placeholder(match):
            placeholder = match.group(1)
            return "{{ " + placeholder + " }}"
        
        # Match {word} but not {{ or }}
        pattern = r'\{(\w+)\}'
        result = re.sub(pattern, replace_placeholder, result)
        
        return result
    
    def render_single(self, template: str, **variables: Any) -> str:
        """
        Render a single prompt without batching.
        
        Convenience method for simple prompt generation.
        
        Args:
            template: Jinja2 template string.
            **variables: Template variables.
        
        Returns:
            Rendered prompt string.
        """
        result = self.from_template_string(template, variables=variables)
        return result[0] if result else ""
    
    def add_filter(self, name: str, func: Callable) -> 'TaskPromptBuilder':
        """
        Add a custom Jinja2 filter.
        
        Args:
            name: Filter name.
            func: Filter function.
        
        Returns:
            Self for chaining.
        """
        self._custom_filters[name] = func
        return self
    
    def set_overlap(self, items: int) -> 'TaskPromptBuilder':
        """
        Set overlap items between batches.
        
        Args:
            items: Number of items to overlap.
        
        Returns:
            Self for chaining.
        """
        self._overlap_items = items
        return self
    
    def set_max_tokens(self, max_tokens: int) -> 'TaskPromptBuilder':
        """
        Set default maximum tokens per prompt.
        
        Args:
            max_tokens: Maximum token count.
        
        Returns:
            Self for chaining.
        """
        self._default_max_tokens = max_tokens
        return self
    
    @property
    def templates_dir(self) -> Path:
        """Get the templates directory path."""
        return self._templates_dir
    
    def list_templates(self) -> List[str]:
        """List available template files."""
        if not self._templates_dir.exists():
            return []
        return [f.name for f in self._templates_dir.glob("*.j2")]


# ============================================================================
# Module-level convenience functions
# ============================================================================

# Default builder instance
_default_builder: Optional[TaskPromptBuilder] = None


def get_builder(**kwargs) -> TaskPromptBuilder:
    """
    Get or create the default TaskPromptBuilder instance.
    
    Args:
        **kwargs: Arguments passed to TaskPromptBuilder if creating new instance.
    
    Returns:
        TaskPromptBuilder instance.
    """
    global _default_builder
    if _default_builder is None or kwargs:
        _default_builder = TaskPromptBuilder(**kwargs)
    return _default_builder


def build_prompt(template: str, **variables: Any) -> str:
    """
    Build a single prompt from a template string.
    
    Args:
        template: Jinja2 template string.
        **variables: Template variables.
    
    Returns:
        Rendered prompt string.
    
    Example:
        >>> prompt = build_prompt(
        ...     "Analyze: {{ items | to_json }}",
        ...     items=["Apple", "Banana"]
        ... )
    """
    return get_builder().render_single(template, **variables)


def build_prompts(
    template: str,
    data: List[Any],
    data_key: str = "data",
    max_tokens: int = 8192,
    **variables: Any
) -> List[str]:
    """
    Build prompts with automatic batching for large datasets.
    
    Args:
        template: Jinja2 template string.
        data: List of data items to process.
        data_key: Variable name for data in template.
        max_tokens: Maximum tokens per prompt.
        **variables: Additional template variables.
    
    Returns:
        List of rendered prompts.
    
    Example:
        >>> prompts = build_prompts(
        ...     "Classify: {{ data | to_json }}",
        ...     data=large_list,
        ...     max_tokens=4096
        ... )
    """
    return get_builder().from_template_string(
        template=template,
        variables=variables,
        data_key=data_key,
        data=data,
        max_tokens=max_tokens
    )


def build_from_file(
    template_name: str,
    data_key: Optional[str] = None,
    data: Optional[List[Any]] = None,
    max_tokens: int = 8192,
    **variables: Any
) -> List[str]:
    """
    Build prompts from a template file.
    
    Args:
        template_name: Template filename.
        data_key: Variable name for batch data.
        data: List of data items for batching.
        max_tokens: Maximum tokens per prompt.
        **variables: Template variables.
    
    Returns:
        List of rendered prompts.
    """
    return get_builder().from_template_file(
        template_name=template_name,
        variables=variables,
        data_key=data_key,
        data=data,
        max_tokens=max_tokens
    )


def build_from_file_blocks(
    template_name: str,
    **variables: Any
) -> Dict[str, str]:
    """
    Build prompts from a template file with system/user blocks.
    
    This function extracts and renders {% block system %} and {% block user %}
    from a template file.
    
    Args:
        template_name: Template filename.
        **variables: Template variables.
    
    Returns:
        Dictionary with 'system' and 'user' keys containing rendered prompts.
    
    Example:
        >>> prompts = build_from_file_blocks("planner.j2", goal="Launch product")
        >>> print(prompts["system"])  # System prompt
        >>> print(prompts["user"])    # User prompt
    """
    return get_builder().from_template_file_blocks(
        template_name=template_name,
        variables=variables
    )


# ============================================================================
# Export all public symbols
# ============================================================================

__all__ = [
    # Main class
    "TaskPromptBuilder",
    # Constants
    "TEMPLATES_DIR",
    "DEFAULT_FILTERS",
    # Convenience functions
    "get_builder",
    "build_prompt",
    "build_prompts",
    "build_from_file",
    "build_from_file_blocks",
    # Default filters (for extension)
    "to_compact_json",
    "to_pretty_json",
    "truncate",
    "escape_quotes",
    "numbered_list",
    "bullet_list",
]
