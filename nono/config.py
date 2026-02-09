"""
Nono Configuration Module

Multi-source configuration with priority resolution.

Priority Order (highest to lowest):
    1. Command-line arguments (via load_args)
    2. Environment variables (via load_env with NONO_ prefix)
    3. Configuration files (JSON, YAML, TOML)
    4. Default values

Usage:
    from nono.config import Config, load_config
    
    # Quick setup
    config = load_config(filepath='config.toml')
    
    # Full control with method chaining
    config = Config(defaults={'timeout': 30})
    config.load_file('config.toml').load_env(prefix='NONO_')
    
    # Access values
    host = config.get('google.default_model')
    timeout = config.get('rate_limits.delay_between_requests', type=float)
    
    # Dictionary-style access
    value = config['ollama.host']
    config['custom.key'] = 'value'
    
    # Validation with schema
    schema = ConfigSchema()
    schema.add_field('google.default_model', type=str, required=True)
    schema.add_field('rate_limits.delay_between_requests', type=float, min_value=0)
    
    config = Config(schema=schema)
    config.load_file('config.toml')
    config.validate()  # Raises ValueError if invalid

Legacy API (backwards compatible):
    from nono.config import NonoConfig, get_templates_dir, get_prompts_dir
    
    NonoConfig.set_templates_dir("/path/to/templates")
    templates_path = get_templates_dir()
"""

from __future__ import annotations

import os
import json
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypeVar, Type
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    # New API
    'Config',
    'ConfigSchema',
    'ConfigValue',
    'ConfigSource',
    'ConfigFormat',
    'load_config',
    'create_sample_config',
    # Legacy API (backwards compatible)
    'NonoConfig',
    'get_templates_dir',
    'get_prompts_dir',
    'set_templates_dir',
    'set_prompts_dir',
]

T = TypeVar('T')

logger = logging.getLogger("Nono.Config")

# Try to import tomllib (Python 3.11+) or tomli as fallback
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore

# Config file names to search (priority order)
STANDARD_CONFIG_NAMES = [
    'config.toml',
    'config.json',
    'config.yaml',
    'config.yml',
]

# Module-level paths
_MODULE_DIR = Path(__file__).parent
_DEFAULT_TEMPLATES_DIR = _MODULE_DIR / "tasker" / "templates"
_DEFAULT_PROMPTS_DIR = _MODULE_DIR / "tasker" / "prompts"
_DEFAULT_CONFIG_FILE = _MODULE_DIR / "config.toml"


class ConfigSource(Enum):
    """Sources for configuration values."""
    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    ARGUMENT = "argument"


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"


@dataclass
class ConfigValue:
    """Configuration value with source metadata."""
    value: Any
    source: ConfigSource = ConfigSource.DEFAULT
    key: str = ""


@dataclass
class ConfigSchema:
    """
    Schema for configuration validation.
    
    Example:
        schema = ConfigSchema()
        schema.add_field('google.default_model', type=str, required=True)
        schema.add_field('rate_limits.delay_between_requests', type=float, min_value=0)
        schema.add_field('ollama.host', type=str, choices=['http://localhost:11434'])
    """
    
    @dataclass
    class Field:
        """Schema field definition."""
        name: str
        type: Type = str
        required: bool = False
        default: Any = None
        choices: List[Any] = field(default_factory=list)
        min_value: Optional[float] = None
        max_value: Optional[float] = None
        
        def validate(self, value: Any) -> tuple[bool, Optional[str]]:
            """Validate a value against this field's constraints."""
            if value is None:
                if self.required:
                    return False, f"Required field '{self.name}' is missing"
                return True, None
            if self.choices and value not in self.choices:
                return False, f"Field '{self.name}' must be one of: {self.choices}"
            if isinstance(value, (int, float)):
                if self.min_value is not None and value < self.min_value:
                    return False, f"Field '{self.name}' must be >= {self.min_value}"
                if self.max_value is not None and value > self.max_value:
                    return False, f"Field '{self.name}' must be <= {self.max_value}"
            return True, None
    
    fields: List[Field] = field(default_factory=list)
    
    def add_field(self, name: str, **kwargs) -> ConfigSchema:
        """Add a field to the schema. Returns self for chaining."""
        self.fields.append(ConfigSchema.Field(name=name, **kwargs))
        return self
    
    def validate(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate config dict against schema. Returns (is_valid, errors)."""
        errors = []
        for field_def in self.fields:
            value = config.get(field_def.name, field_def.default)
            is_valid, error = field_def.validate(value)
            if not is_valid and error:
                errors.append(error)
        return len(errors) == 0, errors


class Config:
    """
    Unified configuration manager with multi-source support.
    
    Example:
        config = Config(defaults={'timeout': 30})
        config.load_file('config.toml').load_env(prefix='NONO_')
        
        model = config.get('google.default_model')
        delay = config.get('rate_limits.delay_between_requests', type=float)
    """
    
    def __init__(
        self,
        defaults: Optional[Dict[str, Any]] = None,
        schema: Optional[ConfigSchema] = None,
        auto_discover: bool = False
    ):
        """
        Initialize configuration manager.
        
        Args:
            defaults: Default values dictionary
            schema: Optional schema for validation
            auto_discover: Whether to search for config files in cwd
        """
        self.schema = schema
        
        # Storage for values from different sources
        self._defaults: Dict[str, ConfigValue] = {}
        self._file_values: Dict[str, ConfigValue] = {}
        self._env_values: Dict[str, ConfigValue] = {}
        self._arg_values: Dict[str, ConfigValue] = {}
        
        # Set defaults
        if defaults:
            for key, value in defaults.items():
                self._defaults[key] = ConfigValue(
                    value=value, source=ConfigSource.DEFAULT, key=key
                )
        
        if auto_discover:
            self._auto_discover()
    
    def _auto_discover(self) -> None:
        """Search for config files in current directory."""
        for config_name in STANDARD_CONFIG_NAMES:
            config_file = Path.cwd() / config_name
            if config_file.exists():
                self.load_file(str(config_file))
                return
    
    def load_file(self, filepath: str, format: Optional[ConfigFormat] = None) -> Config:
        """
        Load configuration from file. Returns self for chaining.
        
        Args:
            filepath: Path to configuration file
            format: File format (auto-detected from extension if not specified)
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Config file not found: {filepath}")
            return self
        
        # Auto-detect format from extension
        if format is None:
            ext = path.suffix.lower()
            format = {
                '.json': ConfigFormat.JSON,
                '.yaml': ConfigFormat.YAML,
                '.yml': ConfigFormat.YAML,
                '.toml': ConfigFormat.TOML,
            }.get(ext, ConfigFormat.TOML)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if format == ConfigFormat.JSON:
                    data = json.load(f)
                elif format == ConfigFormat.YAML:
                    import yaml
                    data = yaml.safe_load(f)
                elif format == ConfigFormat.TOML:
                    if tomllib is None:
                        logger.warning("TOML library not available. Install tomli for Python < 3.11.")
                        return self
                    with open(path, 'rb') as fb:
                        data = tomllib.load(fb)
                else:
                    data = json.load(f)
            
            self._flatten_and_store(data, self._file_values, ConfigSource.FILE)
            logger.debug(f"Loaded config from: {path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")
        
        return self
    
    def _flatten_and_store(
        self,
        data: Dict[str, Any],
        storage: Dict[str, ConfigValue],
        source: ConfigSource,
        prefix: str = ""
    ) -> None:
        """Flatten nested dictionary and store as ConfigValues."""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._flatten_and_store(value, storage, source, full_key)
            else:
                storage[full_key] = ConfigValue(value=value, source=source, key=full_key)
    
    def load_env(self, prefix: str = "NONO_") -> Config:
        """
        Load configuration from environment variables. Returns self for chaining.
        
        Mapping: PREFIX_KEY -> key, PREFIX_NESTED__KEY -> nested.key
        
        Args:
            prefix: Environment variable prefix (e.g., 'NONO_')
        """
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].replace('__', '.').lower()
                parsed_value = self._parse_env_value(value)
                self._env_values[config_key] = ConfigValue(
                    value=parsed_value, source=ConfigSource.ENVIRONMENT, key=config_key
                )
        return self
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value, detecting type."""
        # Boolean
        if value.lower() in ('true', 'false', 'yes', 'no'):
            return value.lower() in ('true', 'yes')
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        # JSON array/object
        if value.startswith(('[', '{')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        return value
    
    def load_args(self, args: Union[Dict[str, Any], object]) -> Config:
        """
        Load configuration from parsed arguments. Returns self for chaining.
        
        Args:
            args: Dictionary or argparse.Namespace with argument values
        """
        if hasattr(args, '__dict__'):
            args = vars(args)
        for key, value in args.items():
            if value is not None:
                self._arg_values[key] = ConfigValue(
                    value=value, source=ConfigSource.ARGUMENT, key=key
                )
        return self
    
    def get(self, key: str, default: Any = None, type: Optional[Type[T]] = None) -> T:
        """
        Get configuration value using priority resolution.
        
        Priority: Arguments > Environment > File > Default
        
        Args:
            key: Configuration key (use dot notation for nested: 'google.default_model')
            default: Default value if not found
            type: Type to coerce value to (e.g., int, bool, float)
        
        Returns:
            Configuration value or default
        """
        sources = [self._arg_values, self._env_values, self._file_values, self._defaults]
        
        for source in sources:
            if key in source:
                value = source[key].value
                if type is not None:
                    try:
                        return type(value)
                    except (ValueError, TypeError):
                        pass
                return value
        return default
    
    def set(self, key: str, value: Any) -> Config:
        """Set a configuration value (highest priority). Returns self for chaining."""
        self._arg_values[key] = ConfigValue(value=value, source=ConfigSource.ARGUMENT, key=key)
        return self
    
    def require(self, key: str, message: Optional[str] = None) -> Any:
        """Get required value, raising ValueError if not found."""
        value = self.get(key)
        if value is None:
            raise ValueError(message or f"Required config '{key}' not found")
        return value
    
    def all(self) -> Dict[str, Any]:
        """Get all configuration values as dictionary."""
        result = {}
        for source in [self._defaults, self._file_values, self._env_values, self._arg_values]:
            for key, cv in source.items():
                result[key] = cv.value
        return result
    
    def validate(self, raise_on_error: bool = True) -> tuple[bool, List[str]]:
        """
        Validate configuration against schema.
        
        Args:
            raise_on_error: If True, raises ValueError on validation failure
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        if not self.schema:
            return True, []
        is_valid, errors = self.schema.validate(self.all())
        if not is_valid and raise_on_error:
            raise ValueError("Validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        return is_valid, errors
    
    def copy(self) -> Config:
        """Create a deep copy of this configuration."""
        new = Config(schema=self.schema, auto_discover=False)
        new._defaults = copy.deepcopy(self._defaults)
        new._file_values = copy.deepcopy(self._file_values)
        new._env_values = copy.deepcopy(self._env_values)
        new._arg_values = copy.deepcopy(self._arg_values)
        return new
    
    def get_source(self, key: str) -> Optional[ConfigSource]:
        """Get the source of a configuration value."""
        sources = [self._arg_values, self._env_values, self._file_values, self._defaults]
        for source in sources:
            if key in source:
                return source[key].source
        return None
    
    # Dictionary-style access
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None
    
    def __repr__(self) -> str:
        return f"Config(keys={len(self.all())})"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_config(
    filepath: Optional[str] = None,
    defaults: Optional[Dict[str, Any]] = None,
    env_prefix: Optional[str] = None
) -> Config:
    """
    Convenience function to create and load configuration.
    
    Args:
        filepath: Path to config file (optional, defaults to config.toml in module dir)
        defaults: Default values dictionary
        env_prefix: Environment variable prefix (e.g., 'NONO_')
    
    Example:
        config = load_config(filepath='config.toml', env_prefix='NONO_')
    """
    config = Config(defaults=defaults, auto_discover=False)
    
    # Load default config file if no filepath specified
    if filepath:
        config.load_file(filepath)
    elif _DEFAULT_CONFIG_FILE.exists():
        config.load_file(str(_DEFAULT_CONFIG_FILE))
    
    if env_prefix:
        config.load_env(prefix=env_prefix)
    
    return config


def create_sample_config(filepath: str) -> bool:
    """
    Create a sample configuration file.
    
    Args:
        filepath: Output file path (e.g., 'config.json')
    
    Returns:
        True if successful, False otherwise
    """
    sample = {
        'google': {
            'default_model': 'gemini-3-flash-preview'
        },
        'openai': {
            'default_model': 'gpt-4o-mini'
        },
        'perplexity': {
            'default_model': 'sonar'
        },
        'deepseek': {
            'default_model': 'deepseek-chat'
        },
        'grok': {
            'default_model': 'grok-1'
        },
        'ollama': {
            'host': 'http://localhost:11434',
            'default_model': 'llama3'
        },
        'rate_limits': {
            'delay_between_requests': 0.5
        },
        'paths': {
            'templates_dir': '',
            'prompts_dir': ''
        }
    }
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sample, f, indent=2)
        return True
    except Exception:
        return False


# ============================================================================
# LEGACY API (Backwards Compatible)
# ============================================================================

class NonoConfig:
    """
    Legacy configuration class for Nono (backwards compatible).
    
    Supports multiple configuration sources with the following priority:
    1. Environment variables (NONO_TEMPLATES_DIR, NONO_PROMPTS_DIR)
    2. Programmatic settings via set_* methods
    3. config.toml file settings
    4. Default paths (nono/tasker/templates, nono/tasker/prompts)
    
    Note: Consider using the new Config class for new code.
    
    Example usage:
        >>> from nono.config import NonoConfig
        >>> NonoConfig.set_templates_dir("/my/project/templates")
        >>> NonoConfig.set_prompts_dir("/my/project/prompts")
    """
    
    # Class-level configuration storage
    _templates_dir: Optional[Path] = None
    _prompts_dir: Optional[Path] = None
    _config_data: Dict[str, Any] = {}
    _config_loaded: bool = False
    
    @classmethod
    def set_templates_dir(cls, path: Union[str, Path]) -> None:
        """
        Set the templates directory programmatically.
        
        Args:
            path: Absolute or relative path to templates directory.
                  Relative paths are resolved from current working directory.
        """
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        cls._templates_dir = resolved
        logger.info(f"Templates directory set to: {resolved}")
    
    @classmethod
    def set_prompts_dir(cls, path: Union[str, Path]) -> None:
        """
        Set the prompts directory programmatically.
        
        Args:
            path: Absolute or relative path to prompts directory.
                  Relative paths are resolved from current working directory.
        """
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        cls._prompts_dir = resolved
        logger.info(f"Prompts directory set to: {resolved}")
    
    @classmethod
    def get_templates_dir(cls) -> Path:
        """
        Get the templates directory with priority resolution.
        
        Resolution order:
        1. NONO_TEMPLATES_DIR environment variable
        2. Programmatic setting via set_templates_dir()
        3. config.toml [paths] templates_dir
        4. Default: nono/tasker/templates
        
        Returns:
            Path to templates directory.
        """
        # 1. Environment variable (highest priority)
        env_path = os.environ.get("NONO_TEMPLATES_DIR")
        if env_path:
            resolved = Path(env_path)
            if not resolved.is_absolute():
                resolved = Path.cwd() / resolved
            return resolved
        
        # 2. Programmatic setting
        if cls._templates_dir is not None:
            return cls._templates_dir
        
        # 3. Config file
        cls._ensure_config_loaded()
        config_path = cls._config_data.get("paths", {}).get("templates_dir", "")
        if config_path:
            resolved = Path(config_path)
            if not resolved.is_absolute():
                resolved = _MODULE_DIR.parent / resolved
            return resolved
        
        # 4. Default
        return _DEFAULT_TEMPLATES_DIR
    
    @classmethod
    def get_prompts_dir(cls) -> Path:
        """
        Get the prompts directory with priority resolution.
        
        Resolution order:
        1. NONO_PROMPTS_DIR environment variable
        2. Programmatic setting via set_prompts_dir()
        3. config.toml [paths] prompts_dir
        4. Default: nono/tasker/prompts
        
        Returns:
            Path to prompts directory.
        """
        # 1. Environment variable (highest priority)
        env_path = os.environ.get("NONO_PROMPTS_DIR")
        if env_path:
            resolved = Path(env_path)
            if not resolved.is_absolute():
                resolved = Path.cwd() / resolved
            return resolved
        
        # 2. Programmatic setting
        if cls._prompts_dir is not None:
            return cls._prompts_dir
        
        # 3. Config file
        cls._ensure_config_loaded()
        config_path = cls._config_data.get("paths", {}).get("prompts_dir", "")
        if config_path:
            resolved = Path(config_path)
            if not resolved.is_absolute():
                resolved = _MODULE_DIR.parent / resolved
            return resolved
        
        # 4. Default
        return _DEFAULT_PROMPTS_DIR
    
    @classmethod
    def get_config_value(cls, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value from config.toml.
        
        Args:
            section: Section name (e.g., "google", "paths").
            key: Key name within the section.
            default: Default value if not found.
        
        Returns:
            Configuration value or default.
        """
        cls._ensure_config_loaded()
        return cls._config_data.get(section, {}).get(key, default)
    
    @classmethod
    def load_from_file(cls, config_path: Union[str, Path]) -> None:
        """
        Load configuration from a custom config file.
        
        Args:
            config_path: Path to TOML configuration file.
        """
        resolved = Path(config_path)
        if not resolved.exists():
            logger.warning(f"Config file not found: {resolved}")
            return
        
        if tomllib is None:
            logger.warning("TOML library not available. Install tomli for Python < 3.11.")
            return
        
        with open(resolved, "rb") as f:
            cls._config_data = tomllib.load(f)
        cls._config_loaded = True
        logger.info(f"Configuration loaded from: {resolved}")
    
    @classmethod
    def _ensure_config_loaded(cls) -> None:
        """Load default config file if not already loaded."""
        if cls._config_loaded:
            return
        
        # Check environment variable for custom config file
        env_config = os.environ.get("NONO_CONFIG_FILE")
        if env_config:
            config_path = Path(env_config)
        else:
            config_path = _DEFAULT_CONFIG_FILE
        
        if config_path.exists() and tomllib is not None:
            try:
                with open(config_path, "rb") as f:
                    cls._config_data = tomllib.load(f)
                logger.debug(f"Loaded config from: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
                cls._config_data = {}
        
        cls._config_loaded = True
    
    @classmethod
    def reset(cls) -> None:
        """Reset all configuration to defaults (useful for testing)."""
        cls._templates_dir = None
        cls._prompts_dir = None
        cls._config_data = {}
        cls._config_loaded = False
        logger.debug("Configuration reset to defaults")
    
    @classmethod
    def get_all_config(cls) -> Dict[str, Any]:
        """
        Get all current configuration values.
        
        Returns:
            Dictionary with all configuration including resolved paths.
        """
        cls._ensure_config_loaded()
        return {
            "templates_dir": str(cls.get_templates_dir()),
            "prompts_dir": str(cls.get_prompts_dir()),
            "config_file": cls._config_data,
            "env_templates_dir": os.environ.get("NONO_TEMPLATES_DIR"),
            "env_prompts_dir": os.environ.get("NONO_PROMPTS_DIR"),
            "env_config_file": os.environ.get("NONO_CONFIG_FILE"),
        }


# Convenience functions for simpler API (Legacy)
def get_templates_dir() -> Path:
    """Get the resolved templates directory path."""
    return NonoConfig.get_templates_dir()


def get_prompts_dir() -> Path:
    """Get the resolved prompts directory path."""
    return NonoConfig.get_prompts_dir()


def set_templates_dir(path: Union[str, Path]) -> None:
    """Set the templates directory programmatically."""
    NonoConfig.set_templates_dir(path)


def set_prompts_dir(path: Union[str, Path]) -> None:
    """Set the prompts directory programmatically."""
    NonoConfig.set_prompts_dir(path)
