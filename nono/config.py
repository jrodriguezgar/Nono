"""
Nono Configuration Module

Provides centralized configuration management with multiple resolution strategies:
1. Environment variables (highest priority)
2. Programmatic configuration via NonoConfig class
3. config.toml file (lowest priority)

Environment Variables:
    NONO_TEMPLATES_DIR: Path to templates directory
    NONO_PROMPTS_DIR: Path to prompts/task definitions directory
    NONO_CONFIG_FILE: Path to config.toml file

Usage from external projects:
    from nono.config import NonoConfig, get_templates_dir, get_prompts_dir
    
    # Option 1: Use environment variables
    os.environ["NONO_TEMPLATES_DIR"] = "/path/to/my/templates"
    
    # Option 2: Configure programmatically
    NonoConfig.set_templates_dir("/path/to/my/templates")
    NonoConfig.set_prompts_dir("/path/to/my/prompts")
    
    # Option 3: Use custom config file
    NonoConfig.load_from_file("/path/to/my/config.toml")
    
    # Get resolved paths
    templates_path = get_templates_dir()
    prompts_path = get_prompts_dir()
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("Nono.Config")

# Try to import tomllib (Python 3.11+) or tomli as fallback
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore

# Module-level paths
_MODULE_DIR = Path(__file__).parent
_DEFAULT_TEMPLATES_DIR = _MODULE_DIR / "tasker" / "templates"
_DEFAULT_PROMPTS_DIR = _MODULE_DIR / "tasker" / "prompts"
_DEFAULT_CONFIG_FILE = _MODULE_DIR / "config.toml"


class NonoConfig:
    """
    Central configuration class for Nono.
    
    Supports multiple configuration sources with the following priority:
    1. Environment variables (NONO_TEMPLATES_DIR, NONO_PROMPTS_DIR)
    2. Programmatic settings via set_* methods
    3. config.toml file settings
    4. Default paths (nono/tasker/templates, nono/tasker/prompts)
    
    Example usage from external projects:
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
        path = Path(path)
        if not path.is_absolute():
            path = Path.cwd() / path
        cls._templates_dir = path
        logger.info(f"Templates directory set to: {path}")
    
    @classmethod
    def set_prompts_dir(cls, path: Union[str, Path]) -> None:
        """
        Set the prompts directory programmatically.
        
        Args:
            path: Absolute or relative path to prompts directory.
                  Relative paths are resolved from current working directory.
        """
        path = Path(path)
        if not path.is_absolute():
            path = Path.cwd() / path
        cls._prompts_dir = path
        logger.info(f"Prompts directory set to: {path}")
    
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
            path = Path(env_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            return path
        
        # 2. Programmatic setting
        if cls._templates_dir is not None:
            return cls._templates_dir
        
        # 3. Config file
        cls._ensure_config_loaded()
        config_path = cls._config_data.get("paths", {}).get("templates_dir", "")
        if config_path:
            path = Path(config_path)
            if not path.is_absolute():
                path = _MODULE_DIR.parent / path
            return path
        
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
            path = Path(env_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            return path
        
        # 2. Programmatic setting
        if cls._prompts_dir is not None:
            return cls._prompts_dir
        
        # 3. Config file
        cls._ensure_config_loaded()
        config_path = cls._config_data.get("paths", {}).get("prompts_dir", "")
        if config_path:
            path = Path(config_path)
            if not path.is_absolute():
                path = _MODULE_DIR.parent / path
            return path
        
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
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return
        
        if tomllib is None:
            logger.warning("TOML library not available. Install tomli for Python < 3.11.")
            return
        
        with open(config_path, "rb") as f:
            cls._config_data = tomllib.load(f)
        cls._config_loaded = True
        logger.info(f"Configuration loaded from: {config_path}")
    
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


# Convenience functions for simpler API
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
