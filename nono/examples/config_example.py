"""
Example usage of the Nono configuration module.

Demonstrates:
    - New Config class with method chaining
    - Dictionary-style access
    - Schema validation
    - Legacy API compatibility

Run: python -m nono.examples.config_example
"""

from pathlib import Path
from nono.config import (
    Config,
    ConfigSchema,
    load_config,
    create_sample_config,
    # Legacy API
    NonoConfig,
    get_templates_dir,
    get_prompts_dir,
)


def example_basic_usage():
    """Basic usage with method chaining."""
    print("=" * 60)
    print("BASIC USAGE")
    print("=" * 60)
    
    # Create config with defaults and load from file
    config = Config(defaults={'custom.timeout': 30})
    config.load_file(str(Path(__file__).parent.parent / 'config.toml'))
    config.load_env(prefix='NONO_')
    
    # Access values with dot notation
    model = config.get('google.default_model')
    delay = config.get('rate_limits.delay_between_requests', type=float)
    timeout = config.get('custom.timeout', type=int)
    
    print(f"Google Model: {model}")
    print(f"Request Delay: {delay}")
    print(f"Custom Timeout: {timeout}")
    print()


def example_method_chaining():
    """Fluent API with method chaining."""
    print("=" * 60)
    print("METHOD CHAINING")
    print("=" * 60)
    
    config = (
        Config(defaults={'app.debug': False})
        .load_file(str(Path(__file__).parent.parent / 'config.toml'))
        .load_env(prefix='NONO_')
        .set('runtime.mode', 'production')
    )
    
    print(f"All config keys: {list(config.all().keys())}")
    print(f"Runtime mode: {config['runtime.mode']}")
    print()


def example_dictionary_access():
    """Dictionary-style access to config values."""
    print("=" * 60)
    print("DICTIONARY ACCESS")
    print("=" * 60)
    
    config = load_config()
    
    # Get with bracket notation
    host = config['ollama.host']
    print(f"Ollama host: {host}")
    
    # Set with bracket notation
    config['custom.key'] = 'custom_value'
    print(f"Custom key: {config['custom.key']}")
    
    # Check if key exists
    print(f"Has 'google.default_model': {'google.default_model' in config}")
    print(f"Has 'nonexistent': {'nonexistent' in config}")
    print()


def example_schema_validation():
    """Configuration validation with schema."""
    print("=" * 60)
    print("SCHEMA VALIDATION")
    print("=" * 60)
    
    # Define schema
    schema = ConfigSchema()
    schema.add_field('google.default_model', type=str, required=True)
    schema.add_field('rate_limits.delay_between_requests', type=float, min_value=0, max_value=10)
    schema.add_field('ollama.host', type=str, required=True)
    
    # Create config with schema
    config = Config(schema=schema)
    config.load_file(str(Path(__file__).parent.parent / 'config.toml'))
    
    # Validate
    is_valid, errors = config.validate(raise_on_error=False)
    print(f"Configuration valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    else:
        print("No validation errors!")
    print()


def example_quick_load():
    """Quick configuration loading with helper function."""
    print("=" * 60)
    print("QUICK LOAD")
    print("=" * 60)
    
    config = load_config(
        filepath=str(Path(__file__).parent.parent / 'config.toml'),
        env_prefix='NONO_',
        defaults={'app.name': 'Nono GenAI'}
    )
    
    print(f"App name: {config.get('app.name')}")
    print(f"Config object: {config}")
    print()


def example_legacy_api():
    """Legacy API for backwards compatibility."""
    print("=" * 60)
    print("LEGACY API (Backwards Compatible)")
    print("=" * 60)
    
    # Get paths using legacy functions
    templates = get_templates_dir()
    prompts = get_prompts_dir()
    
    print(f"Templates dir: {templates}")
    print(f"Prompts dir: {prompts}")
    
    # Get config value using legacy class
    model = NonoConfig.get_config_value('google', 'default_model')
    print(f"Google model (legacy): {model}")
    
    # Get all config
    all_config = NonoConfig.get_all_config()
    print(f"Config file sections: {list(all_config['config_file'].keys())}")
    print()


def example_copy_and_isolation():
    """Configuration copy for isolation (useful for testing)."""
    print("=" * 60)
    print("COPY AND ISOLATION")
    print("=" * 60)
    
    # Original config
    original = load_config()
    original.set('test.value', 'original')
    
    # Create isolated copy
    copied = original.copy()
    copied.set('test.value', 'modified')
    
    print(f"Original value: {original['test.value']}")
    print(f"Copied value: {copied['test.value']}")
    print()


def example_get_source():
    """Track where configuration values come from."""
    print("=" * 60)
    print("VALUE SOURCE TRACKING")
    print("=" * 60)
    
    config = Config(defaults={'app.timeout': 30})
    config.load_file(str(Path(__file__).parent.parent / 'config.toml'))
    config.set('runtime.override', 'from_args')
    
    print(f"Source of 'google.default_model': {config.get_source('google.default_model')}")
    print(f"Source of 'app.timeout': {config.get_source('app.timeout')}")
    print(f"Source of 'runtime.override': {config.get_source('runtime.override')}")
    print()


if __name__ == '__main__':
    example_basic_usage()
    example_method_chaining()
    example_dictionary_access()
    example_schema_validation()
    example_quick_load()
    example_legacy_api()
    example_copy_and_isolation()
    example_get_source()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
