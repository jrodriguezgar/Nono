# Módulo de Configuración - Nono

## Descripción

Gestión unificada de configuración con resolución multi-fuente y diseño basado en instancias para flexibilidad y testabilidad.

## Instalación

El módulo está incluido en el proyecto. No requiere dependencias adicionales para JSON/TOML (Python 3.11+).

Para Python < 3.11:
```bash
pip install tomli
```

Para soporte YAML:
```bash
pip install pyyaml
```

## Inicio Rápido

```python
from nono.config import Config, load_config

# Uso simple
config = load_config(filepath='config.toml', env_prefix='NONO_')
print(config['google.default_model'])

# Con method chaining
config = (
    Config(defaults={'timeout': 30})
    .load_file('config.toml')
    .load_env(prefix='NONO_')
)
```

## Resolución de Prioridad

Los valores se resuelven en orden (el primero encontrado gana):

| Prioridad | Fuente | Método |
|-----------|--------|--------|
| 1 (Alta) | Argumentos | `load_args()` / `set()` |
| 2 | Variables de entorno | `load_env(prefix='NONO_')` |
| 3 | Archivo de configuración | `load_file('config.toml')` |
| 4 (Baja) | Valores por defecto | `Config(defaults={...})` |

## API Principal

### Clase `Config`

```python
from nono.config import Config

# Crear instancia
config = Config(
    defaults={'app.timeout': 30},      # Valores por defecto
    schema=None,                        # Schema para validación (opcional)
    auto_discover=False                 # Buscar config.toml automáticamente
)

# Cargar desde archivo
config.load_file('config.toml')         # TOML
config.load_file('config.json')         # JSON
config.load_file('config.yaml')         # YAML (requiere pyyaml)

# Cargar desde variables de entorno
config.load_env(prefix='NONO_')

# Cargar desde argumentos (argparse.Namespace o dict)
config.load_args({'debug': True, 'port': 8080})

# Obtener valores
model = config.get('google.default_model')
port = config.get('server.port', default=8080, type=int)
api_key = config.require('api.key')  # Lanza ValueError si no existe

# Establecer valores (prioridad máxima)
config.set('runtime.mode', 'production')

# Acceso tipo diccionario
value = config['key']
config['key'] = 'value'
exists = 'key' in config

# Obtener todos los valores
all_config = config.all()

# Obtener fuente de un valor
source = config.get_source('google.default_model')  # ConfigSource.FILE

# Copiar configuración (útil para tests)
isolated = config.copy()
```

### Method Chaining

Todos los métodos de carga retornan `self` para encadenamiento:

```python
config = (
    Config(defaults={'app.debug': False})
    .load_file('config.toml')
    .load_env(prefix='NONO_')
    .set('runtime.mode', 'production')
)
```

### Función `load_config()`

Atajo para crear y cargar configuración:

```python
from nono.config import load_config

config = load_config(
    filepath='config.toml',     # Ruta al archivo (opcional)
    defaults={'timeout': 30},   # Valores por defecto
    env_prefix='NONO_'          # Prefijo para variables de entorno
)
```

## Formatos de Archivo

### TOML (config.toml)

```toml
[google]
default_model = "gemini-3-flash-preview"

[rate_limits]
delay_between_requests = 0.5

[paths]
templates_dir = ""
prompts_dir = ""
```

### JSON (config.json)

```json
{
    "google": {
        "default_model": "gemini-3-flash-preview"
    },
    "rate_limits": {
        "delay_between_requests": 0.5
    }
}
```

### YAML (config.yaml)

```yaml
google:
  default_model: gemini-3-flash-preview

rate_limits:
  delay_between_requests: 0.5
```

## Variables de Entorno

Prefijo: `NONO_`

### Mapeo de claves

| Variable de Entorno | Clave de Configuración |
|---------------------|------------------------|
| `NONO_TIMEOUT` | `timeout` |
| `NONO_GOOGLE__DEFAULT_MODEL` | `google.default_model` |
| `NONO_RATE_LIMITS__DELAY` | `rate_limits.delay` |

> **Nota**: Usa doble guion bajo (`__`) para claves anidadas.

### Conversión de tipos automática

```bash
NONO_DEBUG=true          # → True (bool)
NONO_PORT=8080           # → 8080 (int)
NONO_TIMEOUT=30.5        # → 30.5 (float)
NONO_HOSTS=["a","b"]     # → ["a", "b"] (list)
```

## Validación con Schema

```python
from nono.config import Config, ConfigSchema

# Definir schema
schema = ConfigSchema()
schema.add_field('google.default_model', type=str, required=True)
schema.add_field('rate_limits.delay_between_requests', 
                 type=float, min_value=0, max_value=10)
schema.add_field('ollama.host', type=str, required=True)
schema.add_field('app.mode', type=str, choices=['dev', 'prod'])

# Crear config con schema
config = Config(schema=schema)
config.load_file('config.toml')

# Validar (lanza ValueError si falla)
config.validate()

# O sin excepción
is_valid, errors = config.validate(raise_on_error=False)
if not is_valid:
    for error in errors:
        print(f"Error: {error}")
```

### Opciones de validación

| Parámetro | Descripción |
|-----------|-------------|
| `type` | Tipo esperado (`str`, `int`, `float`, `bool`) |
| `required` | `True` si el campo es obligatorio |
| `default` | Valor por defecto para validación |
| `choices` | Lista de valores permitidos |
| `min_value` | Valor mínimo (numéricos) |
| `max_value` | Valor máximo (numéricos) |

## API Legacy (Compatible hacia atrás)

Para compatibilidad con código existente:

```python
from nono.config import (
    NonoConfig,
    get_templates_dir,
    get_prompts_dir,
    set_templates_dir,
    set_prompts_dir,
)

# Obtener directorios
templates = get_templates_dir()
prompts = get_prompts_dir()

# Configurar directorios
set_templates_dir('/path/to/templates')
set_prompts_dir('/path/to/prompts')

# Obtener valores del archivo TOML
model = NonoConfig.get_config_value('google', 'default_model')

# Cargar archivo personalizado
NonoConfig.load_from_file('/path/to/config.toml')

# Reset (útil para tests)
NonoConfig.reset()
```

### Variables de entorno Legacy

| Variable | Descripción |
|----------|-------------|
| `NONO_TEMPLATES_DIR` | Ruta al directorio de templates |
| `NONO_PROMPTS_DIR` | Ruta al directorio de prompts |
| `NONO_CONFIG_FILE` | Ruta al archivo config.toml |

## Ejemplos de Uso

### Configuración básica

```python
from nono.config import load_config

config = load_config()
model = config['google.default_model']
delay = config.get('rate_limits.delay_between_requests', type=float)
```

### Con valores requeridos

```python
from nono.config import Config

config = Config().load_file('config.toml')

# Lanza ValueError si no existe
api_key = config.require('api.key', message='API key es obligatoria')
```

### Aislamiento para tests

```python
from nono.config import Config

def test_feature():
    # Crear configuración aislada
    config = Config(defaults={'test.mode': True})
    
    # O copiar una existente
    original = load_config()
    isolated = original.copy()
    isolated.set('test.override', 'value')
    
    # El original no se ve afectado
    assert 'test.override' not in original
```

### Tracking de fuentes

```python
from nono.config import Config, ConfigSource

config = Config(defaults={'app.timeout': 30})
config.load_file('config.toml')
config.load_env(prefix='NONO_')

# Saber de dónde viene cada valor
source = config.get_source('google.default_model')
if source == ConfigSource.ENVIRONMENT:
    print("Valor sobrescrito por variable de entorno")
elif source == ConfigSource.FILE:
    print("Valor del archivo de configuración")
```

## Referencia de API

| Método/Función | Descripción |
|----------------|-------------|
| `Config()` | Crear nueva instancia |
| `load_file(path)` | Cargar desde archivo |
| `load_env(prefix)` | Cargar desde variables de entorno |
| `load_args(args)` | Cargar desde argumentos |
| `get(key, default, type)` | Obtener valor con fallback |
| `set(key, value)` | Establecer valor (prioridad máxima) |
| `require(key, message)` | Obtener valor requerido |
| `all()` | Obtener todos los valores resueltos |
| `validate()` | Validar contra schema |
| `copy()` | Crear copia profunda |
| `get_source(key)` | Obtener fuente del valor |
| `load_config()` | Función helper para carga rápida |
| `create_sample_config()` | Crear archivo de ejemplo |

## Enums

```python
from nono.config import ConfigSource, ConfigFormat

# Fuentes de configuración
ConfigSource.DEFAULT      # Valor por defecto
ConfigSource.FILE         # Archivo de configuración
ConfigSource.ENVIRONMENT  # Variable de entorno
ConfigSource.ARGUMENT     # Argumento programático

# Formatos de archivo
ConfigFormat.JSON
ConfigFormat.YAML
ConfigFormat.TOML
```

## Integración con CLI

El módulo de configuración se integra con el [módulo CLI](README_cli.md) para proporcionar una experiencia unificada.

### Carga de Configuración en CLI

El CLI puede cargar configuración desde:

1. **Archivo de configuración** vía `--config-file`
2. **Variables de entorno** con prefijo `NONO_`
3. **Argumentos de línea de comandos** (máxima prioridad)

```bash
# CLI con configuración externa
python -m nono.cli --config-file config.toml --provider gemini --prompt "Hola"
```

### Uso Programático Conjunto

```python
from nono.config import load_config
from nono.cli import CLIBase, print_info

# Cargar configuración
config = load_config(filepath='config.toml', env_prefix='NONO_')

# Crear CLI con valores de config
cli = CLIBase(
    prog="mi_herramienta",
    version=config.get('app.version', '1.0.0')
)

# Los argumentos del CLI tienen prioridad sobre config
args = cli.parse_args()

# Fusionar: args sobreescriben config
config.load_args(vars(args))

# Ahora config tiene la prioridad correcta:
# args > env > file > defaults
model = config['google.default_model']
print_info(f"Usando modelo: {model}")
```

### Diagrama de Flujo

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Defaults   │ -> │  TOML File  │ -> │  Env Vars   │
└─────────────┘    └─────────────┘    └─────────────┘
                                            │
                                            v
                                   ┌─────────────┐
                                   │  CLI Args   │  <- Máxima prioridad
                                   └─────────────┘
                                            │
                                            v
                                   ┌─────────────┐
                                   │   Config    │  <- Valor final
                                   └─────────────┘
```

📖 Ver también: [CLI Documentation](README_cli.md)

## Autor

**DatamanEdge** - MIT License
