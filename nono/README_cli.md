# Módulo CLI - Nono

## Descripción

Interfaz de línea de comandos para Nono GenAI Tasker con soporte multi-proveedor, salida con colores, barras de progreso y parsing unificado de argumentos.

## Instalación

El módulo está incluido en el proyecto. No requiere dependencias adicionales.

Para soporte de colores en Windows sin ANSI nativo:
```bash
pip install colorama
```

## Inicio Rápido

### Uso desde línea de comandos

```bash
# Mostrar ayuda
python -m nono.cli --help

# Ejecutar con prompt directo
python -m nono.cli --provider gemini --prompt "Explica qué es Python"

# Ejecutar con archivo de tarea
python -m nono.cli --provider openai --task summarize --input documento.txt

# Usar Ollama local
python -m nono.cli --provider ollama --model llama3 --prompt "Hola"

# Cargar parámetros desde archivo
python -m nono.cli @params.txt

# Modo simulación (dry-run)
python -m nono.cli --dry-run --provider gemini --prompt "Test"
```

### Uso como módulo Python

```python
from nono.cli import create_cli, print_success, print_error

# Crear CLI con factory function
cli = create_cli(
    prog="mi_herramienta",
    description="Mi herramienta GenAI",
    version="1.0.0",
    with_provider=True,
    with_task=True,
    with_io=True
)

# Agregar ejemplos de uso
cli.add_examples([
    "%(prog)s --provider gemini --prompt 'Hola' -o salida.txt",
    "%(prog)s @params.txt"
])

# Parsear argumentos
args = cli.parse_args()

# Tu lógica aquí...
cli.increment_stat('processed', 100)
print_success("Completado!")
cli.print_final_summary()
```

## Línea de Comandos

### Opciones Globales

| Opción | Descripción |
|--------|-------------|
| `--version`, `-V` | Mostrar versión |
| `--verbose`, `-v` | Aumentar verbosidad (-v=INFO, -vv=DEBUG) |
| `--quiet`, `-q` | Suprimir salida no esencial |
| `--no-color` | Desactivar colores |
| `--dry-run` | Simular sin hacer llamadas API |
| `--output-format`, `-F` | Formato de salida: table, json, csv, text, markdown, summary, quiet |
| `--config-file` | Cargar configuración desde archivo TOML/JSON |
| `--log-file` | Escribir logs a archivo |

### Configuración del Proveedor IA

| Opción | Descripción |
|--------|-------------|
| `--provider`, `-p` | Proveedor: gemini, openai, perplexity, deepseek, grok, ollama |
| `--model`, `-m` | Nombre del modelo |
| `--api-key` | API key (o usar variable de entorno) |
| `--api-key-file` | Leer API key desde archivo |
| `--temperature` | Temperatura de generación 0.0-2.0 (default: 0.7) |
| `--max-tokens` | Tokens máximos en respuesta (default: 4096) |
| `--timeout` | Timeout de request en segundos (default: 60) |
| `--ollama-host` | URL del servidor Ollama (default: http://localhost:11434) |

### Configuración de Tareas

| Opción | Descripción |
|--------|-------------|
| `--task`, `-t` | Nombre de tarea o ruta a archivo JSON |
| `--prompt` | Texto de prompt directo |
| `--system-prompt` | Prompt/instrucciones del sistema |
| `--template` | Archivo de template Jinja2 |
| `--variables`, `--vars` | JSON string o archivo con variables del template |

### Entrada/Salida

| Opción | Descripción |
|--------|-------------|
| `--input`, `-i` | Archivo o datos de entrada |
| `--input-format` | Formato de entrada: text, json, csv, file |
| `--output`, `-o` | Archivo de salida (stdout si no se especifica) |
| `--output-type` | Formato de salida: text, json, csv, markdown |
| `--append` | Añadir al archivo en lugar de sobrescribir |

### Procesamiento por Lotes

| Opción | Descripción |
|--------|-------------|
| `--batch` | Activar modo de procesamiento por lotes |
| `--batch-file` | Archivo con entradas (una por línea o array JSON) |
| `--batch-size` | Número de items por lote (default: 10) |
| `--delay` | Delay entre requests en segundos (default: 0.5) |
| `--retry` | Número de reintentos en fallo (default: 3) |
| `--continue-on-error` | Continuar procesamiento en fallos individuales |

## Archivo de Parámetros (@params.txt)

Puedes cargar argumentos desde un archivo usando `@filename`:

```
--provider
gemini
--model
gemini-3-flash-preview
--prompt
Genera un resumen del siguiente texto
--input
documento.txt
--output
resumen.txt
--verbose
```

## Utilidades de Salida

### Mensajes con Color

```python
from nono.cli import print_success, print_error, print_warning, print_info, cprint, Colors

print_success("Operación completada")  # ✓ Verde
print_error("Algo falló")              # ✗ Rojo
print_warning("Cuidado")               # ⚠ Amarillo
print_info("Información")              # ℹ Cyan

# Personalizado
cprint("Texto en negrita cyan", Colors.CYAN, bold=True)
cprint("Texto tenue", Colors.GRAY, dim=True)
```

### Tablas

```python
from nono.cli import print_table

headers = ["Proveedor", "Modelo", "Estado"]
rows = [
    ["Gemini", "gemini-3-flash", "Activo"],
    ["OpenAI", "gpt-4o-mini", "Activo"],
    ["Ollama", "llama3", "Offline"],
]
print_table(headers, rows)

# Con columna de índice
print_table(headers, rows, show_index=True)
```

### Barra de Progreso

```python
from nono.cli import print_progress

for i in range(101):
    print_progress(i, 100, prefix="Procesando", suffix="Completo")
```

### Spinner

```python
from nono.cli import print_spinner
import time

update = print_spinner("Cargando datos...")
for _ in range(50):
    update()
    time.sleep(0.1)
```

### Resumen de Estadísticas

```python
from nono.cli import print_summary

stats = {
    'total_procesados': 100,
    'exitosos': 95,
    'errores': 5,
    'tokens_usados': 15000,
}
print_summary(stats, title="RESULTADOS")
```

### Confirmación Interactiva

```python
from nono.cli import confirm_action

if confirm_action("¿Deseas continuar?", default=False):
    print("Continuando...")
else:
    print("Cancelado")
```

## Referencia de API

### Clase CLIBase

```python
from nono.cli import CLIBase

cli = CLIBase(
    prog="mi_cli",           # Nombre del programa
    description="Mi CLI",    # Descripción para ayuda
    version="1.0.0"          # Versión
)
```

| Método | Descripción |
|--------|-------------|
| `add_ai_provider_group()` | Añadir args de proveedor IA |
| `add_task_group()` | Añadir args de configuración de tareas |
| `add_io_group(formats)` | Añadir args de entrada/salida |
| `add_batch_group()` | Añadir args de procesamiento por lotes |
| `add_group(name, title)` | Añadir grupo de argumentos personalizado |
| `add_example(example)` | Añadir ejemplo de uso |
| `add_examples(list)` | Añadir múltiples ejemplos |
| `parse_args()` | Parsear línea de comandos |
| `increment_stat(name, value)` | Incrementar estadística |
| `set_stat(name, value)` | Establecer estadística |
| `get_elapsed_time()` | Obtener tiempo transcurrido formateado |
| `print_final_summary(title)` | Imprimir resumen final |
| `exit_success(message)` | Salir con código 0 |
| `exit_with_error(message)` | Salir con error |

### Factory Function

```python
from nono.cli import create_cli

cli = create_cli(
    prog="herramienta",
    description="Mi herramienta",
    version="1.0.0",
    with_provider=True,   # Añadir grupo de proveedor IA
    with_task=True,       # Añadir grupo de tareas
    with_io=True,         # Añadir grupo de E/S
    with_batch=False      # Sin procesamiento por lotes
)
```

### Enums

```python
from nono.cli import OutputFormat, AIProvider, LogLevel

# Formatos de salida
OutputFormat.TABLE
OutputFormat.JSON
OutputFormat.CSV
OutputFormat.TEXT
OutputFormat.MARKDOWN
OutputFormat.SUMMARY
OutputFormat.QUIET

# Proveedores de IA
AIProvider.GEMINI
AIProvider.OPENAI
AIProvider.PERPLEXITY
AIProvider.DEEPSEEK
AIProvider.GROK
AIProvider.OLLAMA

# Niveles de log
LogLevel.DEBUG
LogLevel.INFO
LogLevel.WARNING
LogLevel.ERROR
LogLevel.QUIET
```

### Clase Colors

```python
from nono.cli import Colors

# Colores ANSI
Colors.RED
Colors.GREEN
Colors.YELLOW
Colors.BLUE
Colors.CYAN
Colors.MAGENTA
Colors.WHITE
Colors.GRAY

# Colores semánticos
Colors.SUCCESS  # Verde
Colors.ERROR    # Rojo
Colors.WARNING  # Amarillo
Colors.INFO     # Cyan

# Control
Colors.BOLD
Colors.DIM
Colors.RESET

# Métodos
Colors.disable()  # Desactivar colores
Colors.enable()   # Reactivar colores
Colors.init()     # Inicializar (automático)
```

## Ejemplos Completos

### CLI Simple para Traducción

```python
from nono.cli import create_cli, print_success, print_info

def main():
    cli = create_cli(
        prog="traductor",
        description="Traduce texto usando IA",
        version="1.0.0"
    )
    
    # Añadir argumento personalizado
    lang_group = cli.add_group("language", "Idiomas")
    lang_group.add_argument('--source-lang', default='auto', help="Idioma origen")
    lang_group.add_argument('--target-lang', required=True, help="Idioma destino")
    
    cli.add_examples([
        "%(prog)s --provider gemini --prompt 'Hola mundo' --target-lang en",
    ])
    
    args = cli.parse_args()
    
    print_info(f"Traduciendo de {args.source_lang} a {args.target_lang}")
    
    # Lógica de traducción aquí...
    
    cli.increment_stat('translated', 1)
    print_success("Traducción completada!")
    cli.print_final_summary()

if __name__ == "__main__":
    main()
```

### CLI con Procesamiento por Lotes

```python
from nono.cli import create_cli, print_progress, print_summary
import time

def main():
    cli = create_cli(
        prog="batch_processor",
        description="Procesa múltiples archivos con IA",
        with_batch=True
    )
    
    args = cli.parse_args()
    
    if args.batch and args.batch_file:
        with open(args.batch_file) as f:
            items = f.read().splitlines()
        
        total = len(items)
        for i, item in enumerate(items):
            print_progress(i + 1, total, prefix="Procesando")
            
            # Procesar item...
            time.sleep(args.delay)
            
            cli.increment_stat('processed', 1)
        
        cli.print_final_summary()

if __name__ == "__main__":
    main()
```

## Personalización

### Desactivar Colores

```python
from nono.cli import Colors

# Desactivar globalmente
Colors.disable()

# O usar argumento --no-color en línea de comandos
```

### Cambiar Colores por Defecto

```python
from nono.cli import Colors

Colors.SUCCESS = Colors.BLUE  # Cambiar éxito a azul
Colors.ERROR = Colors.MAGENTA  # Cambiar error a magenta
```

### Grupo de Argumentos Personalizado

```python
from nono.cli import CLIBase

cli = CLIBase(prog="custom", version="1.0.0")

# Añadir grupo personalizado
db_group = cli.add_group("database", "Conexión a Base de Datos")
db_group.add_argument('--db-host', required=True, help="Host de BD")
db_group.add_argument('--db-port', type=int, default=5432, help="Puerto")
db_group.add_argument('--db-name', required=True, help="Nombre de BD")
db_group.add_argument('--db-user', help="Usuario")
db_group.add_argument('--db-password', help="Contraseña")
```

## Variables de Entorno

El CLI soporta API keys mediante variables de entorno:

| Proveedor | Variable |
|-----------|----------|
| Gemini | `GOOGLE_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Perplexity | `PERPLEXITY_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Grok | `GROK_API_KEY` |

También puedes usar `--api-key` o `--api-key-file` para especificar la clave.

## Integración con Config

El CLI se integra con el [módulo de configuración](README_config.md) para gestión centralizada de settings.

### Carga Automática de Configuración

```python
from nono.config import load_config
from nono.cli import CLIBase

# 1. Cargar configuración base desde archivo
config = load_config(filepath='config.toml', env_prefix='NONO_')

# 2. Crear CLI
cli = CLIBase(prog="mi_app", version="1.0.0")
cli.add_ai_provider_group()
args = cli.parse_args()

# 3. Fusionar argumentos CLI en config (máxima prioridad)
config.load_args(vars(args))

# 4. Usar valores finales (CLI > Env > File > Defaults)
provider = config.get('provider', 'gemini')
model = config.get('model') or config.get('google.default_model')
```

### Valores por Defecto desde Config

Puedes usar la configuración para establecer defaults del CLI:

```python
from nono.config import load_config
from nono.cli import CLIConfig, CLIBase

# Leer defaults desde config.toml
config = load_config()

# Crear CLI con defaults de config
cli_config = CLIConfig(
    prog_name="nono",
    default_timeout=config.get('rate_limits.timeout', 60),
)

cli = CLIBase(config=cli_config)
```

### Ejemplo Completo

```python
#!/usr/bin/env python
from nono.config import load_config, ConfigSchema
from nono.cli import CLIBase, print_success, print_error

def main():
    # Configuración con validación
    schema = ConfigSchema()
    schema.add_field('google.default_model', required=True)
    
    config = load_config(filepath='config.toml', env_prefix='NONO_')
    
    # CLI
    cli = CLIBase(prog="app", version="1.0.0")
    cli.add_ai_provider_group()
    args = cli.parse_args()
    
    # Fusionar (CLI tiene prioridad)
    config.load_args(vars(args))
    
    # Validar configuración final
    try:
        config.validate()
    except ValueError as e:
        cli.exit_with_error(str(e))
    
    # Usar configuración
    model = config['google.default_model']
    print_success(f"Usando modelo: {model}")

if __name__ == "__main__":
    main()
```

### Flujo de Prioridad

```
┌──────────────────────────────────────────────────────────────┐
│                    PRIORIDAD DE VALORES                      │
├──────────────────────────────────────────────────────────────┤
│  CLI Arguments (--provider, --model)      <- Máxima         │
│  Environment Variables (NONO_*)                              │
│  Config File (config.toml)                                   │
│  Default Values                           <- Mínima          │
└──────────────────────────────────────────────────────────────┘
```

📖 Ver también: [Configuration Documentation](README_config.md)

## Autor

**DatamanEdge** - MIT License
