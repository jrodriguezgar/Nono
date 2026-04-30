# Nono Integrations — OfficeBridge & ShortFx

> Guía de integración de las librerías del ecosistema DatamanEdge con Nono.

---

## Resumen de Integraciones

| Librería | Tipo | Funciones | Módulo Nono |
|----------|------|-----------|-------------|
| **OfficeBridge** | Automatización documental | 8 tools directas + 3 discovery | `nono.agent.tools.officebridge_tools` |
| **ShortFx** | Funciones deterministas | 8 tools directas + 4 discovery + MCP | `nono.agent.tools.shortfx_tools` |

---

## OfficeBridge

### ¿Qué es?

OfficeBridge es una librería de automatización documental que proporciona un interfaz unificado para Word, PDF, HTML, Markdown, Text y Excel. Incluye motores de traducción AI, censura de PII y conversión N-to-N entre formatos.

### Modelos de Integración

```python
# Modelo 1 — Tools directas (operaciones comunes, máximo rendimiento)
from nono.agent.tools.officebridge_tools import OFFICEBRIDGE_TOOLS
agent = Agent(name="doc_agent", tools=OFFICEBRIDGE_TOOLS, ...)

# Modelo 2 — Discovery (acceso a todas las capacidades)
from nono.agent.tools.officebridge_tools import OFFICEBRIDGE_DISCOVERY_TOOLS
agent = Agent(name="doc_agent", tools=OFFICEBRIDGE_DISCOVERY_TOOLS, ...)

# Modelo 3 — Skill (composable, reutilizable)
from nono.agent.tools.officebridge_tools import OfficeBridgeSkill
agent = Agent(name="doc_agent", skills=[OfficeBridgeSkill()], ...)
```

### Tools Disponibles

| Tool | Descripción |
|------|-------------|
| `ob_convert_document` | Convertir entre Word, PDF, HTML, Markdown y Text |
| `ob_read_document` | Leer contenido de cualquier documento soportado |
| `ob_create_word` | Crear documentos Word con contenido estructurado |
| `ob_create_html` | Crear documentos HTML con título, párrafos y código |
| `ob_create_excel` | Crear hojas Excel con datos tabulares |
| `ob_read_excel` | Leer datos de hojas Excel |
| `ob_translate_document` | Traducir documentos completos con IA |
| `ob_censor_document` | Detectar y censurar información sensible (PII) |

### Fortalezas

- **Conversión N-to-N**: Cualquier formato → cualquier formato vía DocumentTree IR
- **Traducción AI nativa**: Integrada con Nono para traducción de documentos completos
- **Censura PII inteligente**: Detección AI + regex para emails, teléfonos, DNI, IBAN, tarjetas
- **Modelo de documento universal**: `DocumentTree` como representación intermedia format-agnostic
- **Rich formatting**: Estilos, colores, fuentes, alineación y formatos avanzados en Word/PDF/HTML
- **Excel avanzado**: No solo lectura/escritura — también charts (Pie, Bar, Line) y formatos
- **Zero-config**: Detección automática de formatos por extensión de archivo

### Debilidades

- **Solo archivos locales**: No tiene acceso directo a documentos en la nube (S3, GCS)
- **Dependencias pesadas**: `python-docx`, `openpyxl`, `fpdf2`, `PyPDF2` — añaden peso al entorno
- **PDF limitado**: Extracción de texto de PDFs sin OCR nativo (requiere `easyocr` adicional)
- **Sin streaming**: Las operaciones son síncronas y bloquean durante documentos grandes
- **Windows-only para Outlook**: La automatización de Outlook requiere `pywin32` (solo Windows)

---

## ShortFx

### ¿Qué es?

ShortFx es una librería de 3,000+ funciones deterministas diseñadas para agentes AI. Cubre matemáticas, finanzas, fechas, strings, fórmulas Excel, VBA y más. Incluye búsqueda semántica para descubrimiento de funciones y un servidor MCP integrado.

### Modelos de Integración

```python
# Modelo 1 — Tools directas (máximo rendimiento)
from nono.agent.tools.shortfx_tools import SHORTFX_TOOLS
agent = Agent(name="calc", tools=SHORTFX_TOOLS, ...)

# Modelo 2 — Discovery con búsqueda semántica (3,000+ funciones)
from nono.agent.tools.shortfx_tools import SHORTFX_DISCOVERY_TOOLS
agent = Agent(name="calc", tools=SHORTFX_DISCOVERY_TOOLS, ...)

# Modelo 3 — MCP (zero-code)
from nono.agent.tools.shortfx_tools import shortfx_mcp_tools
agent = Agent(name="calc", tools=shortfx_mcp_tools(), ...)

# Modelo 4 — Skill (composable)
from nono.agent.tools.shortfx_tools import ShortFxSkill
agent = Agent(name="calc", skills=[ShortFxSkill()], ...)
```

### Tools Disponibles

| Tool | Descripción |
|------|-------------|
| `fx_future_value` | Valor futuro de una inversión |
| `fx_present_value` | Valor presente de una inversión |
| `fx_add_time` | Sumar tiempo a una fecha |
| `fx_is_valid_date` | Validar si una fecha es correcta |
| `fx_vlookup` | VLOOKUP estilo Excel |
| `fx_calculate` | Evaluación segura de expresiones matemáticas |
| `fx_find_positions` | Encontrar posiciones de un substring |
| `fx_text_similarity` | Similitud entre dos textos (0.0–1.0) |
| `search_shortfx` | Búsqueda semántica de funciones en lenguaje natural |
| `list_shortfx` | Listar funciones por módulo |
| `inspect_shortfx` | Ver esquema de parámetros de una función |
| `call_shortfx` | Ejecutar cualquier función por nombre cualificado |

### Fortalezas

- **3,000+ funciones**: La mayor colección de funciones deterministas para agentes AI
- **Búsqueda semántica**: Encontrar funciones con lenguaje natural (powered by `fastembed`)
- **100% determinista**: Sin inferencia — resultados exactos y reproducibles siempre
- **Servidor MCP nativo**: Compatible con cualquier cliente MCP (Claude, Copilot, etc.)
- **6 dominios completos**: fxDate, fxNumeric, fxString, fxPython, fxExcel, fxVBA
- **OpenAI Schema**: Esquemas JSON automáticos para function-calling
- **Evaluación segura**: Parser AST para expresiones — sin `eval()`, sin riesgos
- **Compatibilidad Excel/VBA**: Funciones idénticas a Excel y VBA para migración sencilla

### Debilidades

- **Solo cálculos**: No produce efectos secundarios (no crea archivos, no conecta a APIs)
- **Sin estado**: Cada llamada es independiente — no mantiene contexto entre invocaciones
- **Búsqueda semántica lenta**: Primera carga descarga modelo `bge-small-en-v1.5` (~130MB)
- **Sin funciones async**: Todas las funciones son síncronas
- **Dominio limitado a cálculos**: No incluye funciones para I/O, networking, o manipulación de archivos

---

## ¿Cuándo Usar Cada Librería?

### Guía de Decisión

| Necesidad | Librería Recomendada | Razón |
|-----------|---------------------|-------|
| Leer/crear documentos Word, PDF, HTML | **OfficeBridge** | Único propósito: automatización documental |
| Convertir entre formatos de documento | **OfficeBridge** | Conversión N-to-N con DocumentTree IR |
| Cálculos financieros (TIR, VAN, FV) | **ShortFx** | 100% determinista, resultados exactos |
| Operaciones con fechas y calendarios | **ShortFx** | Módulo fxDate completo con business logic |
| Traducir documentos completos | **OfficeBridge** | Motor de traducción AI integrado con Nono |
| Fórmulas estilo Excel (VLOOKUP, SUMIF) | **ShortFx** | Módulo fxExcel con 500+ fórmulas |
| Censurar datos sensibles (PII) | **OfficeBridge** | Detección AI + regex de PII |
| Manipulación avanzada de strings | **ShortFx** | fxString con similarity, regex, hashing |
| Crear Excel con charts y formato | **OfficeBridge** | ChartManager con Pie, Bar, Line |
| Operaciones matemáticas complejas | **ShortFx** | Trig, geometría, métodos numéricos |
| Pipeline documental completo | **Ambas** | OfficeBridge (I/O) + ShortFx (cálculos) |

### Combinación Recomendada

Para agentes que necesitan **procesamiento documental + cálculos**, combinar ambas:

```python
from nono.agent import Agent, Runner
from nono.agent.tools.officebridge_tools import OFFICEBRIDGE_TOOLS
from nono.agent.tools.shortfx_tools import SHORTFX_TOOLS

# Agente con capacidades documentales + cálculos deterministas
agent = Agent(
    name="full_analyst",
    instruction="Eres un analista que procesa documentos y realiza cálculos precisos.",
    tools=[*OFFICEBRIDGE_TOOLS, *SHORTFX_TOOLS],
    provider="google",
)

result = Runner(agent).run(
    "Lee el Excel report.xlsx, calcula el valor futuro de cada inversión al 5% "
    "a 10 años y genera un informe en Word con los resultados."
)
```

### Con Skills (mayor autonomía)

```python
from nono.agent import Agent
from nono.agent.tools.officebridge_tools import OfficeBridgeSkill
from nono.agent.tools.shortfx_tools import ShortFxSkill

agent = Agent(
    name="autonomous_analyst",
    instruction="Eres un analista autónomo con acceso a documentos y cálculos.",
    skills=[OfficeBridgeSkill(), ShortFxSkill()],
    provider="google",
)
```

---

## Tabla Comparativa

| Aspecto | OfficeBridge | ShortFx |
|---------|-------------|---------|
| **Propósito** | Automatización documental | Cálculos deterministas |
| **Nº funciones** | ~15 operaciones principales | 3,000+ funciones |
| **Dominio** | Word, PDF, HTML, Excel, Markdown | Math, Finance, Dates, Strings, Excel, VBA |
| **Efectos secundarios** | Sí (crea/modifica archivos) | No (puro cálculo) |
| **Búsqueda semántica** | No | Sí (fastembed) |
| **Servidor MCP** | No | Sí |
| **Requiere GenAI** | Sí (para traducción y censura AI) | No |
| **Dependencias externas** | python-docx, openpyxl, fpdf2 | fastembed (solo para búsqueda) |
| **Modelo de integración** | 3 modelos (Direct, Discovery, Skill) | 4 modelos (Direct, Discovery, MCP, Skill) |
| **Tamaño típico en contexto** | ~2 KB (8 tools) | ~2 KB (8 tools) o ~10 KB (discovery) |
| **Latencia** | Variable (I/O de archivos) | Sub-milisegundo (cálculos puros) |

---

## Instalación

```bash
# OfficeBridge
pip install officebridge

# ShortFx
pip install shortfx

# ShortFx con MCP
pip install shortfx[mcp]

# Ambas
pip install officebridge shortfx
```
