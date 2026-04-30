"""
OfficeBridge integration — document automation as agent tools.

Bridges `OfficeBridge <https://github.com/jrodriguezgar/OfficeBridge>`_ into
Nono's tool system, providing four integration models:

1. **Direct tools** — Curated OfficeBridge functions wrapped as ``FunctionTool``.
2. **Discovery tools** — List, inspect, and use OfficeBridge capabilities via
   meta-tools.
3. **OfficeBridgeSkill** — A reusable skill combining document operations with
   an LLM agent for autonomous document processing.

Requires ``officebridge`` (``pip install officebridge``).

Usage::

    # Model 1 — Curated tools (common document operations)
    from nono.agent.tools.officebridge_tools import OFFICEBRIDGE_TOOLS
    agent = Agent(name="doc_agent", tools=OFFICEBRIDGE_TOOLS, ...)

    # Model 2 — Discovery (access all OfficeBridge capabilities)
    from nono.agent.tools.officebridge_tools import OFFICEBRIDGE_DISCOVERY_TOOLS
    agent = Agent(name="doc_agent", tools=OFFICEBRIDGE_DISCOVERY_TOOLS, ...)

    # Model 3 — Skill (composable, reusable)
    from nono.agent.tools.officebridge_tools import OfficeBridgeSkill
    agent = Agent(name="doc_agent", skills=[OfficeBridgeSkill()], ...)

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..tool import FunctionTool, ToolContext, tool

logger = logging.getLogger("Nono.Agent.Tools.OfficeBridge")

_OFFICEBRIDGE_AVAILABLE: bool | None = None


def _require_officebridge() -> None:
    """Raise ``ImportError`` if OfficeBridge is not installed."""
    global _OFFICEBRIDGE_AVAILABLE

    if _OFFICEBRIDGE_AVAILABLE is None:
        try:
            import officebridge  # noqa: F401

            _OFFICEBRIDGE_AVAILABLE = True
        except ImportError:
            _OFFICEBRIDGE_AVAILABLE = False

    if not _OFFICEBRIDGE_AVAILABLE:
        raise ImportError(
            "OfficeBridge is required for this tool. "
            "Install it with: pip install officebridge  "
            "(or: pip install git+https://github.com/jrodriguezgar/OfficeBridge.git)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Curated Direct Tools
# ═══════════════════════════════════════════════════════════════════════════════
# Pre-wrapped OfficeBridge functions for the most common operations.
# One LLM step: the agent calls the tool directly.  Maximum performance.
# ═══════════════════════════════════════════════════════════════════════════════


@tool(description=(
    "Convert a document between formats. Supported: Word (.docx), PDF (.pdf), "
    "HTML (.html), Markdown (.md), Text (.txt). "
    "Args: source_path (input file), output_path (output file). "
    "Formats are auto-detected from file extensions."
))
def ob_convert_document(source_path: str, output_path: str) -> str:
    """Convert a document from one format to another.

    Args:
        source_path: Path to the source document.
        output_path: Path for the converted output.

    Returns:
        Confirmation message with the output path.
    """
    _require_officebridge()
    from officebridge.gears.converter_doc import DocumentConverter

    DocumentConverter.convert(source_path, output_path)
    return f"Document converted: {source_path} → {output_path}"


@tool(description=(
    "Read the full text content from a document. "
    "Supports: Word (.docx), HTML (.html), Markdown (.md), Text (.txt), PDF (.pdf). "
    "Args: file_path (path to the document)."
))
def ob_read_document(file_path: str) -> str:
    """Extract all text content from a document.

    Args:
        file_path: Path to the document to read.

    Returns:
        The document's text content.
    """
    _require_officebridge()

    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".docx", ".doc"):
        from officebridge.document import WordClient
        client = WordClient(file_path)
        return client.get_full_text()

    elif ext in (".html", ".htm"):
        from officebridge.document import HtmlClient
        client = HtmlClient(file_path)
        return client.get_full_text()

    elif ext in (".md", ".markdown"):
        from officebridge.document import MarkdownClient
        client = MarkdownClient(file_path)
        return client.get_full_text()

    elif ext == ".txt":
        from officebridge.document import TextClient
        client = TextClient(file_path)
        return client.get_full_text()

    elif ext == ".pdf":
        from officebridge.document import PdfBridge
        bridge = PdfBridge()
        return bridge.extract_text(file_path)

    else:
        return f"Unsupported format: {ext}"


@tool(description=(
    "Create a Word document (.docx) from a list of content blocks. "
    "Args: output_path (file path), title (document title), "
    "content_json (JSON array of objects with 'type' and 'text' keys. "
    "type can be: 'heading', 'paragraph', 'bullet'). "
    "Example: '[{\"type\":\"heading\",\"text\":\"Title\"},{\"type\":\"paragraph\",\"text\":\"Body\"}]'"
))
def ob_create_word(output_path: str, title: str, content_json: str) -> str:
    """Create a Word document with structured content.

    Args:
        output_path: Path for the output .docx file.
        title: Document title (added as heading level 1).
        content_json: JSON array of content blocks.

    Returns:
        Confirmation message.
    """
    _require_officebridge()
    from officebridge.document import WordClient

    client = WordClient()
    client.add_heading(title, level=1)

    try:
        blocks = json.loads(content_json)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"

    for block in blocks:
        block_type = block.get("type", "paragraph")
        text = block.get("text", "")
        if block_type == "heading":
            level = block.get("level", 2)
            client.add_heading(text, level=level)
        elif block_type == "bullet":
            client.add_paragraph(text, style="List Bullet")
        else:
            client.add_paragraph(text)

    client.save(output_path)
    return f"Word document created: {output_path}"


@tool(description=(
    "Create an Excel workbook with data. "
    "Args: output_path (file path), sheet_name (worksheet name), "
    "data_json (JSON 2D array: [[\"Header1\",\"Header2\"],[\"val1\",\"val2\"]]). "
    "First row is treated as headers."
))
def ob_create_excel(output_path: str, sheet_name: str, data_json: str) -> str:
    """Create an Excel workbook from tabular data.

    Args:
        output_path: Path for the output .xlsx file.
        sheet_name: Name of the worksheet.
        data_json: JSON 2D array with headers in the first row.

    Returns:
        Confirmation message.
    """
    _require_officebridge()
    from officebridge.excel import ExcelClient

    try:
        data = json.loads(data_json)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"

    client = ExcelClient()
    ws = client.add_sheet(sheet_name)

    for row_idx, row in enumerate(data, start=1):
        for col_idx, value in enumerate(row, start=1):
            ws.cell(row=row_idx, column=col_idx, value=value)

    client.save(output_path)
    return f"Excel workbook created: {output_path}"


@tool(description=(
    "Read data from an Excel workbook. "
    "Args: file_path (path to .xlsx), sheet_name (optional, defaults to active sheet). "
    "Returns JSON 2D array of all cell values."
))
def ob_read_excel(file_path: str, sheet_name: str = "") -> str:
    """Read all data from an Excel worksheet.

    Args:
        file_path: Path to the Excel file.
        sheet_name: Worksheet name (empty for active sheet).

    Returns:
        JSON 2D array of cell values.
    """
    _require_officebridge()
    from officebridge.excel import ExcelClient

    client = ExcelClient(file_path)
    ws = client.get_sheet(sheet_name) if sheet_name else client.active_sheet

    data = []
    for row in ws.iter_rows(values_only=True):
        data.append([str(v) if v is not None else "" for v in row])

    return json.dumps(data, ensure_ascii=False)


@tool(description=(
    "Translate a document to another language using AI. "
    "Args: source_path (input document), output_path (translated output), "
    "target_language (e.g. 'English', 'Spanish', 'French'). "
    "Supports: Word (.docx), HTML, Markdown, Text."
))
def ob_translate_document(
    source_path: str,
    output_path: str,
    target_language: str,
) -> str:
    """Translate a document to another language using GenAI.

    Args:
        source_path: Path to the source document.
        output_path: Path for the translated output.
        target_language: Target language name (e.g. ``"English"``).

    Returns:
        Confirmation message.
    """
    _require_officebridge()
    from officebridge.gears.translator import translate_document

    translate_document(
        source_path=source_path,
        output_path=output_path,
        target_language=target_language,
    )
    return f"Document translated to {target_language}: {output_path}"


@tool(description=(
    "Censor/redact sensitive information (PII) from a document. "
    "Detects names, emails, phones, IDs, addresses, and financial data. "
    "Args: source_path (input), output_path (redacted output), "
    "sensitive_words (optional comma-separated list of additional words to redact)."
))
def ob_censor_document(
    source_path: str,
    output_path: str,
    sensitive_words: str = "",
) -> str:
    """Redact sensitive information from a document.

    Args:
        source_path: Path to the source document.
        output_path: Path for the censored output.
        sensitive_words: Comma-separated list of extra words to redact.

    Returns:
        Confirmation message.
    """
    _require_officebridge()
    from officebridge.gears.censor import censor_document

    words_list = [w.strip() for w in sensitive_words.split(",") if w.strip()] if sensitive_words else None

    censor_document(
        source_path=source_path,
        output_path=output_path,
        sensitive_words=words_list,
    )
    return f"Document censored: {output_path}"


@tool(description=(
    "Generate an HTML document from structured content. "
    "Args: output_path (file path), title (page title), "
    "content_json (JSON array of objects with 'type' and 'text' keys). "
    "Types: 'heading', 'paragraph', 'code', 'list'."
))
def ob_create_html(output_path: str, title: str, content_json: str) -> str:
    """Create an HTML document with structured content.

    Args:
        output_path: Path for the output .html file.
        title: HTML page title.
        content_json: JSON array of content blocks.

    Returns:
        Confirmation message.
    """
    _require_officebridge()
    from officebridge.document import HtmlClient

    client = HtmlClient()
    client.set_title(title)

    try:
        blocks = json.loads(content_json)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"

    for block in blocks:
        block_type = block.get("type", "paragraph")
        text = block.get("text", "")
        if block_type == "heading":
            level = block.get("level", 1)
            client.add_heading(text, level=level)
        elif block_type == "code":
            client.add_code_block(text)
        elif block_type == "list":
            items = block.get("items", [text])
            client.add_list(items)
        else:
            client.add_paragraph(text)

    client.save(output_path)
    return f"HTML document created: {output_path}"


OFFICEBRIDGE_TOOLS: list[FunctionTool] = [
    ob_convert_document,
    ob_read_document,
    ob_create_word,
    ob_create_excel,
    ob_read_excel,
    ob_translate_document,
    ob_censor_document,
    ob_create_html,
]
"""Curated OfficeBridge tools — common document operations, maximum performance."""


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Discovery Tools (meta-tools)
# ═══════════════════════════════════════════════════════════════════════════════
# Expose OfficeBridge's full capabilities via search/inspect/call meta-tools.
# The LLM uses a list → inspect → execute workflow.
# ═══════════════════════════════════════════════════════════════════════════════

_CAPABILITY_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "document.convert",
        "description": "Convert between Word, PDF, HTML, Markdown, and Text formats.",
        "module": "officebridge.gears.converter_doc",
        "function": "DocumentConverter.convert",
        "parameters": {
            "source_path": "str — Input file path",
            "output_path": "str — Output file path",
            "source_format": "str — (optional) Force source format",
            "target_format": "str — (optional) Force target format",
        },
    },
    {
        "name": "document.read_word",
        "description": "Read full text from a Word (.docx) document.",
        "module": "officebridge.document.word_bridge",
        "function": "WordClient(path).get_full_text()",
        "parameters": {"file_path": "str — Path to .docx file"},
    },
    {
        "name": "document.create_word",
        "description": "Create a Word document with headings, paragraphs, tables, and lists.",
        "module": "officebridge.document.word_bridge",
        "function": "WordClient",
        "parameters": {
            "output_path": "str — Output .docx path",
            "content": "list — Content blocks (heading/paragraph/table/bullet)",
        },
    },
    {
        "name": "document.read_html",
        "description": "Read full text from an HTML document.",
        "module": "officebridge.document.html_bridge",
        "function": "HtmlClient(path).get_full_text()",
        "parameters": {"file_path": "str — Path to .html file"},
    },
    {
        "name": "document.create_html",
        "description": "Create an HTML document with title, paragraphs, headings, and code blocks.",
        "module": "officebridge.document.html_bridge",
        "function": "HtmlClient",
        "parameters": {
            "output_path": "str — Output .html path",
            "title": "str — Page title",
            "content": "list — Content blocks",
        },
    },
    {
        "name": "document.read_markdown",
        "description": "Read full text from a Markdown document.",
        "module": "officebridge.document.markdown_bridge",
        "function": "MarkdownClient(path).get_full_text()",
        "parameters": {"file_path": "str — Path to .md file"},
    },
    {
        "name": "document.create_markdown",
        "description": "Create a Markdown document programmatically.",
        "module": "officebridge.document.markdown_bridge",
        "function": "MarkdownClient",
        "parameters": {
            "output_path": "str — Output .md path",
            "content": "list — Content blocks",
        },
    },
    {
        "name": "document.read_pdf",
        "description": "Extract text from a PDF document.",
        "module": "officebridge.document.pdf_bridge",
        "function": "PdfBridge().extract_text(path)",
        "parameters": {"file_path": "str — Path to .pdf file"},
    },
    {
        "name": "document.create_pdf",
        "description": "Create a PDF document with text, headings, and tables.",
        "module": "officebridge.document.pdf_bridge",
        "function": "PdfBridge",
        "parameters": {
            "output_path": "str — Output .pdf path",
            "content": "list — Content blocks",
        },
    },
    {
        "name": "excel.read",
        "description": "Read data from an Excel workbook (.xlsx).",
        "module": "officebridge.excel.excel_bridge",
        "function": "ExcelClient(path)",
        "parameters": {
            "file_path": "str — Path to .xlsx file",
            "sheet_name": "str — (optional) Worksheet name",
        },
    },
    {
        "name": "excel.create",
        "description": "Create an Excel workbook with data, formatting, and charts.",
        "module": "officebridge.excel.excel_bridge",
        "function": "ExcelClient",
        "parameters": {
            "output_path": "str — Output .xlsx path",
            "data": "list[list] — 2D array of values",
            "sheet_name": "str — Worksheet name",
        },
    },
    {
        "name": "excel.chart",
        "description": "Add charts (Pie, Bar, Line) to an Excel workbook.",
        "module": "officebridge.excel.excel_bridge",
        "function": "ChartManager",
        "parameters": {
            "chart_type": "str — 'pie', 'bar', 'line'",
            "data_range": "str — Cell range for chart data",
        },
    },
    {
        "name": "gears.translate",
        "description": "Translate a document to another language using AI (GenAI-powered).",
        "module": "officebridge.gears.translator",
        "function": "translate_document",
        "parameters": {
            "source_path": "str — Input document path",
            "output_path": "str — Translated output path",
            "target_language": "str — Target language (e.g. 'English')",
        },
    },
    {
        "name": "gears.censor",
        "description": "Detect and redact sensitive information (PII) from documents.",
        "module": "officebridge.gears.censor",
        "function": "censor_document",
        "parameters": {
            "source_path": "str — Input document path",
            "output_path": "str — Censored output path",
            "sensitive_words": "list[str] — (optional) Extra words to redact",
        },
    },
    {
        "name": "gears.view",
        "description": "Open a document with the system's default viewer application.",
        "module": "officebridge.gears.viewer",
        "function": "call_viewer",
        "parameters": {
            "document_path": "str — Path to the document",
            "application": "str — (optional) Specific application to use",
        },
    },
]


@tool(description=(
    "List available OfficeBridge capabilities, optionally filtered by category. "
    "Valid categories: document, excel, gears. "
    "Returns name, description, and parameters for each capability. "
    "Example: list_officebridge('document')."
))
def list_officebridge(category: str = "") -> str:
    """Browse OfficeBridge capabilities by category.

    Args:
        category: Category prefix filter (e.g. ``"document"``, ``"excel"``, ``"gears"``).

    Returns:
        JSON array of capability descriptors.
    """
    caps = _CAPABILITY_REGISTRY
    if category:
        caps = [c for c in caps if c["name"].startswith(category)]

    summaries = [
        {"name": c["name"], "description": c["description"]}
        for c in caps
    ]
    return json.dumps(summaries, indent=2, ensure_ascii=False)


@tool(description=(
    "Get full details (parameters, module, function) for a specific OfficeBridge "
    "capability. Use the name from list_officebridge. "
    "Example: inspect_officebridge('gears.translate')."
))
def inspect_officebridge(capability_name: str) -> str:
    """Get the full schema for an OfficeBridge capability.

    Args:
        capability_name: Capability name (e.g. ``"document.convert"``).

    Returns:
        JSON object with full details.
    """
    for cap in _CAPABILITY_REGISTRY:
        if cap["name"] == capability_name:
            return json.dumps(cap, indent=2, ensure_ascii=False)

    return json.dumps({"error": f"Capability '{capability_name}' not found."})


@tool(description=(
    "Execute an OfficeBridge operation by its capability name. "
    "Pass arguments as a JSON object string. "
    "Example: call_officebridge('document.convert', "
    "'{\"source_path\": \"report.docx\", \"output_path\": \"report.pdf\"}')."
))
def call_officebridge(capability_name: str, arguments_json: str = "{}") -> str:
    """Invoke an OfficeBridge capability dynamically.

    Args:
        capability_name: Capability name from the registry.
        arguments_json: JSON string with keyword arguments.

    Returns:
        Operation result as a string.
    """
    _require_officebridge()

    try:
        args = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON arguments: {exc}"

    try:
        # Route to the appropriate OfficeBridge function
        if capability_name == "document.convert":
            from officebridge.gears.converter_doc import DocumentConverter
            DocumentConverter.convert(args["source_path"], args["output_path"])
            return f"Converted: {args['source_path']} → {args['output_path']}"

        elif capability_name == "document.read_word":
            from officebridge.document import WordClient
            return WordClient(args["file_path"]).get_full_text()

        elif capability_name == "document.read_html":
            from officebridge.document import HtmlClient
            return HtmlClient(args["file_path"]).get_full_text()

        elif capability_name == "document.read_markdown":
            from officebridge.document import MarkdownClient
            return MarkdownClient(args["file_path"]).get_full_text()

        elif capability_name == "document.read_pdf":
            from officebridge.document import PdfBridge
            return PdfBridge().extract_text(args["file_path"])

        elif capability_name == "excel.read":
            from officebridge.excel import ExcelClient
            client = ExcelClient(args["file_path"])
            ws = client.get_sheet(args.get("sheet_name")) if args.get("sheet_name") else client.active_sheet
            data = []
            for row in ws.iter_rows(values_only=True):
                data.append([str(v) if v is not None else "" for v in row])
            return json.dumps(data, ensure_ascii=False)

        elif capability_name == "gears.translate":
            from officebridge.gears.translator import translate_document
            translate_document(
                source_path=args["source_path"],
                output_path=args["output_path"],
                target_language=args["target_language"],
            )
            return f"Translated to {args['target_language']}: {args['output_path']}"

        elif capability_name == "gears.censor":
            from officebridge.gears.censor import censor_document
            censor_document(
                source_path=args["source_path"],
                output_path=args["output_path"],
                sensitive_words=args.get("sensitive_words"),
            )
            return f"Censored: {args['output_path']}"

        elif capability_name == "gears.view":
            from officebridge.gears.viewer import call_viewer
            call_viewer(args["document_path"], application=args.get("application"))
            return f"Opened: {args['document_path']}"

        else:
            return f"Unknown capability: {capability_name}"

    except Exception as exc:
        logger.warning("call_officebridge(%s) failed: %s", capability_name, exc)
        return f"Error: {exc}"


OFFICEBRIDGE_DISCOVERY_TOOLS: list[FunctionTool] = [
    list_officebridge,
    inspect_officebridge,
    call_officebridge,
]
"""Discovery meta-tools — browse and use all OfficeBridge capabilities."""


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — OfficeBridgeSkill
# ═══════════════════════════════════════════════════════════════════════════════
# A reusable skill that combines document tools with an LLM agent.
# The agent autonomously decides which document operations to perform.
# ═══════════════════════════════════════════════════════════════════════════════


class OfficeBridgeSkill:
    """Skill that provides access to OfficeBridge document automation.

    The inner agent uses document tools to read, create, convert, translate,
    and censor documents autonomously.

    Can be used standalone or attached to an agent::

        # Standalone
        skill = OfficeBridgeSkill()
        result = skill.run("Convert report.docx to PDF")

        # Attached to an agent
        agent = Agent(name="doc_agent", skills=[OfficeBridgeSkill()], ...)

    Args:
        provider: AI provider (default ``"google"``).
        model: Model override (``None`` for provider default).
    """

    def __init__(
        self,
        *,
        provider: str = "google",
        model: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model

    @property
    def descriptor(self) -> Any:
        """Return the skill's metadata descriptor."""
        from ..skill import SkillDescriptor

        return SkillDescriptor(
            name="officebridge",
            description=(
                "Automate document operations using OfficeBridge: read, create, "
                "convert (Word/PDF/HTML/Markdown/Text), translate, censor PII, "
                "and manage Excel workbooks with charts and formatting."
            ),
            tags=("documents", "word", "excel", "pdf", "html", "markdown",
                  "translation", "censoring", "conversion"),
            input_keys=("input",),
            output_keys=("result",),
        )

    def build_agent(self, **overrides: Any) -> Any:
        """Create the LLM agent that drives OfficeBridge operations.

        Args:
            **overrides: Optional provider/model overrides.

        Returns:
            Configured ``LlmAgent``.
        """
        from ..llm_agent import LlmAgent

        return LlmAgent(
            name="officebridge_agent",
            provider=overrides.get("provider", self._provider),
            model=overrides.get("model", self._model),
            instruction=(
                "You are a document automation assistant powered by OfficeBridge. "
                "OfficeBridge provides tools for:\n\n"
                "- **Document I/O**: Read and create Word, HTML, Markdown, Text, PDF.\n"
                "- **Conversion**: Convert between any supported formats.\n"
                "- **Excel**: Read/write workbooks, add charts and formatting.\n"
                "- **Translation**: AI-powered document translation to any language.\n"
                "- **Censoring**: Detect and redact PII/sensitive information.\n\n"
                "Workflow:\n"
                "1. Understand the user's document task.\n"
                "2. Use the appropriate tool for the operation.\n"
                "3. Confirm the result and provide the output path.\n\n"
                "Always use OfficeBridge tools for document operations — never "
                "try to manipulate file contents directly via string operations."
            ),
            description=self.descriptor.description,
            temperature=0.0,
        )

    def build_tools(self) -> list[FunctionTool]:
        """Return curated tools for the inner agent.

        Returns:
            The curated OfficeBridge tools.
        """
        return list(OFFICEBRIDGE_TOOLS)

    def as_tool(self) -> FunctionTool:
        """Convert this skill to a ``FunctionTool`` for LLM function-calling.

        Returns:
            A ``FunctionTool`` wrapping the full document automation workflow.
        """
        desc = self.descriptor

        def _invoke(input: str) -> str:  # noqa: A002
            return self.run(input)

        return FunctionTool(
            fn=_invoke,
            name=desc.name,
            description=desc.description,
        )

    def run(
        self,
        user_message: str,
        **overrides: Any,
    ) -> str:
        """Execute the skill standalone.

        Args:
            user_message: Document operation request in natural language.
            **overrides: Forwarded to :meth:`build_agent`.

        Returns:
            The agent's final text response.
        """
        from ..runner import Runner

        agent = self.build_agent(**overrides)

        # Inject tools into the agent
        existing = {t.name for t in agent.tools}
        for t in self.build_tools():
            if t.name not in existing:
                agent.tools.append(t)

        return Runner(agent).run(user_message)

    def __repr__(self) -> str:
        return "OfficeBridgeSkill()"
