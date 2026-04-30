"""
SKILL.md file loader — Anthropic Agent Skills standard support.

Loads skills defined as ``SKILL.md`` files with YAML frontmatter,
following the `Anthropic Agent Skills specification
<https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview>`_.

A ``SKILL.md`` file has the structure::

    ---
    name: summarizing-text
    description: >
      Summarize text into concise key points.
      Use when the user asks to summarize, condense, or
      extract key points from text.
    ---

    # Summarize Text

    Produce concise, accurate summaries that capture
    the essential information.

    ## Guidelines

    - Preserve key facts, figures, and conclusions.
    - Organize with clear structure when the source is long.

The YAML frontmatter becomes the ``SkillDescriptor``, and the Markdown
body becomes the system instruction for an ``LlmAgent``.

Usage::

    from nono.agent.skill_loader import load_skill_md, scan_skills_dir

    # Load a single SKILL.md
    skill = load_skill_md("path/to/SKILL.md")
    result = skill.run("Summarize this text...")

    # Scan a directory tree for SKILL.md files
    skills = scan_skills_dir(".claude/skills")
    for s in skills:
        print(s.descriptor.name)

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import importlib
import logging
import re
import sys
from pathlib import Path
from typing import Any

_SKILL_NAME_RE = re.compile(r"^[a-z0-9-]+$")

import yaml

from .base import BaseAgent
from .llm_agent import LlmAgent
from .skill import BaseSkill, SkillDescriptor, SkillRegistry, registry
from .tool import FunctionTool

logger = logging.getLogger("Nono.Agent.SkillLoader")

__all__ = [
    "load_skill_md",
    "scan_skills_dir",
    "MarkdownSkill",
]

# ── YAML frontmatter parser ──────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(
    r"\A\s*---\s*\n(.*?)\n---\s*\n(.*)",
    re.DOTALL,
)


def _parse_skill_md(text: str) -> tuple[dict[str, Any], str]:
    """Parse a SKILL.md file into (frontmatter_dict, body_markdown).

    Args:
        text: Raw content of a SKILL.md file.

    Returns:
        Tuple of (parsed YAML dict, markdown body).

    Raises:
        ValueError: If the file has no valid YAML frontmatter.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(
            "SKILL.md must start with YAML frontmatter "
            "delimited by --- markers."
        )

    raw_yaml = match.group(1)
    body = match.group(2).strip()

    try:
        meta = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML frontmatter: {exc}") from exc

    if not isinstance(meta, dict):
        raise ValueError("YAML frontmatter must be a mapping.")

    return meta, body


def _validate_frontmatter(meta: dict[str, Any]) -> None:
    """Validate frontmatter fields per Agent Skills standard.

    Args:
        meta: Parsed YAML frontmatter dictionary.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    if "name" not in meta:
        raise ValueError("SKILL.md frontmatter must include 'name'.")
    if "description" not in meta:
        raise ValueError("SKILL.md frontmatter must include 'description'.")

    name = meta["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("'name' must be a non-empty string.")
    if len(name) > 64:
        raise ValueError(f"'name' must be <= 64 chars, got {len(name)}.")
    if not _SKILL_NAME_RE.match(name):
        raise ValueError(
            f"'name' must contain only lowercase letters, numbers, "
            f"and hyphens: {name!r}."
        )

    description = meta["description"]
    if not isinstance(description, str) or not description.strip():
        raise ValueError("'description' must be a non-empty string.")
    if len(description) > 1024:
        raise ValueError(
            f"'description' must be <= 1024 chars, got {len(description)}."
        )

    # Optional: compatibility (max 500 chars)
    compatibility = meta.get("compatibility")
    if compatibility is not None:
        if not isinstance(compatibility, str):
            raise ValueError("'compatibility' must be a string.")
        if len(compatibility) > 500:
            raise ValueError(
                f"'compatibility' must be <= 500 chars, "
                f"got {len(compatibility)}."
            )

    # Optional: metadata (dict[str, str])
    metadata = meta.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError("'metadata' must be a mapping.")
        for k, v in metadata.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError(
                    "'metadata' must be a mapping of string keys "
                    "to string values."
                )


# ── Script tool loading ──────────────────────────────────────────────────────

def _load_script_tool(
    entry: dict[str, Any],
    skill_dir: Path,
) -> FunctionTool:
    """Load a single tool from a YAML ``tools`` entry.

    Each entry must have ``name``, ``script``, and ``description``.
    The script must expose a ``main(input: str) -> str`` callable.

    Args:
        entry: ``{name, script, description}`` from YAML frontmatter.
        skill_dir: Directory containing the SKILL.md file.

    Returns:
        A ``FunctionTool`` wrapping the script's ``main`` function.

    Raises:
        ValueError: If the entry is malformed or the script has no ``main``.
        FileNotFoundError: If the script file does not exist.
    """
    for field in ("name", "script", "description"):
        if field not in entry:
            raise ValueError(
                f"Tool entry missing required field {field!r}: {entry}"
            )

    tool_name: str = entry["name"]
    script_rel: str = entry["script"]
    tool_desc: str = entry["description"]

    script_path = (skill_dir / script_rel).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(
            f"Tool script not found: {script_path} "
            f"(referenced by tool {tool_name!r})"
        )

    # Security: only allow scripts within (or under) the skill directory
    try:
        script_path.relative_to(skill_dir.resolve())
    except ValueError:
        raise ValueError(
            f"Tool script {script_rel!r} must be inside the skill "
            f"directory {skill_dir}."
        ) from None

    # Import the script as a module
    module_name = f"_nono_skill_tool_{tool_name}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ValueError(
            f"Cannot load script {script_path} as a Python module."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    del sys.modules[module_name]  # avoid unbounded sys.modules growth

    main_fn = getattr(module, "main", None)
    if main_fn is None or not callable(main_fn):
        raise ValueError(
            f"Tool script {script_path} must define a callable "
            f"``main(input: str) -> str``."
        )

    return FunctionTool(
        fn=main_fn,
        name=tool_name,
        description=tool_desc,
    )


def _load_tools_from_meta(
    meta: dict[str, Any],
    skill_dir: Path | None,
) -> list[FunctionTool]:
    """Parse the ``tools`` YAML field and load all referenced scripts.

    Args:
        meta: Parsed YAML frontmatter.
        skill_dir: Directory containing the SKILL.md file.

    Returns:
        List of ``FunctionTool`` instances.  Empty if no ``tools`` field.
    """
    raw_tools = meta.get("tools")
    if not raw_tools:
        return []

    if not isinstance(raw_tools, list):
        raise ValueError("'tools' must be a list of tool entries.")

    if skill_dir is None:
        raise ValueError(
            "'tools' in YAML requires a file-based SKILL.md "
            "(need skill directory to resolve script paths)."
        )

    tools: list[FunctionTool] = []
    for entry in raw_tools:
        if not isinstance(entry, dict):
            raise ValueError(f"Each tool entry must be a mapping: {entry}")
        tools.append(_load_script_tool(entry, skill_dir))

    return tools


# ── MarkdownSkill ─────────────────────────────────────────────────────────────

class MarkdownSkill(BaseSkill):
    """A skill loaded from a ``SKILL.md`` file.

    The YAML frontmatter provides the descriptor fields, and the Markdown
    body becomes the system instruction for the inner ``LlmAgent``.

    Follows the `Agent Skills <https://agentskills.io/specification>`_
    open standard with full support for:

    - **Standard frontmatter**: ``name``, ``description``, ``license``,
      ``compatibility``, ``metadata``, ``allowed-tools``.
    - **Supporting directories**: ``scripts/``, ``references/``, ``assets/``.
    - **Progressive disclosure**: metadata → instructions → resources.
    - **File references**: ``load_resource()`` reads supporting files.
    - **Tools**: YAML ``tools`` field or programmatic ``FunctionTool``
      instances.

    Args:
        meta: Parsed and validated frontmatter dictionary.
        body: Markdown body (the agent's system instruction).
        source_path: Path to the original SKILL.md file (for debugging).
        tools: Additional ``FunctionTool`` instances to attach.
    """

    def __init__(
        self,
        meta: dict[str, Any],
        body: str,
        source_path: Path | None = None,
        tools: list[FunctionTool] | None = None,
    ) -> None:
        self._meta = meta
        self._body = body
        self._source_path = source_path

        # Parse optional fields with defaults
        tags = meta.get("tags", ())
        if isinstance(tags, list):
            tags = tuple(tags)
        elif isinstance(tags, str):
            tags = tuple(t.strip() for t in tags.split(",") if t.strip())

        # Parse allowed-tools (space-separated string or YAML list)
        raw_allowed = meta.get("allowed-tools", ())
        if isinstance(raw_allowed, str):
            allowed_tools = tuple(
                t.strip() for t in raw_allowed.split() if t.strip()
            )
        elif isinstance(raw_allowed, list):
            allowed_tools = tuple(str(t) for t in raw_allowed)
        else:
            allowed_tools = ()

        # Parse metadata (dict[str, str])
        raw_metadata = meta.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raw_metadata = {}
        parsed_metadata = {
            str(k): str(v) for k, v in raw_metadata.items()
        }

        self._descriptor = SkillDescriptor(
            name=meta["name"],
            description=meta["description"].strip(),
            version=meta.get("version", "1.0.0"),
            tags=tags,
            input_keys=tuple(meta.get("input_keys", ("input",))),
            output_keys=tuple(meta.get("output_keys", ("output",))),
            license=str(meta.get("license", "")),
            compatibility=str(meta.get("compatibility", "")),
            metadata=parsed_metadata,
            allowed_tools=allowed_tools,
        )

        # Optional agent configuration from frontmatter
        self._provider = meta.get("provider", "google")
        self._model = meta.get("model")
        self._temperature = meta.get("temperature", 0.3)
        self._output_format = meta.get("output_format")

        # Tools: YAML-defined scripts + programmatic tools
        skill_dir = source_path.parent if source_path else None
        self._tools: list[FunctionTool] = _load_tools_from_meta(meta, skill_dir)
        if tools:
            self._tools.extend(tools)

    @property
    def descriptor(self) -> SkillDescriptor:
        return self._descriptor

    @property
    def source_path(self) -> Path | None:
        """Path to the SKILL.md file this skill was loaded from."""
        return self._source_path

    @property
    def instruction(self) -> str:
        """The Markdown body used as the agent's system instruction."""
        return self._body

    @property
    def skill_dir(self) -> Path | None:
        """Directory containing the skill's SKILL.md file."""
        if self._source_path:
            return self._source_path.parent
        return None

    # ── Supporting directories (Agent Skills standard) ─────────────

    def list_references(self) -> list[Path]:
        """List files in the ``references/`` directory.

        Returns:
            Sorted list of file paths, or empty list if no directory.
        """
        return self._list_dir("references")

    def list_assets(self) -> list[Path]:
        """List files in the ``assets/`` directory.

        Returns:
            Sorted list of file paths, or empty list if no directory.
        """
        return self._list_dir("assets")

    def list_scripts(self) -> list[Path]:
        """List files in the ``scripts/`` directory.

        Returns:
            Sorted list of file paths, or empty list if no directory.
        """
        return self._list_dir("scripts")

    def _list_dir(self, subdir: str) -> list[Path]:
        """List files in a subdirectory of the skill directory.

        Args:
            subdir: Name of the subdirectory.

        Returns:
            Sorted list of file paths.
        """
        skill_dir = self.skill_dir
        if skill_dir is None:
            return []

        target = skill_dir / subdir
        if not target.is_dir():
            return []

        return sorted(p for p in target.iterdir() if p.is_file())

    def load_resource(self, relative_path: str) -> str:
        """Load a supporting file from the skill directory.

        Implements the Agent Skills standard *file references*: SKILL.md
        can reference supporting files via relative paths such as
        ``references/REFERENCE.md`` or ``assets/template.json``.

        Args:
            relative_path: Path relative to the skill directory
                (e.g. ``"references/REFERENCE.md"``).

        Returns:
            File content as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If path escapes the skill directory
                (path traversal protection).
        """
        skill_dir = self.skill_dir
        if skill_dir is None:
            raise FileNotFoundError(
                "Cannot load resources from a skill without a "
                "source path (not loaded from a file)."
            )

        target = (skill_dir / relative_path).resolve()

        # Security: block path traversal
        try:
            target.relative_to(skill_dir.resolve())
        except ValueError:
            raise ValueError(
                f"Resource path {relative_path!r} escapes the skill "
                f"directory {skill_dir}."
            ) from None

        if not target.is_file():
            raise FileNotFoundError(
                f"Resource not found: {target} "
                f"(referenced as {relative_path!r})"
            )

        return target.read_text(encoding="utf-8")

    def build_tools(self) -> list[FunctionTool]:
        """Return tools defined in YAML frontmatter and/or passed programmatically.

        Returns:
            List of ``FunctionTool`` instances.
        """
        return list(self._tools)

    def build_agent(self, **overrides: Any) -> BaseAgent:
        """Build an LlmAgent using the Markdown body as instruction.

        Args:
            **overrides: Override provider, model, temperature, etc.

        Returns:
            Configured ``LlmAgent``.
        """
        kwargs: dict[str, Any] = {
            "name": self._descriptor.name.replace("-", "_"),
            "provider": overrides.get("provider", self._provider),
            "instruction": self._body,
            "description": self._descriptor.description,
            "temperature": overrides.get("temperature", self._temperature),
        }

        model = overrides.get("model", self._model)
        if model:
            kwargs["model"] = model

        output_format = overrides.get("output_format", self._output_format)
        if output_format:
            kwargs["output_format"] = output_format

        return LlmAgent(**kwargs)

    def __repr__(self) -> str:
        path_info = f", source={self._source_path}" if self._source_path else ""
        return f"MarkdownSkill(name={self._descriptor.name!r}{path_info})"


# ── Public API ────────────────────────────────────────────────────────────────

def load_skill_md(
    path: str | Path,
    *,
    register_in: SkillRegistry | None = None,
    tools: list[FunctionTool] | None = None,
) -> MarkdownSkill:
    """Load a single ``SKILL.md`` file and return a ``MarkdownSkill``.

    Args:
        path: Path to the ``SKILL.md`` file.
        register_in: Optional registry to auto-register the skill in.
            Pass ``registry`` to use the global singleton.
        tools: Additional ``FunctionTool`` instances to attach to the
            skill (combined with any tools declared in YAML frontmatter).

    Returns:
        A ``MarkdownSkill`` instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If frontmatter is missing or invalid.

    Example::

        from nono.agent.skill_loader import load_skill_md
        from nono.agent.skill import registry

        skill = load_skill_md(".claude/skills/pdf-processing/SKILL.md",
                              register_in=registry)
        result = skill.run("Extract text from this PDF content...")
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SKILL.md not found: {path}")

    text = path.read_text(encoding="utf-8")
    meta, body = _parse_skill_md(text)
    _validate_frontmatter(meta)

    skill = MarkdownSkill(meta, body, source_path=path.resolve(), tools=tools)

    if register_in is not None:
        register_in.register(skill)

    logger.info("Loaded skill %r from %s", skill.descriptor.name, path)
    return skill


def scan_skills_dir(
    directory: str | Path,
    *,
    register_in: SkillRegistry | None = None,
) -> list[MarkdownSkill]:
    """Scan a directory tree for ``SKILL.md`` files and load them all.

    Follows the Anthropic convention of searching for files named
    ``SKILL.md`` in immediate subdirectories::

        directory/
        ├── summarizing-text/
        │   └── SKILL.md
        ├── classifying-data/
        │   └── SKILL.md
        └── ...

    Also supports ``SKILL.md`` files directly in the root directory.

    Args:
        directory: Root directory to scan.
        register_in: Optional registry for auto-registration.

    Returns:
        List of loaded ``MarkdownSkill`` instances.

    Example::

        from nono.agent.skill_loader import scan_skills_dir
        from nono.agent.skill import registry

        skills = scan_skills_dir(".claude/skills", register_in=registry)
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.warning("Skills directory does not exist: %s", directory)
        return []

    loaded: list[MarkdownSkill] = []

    # Pattern 1: directory/*/SKILL.md (Anthropic standard)
    for skill_md in sorted(directory.glob("*/SKILL.md")):
        try:
            skill = load_skill_md(skill_md, register_in=register_in)
            loaded.append(skill)
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Skipping %s: %s", skill_md, exc)

    # Pattern 2: directory/SKILL.md (flat layout)
    root_skill = directory / "SKILL.md"
    if root_skill.is_file():
        try:
            skill = load_skill_md(root_skill, register_in=register_in)
            loaded.append(skill)
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Skipping %s: %s", root_skill, exc)

    logger.info(
        "Scanned %s: loaded %d skill(s)", directory, len(loaded)
    )
    return loaded
