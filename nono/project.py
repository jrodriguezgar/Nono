"""
Nono Project System

Organise AI work into isolated, self-contained **projects**.  Each project
owns its own configuration, skills, prompts, templates, and workflows,
keeping concerns separated across different codebases or use-cases.

On disk a Nono project is a directory with a ``nono.toml`` manifest::

    my-project/
    ├── nono.toml            # Project manifest (required)
    ├── config.toml          # Provider / model overrides (optional)
    ├── skills/              # SKILL.md files (Agent Skills standard)
    │   └── my-skill/
    │       └── SKILL.md
    ├── prompts/             # JSON task definitions
    ├── templates/           # Jinja2 prompt templates
    ├── workflows/           # Workflow definitions
    └── data/                # Project data files (CSV, JSON, TXT, …)

Usage::

    from nono.project import Project, init_project, load_project

    # Scaffold a new project
    project = init_project("my-project")

    # Load an existing project
    project = load_project("my-project")

    # Access project resources
    config = project.config()
    skills = project.load_skills()
    print(project.name, project.description)

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, load_config

logger = logging.getLogger("Nono.Project")

__all__ = [
    "Project",
    "ProjectManifest",
    "init_project",
    "load_project",
    "list_projects",
    "MANIFEST_FILE",
    "PROJECT_DIRS",
]

# ── Constants ─────────────────────────────────────────────────────────────────

MANIFEST_FILE = "nono.toml"
"""Name of the project manifest file."""

PROJECT_DIRS = ("skills", "prompts", "templates", "workflows", "data")
"""Standard subdirectories created by :func:`init_project`."""

# Try to import tomllib (Python 3.11+) or tomli as fallback
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


# ── Manifest ──────────────────────────────────────────────────────────────────

@dataclass
class ProjectManifest:
    """Parsed ``nono.toml`` project manifest.

    Args:
        name: Project identifier (human-friendly).
        description: One-line purpose description.
        version: Semantic version string.
        authors: List of ``"Name <email>"`` strings.
        default_provider: Default LLM provider for this project.
        default_model: Default LLM model for this project.
        skills_dir: Relative path to skills directory.
        prompts_dir: Relative path to prompts/task-definitions directory.
        templates_dir: Relative path to Jinja2 templates directory.
        workflows_dir: Relative path to workflows directory.
        data_dir: Relative path to project data directory.
        extra: Arbitrary extra keys preserved from the TOML file.
    """

    name: str = ""
    description: str = ""
    version: str = "0.1.0"
    authors: list[str] = field(default_factory=list)
    default_provider: str = ""
    default_model: str = ""
    skills_dir: str = "skills"
    prompts_dir: str = "prompts"
    templates_dir: str = "templates"
    workflows_dir: str = "workflows"
    data_dir: str = "data"
    extra: dict[str, Any] = field(default_factory=dict)


def _parse_manifest(data: dict[str, Any]) -> ProjectManifest:
    """Build a ``ProjectManifest`` from parsed TOML data.

    Args:
        data: Dictionary loaded from ``nono.toml``.

    Returns:
        Populated ``ProjectManifest``.
    """
    project_section = data.get("project", {})

    known = {
        "name", "description", "version", "authors",
        "default_provider", "default_model",
        "skills_dir", "prompts_dir", "templates_dir", "workflows_dir",
        "data_dir",
    }

    extra = {k: v for k, v in project_section.items() if k not in known}

    # Merge top-level sections (providers, etc.) into extra
    for key, value in data.items():
        if key != "project":
            extra[key] = value

    return ProjectManifest(
        name=project_section.get("name", ""),
        description=project_section.get("description", ""),
        version=project_section.get("version", "0.1.0"),
        authors=project_section.get("authors", []),
        default_provider=project_section.get("default_provider", ""),
        default_model=project_section.get("default_model", ""),
        skills_dir=project_section.get("skills_dir", "skills"),
        prompts_dir=project_section.get("prompts_dir", "prompts"),
        templates_dir=project_section.get("templates_dir", "templates"),
        workflows_dir=project_section.get("workflows_dir", "workflows"),
        data_dir=project_section.get("data_dir", "data"),
        extra=extra,
    )


def _write_manifest(manifest: ProjectManifest, path: Path) -> None:
    """Serialise a ``ProjectManifest`` as TOML and write to *path*.

    Args:
        manifest: The manifest to write.
        path: Destination file (must end with ``nono.toml``).
    """
    def _esc(val: str) -> str:
        """Escape a value for TOML double-quoted string."""
        return val.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")

    lines: list[str] = [
        "# Nono Project Manifest",
        "# https://github.com/DatamanEdge/Nono",
        "",
        "[project]",
        f'name = "{_esc(manifest.name)}"',
        f'description = "{_esc(manifest.description)}"',
        f'version = "{_esc(manifest.version)}"',
    ]

    if manifest.authors:
        authors_str = ", ".join(f'"{_esc(a)}"' for a in manifest.authors)
        lines.append(f"authors = [{authors_str}]")

    if manifest.default_provider:
        lines.append(f'default_provider = "{_esc(manifest.default_provider)}"')

    if manifest.default_model:
        lines.append(f'default_model = "{_esc(manifest.default_model)}"')

    lines.extend([
        "",
        "# Directory layout (relative to project root)",
        f'skills_dir = "{_esc(manifest.skills_dir)}"',
        f'prompts_dir = "{_esc(manifest.prompts_dir)}"',
        f'templates_dir = "{_esc(manifest.templates_dir)}"',
        f'workflows_dir = "{_esc(manifest.workflows_dir)}"',
        f'data_dir = "{_esc(manifest.data_dir)}"',
        "",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Project ───────────────────────────────────────────────────────────────────

class Project:
    """A loaded Nono project bound to a directory on disk.

    Use :func:`load_project` or :func:`init_project` to obtain an instance.

    Args:
        root: Absolute path to the project root directory.
        manifest: Parsed project manifest from ``nono.toml``.
    """

    def __init__(self, root: Path, manifest: ProjectManifest) -> None:
        self._root = root.resolve()
        self._manifest = manifest

    # ── Properties ────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        """Absolute path to the project root directory."""
        return self._root

    @property
    def manifest(self) -> ProjectManifest:
        """The parsed project manifest."""
        return self._manifest

    @property
    def name(self) -> str:
        """Project name from manifest (or directory name as fallback)."""
        return self._manifest.name or self._root.name

    @property
    def description(self) -> str:
        """Project description from manifest."""
        return self._manifest.description

    @property
    def version(self) -> str:
        """Project version from manifest."""
        return self._manifest.version

    # ── Resolved paths ────────────────────────────────────────────

    def _safe_subdir(self, relative: str) -> Path:
        """Resolve a manifest-relative path and verify it stays inside the project root."""
        resolved = (self._root / relative).resolve()
        if not resolved.is_relative_to(self._root):
            raise ValueError(
                f"Path '{relative}' escapes project root '{self._root}'."
            )
        return resolved

    @property
    def skills_dir(self) -> Path:
        """Absolute path to the project's skills directory."""
        return self._safe_subdir(self._manifest.skills_dir)

    @property
    def prompts_dir(self) -> Path:
        """Absolute path to the project's prompts directory."""
        return self._safe_subdir(self._manifest.prompts_dir)

    @property
    def templates_dir(self) -> Path:
        """Absolute path to the project's templates directory."""
        return self._safe_subdir(self._manifest.templates_dir)

    @property
    def workflows_dir(self) -> Path:
        """Absolute path to the project's workflows directory."""
        return self._safe_subdir(self._manifest.workflows_dir)

    @property
    def data_dir(self) -> Path:
        """Absolute path to the project's data directory."""
        return self._safe_subdir(self._manifest.data_dir)

    @property
    def config_file(self) -> Path:
        """Path to the project's ``config.toml`` (may not exist)."""
        return self._root / "config.toml"

    # ── Config ────────────────────────────────────────────────────

    def config(self) -> Config:
        """Load the project's configuration.

        Loads the project-local ``config.toml`` (if present) then
        layers environment variables on top with the ``NONO_`` prefix.
        If the project manifest defines ``default_provider`` or
        ``default_model``, those are set as defaults.

        Returns:
            Populated :class:`Config` instance.
        """
        defaults: dict[str, Any] = {}

        if self._manifest.default_provider:
            defaults["agent.default_provider"] = self._manifest.default_provider

        if self._manifest.default_model:
            defaults["agent.default_model"] = self._manifest.default_model

        cfg = Config(defaults=defaults)

        if self.config_file.is_file():
            cfg.load_file(str(self.config_file))

        cfg.load_env(prefix="NONO_")
        return cfg

    # ── Skills ────────────────────────────────────────────────────

    def load_skills(
        self,
        *,
        register: bool = False,
    ) -> list[Any]:
        """Load all SKILL.md files from the project's skills directory.

        Args:
            register: If ``True``, auto-register each skill in the
                global :data:`nono.agent.skill.registry`.

        Returns:
            List of loaded ``MarkdownSkill`` instances.
        """
        from .agent.skill import registry as global_registry
        from .agent.skill_loader import scan_skills_dir

        reg = global_registry if register else None
        return scan_skills_dir(self.skills_dir, register_in=reg)

    # ── Prompts / Templates ───────────────────────────────────────

    def list_prompts(self) -> list[Path]:
        """List JSON task definitions in the prompts directory.

        Returns:
            Sorted list of ``*.json`` file paths.
        """
        if not self.prompts_dir.is_dir():
            return []

        return sorted(self.prompts_dir.glob("*.json"))

    def list_templates(self) -> list[Path]:
        """List Jinja2 templates in the templates directory.

        Returns:
            Sorted list of ``*.j2`` file paths.
        """
        if not self.templates_dir.is_dir():
            return []

        return sorted(self.templates_dir.glob("*.j2"))

    def list_workflows(self) -> list[Path]:
        """List workflow definition files.

        Returns:
            Sorted list of ``*.json`` and ``*.yaml`` / ``*.yml`` files.
        """
        if not self.workflows_dir.is_dir():
            return []

        patterns = ("*.json", "*.yaml", "*.yml")
        files: list[Path] = []
        for pat in patterns:
            files.extend(self.workflows_dir.glob(pat))

        return sorted(files)

    # ── Data ──────────────────────────────────────────────────────

    def list_data(self, pattern: str = "*") -> list[Path]:
        """List data files in the project's data directory.

        Args:
            pattern: Glob pattern to filter files (default ``"*"``
                matches every file).  Common examples:
                ``"*.csv"``, ``"*.json"``, ``"*.txt"``.

        Returns:
            Sorted list of matching file paths (non-recursive).
        """
        if not self.data_dir.is_dir():
            return []

        return sorted(
            p for p in self.data_dir.glob(pattern) if p.is_file()
        )

    def list_data_recursive(self, pattern: str = "**/*") -> list[Path]:
        """List data files recursively in the project's data directory.

        Args:
            pattern: Glob pattern (default ``"**/*"`` matches all
                files in all subdirectories).

        Returns:
            Sorted list of matching file paths.
        """
        if not self.data_dir.is_dir():
            return []

        return sorted(
            p for p in self.data_dir.glob(pattern) if p.is_file()
        )

    def read_data(self, name: str, *, encoding: str = "utf-8") -> str:
        """Read a data file by name and return its contents as text.

        Args:
            name: Relative path within the data directory
                (e.g. ``"input.csv"`` or ``"subdir/file.txt"``).
            encoding: Text encoding (default ``utf-8``).

        Returns:
            File contents as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the resolved path escapes the data directory.
        """
        target = (self.data_dir / name).resolve()

        # Prevent path traversal
        try:
            target.relative_to(self.data_dir.resolve())
        except ValueError:
            raise ValueError(
                f"Path traversal detected: {name!r} escapes the data directory."
            ) from None

        if not target.is_file():
            raise FileNotFoundError(
                f"Data file not found: {target}"
            )

        return target.read_text(encoding=encoding)

    def read_data_bytes(self, name: str) -> bytes:
        """Read a data file by name and return its raw bytes.

        Args:
            name: Relative path within the data directory.

        Returns:
            File contents as bytes.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the resolved path escapes the data directory.
        """
        target = (self.data_dir / name).resolve()

        try:
            target.relative_to(self.data_dir.resolve())
        except ValueError:
            raise ValueError(
                f"Path traversal detected: {name!r} escapes the data directory."
            ) from None

        if not target.is_file():
            raise FileNotFoundError(
                f"Data file not found: {target}"
            )

        return target.read_bytes()

    # ── Dunder ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Project(name={self.name!r}, root={str(self._root)!r})"
        )

    def __str__(self) -> str:
        desc = f" — {self.description}" if self.description else ""
        return f"{self.name} v{self.version}{desc}"


# ── Public API ────────────────────────────────────────────────────────────────

def init_project(
    path: str | Path,
    *,
    name: str = "",
    description: str = "",
    default_provider: str = "google",
) -> Project:
    """Scaffold a new Nono project on disk.

    Creates the directory, ``nono.toml`` manifest, standard
    subdirectories (``skills/``, ``prompts/``, ``templates/``,
    ``workflows/``), and an optional ``config.toml``.

    Args:
        path: Directory to create the project in.
        name: Project name (defaults to the directory name).
        description: One-line description.
        default_provider: Default LLM provider.

    Returns:
        The newly created :class:`Project`.

    Raises:
        FileExistsError: If ``nono.toml`` already exists in *path*.
    """
    root = Path(path).resolve()
    manifest_path = root / MANIFEST_FILE

    if manifest_path.exists():
        raise FileExistsError(
            f"Project already exists: {manifest_path}"
        )

    root.mkdir(parents=True, exist_ok=True)

    # Create standard subdirectories
    for dirname in PROJECT_DIRS:
        (root / dirname).mkdir(exist_ok=True)

    # Create manifest
    manifest = ProjectManifest(
        name=name or root.name,
        description=description,
        default_provider=default_provider,
    )
    _write_manifest(manifest, manifest_path)

    # Create a starter config.toml with the default provider
    _create_starter_config(root, default_provider)

    logger.info("Initialised project %r at %s", manifest.name, root)
    return load_project(root)


def _create_starter_config(root: Path, provider: str) -> None:
    """Write a minimal ``config.toml`` for a new project.

    Args:
        root: Project root directory.
        provider: Default provider to feature in the config.
    """
    lines = [
        "# Nono project configuration",
        "# Provider and model overrides for this project.",
        "# API keys should be set via environment variables or apikey.txt,",
        "# NOT in this file.",
        "",
        "[agent]",
        f'default_provider = "{provider}"',
        "",
        "[rate_limits]",
        "delay_between_requests = 0.5",
        "",
    ]
    (root / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_project(path: str | Path) -> Project:
    """Load an existing Nono project from a directory.

    Args:
        path: Directory containing ``nono.toml``.

    Returns:
        A :class:`Project` instance.

    Raises:
        FileNotFoundError: If ``nono.toml`` is not found.
        ValueError: If the manifest is malformed.
    """
    root = Path(path).resolve()
    manifest_path = root / MANIFEST_FILE

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No {MANIFEST_FILE} found in {root}.  "
            f"Run 'nono init {root.name}' to create a project."
        )

    if tomllib is None:
        raise RuntimeError(
            "TOML library not available.  Install tomli for Python < 3.11."
        )

    with open(manifest_path, "rb") as f:
        data = tomllib.load(f)

    manifest = _parse_manifest(data)
    logger.info("Loaded project %r from %s", manifest.name, root)
    return Project(root, manifest)


def list_projects(path: str | Path | None = None) -> list[Project]:
    """Scan a directory for Nono projects (immediate children only).

    Looks for subdirectories containing a ``nono.toml`` manifest.

    Args:
        path: Directory to scan (defaults to ``cwd``).

    Returns:
        Sorted list of :class:`Project` instances.
    """
    base = Path(path).resolve() if path else Path.cwd().resolve()

    if not base.is_dir():
        return []

    projects: list[Project] = []

    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue

        manifest = child / MANIFEST_FILE
        if manifest.is_file():
            try:
                projects.append(load_project(child))
            except Exception as exc:
                logger.warning("Skipping %s: %s", child.name, exc)

    return projects
