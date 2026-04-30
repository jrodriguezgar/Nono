"""Tests for nono.project — Project manifest, scaffolding, and discovery.

Covers:
- init_project: directory creation, nono.toml, config.toml, subdirectories
- load_project: manifest parsing, edge cases
- Project class: properties, config, load_skills, list_prompts/templates/workflows
- _parse_manifest / _write_manifest: round-trip fidelity
- Error paths: duplicate init, missing manifest, malformed TOML

Run:
    uv run pytest tests/test_project.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow running as ``python tests/test_project.py``
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Mock jinjapromptpy if not installed
# ---------------------------------------------------------------------------
if "jinjapromptpy" not in sys.modules:
    _jpp_mock = MagicMock()
    sys.modules["jinjapromptpy"] = _jpp_mock
    sys.modules["jinjapromptpy.prompt_generator"] = _jpp_mock
    sys.modules["jinjapromptpy.prompt_template"] = _jpp_mock
    sys.modules["jinjapromptpy.batch_generator"] = _jpp_mock

from nono.project import (
    MANIFEST_FILE,
    PROJECT_DIRS,
    Project,
    ProjectManifest,
    _parse_manifest,
    _write_manifest,
    init_project,
    list_projects,
    load_project,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Return a fresh temporary directory for project scaffolding."""
    return tmp_path / "my-project"


@pytest.fixture
def initialised_project(project_dir: Path) -> Project:
    """Return a fully scaffolded project in *project_dir*."""
    return init_project(
        project_dir,
        name="demo",
        description="A demo project",
        default_provider="google",
    )


# ── init_project ──────────────────────────────────────────────────────────────


class TestInitProject:
    """Tests for :func:`init_project`."""

    def test_creates_manifest_file(self, project_dir: Path) -> None:
        init_project(project_dir)
        assert (project_dir / MANIFEST_FILE).is_file()

    def test_creates_config_toml(self, project_dir: Path) -> None:
        init_project(project_dir)
        assert (project_dir / "config.toml").is_file()

    def test_creates_standard_subdirectories(self, project_dir: Path) -> None:
        init_project(project_dir)
        for dirname in PROJECT_DIRS:
            assert (project_dir / dirname).is_dir()

    def test_default_name_from_directory(self, project_dir: Path) -> None:
        project = init_project(project_dir)
        assert project.name == project_dir.name

    def test_custom_name(self, project_dir: Path) -> None:
        project = init_project(project_dir, name="custom-name")
        assert project.name == "custom-name"

    def test_custom_description(self, project_dir: Path) -> None:
        project = init_project(
            project_dir, description="My custom desc"
        )
        assert project.description == "My custom desc"

    def test_default_provider(self, project_dir: Path) -> None:
        project = init_project(project_dir, default_provider="openai")
        assert project.manifest.default_provider == "openai"

    def test_raises_if_manifest_exists(
        self, initialised_project: Project
    ) -> None:
        with pytest.raises(FileExistsError, match="already exists"):
            init_project(initialised_project.root)

    def test_returns_project_instance(self, project_dir: Path) -> None:
        result = init_project(project_dir)
        assert isinstance(result, Project)

    def test_default_version_is_0_1_0(self, project_dir: Path) -> None:
        project = init_project(project_dir)
        assert project.version == "0.1.0"

    def test_config_toml_contains_provider(
        self, project_dir: Path
    ) -> None:
        init_project(project_dir, default_provider="deepseek")
        text = (project_dir / "config.toml").read_text(encoding="utf-8")
        assert 'default_provider = "deepseek"' in text


# ── load_project ──────────────────────────────────────────────────────────────


class TestLoadProject:
    """Tests for :func:`load_project`."""

    def test_loads_existing_project(
        self, initialised_project: Project
    ) -> None:
        loaded = load_project(initialised_project.root)
        assert loaded.name == initialised_project.name

    def test_raises_if_no_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=MANIFEST_FILE):
            load_project(tmp_path)

    def test_loads_custom_fields(self, tmp_path: Path) -> None:
        manifest_text = dedent("""\
            [project]
            name = "advanced"
            description = "Advanced project"
            version = "2.0.0"
            authors = ["Alice <alice@example.com>"]
            default_provider = "openai"
            default_model = "gpt-4o"
            skills_dir = "my_skills"
            prompts_dir = "my_prompts"
            templates_dir = "my_templates"
            workflows_dir = "my_workflows"
        """)
        (tmp_path / MANIFEST_FILE).write_text(
            manifest_text, encoding="utf-8"
        )
        project = load_project(tmp_path)
        assert project.name == "advanced"
        assert project.version == "2.0.0"
        assert project.manifest.default_model == "gpt-4o"
        assert project.manifest.skills_dir == "my_skills"
        assert project.manifest.authors == ["Alice <alice@example.com>"]

    def test_extra_sections_preserved(self, tmp_path: Path) -> None:
        manifest_text = dedent("""\
            [project]
            name = "extra-test"

            [providers.google]
            api_key_env = "MY_KEY"
        """)
        (tmp_path / MANIFEST_FILE).write_text(
            manifest_text, encoding="utf-8"
        )
        project = load_project(tmp_path)
        assert "providers" in project.manifest.extra


# ── list_projects ──────────────────────────────────────────────────────────


class TestListProjects:
    """Tests for :func:`list_projects`."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert list_projects(tmp_path) == []

    def test_finds_multiple_projects(self, tmp_path: Path) -> None:
        init_project(tmp_path / "alpha", name="Alpha")
        init_project(tmp_path / "beta", name="Beta")
        projects = list_projects(tmp_path)
        names = [p.name for p in projects]
        assert "Alpha" in names
        assert "Beta" in names
        assert len(projects) == 2

    def test_ignores_non_project_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "not-a-project").mkdir()
        init_project(tmp_path / "real", name="Real")
        projects = list_projects(tmp_path)
        assert len(projects) == 1
        assert projects[0].name == "Real"

    def test_ignores_files(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("hello", encoding="utf-8")
        assert list_projects(tmp_path) == []

    def test_returns_sorted(self, tmp_path: Path) -> None:
        init_project(tmp_path / "zzz")
        init_project(tmp_path / "aaa")
        projects = list_projects(tmp_path)
        roots = [p.root.name for p in projects]
        assert roots == sorted(roots)

    def test_nonexistent_path(self, tmp_path: Path) -> None:
        assert list_projects(tmp_path / "does-not-exist") == []


# ── Project class ─────────────────────────────────────────────────────────────


class TestProject:
    """Tests for the :class:`Project` instance."""

    def test_root_is_absolute(self, initialised_project: Project) -> None:
        assert initialised_project.root.is_absolute()

    def test_skills_dir_path(self, initialised_project: Project) -> None:
        assert initialised_project.skills_dir == (
            initialised_project.root / "skills"
        )

    def test_prompts_dir_path(self, initialised_project: Project) -> None:
        assert initialised_project.prompts_dir == (
            initialised_project.root / "prompts"
        )

    def test_templates_dir_path(
        self, initialised_project: Project
    ) -> None:
        assert initialised_project.templates_dir == (
            initialised_project.root / "templates"
        )

    def test_workflows_dir_path(
        self, initialised_project: Project
    ) -> None:
        assert initialised_project.workflows_dir == (
            initialised_project.root / "workflows"
        )

    def test_config_file_path(
        self, initialised_project: Project
    ) -> None:
        assert initialised_project.config_file == (
            initialised_project.root / "config.toml"
        )

    def test_repr(self, initialised_project: Project) -> None:
        r = repr(initialised_project)
        assert "Project(" in r
        assert "demo" in r

    def test_str(self, initialised_project: Project) -> None:
        s = str(initialised_project)
        assert "demo" in s
        assert "0.1.0" in s

    def test_str_with_description(
        self, initialised_project: Project
    ) -> None:
        s = str(initialised_project)
        assert "A demo project" in s


# ── Project.config() ─────────────────────────────────────────────────────────


class TestProjectConfig:
    """Tests for :meth:`Project.config`."""

    def test_config_returns_config_instance(
        self, initialised_project: Project
    ) -> None:
        from nono.config import Config

        cfg = initialised_project.config()
        assert isinstance(cfg, Config)

    def test_config_loads_config_toml(
        self, initialised_project: Project
    ) -> None:
        cfg = initialised_project.config()
        # The starter config.toml contains agent.default_provider
        val = cfg.get("agent.default_provider")
        assert val == "google"


# ── Project.load_skills() ────────────────────────────────────────────────────


class TestProjectLoadSkills:
    """Tests for :meth:`Project.load_skills`."""

    def test_empty_skills_dir_returns_empty_list(
        self, initialised_project: Project
    ) -> None:
        skills = initialised_project.load_skills()
        assert skills == []

    def test_loads_skill_from_project_dir(
        self, initialised_project: Project
    ) -> None:
        skill_dir = initialised_project.skills_dir / "greet"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            dedent("""\
                ---
                name: greet
                description: Greet the user.
                ---

                # Greet

                Say hello.
            """),
            encoding="utf-8",
        )
        skills = initialised_project.load_skills()
        assert len(skills) == 1
        assert skills[0].descriptor.name == "greet"


# ── Project.list_prompts/templates/workflows ──────────────────────────────────


class TestProjectListResources:
    """Tests for list_prompts, list_templates, list_workflows."""

    def test_list_prompts_empty(
        self, initialised_project: Project
    ) -> None:
        assert initialised_project.list_prompts() == []

    def test_list_prompts_finds_json_files(
        self, initialised_project: Project
    ) -> None:
        (initialised_project.prompts_dir / "task.json").write_text(
            "{}", encoding="utf-8"
        )
        prompts = initialised_project.list_prompts()
        assert len(prompts) == 1
        assert prompts[0].name == "task.json"

    def test_list_templates_empty(
        self, initialised_project: Project
    ) -> None:
        assert initialised_project.list_templates() == []

    def test_list_templates_finds_j2_files(
        self, initialised_project: Project
    ) -> None:
        (initialised_project.templates_dir / "prompt.j2").write_text(
            "Hello {{ name }}", encoding="utf-8"
        )
        templates = initialised_project.list_templates()
        assert len(templates) == 1
        assert templates[0].name == "prompt.j2"

    def test_list_workflows_empty(
        self, initialised_project: Project
    ) -> None:
        assert initialised_project.list_workflows() == []

    def test_list_workflows_finds_yaml_and_json(
        self, initialised_project: Project
    ) -> None:
        (initialised_project.workflows_dir / "flow.json").write_text(
            "{}", encoding="utf-8"
        )
        (initialised_project.workflows_dir / "flow.yaml").write_text(
            "steps: []", encoding="utf-8"
        )
        workflows = initialised_project.list_workflows()
        assert len(workflows) == 2


# ── Project.list_data / read_data ─────────────────────────────────────────────


class TestProjectDataResources:
    """Tests for data directory methods."""

    def test_data_dir_path(self, initialised_project: Project) -> None:
        assert initialised_project.data_dir == (
            initialised_project.root / "data"
        )

    def test_data_dir_created_on_init(
        self, initialised_project: Project
    ) -> None:
        assert initialised_project.data_dir.is_dir()

    def test_list_data_empty(
        self, initialised_project: Project
    ) -> None:
        files = initialised_project.list_data("*.csv")
        assert files == []

    def test_list_data_finds_csv(
        self, initialised_project: Project
    ) -> None:
        (initialised_project.data_dir / "input.csv").write_text(
            "a,b\n1,2\n", encoding="utf-8"
        )
        files = initialised_project.list_data("*.csv")
        assert len(files) == 1
        assert files[0].name == "input.csv"

    def test_list_data_default_pattern(
        self, initialised_project: Project
    ) -> None:
        (initialised_project.data_dir / "a.json").write_text(
            "{}", encoding="utf-8"
        )
        (initialised_project.data_dir / "b.txt").write_text(
            "hello", encoding="utf-8"
        )
        files = initialised_project.list_data()
        names = {f.name for f in files}
        assert "a.json" in names
        assert "b.txt" in names

    def test_list_data_recursive(
        self, initialised_project: Project
    ) -> None:
        sub = initialised_project.data_dir / "sub"
        sub.mkdir()
        (sub / "deep.csv").write_text("x\n1\n", encoding="utf-8")
        (initialised_project.data_dir / "top.csv").write_text(
            "y\n2\n", encoding="utf-8"
        )
        files = initialised_project.list_data_recursive("**/*.csv")
        assert len(files) == 2

    def test_read_data_text(
        self, initialised_project: Project
    ) -> None:
        (initialised_project.data_dir / "notes.txt").write_text(
            "hello world", encoding="utf-8"
        )
        content = initialised_project.read_data("notes.txt")
        assert content == "hello world"

    def test_read_data_bytes(
        self, initialised_project: Project
    ) -> None:
        payload = b"\x00\x01\x02"
        (initialised_project.data_dir / "bin.dat").write_bytes(payload)
        assert initialised_project.read_data_bytes("bin.dat") == payload

    def test_read_data_not_found(
        self, initialised_project: Project
    ) -> None:
        with pytest.raises(FileNotFoundError):
            initialised_project.read_data("missing.csv")

    def test_read_data_path_traversal_blocked(
        self, initialised_project: Project
    ) -> None:
        with pytest.raises(ValueError, match="traversal"):
            initialised_project.read_data("../../config.toml")

    def test_read_data_bytes_path_traversal_blocked(
        self, initialised_project: Project
    ) -> None:
        with pytest.raises(ValueError, match="traversal"):
            initialised_project.read_data_bytes("../../nono.toml")

    def test_custom_data_dir_in_manifest(self, tmp_path: Path) -> None:
        manifest_text = dedent("""\
            [project]
            name = "custom-data"
            data_dir = "datasets"
        """)
        root = tmp_path / "proj"
        root.mkdir()
        (root / MANIFEST_FILE).write_text(
            manifest_text, encoding="utf-8"
        )
        (root / "datasets").mkdir()
        (root / "datasets" / "f.csv").write_text("a\n1\n", encoding="utf-8")

        project = load_project(root)
        assert project.data_dir == root / "datasets"
        assert len(project.list_data("*.csv")) == 1


# ── _parse_manifest / _write_manifest round-trip ──────────────────────────────


class TestManifestRoundTrip:
    """Tests for manifest parsing and serialisation."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        original = ProjectManifest(
            name="roundtrip",
            description="Test roundtrip",
            version="1.2.3",
            authors=["Bob"],
            default_provider="google",
            default_model="gemini-pro",
        )
        path = tmp_path / MANIFEST_FILE
        _write_manifest(original, path)

        # Re-parse
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(path, "rb") as f:
            data = tomllib.load(f)

        restored = _parse_manifest(data)
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.version == original.version
        assert restored.authors == original.authors
        assert restored.default_provider == original.default_provider
        assert restored.default_model == original.default_model

    def test_parse_minimal(self) -> None:
        data: dict = {"project": {"name": "minimal"}}
        m = _parse_manifest(data)
        assert m.name == "minimal"
        assert m.version == "0.1.0"
        assert m.skills_dir == "skills"

    def test_parse_empty_project_section(self) -> None:
        m = _parse_manifest({})
        assert m.name == ""
        assert m.version == "0.1.0"
