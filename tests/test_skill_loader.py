"""Tests for the SKILL.md file loader (Anthropic standard)."""

from __future__ import annotations

import pytest
from pathlib import Path
from textwrap import dedent

from nono.agent.skill import SkillRegistry
from nono.agent.skill_loader import (
    MarkdownSkill,
    _parse_skill_md,
    _validate_frontmatter,
    load_skill_md,
    scan_skills_dir,
)
from nono.agent.base import BaseAgent
from nono.agent.tool import FunctionTool


# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_SKILL_MD = dedent("""\
    ---
    name: test-skill
    description: A test skill for unit testing.
    tags:
      - text
      - testing
    version: "2.0.0"
    temperature: 0.5
    output_format: json
    ---

    # Test Skill

    You are a test skill. Return JSON.

    ## Guidelines

    - Be concise.
    - Return valid JSON.
""")

MINIMAL_SKILL_MD = dedent("""\
    ---
    name: minimal
    description: Minimal skill with only required fields.
    ---

    Just do the thing.
""")


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with SKILL.md files."""
    # skill 1: standard layout
    d1 = tmp_path / "analyzing-logs"
    d1.mkdir()
    (d1 / "SKILL.md").write_text(dedent("""\
        ---
        name: analyzing-logs
        description: Analyze application logs for errors and patterns.
        tags:
          - logs
          - analysis
        ---

        # Analyze Logs

        Parse logs and identify errors, patterns, anomalies.
    """), encoding="utf-8")

    # skill 2: another subdirectory
    d2 = tmp_path / "generating-reports"
    d2.mkdir()
    (d2 / "SKILL.md").write_text(dedent("""\
        ---
        name: generating-reports
        description: Generate structured reports from data.
        tags:
          - reporting
        temperature: 0.2
        ---

        # Generate Reports

        Create structured markdown reports.
    """), encoding="utf-8")

    return tmp_path


@pytest.fixture
def single_skill_file(tmp_path: Path) -> Path:
    """Create a single SKILL.md file."""
    skill_path = tmp_path / "my-skill" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(VALID_SKILL_MD, encoding="utf-8")
    return skill_path


# ── _parse_skill_md ──────────────────────────────────────────────────────────

class TestParseFrontmatter:

    def test_valid_frontmatter(self):
        meta, body = _parse_skill_md(VALID_SKILL_MD)
        assert meta["name"] == "test-skill"
        assert meta["description"] == "A test skill for unit testing."
        assert meta["tags"] == ["text", "testing"]
        assert "# Test Skill" in body

    def test_minimal_frontmatter(self):
        meta, body = _parse_skill_md(MINIMAL_SKILL_MD)
        assert meta["name"] == "minimal"
        assert "Just do the thing." in body

    def test_no_frontmatter(self):
        with pytest.raises(ValueError, match="YAML frontmatter"):
            _parse_skill_md("# No frontmatter here\nJust markdown.")

    def test_incomplete_frontmatter(self):
        with pytest.raises(ValueError, match="YAML frontmatter"):
            _parse_skill_md("---\nname: test\n# Missing closing ---")

    def test_invalid_yaml(self):
        bad = "---\n: invalid: yaml: [broken\n---\nBody."
        with pytest.raises(ValueError, match="Invalid YAML"):
            _parse_skill_md(bad)

    def test_non_dict_yaml(self):
        bad = "---\n- a list\n- not a dict\n---\nBody."
        with pytest.raises(ValueError, match="must be a mapping"):
            _parse_skill_md(bad)


# ── _validate_frontmatter ────────────────────────────────────────────────────

class TestValidateFrontmatter:

    def test_missing_name(self):
        with pytest.raises(ValueError, match="name"):
            _validate_frontmatter({"description": "test"})

    def test_missing_description(self):
        with pytest.raises(ValueError, match="description"):
            _validate_frontmatter({"name": "test"})

    def test_empty_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_frontmatter({"name": "", "description": "test"})

    def test_name_too_long(self):
        with pytest.raises(ValueError, match="<= 64"):
            _validate_frontmatter({
                "name": "a" * 65,
                "description": "test",
            })

    def test_name_invalid_chars(self):
        with pytest.raises(ValueError, match="lowercase"):
            _validate_frontmatter({
                "name": "Invalid_Name",
                "description": "test",
            })

    def test_name_uppercase(self):
        with pytest.raises(ValueError, match="lowercase"):
            _validate_frontmatter({
                "name": "MySkill",
                "description": "test",
            })

    def test_description_too_long(self):
        with pytest.raises(ValueError, match="<= 1024"):
            _validate_frontmatter({
                "name": "test",
                "description": "x" * 1025,
            })

    def test_valid_passes(self):
        _validate_frontmatter({
            "name": "valid-name-123",
            "description": "A valid description.",
        })

    def test_name_with_numbers_and_hyphens(self):
        _validate_frontmatter({
            "name": "my-skill-v2",
            "description": "Valid.",
        })


# ── MarkdownSkill ─────────────────────────────────────────────────────────────

class TestMarkdownSkill:

    def test_descriptor_from_frontmatter(self):
        meta, body = _parse_skill_md(VALID_SKILL_MD)
        skill = MarkdownSkill(meta, body)
        d = skill.descriptor
        assert d.name == "test-skill"
        assert d.description == "A test skill for unit testing."
        assert d.version == "2.0.0"
        assert d.tags == ("text", "testing")

    def test_instruction_property(self):
        meta, body = _parse_skill_md(VALID_SKILL_MD)
        skill = MarkdownSkill(meta, body)
        assert "# Test Skill" in skill.instruction
        assert "Be concise" in skill.instruction

    def test_build_agent_returns_agent(self):
        meta, body = _parse_skill_md(VALID_SKILL_MD)
        skill = MarkdownSkill(meta, body)
        agent = skill.build_agent()
        assert isinstance(agent, BaseAgent)
        assert agent.name == "test_skill"  # hyphens converted to underscores

    def test_build_agent_with_overrides(self):
        meta, body = _parse_skill_md(VALID_SKILL_MD)
        skill = MarkdownSkill(meta, body)
        agent = skill.build_agent(provider="openai", temperature=0.9)
        assert isinstance(agent, BaseAgent)

    def test_source_path(self):
        meta, body = _parse_skill_md(VALID_SKILL_MD)
        p = Path("/fake/path/SKILL.md")
        skill = MarkdownSkill(meta, body, source_path=p)
        assert skill.source_path == p

    def test_repr(self):
        meta, body = _parse_skill_md(VALID_SKILL_MD)
        skill = MarkdownSkill(meta, body)
        assert "test-skill" in repr(skill)
        assert "MarkdownSkill" in repr(skill)

    def test_defaults_for_optional_fields(self):
        meta, body = _parse_skill_md(MINIMAL_SKILL_MD)
        skill = MarkdownSkill(meta, body)
        d = skill.descriptor
        assert d.version == "1.0.0"
        assert d.tags == ()
        assert d.input_keys == ("input",)
        assert d.output_keys == ("output",)

    def test_tags_from_string(self):
        """Tags can be a comma-separated string."""
        meta = {
            "name": "test",
            "description": "Test.",
            "tags": "a, b, c",
        }
        skill = MarkdownSkill(meta, "body")
        assert skill.descriptor.tags == ("a", "b", "c")

    def test_as_tool(self):
        meta, body = _parse_skill_md(VALID_SKILL_MD)
        skill = MarkdownSkill(meta, body)
        tool = skill.as_tool()
        assert tool.name == "test-skill"
        assert "test skill" in tool.description.lower()


# ── load_skill_md ─────────────────────────────────────────────────────────────

class TestLoadSkillMd:

    def test_load_valid_file(self, single_skill_file: Path):
        skill = load_skill_md(single_skill_file)
        assert isinstance(skill, MarkdownSkill)
        assert skill.descriptor.name == "test-skill"
        assert skill.source_path == single_skill_file.resolve()

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_skill_md("/nonexistent/SKILL.md")

    def test_load_with_registry(self, single_skill_file: Path):
        reg = SkillRegistry()
        load_skill_md(single_skill_file, register_in=reg)
        assert "test-skill" in reg

    def test_load_without_registry(self, single_skill_file: Path):
        reg = SkillRegistry()
        skill = load_skill_md(single_skill_file)
        assert "test-skill" not in reg
        assert skill.descriptor.name == "test-skill"

    def test_load_invalid_content(self, tmp_path: Path):
        bad_file = tmp_path / "SKILL.md"
        bad_file.write_text("No frontmatter here.", encoding="utf-8")
        with pytest.raises(ValueError):
            load_skill_md(bad_file)


# ── scan_skills_dir ──────────────────────────────────────────────────────────

class TestScanSkillsDir:

    def test_scan_loads_all(self, skill_dir: Path):
        skills = scan_skills_dir(skill_dir)
        assert len(skills) == 2
        names = {s.descriptor.name for s in skills}
        assert names == {"analyzing-logs", "generating-reports"}

    def test_scan_with_registry(self, skill_dir: Path):
        reg = SkillRegistry()
        scan_skills_dir(skill_dir, register_in=reg)
        assert "analyzing-logs" in reg
        assert "generating-reports" in reg

    def test_scan_empty_dir(self, tmp_path: Path):
        skills = scan_skills_dir(tmp_path)
        assert skills == []

    def test_scan_nonexistent_dir(self):
        skills = scan_skills_dir("/nonexistent/dir")
        assert skills == []

    def test_scan_skips_invalid(self, skill_dir: Path):
        """Invalid SKILL.md files are skipped with a warning."""
        bad_dir = skill_dir / "broken-skill"
        bad_dir.mkdir()
        (bad_dir / "SKILL.md").write_text("no frontmatter", encoding="utf-8")

        skills = scan_skills_dir(skill_dir)
        # 2 valid + 0 invalid (broken is skipped)
        assert len(skills) == 2

    def test_scan_flat_layout(self, tmp_path: Path):
        """SKILL.md directly in the root directory."""
        (tmp_path / "SKILL.md").write_text(MINIMAL_SKILL_MD, encoding="utf-8")
        skills = scan_skills_dir(tmp_path)
        assert len(skills) == 1
        assert skills[0].descriptor.name == "minimal"

    def test_scan_mixed_layout(self, skill_dir: Path):
        """Both subdirectory and root SKILL.md files."""
        (skill_dir / "SKILL.md").write_text(
            MINIMAL_SKILL_MD, encoding="utf-8"
        )
        skills = scan_skills_dir(skill_dir)
        assert len(skills) == 3  # 2 subdirs + 1 root


# ── Integration: built-in SKILL.md files ──────────────────────────────────────

class TestBuiltInSkillMd:

    @pytest.fixture
    def skills_dir(self) -> Path:
        """Path to the built-in SKILL.md files."""
        return Path(__file__).resolve().parent.parent / "nono" / "agent" / "skills"

    def test_built_in_skill_md_files_exist(self, skills_dir: Path):
        expected = [
            "summarizing-text",
            "classifying-data",
            "extracting-data",
            "reviewing-code",
            "translating-text",
        ]
        for name in expected:
            skill_file = skills_dir / name / "SKILL.md"
            assert skill_file.is_file(), f"Missing: {skill_file}"

    def test_built_in_skill_md_loadable(self, skills_dir: Path):
        skills = scan_skills_dir(skills_dir)
        loaded_names = {s.descriptor.name for s in skills}
        expected = {
            "summarizing-text",
            "classifying-data",
            "extracting-data",
            "reviewing-code",
            "translating-text",
        }
        assert expected <= loaded_names

    def test_built_in_skill_md_valid_names(self, skills_dir: Path):
        """All SKILL.md names follow Anthropic naming conventions."""
        import re
        skills = scan_skills_dir(skills_dir)
        for skill in skills:
            name = skill.descriptor.name
            assert len(name) <= 64
            assert re.match(r"^[a-z0-9-]+$", name), f"Invalid name: {name}"

    def test_built_in_skill_md_have_descriptions(self, skills_dir: Path):
        skills = scan_skills_dir(skills_dir)
        for skill in skills:
            d = skill.descriptor
            assert len(d.description) > 10, f"Short description: {d.name}"
            assert len(d.description) <= 1024

    def test_built_in_skill_md_have_tags(self, skills_dir: Path):
        skills = scan_skills_dir(skills_dir)
        for skill in skills:
            assert len(skill.descriptor.tags) > 0, (
                f"No tags: {skill.descriptor.name}"
            )

    def test_built_in_skill_md_build_agent(self, skills_dir: Path):
        skills = scan_skills_dir(skills_dir)
        for skill in skills:
            agent = skill.build_agent()
            assert isinstance(agent, BaseAgent)


# ── Tools in SKILL.md ────────────────────────────────────────────────────────

SKILL_MD_WITH_TOOLS = dedent("""\
    ---
    name: form-processor
    description: Process PDF forms with validation tools.
    tools:
      - name: validate_form
        script: scripts/validate_form.py
        description: Validate form field mappings.
      - name: analyze_pdf
        script: scripts/analyze_pdf.py
        description: Extract form fields from PDF.
    ---

    # Form Processor

    Use the validate_form and analyze_pdf tools.
""")


@pytest.fixture
def skill_with_tools_dir(tmp_path: Path) -> Path:
    """Create a skill directory with tool scripts."""
    skill_dir = tmp_path / "form-processing"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(SKILL_MD_WITH_TOOLS, encoding="utf-8")

    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "validate_form.py").write_text(
        dedent("""\
            def main(input: str) -> str:
                return f"validated: {input}"
        """),
        encoding="utf-8",
    )
    (scripts_dir / "analyze_pdf.py").write_text(
        dedent("""\
            def main(input: str) -> str:
                return f"fields: {input}"
        """),
        encoding="utf-8",
    )
    return skill_dir


class TestMarkdownSkillTools:

    def test_yaml_tools_loaded(self, skill_with_tools_dir: Path):
        """Tools declared in YAML frontmatter are loaded as FunctionTools."""
        skill_md = skill_with_tools_dir / "SKILL.md"
        skill = load_skill_md(skill_md)
        tools = skill.build_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"validate_form", "analyze_pdf"}

    def test_yaml_tools_callable(self, skill_with_tools_dir: Path):
        """YAML-loaded tools invoke the script's main() function."""
        skill_md = skill_with_tools_dir / "SKILL.md"
        skill = load_skill_md(skill_md)
        tools = skill.build_tools()
        validate = next(t for t in tools if t.name == "validate_form")
        result = validate.invoke({"input": "test"})
        assert result == "validated: test"

    def test_programmatic_tools(self):
        """Tools passed programmatically are available via build_tools()."""
        def my_fn(input: str) -> str:
            return f"result: {input}"

        tool = FunctionTool(fn=my_fn, name="my_tool", description="A tool")
        meta, body = _parse_skill_md(MINIMAL_SKILL_MD)
        skill = MarkdownSkill(meta, body, tools=[tool])
        tools = skill.build_tools()
        assert len(tools) == 1
        assert tools[0].name == "my_tool"

    def test_combined_yaml_and_programmatic(self, skill_with_tools_dir: Path):
        """YAML tools and programmatic tools are combined."""
        def extra_fn(input: str) -> str:
            return "extra"

        extra_tool = FunctionTool(fn=extra_fn, name="extra", description="Extra")
        skill_md = skill_with_tools_dir / "SKILL.md"
        skill = load_skill_md(skill_md, tools=[extra_tool])
        tools = skill.build_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"validate_form", "analyze_pdf", "extra"}

    def test_missing_script_raises(self, tmp_path: Path):
        """Non-existent script file raises FileNotFoundError."""
        skill_md_text = dedent("""\
            ---
            name: broken
            description: Skill with missing script.
            tools:
              - name: missing
                script: scripts/ghost.py
                description: Does not exist.
            ---

            Nothing here.
        """)
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(skill_md_text, encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="ghost.py"):
            load_skill_md(skill_file)

    def test_script_without_main_raises(self, tmp_path: Path):
        """Script without main() function raises ValueError."""
        skill_dir = tmp_path / "bad-script"
        skill_dir.mkdir()
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "no_main.py").write_text(
            "x = 42\n", encoding="utf-8"
        )
        skill_md_text = dedent("""\
            ---
            name: bad-script
            description: Skill with script missing main.
            tools:
              - name: no_main
                script: scripts/no_main.py
                description: Has no main function.
            ---

            Body.
        """)
        (skill_dir / "SKILL.md").write_text(skill_md_text, encoding="utf-8")
        with pytest.raises(ValueError, match="main"):
            load_skill_md(skill_dir / "SKILL.md")

    def test_tool_entry_missing_fields(self, tmp_path: Path):
        """Tool entry without required fields raises ValueError."""
        skill_md_text = dedent("""\
            ---
            name: bad-tools
            description: Tool entry missing fields.
            tools:
              - name: incomplete
            ---

            Body.
        """)
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(skill_md_text, encoding="utf-8")
        with pytest.raises(ValueError, match="script"):
            load_skill_md(skill_file)

    def test_no_tools_returns_empty(self):
        """Skill without tools field returns empty build_tools()."""
        meta, body = _parse_skill_md(MINIMAL_SKILL_MD)
        skill = MarkdownSkill(meta, body)
        assert skill.build_tools() == []

    def test_scan_loads_tools(self, skill_with_tools_dir: Path):
        """scan_skills_dir loads tools from YAML frontmatter."""
        parent = skill_with_tools_dir.parent
        skills = scan_skills_dir(parent)
        assert len(skills) == 1
        tools = skills[0].build_tools()
        assert len(tools) == 2

    def test_path_traversal_rejected(self, tmp_path: Path):
        """Script paths outside the skill directory are rejected."""
        # Create an "evil" script outside the skill dir
        evil_script = tmp_path / "evil.py"
        evil_script.write_text("def main(input): return 'pwned'", encoding="utf-8")

        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_md_text = dedent("""\
            ---
            name: evil-skill
            description: Tries path traversal.
            tools:
              - name: evil
                script: ../evil.py
                description: Path traversal attempt.
            ---

            Body.
        """)
        (skill_dir / "SKILL.md").write_text(skill_md_text, encoding="utf-8")
        with pytest.raises(ValueError, match="inside the skill"):
            load_skill_md(skill_dir / "SKILL.md")


# ── Agent Skills standard fields ─────────────────────────────────────────────

FULL_STANDARD_SKILL_MD = dedent("""\
    ---
    name: pdf-processing
    description: Extract PDF text, fill forms, merge files. Use when handling PDFs.
    license: Apache-2.0
    compatibility: Requires Python 3.10+ and pdfplumber
    metadata:
      author: example-org
      version: "1.0"
    allowed-tools: Read Grep Bash
    tags:
      - pdf
      - extraction
    ---

    # PDF Processing

    Extract text and tables from PDF files.

    ## References

    See [references/REFERENCE.md](references/REFERENCE.md) for API details.
    See [assets/template.json](assets/template.json) for output format.
""")


@pytest.fixture
def full_standard_skill_dir(tmp_path: Path) -> Path:
    """Create a skill directory following the full Agent Skills standard."""
    skill_dir = tmp_path / "pdf-processing"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        FULL_STANDARD_SKILL_MD, encoding="utf-8"
    )

    # references/ directory
    refs_dir = skill_dir / "references"
    refs_dir.mkdir()
    (refs_dir / "REFERENCE.md").write_text(
        "# API Reference\n\nDetailed API docs.", encoding="utf-8"
    )
    (refs_dir / "FORMS.md").write_text(
        "# Form Templates\n\nForm definitions.", encoding="utf-8"
    )

    # assets/ directory
    assets_dir = skill_dir / "assets"
    assets_dir.mkdir()
    (assets_dir / "template.json").write_text(
        '{"output": "schema"}', encoding="utf-8"
    )
    (assets_dir / "logo.txt").write_text("LOGO", encoding="utf-8")

    # scripts/ directory
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "extract.py").write_text(
        'def main(input: str) -> str:\n    return "extracted"',
        encoding="utf-8",
    )

    return skill_dir


class TestStandardFrontmatterFields:
    """Tests for Agent Skills standard fields: license, compatibility, metadata,
    allowed-tools."""

    def test_license_field(self, full_standard_skill_dir: Path):
        skill = load_skill_md(full_standard_skill_dir / "SKILL.md")
        assert skill.descriptor.license == "Apache-2.0"

    def test_compatibility_field(self, full_standard_skill_dir: Path):
        skill = load_skill_md(full_standard_skill_dir / "SKILL.md")
        assert skill.descriptor.compatibility == (
            "Requires Python 3.10+ and pdfplumber"
        )

    def test_metadata_field(self, full_standard_skill_dir: Path):
        skill = load_skill_md(full_standard_skill_dir / "SKILL.md")
        assert skill.descriptor.metadata == {
            "author": "example-org",
            "version": "1.0",
        }

    def test_allowed_tools_from_string(self, full_standard_skill_dir: Path):
        skill = load_skill_md(full_standard_skill_dir / "SKILL.md")
        assert skill.descriptor.allowed_tools == ("Read", "Grep", "Bash")

    def test_allowed_tools_from_list(self, tmp_path: Path):
        """allowed-tools as a YAML list."""
        md = dedent("""\
            ---
            name: reader
            description: Read files.
            allowed-tools:
              - Read
              - Grep
            ---

            Read things.
        """)
        p = tmp_path / "SKILL.md"
        p.write_text(md, encoding="utf-8")
        skill = load_skill_md(p)
        assert skill.descriptor.allowed_tools == ("Read", "Grep")

    def test_defaults_when_missing(self):
        meta, body = _parse_skill_md(MINIMAL_SKILL_MD)
        skill = MarkdownSkill(meta, body)
        assert skill.descriptor.license == ""
        assert skill.descriptor.compatibility == ""
        assert skill.descriptor.metadata == {}
        assert skill.descriptor.allowed_tools == ()

    def test_validation_compatibility_too_long(self):
        with pytest.raises(ValueError, match="<= 500"):
            _validate_frontmatter({
                "name": "test",
                "description": "test",
                "compatibility": "x" * 501,
            })

    def test_validation_metadata_not_dict(self):
        with pytest.raises(ValueError, match="mapping"):
            _validate_frontmatter({
                "name": "test",
                "description": "test",
                "metadata": "not a dict",
            })

    def test_validation_metadata_non_string_values(self):
        with pytest.raises(ValueError, match="string"):
            _validate_frontmatter({
                "name": "test",
                "description": "test",
                "metadata": {"key": 123},
            })


# ── Supporting directories & file references ─────────────────────────────────

class TestSupportingDirectories:
    """Tests for references/, assets/, scripts/ directory listing."""

    def test_list_references(self, full_standard_skill_dir: Path):
        skill = load_skill_md(full_standard_skill_dir / "SKILL.md")
        refs = skill.list_references()
        names = [p.name for p in refs]
        assert "FORMS.md" in names
        assert "REFERENCE.md" in names

    def test_list_assets(self, full_standard_skill_dir: Path):
        skill = load_skill_md(full_standard_skill_dir / "SKILL.md")
        assets = skill.list_assets()
        names = [p.name for p in assets]
        assert "template.json" in names
        assert "logo.txt" in names

    def test_list_scripts(self, full_standard_skill_dir: Path):
        skill = load_skill_md(full_standard_skill_dir / "SKILL.md")
        scripts = skill.list_scripts()
        assert len(scripts) == 1
        assert scripts[0].name == "extract.py"

    def test_list_references_empty(self, single_skill_file: Path):
        """No references/ directory returns empty list."""
        skill = load_skill_md(single_skill_file)
        assert skill.list_references() == []

    def test_list_assets_empty(self, single_skill_file: Path):
        skill = load_skill_md(single_skill_file)
        assert skill.list_assets() == []

    def test_list_dirs_no_source_path(self):
        """Skills without source_path return empty lists."""
        meta, body = _parse_skill_md(MINIMAL_SKILL_MD)
        skill = MarkdownSkill(meta, body)
        assert skill.list_references() == []
        assert skill.list_assets() == []
        assert skill.list_scripts() == []

    def test_skill_dir_property(self, full_standard_skill_dir: Path):
        skill = load_skill_md(full_standard_skill_dir / "SKILL.md")
        assert skill.skill_dir == full_standard_skill_dir

    def test_skill_dir_none_without_source(self):
        meta, body = _parse_skill_md(MINIMAL_SKILL_MD)
        skill = MarkdownSkill(meta, body)
        assert skill.skill_dir is None


class TestLoadResource:
    """Tests for load_resource() file reference loading."""

    def test_load_reference_file(self, full_standard_skill_dir: Path):
        skill = load_skill_md(full_standard_skill_dir / "SKILL.md")
        content = skill.load_resource("references/REFERENCE.md")
        assert "API Reference" in content

    def test_load_asset_file(self, full_standard_skill_dir: Path):
        skill = load_skill_md(full_standard_skill_dir / "SKILL.md")
        content = skill.load_resource("assets/template.json")
        assert '"output"' in content

    def test_load_nonexistent_raises(self, full_standard_skill_dir: Path):
        skill = load_skill_md(full_standard_skill_dir / "SKILL.md")
        with pytest.raises(FileNotFoundError, match="ghost.md"):
            skill.load_resource("references/ghost.md")

    def test_load_resource_path_traversal(
        self, full_standard_skill_dir: Path, tmp_path: Path
    ):
        """Path traversal outside skill directory is blocked."""
        secret = tmp_path / "secret.txt"
        secret.write_text("secret!", encoding="utf-8")

        skill = load_skill_md(full_standard_skill_dir / "SKILL.md")
        with pytest.raises(ValueError, match="escapes"):
            skill.load_resource("../../secret.txt")

    def test_load_resource_no_source_path(self):
        meta, body = _parse_skill_md(MINIMAL_SKILL_MD)
        skill = MarkdownSkill(meta, body)
        with pytest.raises(FileNotFoundError, match="source path"):
            skill.load_resource("references/any.md")
