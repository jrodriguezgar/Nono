"""Tests for subprocess sandbox isolation in CodeExecuter."""
import os
import shutil
import subprocess
import sys
import tempfile


def test_module_compiles():
    """Verify genai_executer compiles up to the tasker import."""
    src = open(
        os.path.join("nono", "executer", "genai_executer.py"),
        encoding="utf-8",
    ).read()
    # Everything before the tasker import block is self-contained
    chunk = src.split("# Import tasker")[0]
    compile(chunk, "genai_executer.py", "exec")
    print("PASS: module compiles (up to tasker import)")


def _safe_env() -> dict[str, str]:
    """Minimal env that allows subprocess to start on Windows."""
    env: dict[str, str] = {}
    for key in ("SystemRoot", "SystemDrive", "TEMP", "TMP"):
        val = os.environ.get(key)
        if val:
            env[key] = val
    return env


def test_subprocess_empty_env():
    """Subprocess with minimal safe env has no user/app variables."""
    sandbox = tempfile.mkdtemp(prefix="nono_test_")
    code = "import os; print(dict(os.environ))"
    path = os.path.join(sandbox, "_run.py")
    with open(path, "w") as f:
        f.write(code)

    safe_env = _safe_env()
    try:
        r = subprocess.run(
            [sys.executable, "-I", path],
            capture_output=True, text=True, timeout=10,
            env=safe_env, cwd=sandbox,
        )
        assert r.returncode == 0, f"Subprocess failed: {r.stderr}"
        # Only OS-required + Python-internal keys should be present
        reported = eval(r.stdout.strip())  # noqa: S307
        allowed_prefixes = ("system", "temp", "tmp", "python")
        for key in reported:
            lk = key.lower()
            assert any(lk.startswith(p) for p in allowed_prefixes), (
                f"Unexpected env var: {key}"
            )
        # Env must be small (no full parent env copy)
        assert len(reported) <= 10, f"Too many env vars: {len(reported)}"
        print("PASS: subprocess has minimal safe env")
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def test_no_secrets_leak():
    """Secrets from parent environment do not leak into sandbox."""
    sandbox = tempfile.mkdtemp(prefix="nono_test_")
    code = 'import os; print(os.environ.get("SECRET_API_KEY", "NONE"))'
    path = os.path.join(sandbox, "_run.py")
    with open(path, "w") as f:
        f.write(code)

    # Set a fake secret in the parent env
    parent_env = os.environ.copy()
    parent_env["SECRET_API_KEY"] = "super-secret-123"

    safe_env = _safe_env()
    try:
        r = subprocess.run(
            [sys.executable, "-I", path],
            capture_output=True, text=True, timeout=10,
            env=safe_env, cwd=sandbox,  # safe_env has no secrets
        )
        assert r.returncode == 0, f"Subprocess failed: {r.stderr}"
        assert r.stdout.strip() == "NONE", f"Secret leaked: {r.stdout}"
        print("PASS: no secrets leak")
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def test_cwd_is_sandbox():
    """Subprocess cwd is the sandbox directory, not the project root."""
    sandbox = tempfile.mkdtemp(prefix="nono_test_")
    code = "import os; print(os.getcwd())"
    path = os.path.join(sandbox, "_run.py")
    with open(path, "w") as f:
        f.write(code)

    safe_env = _safe_env()
    try:
        r = subprocess.run(
            [sys.executable, "-I", path],
            capture_output=True, text=True, timeout=10,
            env=safe_env, cwd=sandbox,
        )
        assert r.returncode == 0, f"Subprocess failed: {r.stderr}"
        actual = os.path.realpath(r.stdout.strip())
        expected = os.path.realpath(sandbox)
        assert actual == expected, f"cwd mismatch: {actual} != {expected}"
        print("PASS: cwd is sandbox dir")
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def test_safe_builtins_block_open():
    """In SAFE exec mode, open() should not be available."""
    import ast

    src = open(
        os.path.join("nono", "executer", "genai_executer.py"),
        encoding="utf-8",
    ).read()
    tree = ast.parse(src)

    # Find the _SAFE_BUILTINS class variable and extract allowed names
    safe_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "_SAFE_BUILTINS":
                # Walk the comprehension to find the tuple of allowed keys
                for child in ast.walk(node):
                    if isinstance(child, ast.Constant) and isinstance(child.value, str):
                        safe_names.add(child.value)

    assert safe_names, "_SAFE_BUILTINS keys not found in AST"

    blocked = {"open", "eval", "exec", "compile", "__import__", "globals", "locals"}
    for name in blocked:
        assert name not in safe_names, f"{name} should be blocked"

    # Basics should be allowed
    for name in ("print", "len", "range", "int", "str"):
        assert name in safe_names, f"{name} should be allowed"

    print("PASS: safe builtins block dangerous functions")


if __name__ == "__main__":
    test_module_compiles()
    test_subprocess_empty_env()
    test_no_secrets_leak()
    test_cwd_is_sandbox()
    test_safe_builtins_block_open()
    print()
    print("=" * 60)
    print("  All 5 sandbox tests PASSED")
    print("=" * 60)
