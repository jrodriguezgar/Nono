"""
E2B sandbox client.

Executes code inside `E2B <https://e2b.dev>`_ cloud sandboxes.  Requires
the ``e2b-code-interpreter`` package and the ``E2B_API_KEY`` env-var.

Example:
    >>> from nono.sandbox import E2BSandboxClient, SandboxRunConfig
    >>> client = E2BSandboxClient()
    >>> result = client.execute("print('hello from E2B')", SandboxRunConfig())
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..base import (
    BaseSandboxClient,
    SandboxResult,
    SandboxRunConfig,
    SandboxStatus,
)

logger = logging.getLogger("Nono.Sandbox.E2B")

_E2B_API_KEY_ENV = "E2B_API_KEY"


class E2BSandboxClient(BaseSandboxClient):
    """Sandbox client backed by E2B cloud sandboxes.

    Args:
        api_key_env: Env-var holding the E2B API key.
        template: E2B sandbox template name (default ``"base"``).
    """

    def __init__(
        self,
        *,
        api_key_env: str = _E2B_API_KEY_ENV,
        template: str = "base",
    ) -> None:
        super().__init__(api_key_env=api_key_env)
        self._template = template

    def execute(
        self,
        code: str,
        config: SandboxRunConfig,
    ) -> SandboxResult:
        """Run *code* inside an E2B sandbox.

        Args:
            code: Python code to execute.
            config: Execution configuration.

        Returns:
            A ``SandboxResult`` with captured output and status.
        """
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError as exc:
            raise ImportError(
                "E2B support requires the 'e2b-code-interpreter' package. "
                "Install it with: pip install e2b-code-interpreter"
            ) from exc

        api_key = self._get_api_key()
        start = time.monotonic()
        sandbox = None

        try:
            sandbox = Sandbox(
                template=self._template,
                api_key=api_key,
                timeout=config.timeout,
                metadata=config.metadata,
            )

            self._upload_manifest(sandbox, config)
            self._install_packages(sandbox, config.packages)

            execution = sandbox.run_code(code, timeout=config.timeout)

            stdout = "\n".join(
                str(r.text) for r in (execution.results or []) if hasattr(r, "text")
            )

            if execution.logs:
                log_stdout = "\n".join(execution.logs.stdout or [])
                log_stderr = "\n".join(execution.logs.stderr or [])
            else:
                log_stdout, log_stderr = "", ""

            combined_stdout = "\n".join(filter(None, [log_stdout, stdout]))
            has_error = execution.error is not None

            output_files = self._collect_outputs(sandbox, config)

            return SandboxResult(
                status=SandboxStatus.FAILED if has_error else SandboxStatus.COMPLETED,
                stdout=combined_stdout,
                stderr=log_stderr if not has_error else str(execution.error),
                exit_code=1 if has_error else 0,
                output_files=output_files,
                sandbox_id=sandbox.sandbox_id,
                duration_seconds=time.monotonic() - start,
            )

        except Exception as exc:
            logger.error("E2B execution failed: %s", exc)
            return SandboxResult(
                status=SandboxStatus.FAILED,
                stderr=str(exc),
                exit_code=1,
                duration_seconds=time.monotonic() - start,
            )

        finally:
            if sandbox and not config.keep_alive:
                try:
                    sandbox.kill()
                except Exception:
                    logger.warning("Failed to kill E2B sandbox.")

    def terminate(self, sandbox_id: str) -> None:
        """Terminate an E2B sandbox by ID.

        Args:
            sandbox_id: E2B sandbox identifier.
        """
        try:
            from e2b_code_interpreter import Sandbox

            Sandbox.kill(sandbox_id, api_key=self._get_api_key())
        except Exception as exc:
            logger.warning("E2B terminate failed for %s: %s", sandbox_id, exc)

    # ── private helpers ───────────────────────────────────────────────

    def _upload_manifest(self, sandbox: Any, config: SandboxRunConfig) -> None:
        """Materialise manifest entries inside the sandbox."""
        from ..manifest import LocalDir, LocalFile

        if not config.manifest:
            return

        for mount_path, entry in config.manifest.entries.items():
            target = f"{config.working_dir}/{mount_path}"
            sandbox.commands.run(f"mkdir -p {target}")

            if isinstance(entry, LocalDir):
                self._upload_local_dir(sandbox, str(entry.src), target)
            elif isinstance(entry, LocalFile):
                self._upload_local_file(sandbox, str(entry.src), target)
            else:
                logger.info(
                    "E2B: cloud entry %s (%s) — download inside sandbox.",
                    mount_path,
                    entry.entry_type(),
                )

    def _upload_local_dir(self, sandbox: Any, src: str, target: str) -> None:
        """Upload a local directory recursively."""
        from pathlib import Path

        src_path = Path(src)

        if not src_path.is_dir():
            logger.warning("E2B: local dir %s does not exist, skipping.", src)
            return

        for file_path in src_path.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(src_path)
                remote_path = f"{target}/{relative}"
                sandbox.files.write(remote_path, file_path.read_bytes())

    def _upload_local_file(self, sandbox: Any, src: str, target: str) -> None:
        """Upload a single local file."""
        from pathlib import Path

        file_path = Path(src)

        if not file_path.is_file():
            logger.warning("E2B: local file %s does not exist, skipping.", src)
            return

        remote_path = f"{target}/{file_path.name}"
        sandbox.files.write(remote_path, file_path.read_bytes())

    def _install_packages(self, sandbox: Any, packages: list[str]) -> None:
        """Install Python packages inside the sandbox."""
        if not packages:
            return

        pkg_str = " ".join(packages)
        sandbox.commands.run(f"pip install {pkg_str}")

    def _collect_outputs(
        self,
        sandbox: Any,
        config: SandboxRunConfig,
    ) -> dict[str, bytes]:
        """Collect files from the output directory."""
        output_files: dict[str, bytes] = {}

        if not config.manifest or not config.manifest.output:
            return output_files

        output_path = f"{config.working_dir}/{config.manifest.output.path}"

        try:
            files = sandbox.files.list(output_path)

            for f in files:
                if not f.is_dir:
                    content = sandbox.files.read(f"{output_path}/{f.name}")
                    output_files[f.name] = (
                        content if isinstance(content, bytes) else content.encode()
                    )
        except Exception as exc:
            logger.warning("E2B: failed to collect outputs: %s", exc)

        return output_files
