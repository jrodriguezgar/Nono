"""
Modal sandbox client.

Executes code inside `Modal <https://modal.com>`_ serverless containers.
Requires the ``modal`` package and the ``MODAL_TOKEN_ID`` /
``MODAL_TOKEN_SECRET`` env-vars.

Example:
    >>> from nono.sandbox import ModalSandboxClient, SandboxRunConfig
    >>> client = ModalSandboxClient()
    >>> result = client.execute("print('hello from Modal')", SandboxRunConfig())
"""

from __future__ import annotations

import logging
import time

from ..base import (
    BaseSandboxClient,
    SandboxResult,
    SandboxRunConfig,
    SandboxStatus,
)

logger = logging.getLogger("Nono.Sandbox.Modal")

_MODAL_TOKEN_ID_ENV = "MODAL_TOKEN_ID"
_MODAL_TOKEN_SECRET_ENV = "MODAL_TOKEN_SECRET"


class ModalSandboxClient(BaseSandboxClient):
    """Sandbox client backed by Modal serverless containers.

    Args:
        api_key_env: Env-var holding the Modal token ID.
        image: Modal image specification (default ``"python:3.12"``).
    """

    def __init__(
        self,
        *,
        api_key_env: str = _MODAL_TOKEN_ID_ENV,
        image: str = "python:3.12",
    ) -> None:
        super().__init__(api_key_env=api_key_env)
        self._image = image

    def execute(
        self,
        code: str,
        config: SandboxRunConfig,
    ) -> SandboxResult:
        """Run *code* inside a Modal sandbox.

        Args:
            code: Python code to execute.
            config: Execution configuration.

        Returns:
            A ``SandboxResult`` with captured output and status.
        """
        try:
            import modal
        except ImportError as exc:
            raise ImportError(
                "Modal support requires the 'modal' package. "
                "Install it with: pip install modal"
            ) from exc

        start = time.monotonic()

        try:
            app = modal.App.lookup("nono-sandbox", create_if_missing=True)
            image = modal.Image.debian_slim(python_version="3.12")

            if config.packages:
                image = image.pip_install(*config.packages)

            sb = modal.Sandbox.create(
                image=image,
                timeout=config.timeout,
                app=app,
            )

            process = sb.exec("python", "-c", code)
            process.wait()

            stdout = process.stdout.read()
            stderr = process.stderr.read()
            exit_code = process.returncode or 0

            sandbox_id = sb.object_id if hasattr(sb, "object_id") else ""

            return SandboxResult(
                status=SandboxStatus.COMPLETED if exit_code == 0 else SandboxStatus.FAILED,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                sandbox_id=str(sandbox_id),
                duration_seconds=time.monotonic() - start,
            )

        except Exception as exc:
            logger.error("Modal execution failed: %s", exc)
            return SandboxResult(
                status=SandboxStatus.FAILED,
                stderr=str(exc),
                exit_code=1,
                duration_seconds=time.monotonic() - start,
            )

    def terminate(self, sandbox_id: str) -> None:
        """Terminate a Modal sandbox.

        Args:
            sandbox_id: Modal sandbox identifier.
        """
        try:
            import modal

            sb = modal.Sandbox.from_id(sandbox_id)
            sb.terminate()
        except Exception as exc:
            logger.warning("Modal terminate failed for %s: %s", sandbox_id, exc)
