"""
Blaxel sandbox client.

Executes code inside `Blaxel <https://blaxel.ai>`_ sandboxes.  Requires
the ``blaxel`` package and the ``BLAXEL_API_KEY`` env-var.

Example:
    >>> from nono.sandbox import BlaxelSandboxClient, SandboxRunConfig
    >>> client = BlaxelSandboxClient()
    >>> result = client.execute("print('hello from Blaxel')", SandboxRunConfig())
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

logger = logging.getLogger("Nono.Sandbox.Blaxel")

_BLAXEL_API_KEY_ENV = "BLAXEL_API_KEY"


class BlaxelSandboxClient(BaseSandboxClient):
    """Sandbox client backed by Blaxel cloud environments.

    Args:
        api_key_env: Env-var holding the Blaxel API key.
    """

    def __init__(self, *, api_key_env: str = _BLAXEL_API_KEY_ENV) -> None:
        super().__init__(api_key_env=api_key_env)

    def execute(
        self,
        code: str,
        config: SandboxRunConfig,
    ) -> SandboxResult:
        """Run *code* inside a Blaxel sandbox.

        Args:
            code: Python code to execute.
            config: Execution configuration.

        Returns:
            A ``SandboxResult`` with captured output and status.
        """
        try:
            from blaxel.sandbox import SandboxInstance
        except ImportError as exc:
            raise ImportError(
                "Blaxel support requires the 'blaxel' package. "
                "Install it with: pip install blaxel"
            ) from exc

        api_key = self._get_api_key()
        start = time.monotonic()

        try:
            sandbox = SandboxInstance.create(
                api_key=api_key,
                timeout=config.timeout,
                environment=config.environment,
            )

            if config.packages:
                pkg_str = " ".join(config.packages)
                sandbox.exec(f"pip install {pkg_str}")

            result = sandbox.exec(f"python -c {code!r}")

            stdout = result.stdout if hasattr(result, "stdout") else str(result)
            stderr = result.stderr if hasattr(result, "stderr") else ""
            exit_code = result.exit_code if hasattr(result, "exit_code") else 0

            sandbox_id = sandbox.id if hasattr(sandbox, "id") else ""

            return SandboxResult(
                status=SandboxStatus.COMPLETED if exit_code == 0 else SandboxStatus.FAILED,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                sandbox_id=str(sandbox_id),
                duration_seconds=time.monotonic() - start,
            )

        except Exception as exc:
            logger.error("Blaxel execution failed: %s", exc)
            return SandboxResult(
                status=SandboxStatus.FAILED,
                stderr=str(exc),
                exit_code=1,
                duration_seconds=time.monotonic() - start,
            )

    def terminate(self, sandbox_id: str) -> None:
        """Terminate a Blaxel sandbox.

        Args:
            sandbox_id: Blaxel sandbox identifier.
        """
        logger.info("Blaxel terminate requested for %s.", sandbox_id)
