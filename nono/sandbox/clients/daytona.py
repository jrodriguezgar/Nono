"""
Daytona sandbox client.

Executes code inside `Daytona <https://daytona.io>`_ development environments.
Requires the ``daytona-sdk`` package and the ``DAYTONA_API_KEY`` env-var.

Example:
    >>> from nono.sandbox import DaytonaSandboxClient, SandboxRunConfig
    >>> client = DaytonaSandboxClient()
    >>> result = client.execute("print('hello from Daytona')", SandboxRunConfig())
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

logger = logging.getLogger("Nono.Sandbox.Daytona")

_DAYTONA_API_KEY_ENV = "DAYTONA_API_KEY"


class DaytonaSandboxClient(BaseSandboxClient):
    """Sandbox client backed by Daytona development environments.

    Args:
        api_key_env: Env-var holding the Daytona API key.
        target: Daytona target (default ``"us"``).
    """

    def __init__(
        self,
        *,
        api_key_env: str = _DAYTONA_API_KEY_ENV,
        target: str = "us",
    ) -> None:
        super().__init__(api_key_env=api_key_env)
        self._target = target

    def execute(
        self,
        code: str,
        config: SandboxRunConfig,
    ) -> SandboxResult:
        """Run *code* inside a Daytona sandbox.

        Args:
            code: Python code to execute.
            config: Execution configuration.

        Returns:
            A ``SandboxResult`` with captured output and status.
        """
        try:
            from daytona_sdk import Daytona, DaytonaConfig, CreateSandboxParams
        except ImportError as exc:
            raise ImportError(
                "Daytona support requires the 'daytona-sdk' package. "
                "Install it with: pip install daytona-sdk"
            ) from exc

        api_key = self._get_api_key()
        start = time.monotonic()
        sandbox = None

        try:
            daytona = Daytona(DaytonaConfig(api_key=api_key, target=self._target))

            params = CreateSandboxParams(language="python")
            sandbox = daytona.create(params)

            if config.packages:
                pkg_str = " ".join(config.packages)
                sandbox.process.exec(f"pip install {pkg_str}")

            response = sandbox.process.code_run(code)

            return SandboxResult(
                status=SandboxStatus.COMPLETED if response.exit_code == 0 else SandboxStatus.FAILED,
                stdout=response.result or "",
                stderr="",
                exit_code=response.exit_code,
                sandbox_id=sandbox.id if hasattr(sandbox, "id") else "",
                duration_seconds=time.monotonic() - start,
            )

        except Exception as exc:
            logger.error("Daytona execution failed: %s", exc)
            return SandboxResult(
                status=SandboxStatus.FAILED,
                stderr=str(exc),
                exit_code=1,
                duration_seconds=time.monotonic() - start,
            )

        finally:
            if sandbox and not config.keep_alive:
                try:
                    daytona.remove(sandbox)
                except Exception:
                    logger.warning("Failed to remove Daytona sandbox.")

    def terminate(self, sandbox_id: str) -> None:
        """Terminate a Daytona sandbox.

        Args:
            sandbox_id: Daytona sandbox identifier.
        """
        logger.info("Daytona terminate: use daytona.remove() with sandbox object.")
