"""
Runloop sandbox client.

Executes code inside `Runloop <https://runloop.ai>`_ dev-boxes.  Requires
the ``runloop-api-client`` package and the ``RUNLOOP_API_KEY`` env-var.

Example:
    >>> from nono.sandbox import RunloopSandboxClient, SandboxRunConfig
    >>> client = RunloopSandboxClient()
    >>> result = client.execute("print('hello from Runloop')", SandboxRunConfig())
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

logger = logging.getLogger("Nono.Sandbox.Runloop")

_RUNLOOP_API_KEY_ENV = "RUNLOOP_API_KEY"


class RunloopSandboxClient(BaseSandboxClient):
    """Sandbox client backed by Runloop dev-boxes.

    Args:
        api_key_env: Env-var holding the Runloop API key.
        blueprint: Runloop blueprint name.
    """

    def __init__(
        self,
        *,
        api_key_env: str = _RUNLOOP_API_KEY_ENV,
        blueprint: str = "default",
    ) -> None:
        super().__init__(api_key_env=api_key_env)
        self._blueprint = blueprint

    def execute(
        self,
        code: str,
        config: SandboxRunConfig,
    ) -> SandboxResult:
        """Run *code* inside a Runloop devbox.

        Args:
            code: Python code to execute.
            config: Execution configuration.

        Returns:
            A ``SandboxResult`` with captured output and status.
        """
        try:
            from runloop_api_client import Runloop
        except ImportError as exc:
            raise ImportError(
                "Runloop support requires the 'runloop-api-client' package. "
                "Install it with: pip install runloop-api-client"
            ) from exc

        api_key = self._get_api_key()
        start = time.monotonic()
        devbox = None

        try:
            client = Runloop(api_key=api_key)
            devbox = client.devboxes.create(
                blueprint_name=self._blueprint,
                environment_variables={
                    k: v for k, v in config.environment.items()
                },
            )

            client.devboxes.await_running(devbox.id)

            if config.packages:
                pkg_str = " ".join(config.packages)
                client.devboxes.execute_sync(
                    devbox.id,
                    command=f"pip install {pkg_str}",
                )

            result = client.devboxes.execute_sync(
                devbox.id,
                command=f"python -c {code!r}",
            )

            stdout = result.stdout if hasattr(result, "stdout") else ""
            stderr = result.stderr if hasattr(result, "stderr") else ""
            exit_code = result.exit_code if hasattr(result, "exit_code") else 0

            return SandboxResult(
                status=SandboxStatus.COMPLETED if exit_code == 0 else SandboxStatus.FAILED,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                sandbox_id=devbox.id,
                duration_seconds=time.monotonic() - start,
            )

        except Exception as exc:
            logger.error("Runloop execution failed: %s", exc)
            return SandboxResult(
                status=SandboxStatus.FAILED,
                stderr=str(exc),
                exit_code=1,
                duration_seconds=time.monotonic() - start,
            )

        finally:
            if devbox and not config.keep_alive:
                try:
                    client.devboxes.shutdown(devbox.id)
                except Exception:
                    logger.warning("Failed to shutdown Runloop devbox.")

    def terminate(self, sandbox_id: str) -> None:
        """Terminate a Runloop devbox.

        Args:
            sandbox_id: Runloop devbox identifier.
        """
        try:
            from runloop_api_client import Runloop

            client = Runloop(api_key=self._get_api_key())
            client.devboxes.shutdown(sandbox_id)
        except Exception as exc:
            logger.warning("Runloop terminate failed for %s: %s", sandbox_id, exc)
