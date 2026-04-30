"""
Cloudflare Workers sandbox client.

Executes code inside `Cloudflare Workers <https://workers.cloudflare.com>`_
containers.  Requires the ``cloudflare`` package and the
``CLOUDFLARE_API_TOKEN`` env-var.

Example:
    >>> from nono.sandbox import CloudflareSandboxClient, SandboxRunConfig
    >>> client = CloudflareSandboxClient()
    >>> result = client.execute("print('hello from Cloudflare')", SandboxRunConfig())
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

logger = logging.getLogger("Nono.Sandbox.Cloudflare")

_CF_API_TOKEN_ENV = "CLOUDFLARE_API_TOKEN"
_CF_ACCOUNT_ID_ENV = "CLOUDFLARE_ACCOUNT_ID"


class CloudflareSandboxClient(BaseSandboxClient):
    """Sandbox client backed by Cloudflare Workers / Containers.

    Args:
        api_key_env: Env-var holding the Cloudflare API token.
        account_id_env: Env-var holding the Cloudflare account ID.
    """

    def __init__(
        self,
        *,
        api_key_env: str = _CF_API_TOKEN_ENV,
        account_id_env: str = _CF_ACCOUNT_ID_ENV,
    ) -> None:
        super().__init__(api_key_env=api_key_env)
        self._account_id_env = account_id_env

    def execute(
        self,
        code: str,
        config: SandboxRunConfig,
    ) -> SandboxResult:
        """Run *code* inside a Cloudflare sandbox.

        Args:
            code: Python code to execute.
            config: Execution configuration.

        Returns:
            A ``SandboxResult`` with captured output and status.
        """
        try:
            import cloudflare
        except ImportError as exc:
            raise ImportError(
                "Cloudflare support requires the 'cloudflare' package. "
                "Install it with: pip install cloudflare"
            ) from exc

        import os

        api_token = self._get_api_key()
        account_id = os.environ.get(self._account_id_env, "")

        if not account_id:
            raise EnvironmentError(
                f"Cloudflare account ID not found. Set {self._account_id_env!r}."
            )

        start = time.monotonic()

        try:
            client = cloudflare.Cloudflare(api_token=api_token)

            container = client.containers.create(
                account_id=account_id,
                image="python:3.12-slim",
                memory_mb=512,
                timeout_seconds=config.timeout,
                environment_variables={
                    k: {"value": v} for k, v in config.environment.items()
                },
            )

            sandbox_id = container.id if hasattr(container, "id") else ""

            exec_result = client.containers.exec(
                account_id=account_id,
                container_id=sandbox_id,
                command=["python", "-c", code],
            )

            stdout = exec_result.stdout if hasattr(exec_result, "stdout") else ""
            stderr = exec_result.stderr if hasattr(exec_result, "stderr") else ""
            exit_code = exec_result.exit_code if hasattr(exec_result, "exit_code") else 0

            return SandboxResult(
                status=SandboxStatus.COMPLETED if exit_code == 0 else SandboxStatus.FAILED,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                sandbox_id=str(sandbox_id),
                duration_seconds=time.monotonic() - start,
            )

        except Exception as exc:
            logger.error("Cloudflare execution failed: %s", exc)
            return SandboxResult(
                status=SandboxStatus.FAILED,
                stderr=str(exc),
                exit_code=1,
                duration_seconds=time.monotonic() - start,
            )

    def terminate(self, sandbox_id: str) -> None:
        """Terminate a Cloudflare container.

        Args:
            sandbox_id: Cloudflare container identifier.
        """
        try:
            import os
            import cloudflare

            client = cloudflare.Cloudflare(api_token=self._get_api_key())
            account_id = os.environ.get(self._account_id_env, "")
            client.containers.delete(account_id=account_id, container_id=sandbox_id)
        except Exception as exc:
            logger.warning("Cloudflare terminate failed for %s: %s", sandbox_id, exc)
