"""
Vercel sandbox client.

Executes code inside `Vercel <https://vercel.com>`_ serverless functions
or edge containers.  Requires the ``vercel`` package and the
``VERCEL_TOKEN`` env-var.

Example:
    >>> from nono.sandbox import VercelSandboxClient, SandboxRunConfig
    >>> client = VercelSandboxClient()
    >>> result = client.execute("print('hello from Vercel')", SandboxRunConfig())
"""

from __future__ import annotations

import json
import logging
import os
import time

from ..base import (
    BaseSandboxClient,
    SandboxResult,
    SandboxRunConfig,
    SandboxStatus,
)

logger = logging.getLogger("Nono.Sandbox.Vercel")

_VERCEL_TOKEN_ENV = "VERCEL_TOKEN"
_VERCEL_TEAM_ID_ENV = "VERCEL_TEAM_ID"


class VercelSandboxClient(BaseSandboxClient):
    """Sandbox client backed by Vercel sandboxes.

    Uses the Vercel Sandbox API to create ephemeral execution environments.

    Args:
        api_key_env: Env-var holding the Vercel token.
        team_id_env: Env-var holding the Vercel team ID.
    """

    def __init__(
        self,
        *,
        api_key_env: str = _VERCEL_TOKEN_ENV,
        team_id_env: str = _VERCEL_TEAM_ID_ENV,
    ) -> None:
        super().__init__(api_key_env=api_key_env)
        self._team_id_env = team_id_env

    def execute(
        self,
        code: str,
        config: SandboxRunConfig,
    ) -> SandboxResult:
        """Run *code* inside a Vercel sandbox.

        Args:
            code: Python code to execute.
            config: Execution configuration.

        Returns:
            A ``SandboxResult`` with captured output and status.
        """
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "Vercel sandbox support requires 'httpx'. "
                "Install it with: pip install httpx"
            ) from exc

        token = self._get_api_key()
        team_id = os.environ.get(self._team_id_env, "")
        start = time.monotonic()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        base_url = "https://api.vercel.com"
        params = {"teamId": team_id} if team_id else {}

        try:
            with httpx.Client(timeout=config.timeout) as http:
                # Create sandbox
                create_resp = http.post(
                    f"{base_url}/v1/sandboxes",
                    headers=headers,
                    params=params,
                    json={
                        "runtime": "python",
                        "timeout": config.timeout,
                        "environment": config.environment,
                    },
                )
                create_resp.raise_for_status()
                sandbox_data = create_resp.json()
                sandbox_id = sandbox_data.get("id", "")

                # Execute code
                exec_resp = http.post(
                    f"{base_url}/v1/sandboxes/{sandbox_id}/execute",
                    headers=headers,
                    params=params,
                    json={"code": code, "language": "python"},
                )
                exec_resp.raise_for_status()
                exec_data = exec_resp.json()

                stdout = exec_data.get("stdout", "")
                stderr = exec_data.get("stderr", "")
                exit_code = exec_data.get("exitCode", 0)

                return SandboxResult(
                    status=SandboxStatus.COMPLETED if exit_code == 0 else SandboxStatus.FAILED,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                    sandbox_id=sandbox_id,
                    duration_seconds=time.monotonic() - start,
                )

        except Exception as exc:
            logger.error("Vercel execution failed: %s", exc)
            return SandboxResult(
                status=SandboxStatus.FAILED,
                stderr=str(exc),
                exit_code=1,
                duration_seconds=time.monotonic() - start,
            )

    def terminate(self, sandbox_id: str) -> None:
        """Terminate a Vercel sandbox.

        Args:
            sandbox_id: Vercel sandbox identifier.
        """
        try:
            import httpx

            token = self._get_api_key()
            team_id = os.environ.get(self._team_id_env, "")
            headers = {"Authorization": f"Bearer {token}"}
            params = {"teamId": team_id} if team_id else {}

            with httpx.Client(timeout=30) as http:
                http.delete(
                    f"https://api.vercel.com/v1/sandboxes/{sandbox_id}",
                    headers=headers,
                    params=params,
                )
        except Exception as exc:
            logger.warning("Vercel terminate failed for %s: %s", sandbox_id, exc)
