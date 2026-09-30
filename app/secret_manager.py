# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Secure Secret Management integration via Google Cloud Secret Manager.

Ensures zero hardcoded API keys, credentials, or tokens in the codebase.
Retrieves secrets dynamically from Google Cloud Secret Manager with caching
and environment variable fallback for local testing and CI/CD pipelines.
"""

from __future__ import annotations

import functools
import logging
import os

from google.api_core import exceptions as google_exceptions
from google.cloud import secretmanager

logger = logging.getLogger(__name__)


class SecretManagerClient:
    """Client for retrieving sensitive configuration from Google Cloud Secret Manager."""

    def __init__(self, project_id: str | None = None) -> None:
        """Initialize the Secret Manager client.

        Args:
            project_id: Optional GCP project ID. Defaults to GOOGLE_CLOUD_PROJECT
                environment variable.
        """
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._client: secretmanager.SecretManagerServiceClient | None = None

    @property
    def client(self) -> secretmanager.SecretManagerServiceClient:
        """Lazy-initialize the underlying SecretManagerServiceClient."""
        if self._client is None:
            self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    @functools.lru_cache(maxsize=64)  # noqa: B019
    def get_secret(
        self,
        secret_id: str,
        version_id: str = "latest",
        default: str | None = None,
    ) -> str:
        """Retrieve a secret value from Secret Manager or environment fallback.

        Args:
            secret_id: Identifier of the secret in GCP Secret Manager or environment
                variable name (e.g., 'PAGERDUTY_WEBHOOK_SECRET').
            version_id: Secret version to access (defaults to 'latest').
            default: Non-sensitive fallback value if the secret is not configured.

        Returns:
            The resolved secret string.

        Raises:
            RuntimeError: If the secret cannot be resolved and no default is provided.
        """
        # 1. Check environment variable injected by Cloud Run Secret Manager volume/env mount
        env_val = os.environ.get(secret_id)
        if env_val:
            return env_val

        # 2. Fetch directly from Google Cloud Secret Manager API when project_id is set
        if self.project_id and os.environ.get("ENABLE_GCP_SECRET_MANAGER", "true").lower() in (
            "true",
            "1",
            "yes",
        ):
            resource_name = (
                f"projects/{self.project_id}/secrets/{secret_id}/versions/{version_id}"
            )
            try:
                response = self.client.access_secret_version(
                    request={"name": resource_name}
                )
                payload = response.payload.data.decode("UTF-8")
                return payload.strip()
            except (google_exceptions.GoogleAPICallError, Exception) as exc:
                logger.warning(
                    "Secret Manager lookup fallback for secret_id=%s: %s",
                    secret_id,
                    type(exc).__name__,
                )

        if default is not None:
            return default

        raise RuntimeError(
            f"Secret '{secret_id}' is not set in environment or Google Cloud Secret Manager."
        )


@functools.cache
def get_secret_manager() -> SecretManagerClient:
    """Return a singleton SecretManagerClient instance."""
    return SecretManagerClient()
