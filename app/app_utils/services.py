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

"""Process-wide ADK session, memory, and artifact services shared by every serving surface.

Registered under ``shared://`` so the ADK web routes, the A2A path, and the
reasoning_engine adapter share one instance: a session created on any surface
is visible to the others.
"""

from __future__ import annotations

import functools
import os

from agentplatform import agent_engines
from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService
from google.adk.cli.service_registry import get_service_registry
from google.adk.cli.utils.service_factory import create_session_service_from_options
from google.adk.memory import InMemoryMemoryService, VertexAiMemoryBankService
from google.adk.sessions import DatabaseSessionService, InMemorySessionService

SESSION_SERVICE_URI = "shared://session"
ARTIFACT_SERVICE_URI = "shared://artifact"
MEMORY_SERVICE_URI = "shared://memory"

_AGENT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


@functools.cache
def get_session_service():
    """Process-wide persistent session service (SQL Database, Vertex AI Agent Engine, or In-Memory)."""
    if db_url := os.environ.get("SESSION_DB_URL"):
        return DatabaseSessionService(db_url=db_url)

    use_in_memory_session = os.environ.get("USE_IN_MEMORY_SESSION", "").lower() in (
        "true",
        "1",
        "yes",
    )
    if not use_in_memory_session:
        default_agent_name = "l200-agent"
        agent_name = os.environ.get("AGENT_ENGINE_SESSION_NAME", default_agent_name)
        existing_agents = list(agent_engines.list(filter=f"display_name={agent_name}"))
        engine = (
            existing_agents[0]
            if existing_agents
            else agent_engines.create(display_name=agent_name)
        )
        return create_session_service_from_options(
            base_dir=_AGENT_DIR,
            session_service_uri=f"agentengine://{engine.resource_name}",
        )

    return InMemorySessionService()


@functools.cache
def get_memory_service():
    """Process-wide long-term memory service: VertexAiMemoryBankService when configured, else InMemory."""
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    agent_engine_id = os.environ.get("AGENT_ENGINE_ID")
    if project_id and agent_engine_id:
        return VertexAiMemoryBankService(
            project=project_id,
            location=location,
            agent_engine_id=agent_engine_id,
        )
    return InMemoryMemoryService()


@functools.cache
def get_artifact_service():
    """Process-wide artifact service: GCS when a bucket is set, else in-memory."""
    if bucket := os.environ.get("LOGS_BUCKET_NAME"):
        return GcsArtifactService(bucket_name=bucket)
    return InMemoryArtifactService()


_registry = get_service_registry()
_registry.register_session_service("shared", lambda uri, **kw: get_session_service())
_registry.register_artifact_service("shared", lambda uri, **kw: get_artifact_service())
_registry.register_memory_service("shared", lambda uri, **kw: get_memory_service())
