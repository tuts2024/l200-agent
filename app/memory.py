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

"""Context Compaction, Persistent Session State, and Async Memory Operations.

Provides:
- History compaction via ADK `EventsCompactionConfig`, `LlmEventSummarizer`,
  and `ContextCacheConfig`.
- Persistent session state backed by `VertexAiSessionService` or
  `DatabaseSessionService` (PostgreSQL/SQLite) with `VertexAiMemoryBankService`.
- Non-blocking `async` memory generation and consolidation using
  `asyncio.create_task` so UI responses never block on memory indexing.
"""

from __future__ import annotations

import asyncio
import functools
import os
from typing import Any

from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.memory import InMemoryMemoryService, VertexAiMemoryBankService
from google.adk.models import Gemini
from google.adk.sessions import (
    DatabaseSessionService,
    InMemorySessionService,
    VertexAiSessionService,
)
from google.adk.tools.load_memory_tool import LoadMemoryTool
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

from app.observability import (
    log_agent_outcome_after_callback,
    log_structured_event,
    pii_redactor,
)

# Track background memory tasks to prevent garbage collection mid-flight
_BACKGROUND_MEMORY_TASKS: set[asyncio.Task[Any]] = set()


def build_events_compaction_config(
    summarizer_model: str = "gemini-2.5-flash",
) -> EventsCompactionConfig:
    """Create ADK EventsCompactionConfig with an LLM event summarizer to prevent context bloat."""
    summarizer = LlmEventSummarizer(
        llm=Gemini(model=summarizer_model),
    )
    return EventsCompactionConfig(
        compaction_interval=6,
        overlap_size=2,
        token_threshold=12000,
        event_retention_size=8,
        summarizer=summarizer,
    )


def build_context_cache_config() -> ContextCacheConfig:
    """Create ADK ContextCacheConfig to reuse system instructions and static context."""
    return ContextCacheConfig(
        min_tokens=2048,
        ttl_seconds=1800,
        cache_intervals=10,
    )


@functools.cache
def create_persistent_session_service() -> Any:
    """Instantiate a persistent session backend (Vertex AI Agent Engine or SQL Database).

    Priority order:
    1. `SESSION_DB_URL`: SQLAlchemy-backed `DatabaseSessionService` (PostgreSQL / SQLite).
    2. `AGENT_ENGINE_ID` + `GOOGLE_CLOUD_PROJECT`: `VertexAiSessionService`.
    3. Local SQLite `DatabaseSessionService` or `InMemorySessionService` fallback for unit tests.
    """
    db_url = os.environ.get("SESSION_DB_URL")
    if db_url:
        return DatabaseSessionService(db_url=db_url)

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    agent_engine_id = os.environ.get("AGENT_ENGINE_ID")
    if project_id and agent_engine_id:
        return VertexAiSessionService(
            project=project_id,
            location=location,
            agent_engine_id=agent_engine_id,
        )

    return InMemorySessionService()


@functools.cache
def create_persistent_memory_service() -> Any:
    """Instantiate long-term cross-session memory backend (`VertexAiMemoryBankService`)."""
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


async def _consolidate_session_memory_async(callback_context: Any) -> None:
    """Background coroutine that scrubs PII from session state and persists to Memory Bank."""
    agent_name = getattr(callback_context, "agent_name", "root_agent")
    try:
        state = getattr(callback_context, "state", None)
        if state is not None and hasattr(state, "to_dict"):
            raw_dict = state.to_dict()
            for key, val in raw_dict.items():
                if isinstance(val, str):
                    state[key] = pii_redactor.redact_text(val)

        # Trigger ADK's async memory consolidation if supported by the context
        add_memory_fn = getattr(callback_context, "add_session_to_memory", None)
        if callable(add_memory_fn):
            await add_memory_fn()

        log_structured_event(
            event_type="ASYNC_MEMORY_CONSOLIDATION_SUCCESS",
            message=f"Async session memory consolidation completed for {agent_name}.",
            phase="OUTCOME",
            agent_name=agent_name,
            outcome={"status": "consolidated", "background_task": True},
        )
    except Exception as exc:
        log_structured_event(
            event_type="ASYNC_MEMORY_CONSOLIDATION_SKIPPED",
            message=f"Async memory consolidation gracefully skipped: {type(exc).__name__}",
            phase="OUTCOME",
            agent_name=agent_name,
            outcome={"status": "skipped", "reason": str(exc)},
        )


async def async_after_agent_memory_callback(callback_context: Any) -> None:
    """Non-blocking async after_agent_callback that schedules background memory consolidation.

    Records the agent execution outcome immediately and dispatches expensive
    memory generation/consolidation via `asyncio.create_task` to avoid blocking
    the user-facing response stream.
    """
    log_agent_outcome_after_callback(callback_context)

    task = asyncio.create_task(_consolidate_session_memory_async(callback_context))
    _BACKGROUND_MEMORY_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_MEMORY_TASKS.discard)


def get_memory_retrieval_tools() -> list[Any]:
    """Return ADK memory tools for automatic and on-demand cross-turn recall."""
    return [PreloadMemoryTool(), LoadMemoryTool()]
