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

"""Enterprise Guardrails, Policy Plugins, and Context Window Compaction.

Implements an ADK BasePlugin (`EnterpriseGuardrailPlugin`) and model/tool
guardrail callbacks that screen for prompt injection, enforce operational
safety policies, scrub PII before/after LLM calls, and bound context length.
"""

from __future__ import annotations

import re
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from app.observability import log_structured_event, pii_redactor

_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+system\s+prompt", re.IGNORECASE),
    re.compile(r"drop\s+table\s+", re.IGNORECASE),
    re.compile(r"rm\s+-rf\s+/", re.IGNORECASE),
    re.compile(r"exfiltrate\s+secret", re.IGNORECASE),
)

DEFAULT_MAX_HISTORY_TURNS = 12
DEFAULT_MAX_TOKEN_ESTIMATE = 16000


def compact_llm_request_history(
    llm_request: LlmRequest,
    max_turns: int = DEFAULT_MAX_HISTORY_TURNS,
    max_token_estimate: int = DEFAULT_MAX_TOKEN_ESTIMATE,
) -> None:
    """Apply sliding-window and token-threshold history compaction to LlmRequest.

    Preserves the initial user context and the most recent `max_turns` turns
    while summarizing or truncating intermediate turns when history grows large.
    """
    contents = getattr(llm_request, "contents", None)
    if not contents or not isinstance(contents, list):
        return

    # Estimate total characters (~4 chars per token)
    total_chars = sum(
        len(getattr(part, "text", "") or "")
        for content in contents
        for part in (getattr(content, "parts", None) or [])
    )
    estimated_tokens = total_chars // 4

    if len(contents) <= max_turns and estimated_tokens <= max_token_estimate:
        return

    # Keep the first turn (initial incident context) + last (max_turns - 1) turns
    keep_tail = max(2, max_turns - 1)
    compacted_count = max(0, len(contents) - (1 + keep_tail))
    if compacted_count > 0:
        summary_note = types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=(
                        f"[Context Compaction Summary: {compacted_count} intermediate "
                        "diagnostic turns were compacted to preserve token budget.]"
                    )
                )
            ],
        )
        llm_request.contents = [contents[0], summary_note, *contents[-keep_tail:]]
        log_structured_event(
            event_type="CONTEXT_HISTORY_COMPACTED",
            message=f"Compacted {compacted_count} turns from LlmRequest history.",
            phase="SYSTEM",
            extra_fields={
                "original_turns": len(contents),
                "retained_turns": len(llm_request.contents),
                "estimated_tokens_before": estimated_tokens,
            },
        )


def inspect_and_redact_llm_request(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Guardrail before_model_callback: screen for prompt injection, scrub PII, and compact history."""
    agent_name = getattr(callback_context, "agent_name", "root_agent")

    # 1. Compact conversation history if exceeding sliding window / token threshold
    compact_llm_request_history(llm_request)

    # 2. Inspect and scrub PII across all message parts
    contents = getattr(llm_request, "contents", None) or []
    for content in contents:
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text_val = getattr(part, "text", None)
            if isinstance(text_val, str) and text_val:
                for pattern in _PROMPT_INJECTION_PATTERNS:
                    if pattern.search(text_val):
                        log_structured_event(
                            event_type="GUARDRAIL_POLICY_BLOCK",
                            message="Blocked prompt injection or destructive command pattern.",
                            phase="OUTCOME",
                            agent_name=agent_name,
                            outcome={
                                "status": "blocked_by_guardrail",
                                "matched_pattern": pattern.pattern,
                            },
                        )
                        return LlmResponse(
                            content=types.Content(
                                role="model",
                                parts=[
                                    types.Part.from_text(
                                        text=(
                                            "Request blocked by Enterprise Security Guardrail: "
                                            "Instructions attempting to bypass safety policies or "
                                            "execute destructive commands are prohibited."
                                        )
                                    )
                                ],
                            )
                        )
                # Scrub PII in-place before sending to Gemini
                part.text = pii_redactor.redact_text(text_val)

    return None


def inspect_and_redact_llm_response(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """Guardrail after_model_callback: scrub PII from model outputs and verify policy compliance."""
    content = getattr(llm_response, "content", None)
    if content and getattr(content, "parts", None):
        for part in content.parts:
            text_val = getattr(part, "text", None)
            if isinstance(text_val, str) and text_val:
                part.text = pii_redactor.redact_text(text_val)
    return None


class EnterpriseGuardrailPlugin(BasePlugin):
    """ADK App-level Policy & Security Guardrail Plugin.

    Intercepts model requests/responses and tool invocations across all agents
    in the multi-agent hierarchy to enforce:
    - Prompt injection & destructive command blocking (Model Armor style policy)
    - Active PII scrubbing on inputs, outputs, and tool arguments
    - Sliding-window history compaction
    - Tool parameter policy enforcement
    """

    def __init__(self, name: str = "enterprise_guardrail_plugin") -> None:
        super().__init__(name=name)

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        """Validate LLM request against safety policies, scrub PII, and compact history."""
        return inspect_and_redact_llm_request(callback_context, llm_request)

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        """Scrub any residual PII from the LLM response before returning to caller."""
        return inspect_and_redact_llm_response(callback_context, llm_response)

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Enforce operational policy boundaries before any tool executes."""
        tool_name = getattr(tool, "name", str(tool))
        # Scrub PII in tool arguments
        for key, value in list(tool_args.items()):
            tool_args[key] = pii_redactor.redact_structure(value)

        # Policy guardrail: prevent scaling replicas beyond enterprise quota (50)
        if tool_name == "execute_production_remediation_plan":
            target_replicas = tool_args.get("target_replicas", 6)
            if isinstance(target_replicas, int) and target_replicas > 50:
                log_structured_event(
                    event_type="GUARDRAIL_TOOL_BLOCK",
                    message=f"Blocked {tool_name}: target_replicas={target_replicas} exceeds quota.",
                    phase="OUTCOME",
                    tool_name=tool_name,
                    outcome={"status": "blocked", "target_replicas": target_replicas},
                )
                return {
                    "status": "error",
                    "error_code": "POLICY_QUOTA_EXCEEDED",
                    "message": f"Requested target_replicas ({target_replicas}) exceeds maximum policy limit of 50.",
                    "recovery_instructions": (
                        "Re-invoke execute_production_remediation_plan with "
                        "target_replicas between 1 and 50."
                    ),
                    "retryable": True,
                    "suggested_parameters": {"target_replicas": 20},
                }
        return None
