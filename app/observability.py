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

"""Enterprise Observability: Structured JSON Logging, Distributed Tracing,
Intent-vs-Outcome Capture, and PII Redaction (Regex + Google Cloud DLP).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import time
import uuid
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pythonjsonlogger.json import JsonFormatter

try:
    from google.cloud import dlp_v2
except ImportError:  # pragma: no cover
    dlp_v2 = None  # type: ignore[assignment]

try:
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
except ImportError:  # pragma: no cover
    CloudTraceSpanExporter = None  # type: ignore[assignment]


class PiiRedactor:
    """Active PII scrubbing engine combining regex patterns and Google Cloud DLP."""

    _PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        (
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            "[REDACTED_EMAIL]",
        ),
        (
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "[REDACTED_SSN]",
        ),
        (
            re.compile(
                r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
            ),
            "[REDACTED_PHONE]",
        ),
        (
            re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
            "[REDACTED_CREDIT_CARD]",
        ),
        (
            re.compile(
                r"\b(?:AIza[0-9A-Za-z\-_]{20,45}|ya29\.[0-9A-Za-z\-_]+|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,45})\b"
            ),
            "[REDACTED_API_KEY]",
        ),
    )

    def __init__(self, project_id: str | None = None, enable_cloud_dlp: bool = False) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.enable_cloud_dlp = enable_cloud_dlp or (
            os.environ.get("ENABLE_CLOUD_DLP", "false").lower() in ("true", "1", "yes")
        )
        self._dlp_client: Any = None

    def redact_text(self, text: str) -> str:
        """Scrub PII tokens from a raw string using regex and optional Cloud DLP."""
        if not text:
            return text

        scrubbed = text
        for pattern, replacement in self._PATTERNS:
            scrubbed = pattern.sub(replacement, scrubbed)

        if self.enable_cloud_dlp and self.project_id and dlp_v2 is not None:
            scrubbed = self._deidentify_with_cloud_dlp(scrubbed)

        return scrubbed

    def _deidentify_with_cloud_dlp(self, text: str) -> str:
        """Call Google Cloud DLP API to de-identify remaining sensitive info types."""
        try:
            if self._dlp_client is None:
                self._dlp_client = dlp_v2.DlpServiceClient()
            parent = f"projects/{self.project_id}/locations/global"
            inspect_config = {
                "info_types": [
                    {"name": "EMAIL_ADDRESS"},
                    {"name": "PHONE_NUMBER"},
                    {"name": "US_SOCIAL_SECURITY_NUMBER"},
                    {"name": "CREDIT_CARD_NUMBER"},
                ]
            }
            deidentify_config = {
                "info_type_transformations": {
                    "transformations": [
                        {
                            "primitive_transformation": {
                                "replace_with_info_type_config": {}
                            }
                        }
                    ]
                }
            }
            response = self._dlp_client.deidentify_content(
                request={
                    "parent": parent,
                    "inspect_config": inspect_config,
                    "deidentify_config": deidentify_config,
                    "item": {"value": text},
                }
            )
            return response.item.value
        except Exception:
            return text

    def redact_structure(self, data: Any) -> Any:
        """Recursively scrub PII from dicts, lists, tuples, and strings."""
        if isinstance(data, str):
            return self.redact_text(data)
        if isinstance(data, dict):
            return {str(k): self.redact_structure(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self.redact_structure(item) for item in data]
        if isinstance(data, tuple):
            return tuple(self.redact_structure(item) for item in data)
        return data


pii_redactor = PiiRedactor()


class StructuredJsonFormatter(JsonFormatter):
    """JSON log formatter that injects OpenTelemetry trace context and scrubs PII."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        log_record["severity"] = record.levelname
        log_record["logger"] = record.name

        current_span = trace.get_current_span()
        span_ctx = current_span.get_span_context() if current_span else None
        if span_ctx and span_ctx.is_valid:
            log_record["trace_id"] = format(span_ctx.trace_id, "032x")
            log_record["span_id"] = format(span_ctx.span_id, "016x")
        else:
            log_record.setdefault("trace_id", "0" * 32)
            log_record.setdefault("span_id", "0" * 16)

        # Ensure all logged fields are scrubbed for PII before serialization
        for key, val in list(log_record.items()):
            log_record[key] = pii_redactor.redact_structure(val)


def configure_structured_logging(service_name: str = "l200-agent") -> logging.Logger:
    """Configure root/service logger with structured JSON output and PII redaction."""
    logger = logging.getLogger(service_name)
    logger.setLevel(logging.INFO)
    if not any(
        isinstance(getattr(h, "formatter", None), StructuredJsonFormatter)
        for h in logger.handlers
    ):
        handler = logging.StreamHandler()
        formatter = StructuredJsonFormatter(
            "%(timestamp)s %(severity)s %(name)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.handlers = [handler]
        logger.propagate = False
    return logger


def configure_distributed_tracing(service_name: str = "l200-agent") -> trace.Tracer:
    """Configure OpenTelemetry TracerProvider with optional Cloud Trace export."""
    current_provider = trace.get_tracer_provider()
    if not isinstance(current_provider, TracerProvider):
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": "1.0.0",
            }
        )
        provider = TracerProvider(resource=resource)
        if (
            CloudTraceSpanExporter is not None
            and os.environ.get("ENABLE_CLOUD_TRACE_EXPORT", "false").lower()
            in ("true", "1", "yes")
        ):
            try:
                cloud_exporter = CloudTraceSpanExporter()
                provider.add_span_processor(BatchSpanProcessor(cloud_exporter))
            except Exception:
                pass
        trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


structured_logger = configure_structured_logging("l200-agent")
tracer = configure_distributed_tracing("l200-agent")


def log_structured_event(
    event_type: str,
    message: str,
    *,
    phase: str = "SYSTEM",
    agent_name: str = "root_agent",
    tool_name: str | None = None,
    intent: dict[str, Any] | None = None,
    outcome: dict[str, Any] | None = None,
    latency_ms: float | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit a PII-redacted structured JSON log entry capturing Intent vs. Outcome."""
    payload: dict[str, Any] = {
        "event_type": event_type,
        "phase": phase,
        "agent_name": agent_name,
    }
    if tool_name:
        payload["tool_name"] = tool_name
    if intent is not None:
        payload["intent"] = pii_redactor.redact_structure(intent)
    if outcome is not None:
        payload["outcome"] = pii_redactor.redact_structure(outcome)
    if latency_ms is not None:
        payload["latency_ms"] = round(latency_ms, 2)
    if extra_fields:
        payload.update(pii_redactor.redact_structure(extra_fields))

    redacted_message = pii_redactor.redact_text(message)
    structured_logger.info(redacted_message, extra=payload)
    return payload


def log_agent_intent_before_callback(callback_context: Any) -> None:
    """ADK before_agent_callback: record agent execution INTENT and start span timer."""
    agent_name = getattr(callback_context, "agent_name", "unknown_agent")
    state = getattr(callback_context, "state", None)
    correlation_id = str(uuid.uuid4())
    if isinstance(state, dict) or hasattr(state, "__setitem__"):
        try:
            state[f"_start_ts_{agent_name}"] = time.perf_counter()
            state[f"_corr_id_{agent_name}"] = correlation_id
        except Exception:
            pass

    log_structured_event(
        event_type="AGENT_EXECUTION_START",
        message=f"Agent '{agent_name}' starting turn with declared intent.",
        phase="INTENT",
        agent_name=agent_name,
        intent={
            "action": "invoke_agent",
            "agent_name": agent_name,
            "correlation_id": correlation_id,
        },
    )


def log_agent_outcome_after_callback(callback_context: Any) -> None:
    """ADK after_agent_callback: record agent execution OUTCOME and latency."""
    agent_name = getattr(callback_context, "agent_name", "unknown_agent")
    state = getattr(callback_context, "state", None)
    latency_ms = 0.0
    correlation_id = "unknown"
    if state is not None and hasattr(state, "get"):
        start_ts = state.get(f"_start_ts_{agent_name}")
        correlation_id = state.get(f"_corr_id_{agent_name}", "unknown")
        if isinstance(start_ts, (int, float)):
            latency_ms = (time.perf_counter() - start_ts) * 1000.0

    log_structured_event(
        event_type="AGENT_EXECUTION_COMPLETE",
        message=f"Agent '{agent_name}' completed execution.",
        phase="OUTCOME",
        agent_name=agent_name,
        outcome={
            "status": "completed",
            "agent_name": agent_name,
            "correlation_id": correlation_id,
        },
        latency_ms=latency_ms,
    )


def log_tool_intent_before_callback(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
) -> dict[str, Any] | None:
    """ADK before_tool_callback: scrub PII from args and log tool INTENT before execution."""
    tool_name = getattr(tool, "name", str(tool))
    agent_name = getattr(tool_context, "agent_name", "root_agent")

    # Scrub any PII in tool arguments in-place before execution
    for key, val in list(args.items()):
        args[key] = pii_redactor.redact_structure(val)

    state = getattr(tool_context, "state", None)
    if state is not None and hasattr(state, "__setitem__"):
        try:
            state[f"_tool_start_{tool_name}"] = time.perf_counter()
        except Exception:
            pass

    log_structured_event(
        event_type="TOOL_INVOCATION_INTENT",
        message=f"Tool '{tool_name}' invoked with validated intent.",
        phase="INTENT",
        agent_name=agent_name,
        tool_name=tool_name,
        intent={
            "planned_tool": tool_name,
            "sanitized_arguments": pii_redactor.redact_structure(args),
        },
    )
    return None


def log_tool_outcome_after_callback(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
    tool_response: dict[str, Any] | Any,
) -> dict[str, Any] | None:
    """ADK after_tool_callback: log actual tool OUTCOME vs. intent and scrub PII."""
    tool_name = getattr(tool, "name", str(tool))
    agent_name = getattr(tool_context, "agent_name", "root_agent")
    state = getattr(tool_context, "state", None)
    latency_ms = 0.0
    if state is not None and hasattr(state, "get"):
        start_ts = state.get(f"_tool_start_{tool_name}")
        if isinstance(start_ts, (int, float)):
            latency_ms = (time.perf_counter() - start_ts) * 1000.0

    sanitized_response = pii_redactor.redact_structure(tool_response)
    status = (
        sanitized_response.get("status", "success")
        if isinstance(sanitized_response, dict)
        else "success"
    )

    log_structured_event(
        event_type="TOOL_INVOCATION_OUTCOME",
        message=f"Tool '{tool_name}' finished with status={status}.",
        phase="OUTCOME",
        agent_name=agent_name,
        tool_name=tool_name,
        intent={
            "planned_tool": tool_name,
            "sanitized_arguments": pii_redactor.redact_structure(args),
        },
        outcome={
            "status": status,
            "result_summary": json.dumps(sanitized_response, default=str)[:500],
        },
        latency_ms=latency_ms,
    )
    if isinstance(sanitized_response, dict):
        return sanitized_response
    return None
