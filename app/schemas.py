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

"""Explicit Pydantic JSON schemas for tool inputs, tool outputs, and agent I/O.

Enforces strict JSON schema validation across all tool boundaries and LLM
structured outputs to prevent hallucinated parameters and malformed responses.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SeverityLevel(StrEnum):
    """Standardized incident severity classifications."""

    SEV1 = "SEV-1"
    SEV2 = "SEV-2"
    SEV3 = "SEV-3"
    SEV4 = "SEV-4"


class RemediationActionType(StrEnum):
    """Supported production remediation action types."""

    SCALE_REPLICAS = "scale_replicas"
    RESTART_PODS = "restart_pods"
    DRAIN_TRAFFIC = "drain_traffic"
    FLUSH_CACHE = "flush_cache"
    ENABLE_CIRCUIT_BREAKER = "enable_circuit_breaker"


class GuidedToolErrorResponse(BaseModel):
    """Structured error response providing actionable recovery instructions to the LLM."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        default="error",
        description="Execution status indicator, always 'error' for failed tool calls.",
    )
    error_code: str = Field(
        ...,
        description="Machine-readable error classification code (e.g., INVALID_SERVICE_NAME).",
    )
    message: str = Field(
        ...,
        description="Human-readable explanation of why the tool invocation failed.",
    )
    recovery_instructions: str = Field(
        ...,
        description=(
            "Step-by-step instructions telling the LLM how to self-correct its "
            "arguments or which alternative tool to invoke next."
        ),
    )
    retryable: bool = Field(
        default=True,
        description="Whether the LLM should retry the call after correcting arguments.",
    )
    suggested_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Recommended corrected parameter values for the next invocation.",
    )


class ServiceTelemetryQueryInput(BaseModel):
    """Input schema for querying golden signal telemetry metrics of a cloud service."""

    model_config = ConfigDict(extra="forbid")

    service_name: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^[a-z0-9-]+$",
        description=(
            "Canonical kebab-case identifier of the microservice "
            "(e.g., 'checkout-service', 'payment-gateway', 'auth-service', 'inventory-api')."
        ),
    )
    environment: str = Field(
        default="prod",
        pattern=r"^(prod|staging|canary|dev)$",
        description="Target deployment environment ('prod', 'staging', 'canary', or 'dev').",
    )
    lookback_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,
        description="Time window in minutes to aggregate golden signals (5 to 1440).",
    )

    @field_validator("service_name")
    @classmethod
    def normalize_service_name(cls, value: str) -> str:
        """Ensure service name is stripped and lowercased."""
        return value.strip().lower()


class ServiceTelemetryOutput(BaseModel):
    """Output schema containing golden signal metrics and anomaly indicators."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="success", description="Tool execution status.")
    service_name: str = Field(..., description="Queried microservice identifier.")
    environment: str = Field(..., description="Queried deployment environment.")
    lookback_minutes: int = Field(..., description="Lookback window in minutes.")
    latency_p95_ms: float = Field(..., description="95th percentile latency in ms.")
    latency_p99_ms: float = Field(..., description="99th percentile latency in ms.")
    error_rate_percent: float = Field(..., description="HTTP 5xx error rate percentage.")
    cpu_utilization_percent: float = Field(..., description="Mean pod CPU utilization.")
    memory_utilization_percent: float = Field(
        ..., description="Mean pod memory utilization."
    )
    active_replicas: int = Field(..., description="Number of healthy serving pods.")
    recent_deployment_version: str = Field(
        ..., description="Currently deployed container release tag."
    )
    anomalies_detected: list[str] = Field(
        default_factory=list,
        description="List of detected golden-signal anomalies and threshold breaches.",
    )


class IncidentRunbookSearchInput(BaseModel):
    """Input schema for searching standard operating runbooks and postmortems."""

    model_config = ConfigDict(extra="forbid")

    symptom_query: str = Field(
        ...,
        min_length=3,
        max_length=256,
        description="Search query describing failure symptoms, error codes, or service name.",
    )
    severity_filter: SeverityLevel | None = Field(
        default=None,
        description="Optional severity level filter ('SEV-1', 'SEV-2', 'SEV-3', 'SEV-4').",
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of matching runbooks to return (1 to 10).",
    )


class RunbookEntry(BaseModel):
    """Structured runbook record returned from the knowledge base."""

    model_config = ConfigDict(extra="forbid")

    runbook_id: str = Field(..., description="Unique runbook identifier (e.g., RB-104).")
    title: str = Field(..., description="Runbook title.")
    applicable_services: list[str] = Field(
        ..., description="Services covered by this runbook."
    )
    root_cause_signature: str = Field(
        ..., description="Diagnostic signature matching the incident."
    )
    mitigation_steps: list[str] = Field(
        ..., description="Ordered remediation steps verified by SRE."
    )
    recommended_action_type: RemediationActionType = Field(
        ..., description="Primary automated remediation action type."
    )


class IncidentRunbookSearchOutput(BaseModel):
    """Output schema for runbook search results."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="success", description="Tool execution status.")
    query: str = Field(..., description="Executed symptom search query.")
    total_matches: int = Field(..., description="Number of runbooks matched.")
    runbooks: list[RunbookEntry] = Field(
        default_factory=list, description="Matched SRE runbook entries."
    )


class IncidentTicketCreateInput(BaseModel):
    """Input schema for creating a formal incident tracking ticket."""

    model_config = ConfigDict(extra="forbid")

    service_name: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Affected microservice name.",
    )
    severity: SeverityLevel = Field(
        ...,
        description="Incident severity level ('SEV-1', 'SEV-2', 'SEV-3', 'SEV-4').",
    )
    summary: str = Field(
        ...,
        min_length=10,
        max_length=200,
        description="Concise incident headline summarizing impact.",
    )
    probable_root_cause: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Evidence-backed root cause hypothesis from telemetry and runbooks.",
    )
    commander_email: str = Field(
        default="sre-oncall@example.com",
        description="Incident commander contact email (scrubbed if PII policy applies).",
    )


class IncidentTicketCreateOutput(BaseModel):
    """Output schema for a created incident ticket."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="success", description="Tool execution status.")
    incident_id: str = Field(..., description="Generated incident tracking ID.")
    severity: SeverityLevel = Field(..., description="Assigned severity level.")
    service_name: str = Field(..., description="Affected microservice.")
    bridge_url: str = Field(..., description="Incident response war-room URL.")
    created_at_iso: str = Field(..., description="ISO-8601 creation timestamp.")


class RemediationPlanExecutionInput(BaseModel):
    """Input schema for executing a high-stakes production remediation action."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str = Field(
        ...,
        pattern=r"^INC-[0-9]{4,8}$",
        description="Tracking incident ID (e.g., 'INC-4021').",
    )
    service_name: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Target production microservice.",
    )
    action_type: RemediationActionType = Field(
        ...,
        description="Remediation operation to execute.",
    )
    target_replicas: int = Field(
        default=6,
        ge=1,
        le=50,
        description="Desired replica count when scaling replicas (1 to 50).",
    )
    runbook_id: str = Field(
        ...,
        pattern=r"^RB-[0-9]{3,5}$",
        description="Authorizing SRE runbook ID (e.g., 'RB-101').",
    )
    justification: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Engineering justification explaining why this action is safe.",
    )


class RemediationPlanExecutionOutput(BaseModel):
    """Output schema for production remediation plan execution."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        ..., description="Execution status ('success', 'pending_approval', 'error')."
    )
    execution_id: str = Field(..., description="Unique remediation execution ID.")
    incident_id: str = Field(..., description="Associated incident ID.")
    service_name: str = Field(..., description="Target microservice.")
    action_type: RemediationActionType = Field(..., description="Executed action type.")
    human_approved: bool = Field(
        ..., description="Whether Human-in-the-Loop confirmation was verified."
    )
    verification_metrics: dict[str, float | int | str] = Field(
        default_factory=dict,
        description="Post-remediation telemetry snapshot.",
    )


class CanaryRollbackInput(BaseModel):
    """Input schema for rolling back a faulty canary deployment."""

    model_config = ConfigDict(extra="forbid")

    service_name: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Microservice whose canary deployment should be rolled back.",
    )
    faulty_version: str = Field(
        ...,
        min_length=2,
        max_length=32,
        description="Faulty release tag to roll back from (e.g., 'v2.14.0-canary').",
    )
    stable_version: str = Field(
        ...,
        min_length=2,
        max_length=32,
        description="Last known good release tag to restore (e.g., 'v2.13.8').",
    )
    reason: str = Field(
        ...,
        min_length=10,
        max_length=400,
        description="Reason for triggering the canary rollback.",
    )


class CanaryRollbackOutput(BaseModel):
    """Output schema for canary deployment rollback."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="Rollback status.")
    rollback_id: str = Field(..., description="Unique rollback operation ID.")
    service_name: str = Field(..., description="Target microservice.")
    restored_version: str = Field(..., description="Restored stable version tag.")
    traffic_shifted_percent: int = Field(
        ..., description="Percentage of traffic routed to stable version."
    )
    human_approved: bool = Field(
        ..., description="Whether Human-in-the-Loop confirmation was granted."
    )
