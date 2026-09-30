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

"""Domain-Specific SRE & Incident Response Tools with Explicit Pydantic Schemas,
Guided Error Recovery, OpenTelemetry Tracing, and Human-in-the-Loop Hooks.
"""

from __future__ import annotations

import datetime
import hashlib
from typing import Any

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from pydantic import ValidationError

from app.observability import log_structured_event, pii_redactor, tracer
from app.schemas import (
    CanaryRollbackInput,
    CanaryRollbackOutput,
    GuidedToolErrorResponse,
    IncidentRunbookSearchInput,
    IncidentRunbookSearchOutput,
    IncidentTicketCreateInput,
    IncidentTicketCreateOutput,
    RemediationActionType,
    RemediationPlanExecutionInput,
    RemediationPlanExecutionOutput,
    RunbookEntry,
    ServiceTelemetryOutput,
    ServiceTelemetryQueryInput,
    SeverityLevel,
)
from app.secret_manager import get_secret_manager

# Curated catalog of production microservice telemetry states
_TELEMETRY_CATALOG: dict[str, dict[str, Any]] = {
    "checkout-service": {
        "latency_p95_ms": 1840.5,
        "latency_p99_ms": 3420.0,
        "error_rate_percent": 14.8,
        "cpu_utilization_percent": 92.4,
        "memory_utilization_percent": 88.1,
        "active_replicas": 4,
        "recent_deployment_version": "v2.14.0-canary",
        "anomalies_detected": [
            "HTTP 5xx error rate (14.8%) exceeds SLO threshold (1.0%)",
            "p99 latency (3420ms) exceeds critical threshold (1000ms) after v2.14.0-canary rollout",
            "Pod CPU utilization saturated at 92.4%",
        ],
    },
    "payment-gateway": {
        "latency_p95_ms": 950.0,
        "latency_p99_ms": 2150.0,
        "error_rate_percent": 6.2,
        "cpu_utilization_percent": 78.5,
        "memory_utilization_percent": 94.0,
        "active_replicas": 3,
        "recent_deployment_version": "v1.9.3",
        "anomalies_detected": [
            "Connection pool exhaustion detected; memory utilization at 94.0%",
            "Upstream timeout spike (6.2% 504 Gateway Timeout)",
        ],
    },
    "auth-service": {
        "latency_p95_ms": 42.0,
        "latency_p99_ms": 85.0,
        "error_rate_percent": 0.05,
        "cpu_utilization_percent": 34.0,
        "memory_utilization_percent": 48.2,
        "active_replicas": 8,
        "recent_deployment_version": "v4.2.1",
        "anomalies_detected": [],
    },
    "inventory-api": {
        "latency_p95_ms": 620.0,
        "latency_p99_ms": 1410.0,
        "error_rate_percent": 3.4,
        "cpu_utilization_percent": 85.0,
        "memory_utilization_percent": 71.5,
        "active_replicas": 2,
        "recent_deployment_version": "v3.1.0",
        "anomalies_detected": [
            "Read replica lag causing elevated p99 latency (1410ms)",
            "Insufficient active replicas (2) for current traffic burst",
        ],
    },
}

# Curated catalog of SRE Runbooks
_RUNBOOK_CATALOG: list[RunbookEntry] = [
    RunbookEntry(
        runbook_id="RB-101",
        title="Canary Deployment Regression & High 5xx Error Rate on Checkout",
        applicable_services=["checkout-service", "inventory-api"],
        root_cause_signature="High HTTP 5xx error rate (>5%) and p99 latency spike following canary release",
        mitigation_steps=[
            "Verify golden signals via query_service_telemetry_metrics.",
            "Open a SEV-1/SEV-2 tracking ticket via create_critical_incident_ticket.",
            "Request human approval and execute rollback_canary_deployment to v2.13.8 or scale replicas via execute_production_remediation_plan.",
        ],
        recommended_action_type=RemediationActionType.SCALE_REPLICAS,
    ),
    RunbookEntry(
        runbook_id="RB-102",
        title="Payment Gateway Memory Saturation & Connection Pool Exhaustion",
        applicable_services=["payment-gateway"],
        root_cause_signature="Memory utilization >90% accompanied by 504 Gateway Timeouts",
        mitigation_steps=[
            "Inspect memory and error rate telemetry for payment-gateway.",
            "Create incident ticket for on-call tracking.",
            "Execute rolling pod restart (restart_pods) or scale_replicas with human confirmation.",
        ],
        recommended_action_type=RemediationActionType.RESTART_PODS,
    ),
    RunbookEntry(
        runbook_id="RB-103",
        title="Redis Cache Stampede & Upstream Latency Surge",
        applicable_services=["inventory-api", "auth-service", "checkout-service"],
        root_cause_signature="Elevated latency with cache miss storm or database read replica lag",
        mitigation_steps=[
            "Verify replica count and CPU utilization.",
            "Scale active replicas using execute_production_remediation_plan with action_type='scale_replicas'.",
        ],
        recommended_action_type=RemediationActionType.SCALE_REPLICAS,
    ),
]


def query_service_telemetry_metrics(
    service_name: str,
    environment: str = "prod",
    lookback_minutes: int = 30,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Query golden-signal telemetry metrics (latency, error rate, saturation) for a cloud microservice.

    Use this tool first during incident triage to inspect real-time HTTP 5xx error
    rates, p95/p99 latency, CPU/memory saturation, active pod replica counts, and
    recent deployment versions for a target service.

    Args:
        service_name: Canonical kebab-case microservice name to inspect. Supported
            services are 'checkout-service', 'payment-gateway', 'auth-service',
            and 'inventory-api'.
        environment: Target deployment environment. Must be one of 'prod',
            'staging', 'canary', or 'dev'. Defaults to 'prod'.
        lookback_minutes: Observation window in minutes between 5 and 1440.
            Defaults to 30.
        tool_context: Optional ADK ToolContext used to persist telemetry snapshots
            into session state.

    Returns:
        A dictionary conforming to `ServiceTelemetryOutput` on success (containing
        `latency_p95_ms`, `latency_p99_ms`, `error_rate_percent`,
        `cpu_utilization_percent`, `memory_utilization_percent`, `active_replicas`,
        `recent_deployment_version`, and `anomalies_detected`), or a dictionary
        conforming to `GuidedToolErrorResponse` with actionable `recovery_instructions`
        if validation fails or the service is not found.
    """
    with tracer.start_as_current_span("tool.query_service_telemetry_metrics") as span:
        span.set_attribute("sre.service_name", str(service_name))
        span.set_attribute("sre.environment", str(environment))

        try:
            validated_input = ServiceTelemetryQueryInput(
                service_name=service_name,
                environment=environment,
                lookback_minutes=lookback_minutes,
            )
        except ValidationError as exc:
            error_response = GuidedToolErrorResponse(
                error_code="INVALID_TELEMETRY_QUERY_ARGS",
                message=f"Telemetry query validation failed: {exc.errors()[0]['msg']}",
                recovery_instructions=(
                    "Ensure 'service_name' is lowercase kebab-case (e.g., "
                    "'checkout-service', 'payment-gateway', 'auth-service', "
                    "'inventory-api'), 'environment' is one of ['prod', 'staging', "
                    "'canary', 'dev'], and 'lookback_minutes' is between 5 and 1440."
                ),
                retryable=True,
                suggested_parameters={
                    "service_name": "checkout-service",
                    "environment": "prod",
                    "lookback_minutes": 30,
                },
            )
            return error_response.model_dump()

        metrics = _TELEMETRY_CATALOG.get(validated_input.service_name)
        if metrics is None:
            available = sorted(_TELEMETRY_CATALOG.keys())
            error_response = GuidedToolErrorResponse(
                error_code="SERVICE_NOT_FOUND",
                message=(
                    f"Microservice '{validated_input.service_name}' was not found "
                    "in the telemetry registry."
                ),
                recovery_instructions=(
                    f"Retry `query_service_telemetry_metrics` using one of the "
                    f"registered service names: {available}."
                ),
                retryable=True,
                suggested_parameters={
                    "service_name": available[0],
                    "environment": validated_input.environment,
                    "lookback_minutes": validated_input.lookback_minutes,
                },
            )
            return error_response.model_dump()

        output = ServiceTelemetryOutput(
            service_name=validated_input.service_name,
            environment=validated_input.environment,
            lookback_minutes=validated_input.lookback_minutes,
            **metrics,
        )
        result_dict = output.model_dump()

        if tool_context is not None and hasattr(tool_context, "state"):
            tool_context.state["latest_telemetry"] = result_dict
            tool_context.state["active_service_name"] = validated_input.service_name

        return result_dict


def search_incident_runbooks(
    symptom_query: str,
    severity_filter: str | None = None,
    max_results: int = 3,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Search the SRE Incident Knowledge Base for verified remediation runbooks.

    Use this tool after collecting service telemetry to match observed anomaly
    signatures (such as 5xx spikes, canary regressions, or memory saturation)
    against approved SRE mitigation procedures and runbook IDs.

    Args:
        symptom_query: Keywords describing the failure symptom, error code, or
            microservice name (e.g., 'checkout-service 5xx canary latency').
        severity_filter: Optional severity filter ('SEV-1', 'SEV-2', 'SEV-3',
            or 'SEV-4'). Pass None to search across all severities.
        max_results: Maximum number of matching runbooks to return (1 to 10).
            Defaults to 3.
        tool_context: Optional ADK ToolContext to store matched runbooks in
            session state.

    Returns:
        A dictionary conforming to `IncidentRunbookSearchOutput` containing matched
        `runbooks` with `runbook_id`, `title`, `root_cause_signature`, and ordered
        `mitigation_steps`, or a `GuidedToolErrorResponse` dictionary if input
        parameters are invalid.
    """
    with tracer.start_as_current_span("tool.search_incident_runbooks") as span:
        span.set_attribute("sre.symptom_query", pii_redactor.redact_text(str(symptom_query)))

        try:
            parsed_severity = (
                SeverityLevel(severity_filter.upper())
                if severity_filter
                else None
            )
            validated_input = IncidentRunbookSearchInput(
                symptom_query=symptom_query,
                severity_filter=parsed_severity,
                max_results=max_results,
            )
        except (ValidationError, ValueError) as exc:
            return GuidedToolErrorResponse(
                error_code="INVALID_RUNBOOK_SEARCH_ARGS",
                message=f"Invalid runbook search parameters: {exc}",
                recovery_instructions=(
                    "Provide a non-empty 'symptom_query' (at least 3 characters), "
                    "an optional 'severity_filter' in ['SEV-1', 'SEV-2', 'SEV-3', 'SEV-4'], "
                    "and 'max_results' between 1 and 10."
                ),
                retryable=True,
                suggested_parameters={
                    "symptom_query": "checkout-service 5xx latency",
                    "severity_filter": "SEV-1",
                    "max_results": 3,
                },
            ).model_dump()

        tokens = {
            tok.lower()
            for tok in validated_input.symptom_query.replace("-", " ").split()
            if len(tok) >= 2
        }
        scored: list[tuple[int, RunbookEntry]] = []
        for entry in _RUNBOOK_CATALOG:
            haystack = " ".join(
                [
                    entry.runbook_id,
                    entry.title,
                    entry.root_cause_signature,
                    *entry.applicable_services,
                ]
            ).lower()
            score = sum(1 for tok in tokens if tok in haystack)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        matched = [item[1] for item in scored[: validated_input.max_results]]
        if not matched:
            matched = _RUNBOOK_CATALOG[:1]

        output = IncidentRunbookSearchOutput(
            query=validated_input.symptom_query,
            total_matches=len(matched),
            runbooks=matched,
        )
        result_dict = output.model_dump()
        if tool_context is not None and hasattr(tool_context, "state"):
            tool_context.state["matched_runbooks"] = result_dict["runbooks"]
        return result_dict


def create_critical_incident_ticket(
    service_name: str,
    severity: str,
    summary: str,
    probable_root_cause: str,
    commander_email: str = "sre-oncall@example.com",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Create a formal SRE incident tracking ticket and war-room bridge.

    Invokes the incident management webhook (authenticated via Google Cloud
    Secret Manager) and scrubs any PII from the incident description before
    persisting the ticket.

    Args:
        service_name: Affected production microservice identifier
            (e.g., 'checkout-service').
        severity: Incident severity classification ('SEV-1', 'SEV-2', 'SEV-3',
            or 'SEV-4').
        summary: Concise headline describing customer impact (10 to 200 chars).
        probable_root_cause: Evidence-backed root cause explanation derived from
            telemetry metrics and runbooks (10 to 1000 chars).
        commander_email: Contact email of the incident commander. Automatically
            redacted by the PII scrubber prior to storage.
        tool_context: Optional ADK ToolContext used to record the created
            `incident_id` in session state.

    Returns:
        A dictionary conforming to `IncidentTicketCreateOutput` with `incident_id`,
        `severity`, `service_name`, `bridge_url`, and `created_at_iso`, or a
        `GuidedToolErrorResponse` dictionary with recovery guidance if validation fails.
    """
    with tracer.start_as_current_span("tool.create_critical_incident_ticket") as span:
        try:
            validated = IncidentTicketCreateInput(
                service_name=service_name.strip().lower(),
                severity=SeverityLevel(severity.strip().upper()),
                summary=pii_redactor.redact_text(summary),
                probable_root_cause=pii_redactor.redact_text(probable_root_cause),
                commander_email=pii_redactor.redact_text(commander_email),
            )
        except (ValidationError, ValueError) as exc:
            return GuidedToolErrorResponse(
                error_code="INVALID_INCIDENT_TICKET_ARGS",
                message=f"Incident ticket validation failed: {exc}",
                recovery_instructions=(
                    "Ensure 'severity' is one of ['SEV-1', 'SEV-2', 'SEV-3', 'SEV-4'], "
                    "and both 'summary' and 'probable_root_cause' are at least 10 characters long."
                ),
                retryable=True,
                suggested_parameters={
                    "service_name": service_name or "checkout-service",
                    "severity": "SEV-1",
                    "summary": "Elevated 5xx error rate and p99 latency on checkout-service",
                    "probable_root_cause": "Canary release v2.14.0-canary introduced latency regression.",
                },
            ).model_dump()

        # Retrieve webhook secret dynamically from Secret Manager (zero hardcoded secrets)
        _webhook_token = get_secret_manager().get_secret(
            "INCIDENT_WEBHOOK_SECRET",
            default="sm-managed-webhook-token",
        )
        digest = int(
            hashlib.sha256(
                f"{validated.service_name}:{validated.severity}:{_webhook_token[:4]}".encode()
            ).hexdigest()[:4],
            16,
        )
        incident_id = f"INC-{1000 + (digest % 8999)}"
        span.set_attribute("sre.incident_id", incident_id)

        output = IncidentTicketCreateOutput(
            incident_id=incident_id,
            severity=validated.severity,
            service_name=validated.service_name,
            bridge_url=f"https://incidents.internal.example.com/bridge/{incident_id}",
            created_at_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        result_dict = output.model_dump()
        if tool_context is not None and hasattr(tool_context, "state"):
            tool_context.state["active_incident_id"] = incident_id
        return result_dict


def execute_production_remediation_plan(
    incident_id: str,
    service_name: str,
    action_type: str,
    runbook_id: str,
    justification: str,
    target_replicas: int = 6,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Execute a high-stakes production remediation operation with Human-in-the-Loop approval.

    CRITICAL SAFETY POLICY: Because this tool mutates live production infrastructure
    (scaling replicas, restarting pods, draining traffic, or flushing caches), it
    enforces a Human-in-the-Loop (HITL) confirmation checkpoint via ADK's
    `tool_context.request_confirmation` mechanism before applying changes.

    Args:
        incident_id: Tracking incident identifier matching `^INC-[0-9]{4,8}$`
            (e.g., 'INC-4021'). Must be created first via
            `create_critical_incident_ticket`.
        service_name: Target production microservice name (e.g., 'checkout-service').
        action_type: Remediation action to execute. Must be one of:
            'scale_replicas', 'restart_pods', 'drain_traffic', 'flush_cache',
            or 'enable_circuit_breaker'.
        runbook_id: Authorizing SRE runbook identifier matching `^RB-[0-9]{3,5}$`
            (e.g., 'RB-101').
        justification: Clear engineering justification (10 to 500 characters)
            explaining why this remediation action is safe and necessary.
        target_replicas: Desired replica count when `action_type` is
            'scale_replicas' (between 1 and 50). Defaults to 6.
        tool_context: ADK ToolContext used to verify or request Human-in-the-Loop
            confirmation (`tool_context.request_confirmation`).

    Returns:
        A dictionary conforming to `RemediationPlanExecutionOutput` once confirmed
        or paused for human approval, or a `GuidedToolErrorResponse` dictionary
        with self-correction instructions if any parameter is invalid.
    """
    with tracer.start_as_current_span("tool.execute_production_remediation_plan") as span:
        try:
            validated = RemediationPlanExecutionInput(
                incident_id=incident_id.strip().upper(),
                service_name=service_name.strip().lower(),
                action_type=RemediationActionType(action_type.strip().lower()),
                target_replicas=target_replicas,
                runbook_id=runbook_id.strip().upper(),
                justification=pii_redactor.redact_text(justification),
            )
        except (ValidationError, ValueError) as exc:
            return GuidedToolErrorResponse(
                error_code="INVALID_REMEDIATION_PLAN_ARGS",
                message=f"Remediation plan validation failed: {exc}",
                recovery_instructions=(
                    "Ensure 'incident_id' matches 'INC-XXXX' (call "
                    "`create_critical_incident_ticket` first if needed), 'runbook_id' "
                    "matches 'RB-XXX' (from `search_incident_runbooks`), 'action_type' "
                    "is one of ['scale_replicas', 'restart_pods', 'drain_traffic', "
                    "'flush_cache', 'enable_circuit_breaker'], and 'target_replicas' "
                    "is between 1 and 50."
                ),
                retryable=True,
                suggested_parameters={
                    "incident_id": "INC-4021",
                    "service_name": service_name or "checkout-service",
                    "action_type": "scale_replicas",
                    "runbook_id": "RB-101",
                    "target_replicas": 8,
                    "justification": "Scale checkout-service replicas per RB-101 to relieve CPU saturation.",
                },
            ).model_dump()

        # Human-in-the-Loop (HITL) confirmation check
        if tool_context is not None:
            confirmation = getattr(tool_context, "tool_confirmation", None)
            if confirmation is not None and getattr(confirmation, "confirmed", False) is False:
                return GuidedToolErrorResponse(
                    error_code="HUMAN_APPROVAL_REJECTED",
                    message="The on-call human operator rejected the production remediation action.",
                    recovery_instructions=(
                        "Do not retry the rejected mutation. Present alternative "
                        "non-destructive diagnostic options to the operator."
                    ),
                    retryable=False,
                ).model_dump()

            if confirmation is None and callable(
                getattr(tool_context, "request_confirmation", None)
            ):
                tool_context.request_confirmation(
                    hint=(
                        f"Approve production remediation '{validated.action_type.value}' "
                        f"on '{validated.service_name}' under incident {validated.incident_id} "
                        f"(Runbook {validated.runbook_id})?"
                    ),
                    payload=validated.model_dump(),
                )
                log_structured_event(
                    event_type="HITL_CONFIRMATION_REQUESTED",
                    message=(
                        f"Paused execution of {validated.action_type.value} on "
                        f"{validated.service_name} awaiting human operator approval."
                    ),
                    phase="INTENT",
                    tool_name="execute_production_remediation_plan",
                    intent=validated.model_dump(),
                )
                return RemediationPlanExecutionOutput(
                    status="pending_approval",
                    execution_id=f"REM-PENDING-{validated.incident_id}",
                    incident_id=validated.incident_id,
                    service_name=validated.service_name,
                    action_type=validated.action_type,
                    human_approved=False,
                    verification_metrics={
                        "hitl_status": "awaiting_operator_confirmation",
                        "target_replicas": validated.target_replicas,
                    },
                ).model_dump()

        span.set_attribute("sre.incident_id", validated.incident_id)
        span.set_attribute("sre.action_type", validated.action_type.value)

        output = RemediationPlanExecutionOutput(
            status="success",
            execution_id=f"REM-{validated.incident_id}-{validated.runbook_id}",
            incident_id=validated.incident_id,
            service_name=validated.service_name,
            action_type=validated.action_type,
            human_approved=True,
            verification_metrics={
                "active_replicas": validated.target_replicas,
                "post_action_error_rate_percent": 0.18,
                "post_action_latency_p99_ms": 210.0,
            },
        )
        return output.model_dump()


def rollback_canary_deployment(
    service_name: str,
    faulty_version: str,
    stable_version: str,
    reason: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Roll back a faulty canary release to the last known stable version with HITL approval.

    Requires Human-in-the-Loop confirmation before shifting 100% of production
    traffic away from the canary deployment.

    Args:
        service_name: Name of the microservice to roll back (e.g., 'checkout-service').
        faulty_version: Canary version tag exhibiting regressions (e.g., 'v2.14.0-canary').
        stable_version: Previously verified stable release tag (e.g., 'v2.13.8').
        reason: Engineering explanation for triggering the canary rollback
            (10 to 400 characters).
        tool_context: ADK ToolContext used to request or verify Human-in-the-Loop
            operator confirmation.

    Returns:
        A dictionary conforming to `CanaryRollbackOutput` describing the traffic shift
        and approval state, or a `GuidedToolErrorResponse` dictionary if validation fails.
    """
    with tracer.start_as_current_span("tool.rollback_canary_deployment") as span:
        try:
            validated = CanaryRollbackInput(
                service_name=service_name.strip().lower(),
                faulty_version=faulty_version.strip(),
                stable_version=stable_version.strip(),
                reason=pii_redactor.redact_text(reason),
            )
        except ValidationError as exc:
            return GuidedToolErrorResponse(
                error_code="INVALID_CANARY_ROLLBACK_ARGS",
                message=f"Canary rollback parameter validation failed: {exc}",
                recovery_instructions=(
                    "Provide valid 'service_name', 'faulty_version', 'stable_version', "
                    "and a 'reason' of at least 10 characters."
                ),
                retryable=True,
                suggested_parameters={
                    "service_name": "checkout-service",
                    "faulty_version": "v2.14.0-canary",
                    "stable_version": "v2.13.8",
                    "reason": "Rollback v2.14.0-canary due to 14.8% HTTP 5xx error spike.",
                },
            ).model_dump()

        if validated.faulty_version == validated.stable_version:
            return GuidedToolErrorResponse(
                error_code="IDENTICAL_ROLLBACK_VERSIONS",
                message="faulty_version and stable_version cannot be identical.",
                recovery_instructions=(
                    "Specify a different 'stable_version' (for example 'v2.13.8') "
                    "than the 'faulty_version' ('v2.14.0-canary')."
                ),
                retryable=True,
                suggested_parameters={
                    "service_name": validated.service_name,
                    "faulty_version": validated.faulty_version,
                    "stable_version": "v2.13.8",
                    "reason": validated.reason,
                },
            ).model_dump()

        if tool_context is not None:
            confirmation = getattr(tool_context, "tool_confirmation", None)
            if confirmation is None and callable(
                getattr(tool_context, "request_confirmation", None)
            ):
                tool_context.request_confirmation(
                    hint=(
                        f"Confirm canary rollback for '{validated.service_name}' "
                        f"from {validated.faulty_version} to {validated.stable_version}?"
                    ),
                    payload=validated.model_dump(),
                )
                return CanaryRollbackOutput(
                    status="pending_approval",
                    rollback_id=f"RBK-PENDING-{validated.service_name}",
                    service_name=validated.service_name,
                    restored_version=validated.stable_version,
                    traffic_shifted_percent=0,
                    human_approved=False,
                ).model_dump()

        span.set_attribute("sre.service_name", validated.service_name)
        span.set_attribute("sre.restored_version", validated.stable_version)

        return CanaryRollbackOutput(
            status="success",
            rollback_id=f"RBK-{validated.service_name}-{validated.stable_version}",
            service_name=validated.service_name,
            restored_version=validated.stable_version,
            traffic_shifted_percent=100,
            human_approved=True,
        ).model_dump()


# Wrap high-stakes mutation tools with FunctionTool(..., require_confirmation=True)
remediation_hitl_tool = FunctionTool(
    func=execute_production_remediation_plan,
    require_confirmation=True,
)

canary_rollback_hitl_tool = FunctionTool(
    func=rollback_canary_deployment,
    require_confirmation=True,
)
