from __future__ import annotations

import json
import re
from typing import Any


_EVENT_TYPES = {
    "run_started": "run.started",
    "run_completed": "run.completed",
    "run_failed": "run.failed",
    "model_started": "model.started",
    "model_response": "model.response",
    "model_error": "model.failed",
    "tool_use": "tool.started",
    "tool_result": "tool.completed",
    "tool_error": "tool.failed",
}
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "access_token",
    "refresh_token",
    "password",
    "passwd",
    "secret",
    "credential",
    "cookie",
    "set_cookie",
}
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([a-z0-9_-]*(?:api[_-]?key|auth(?:orization)?[_-]?token|access[_-]?token|"
    r"refresh[_-]?token|password|secret))\b(\s*[:=]\s*)([^\s'\"\\]+)"
)
_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]+")


def redact_trace_value(value: Any) -> Any:
    """Return a JSON-safe copy with common credential fields redacted."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _SENSITIVE_KEYS or (
                normalized.endswith("_token") and not normalized.endswith("_tokens")
            ):
                redacted[str(key)] = "[REDACTED]"
            else:
                redacted[str(key)] = redact_trace_value(item)
        return redacted
    if isinstance(value, (list, tuple, set)):
        return [redact_trace_value(item) for item in value]
    if isinstance(value, str):
        value = _ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", value)
        return _BEARER_RE.sub(lambda match: f"{match.group(1)}[REDACTED]", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


class TeamTraceRecorder:
    """Attach agent-loop events to the team created or active during a run."""

    def __init__(self, tool_context: Any):
        self.context = tool_context
        self.team_id: str | None = None
        self.pending: list[dict[str, Any]] = []
        self._bind_existing_team()

    def _bind_existing_team(self) -> None:
        try:
            team = self.context.team_store.load_active_team()
        except Exception:
            return
        if team is None:
            return
        is_child_run = bool(self.context.actor_id or self.context.current_task_id)
        if is_child_run or team.status in {"created", "running", "failed"}:
            self.team_id = team.team_id

    def _try_bind_new_team(self) -> None:
        if self.team_id is not None:
            return
        try:
            team = self.context.team_store.load_active_team()
        except Exception:
            return
        if team is not None and team.status in {"created", "running", "failed"}:
            self.team_id = team.team_id

    def record(self, event: Any) -> None:
        try:
            packed = self._pack(event)
            self._try_bind_new_team()
            if self.team_id is None:
                self.pending.append(packed)
                return
            queued = [*self.pending, packed]
            self.pending.clear()
            for item in queued:
                self.context.team_store.append_event(
                    self.team_id,
                    item["type"],
                    item["data"],
                    created_at=item["created_at"],
                )
        except Exception:
            # Tracing must never make the underlying agent run fail.
            return

    def _pack(self, event: Any) -> dict[str, Any]:
        event_type = _EVENT_TYPES.get(event.kind, str(event.kind).replace("_", "."))
        if event.kind == "tool_result" and event.is_error:
            event_type = "tool.failed"
        actor_id, actor_name = self._actor_identity()
        data: dict[str, Any] = {
            "actor_id": actor_id,
            "actor_name": actor_name,
            "task_id": self.context.current_task_id,
        }
        for source, target in (
            ("turn", "turn"),
            ("model", "model"),
            ("finish_reason", "finish_reason"),
            ("content", "content"),
            ("usage", "usage"),
            ("tool_name", "tool_name"),
            ("tool_input", "tool_input"),
            ("tool_output", "tool_output"),
            ("tool_use_id", "tool_use_id"),
            ("duration_ms", "duration_ms"),
            ("error", "error"),
        ):
            value = getattr(event, source, None)
            if value is not None:
                data[target] = value
        if getattr(event, "is_error", False):
            data["is_error"] = True
        safe = redact_trace_value(data)
        return {
            "type": event_type,
            "created_at": event.created_at,
            "data": json.loads(json.dumps(safe, ensure_ascii=False, default=str)),
        }

    def _actor_identity(self) -> tuple[str | None, str | None]:
        try:
            team = self.context.team_store.load_team(self.team_id) if self.team_id else None
            if team is None:
                team = self.context.team_store.load_active_team()
            if team is None:
                return self.context.actor_id, None
            actor_id = self.context.actor_id or team.lead_agent_id
            if actor_id == team.lead_agent_id:
                return actor_id, "lead"
            agent = self.context.team_store.load_agent(team.team_id, actor_id)
            return actor_id, agent.name if agent is not None else actor_id
        except Exception:
            return self.context.actor_id, None
