from __future__ import annotations

import json
import time
from typing import Any

from ..teammate.trace import redact_trace_value


_EVENT_TYPES = {
    "run_started": "agent_loop.started",
    "run_completed": "agent_loop.completed",
    "run_failed": "agent_loop.failed",
    "run_cancelled": "agent_loop.cancelled",
    "model_started": "model.started",
    "model_response": "model.response",
    "model_error": "model.failed",
    "tool_use": "tool.started",
    "tool_result": "tool.completed",
    "tool_error": "tool.failed",
}


class PeerTraceRecorder:
    def __init__(self, context: Any):
        self.context = context

    def record(self, event: Any) -> None:
        if self.context.peer_store is None or self.context.peer_run_id is None:
            return
        event_type = _EVENT_TYPES.get(event.kind, str(event.kind).replace("_", "."))
        if event.kind == "tool_result" and event.is_error:
            event_type = "tool.failed"
        data: dict[str, Any] = {
            "peer_id": self.context.peer_id,
            "actor_id": self.context.peer_id,
        }
        for source in (
            "turn",
            "model",
            "finish_reason",
            "content",
            "usage",
            "tool_name",
            "tool_input",
            "tool_output",
            "tool_use_id",
            "duration_ms",
            "error",
        ):
            value = getattr(event, source, None)
            if value is not None:
                data[source] = value
        if getattr(event, "is_error", False):
            data["is_error"] = True
        safe = json.loads(
            json.dumps(redact_trace_value(data), ensure_ascii=False, default=str)
        )
        self.context.peer_store.append_event(
            self.context.peer_run_id,
            event_type,
            safe,
            created_at=event.created_at,
            monotonic_ns=time.monotonic_ns(),
        )
