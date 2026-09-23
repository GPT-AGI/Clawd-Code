from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class WebhookEvent:
    tenant_id: str
    event_id: str
    sequence: int
    payload: dict[str, Any]


class WebhookProcessor:
    def __init__(self) -> None:
        self._processed: set[str] = set()
        self._last_sequence = 0

    def process(
        self,
        event: WebhookEvent,
        handler: Callable[[WebhookEvent], None],
    ) -> str:
        if event.event_id in self._processed:
            return "duplicate"
        self._processed.add(event.event_id)
        self._last_sequence = max(self._last_sequence, event.sequence)
        handler(event)
        return "processed"

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            "global": {
                "processed_ids": list(self._processed),
                "last_sequence": self._last_sequence,
            }
        }
