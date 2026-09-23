from __future__ import annotations

import threading
import time
from typing import Any

from .policy import CommunicationPolicy
from .store import PeerStore
from .workspace import PeerWorkspaceManager


class PeerRunControl:
    def __init__(
        self,
        run_id: str,
        store: PeerStore,
        policy: CommunicationPolicy,
        workspace: PeerWorkspaceManager,
        *,
        timeout_seconds: float,
        token_budget: int | None,
        turn_budget: int | None,
    ) -> None:
        self.run_id = run_id
        self.store = store
        self.policy = policy
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds
        self.token_budget = token_budget
        self.turn_budget = turn_budget
        self.started_monotonic = time.monotonic()
        self.stop_event = threading.Event()
        self._lock = threading.RLock()
        self._reason: str | None = None

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def should_stop(self) -> bool:
        return self.stop_event.is_set()

    def remaining_seconds(self) -> float:
        return max(0.0, self.timeout_seconds - (time.monotonic() - self.started_monotonic))

    def request_stop(self, reason: str) -> None:
        with self._lock:
            if self._reason is None:
                self._reason = reason
            self.stop_event.set()
            self.store.signal_bus.notify_run(self.run_id)

    def submit(self, peer_id: str, revision: str, summary: str) -> dict[str, Any]:
        validation = self.workspace.validate_revision(revision)
        submission, accepted = self.store.attempt_submission(
            self.run_id, peer_id, revision, summary, validation
        )
        if submission.status == "accepted":
            self.request_stop("submitted")
        return {
            "status": submission.status,
            "attempt": submission.to_dict(),
            "accepted_submission": accepted,
        }

    def record_usage(
        self, peer_id: str, usage_delta: dict[str, int]
    ) -> tuple[dict[str, int], dict[str, int]]:
        run_usage, peer_usage = self.store.update_usage(
            self.run_id, peer_id, usage_delta
        )
        if self.token_budget is not None and run_usage["total_tokens"] >= self.token_budget:
            self.request_stop("budget_exhausted")
        if self.turn_budget is not None and run_usage["turns"] >= self.turn_budget:
            self.request_stop("budget_exhausted")
        return run_usage, peer_usage
