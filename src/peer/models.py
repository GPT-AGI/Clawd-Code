from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _from_dict(cls: type[Any], data: dict[str, Any]) -> Any:
    return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


@dataclass(frozen=True)
class PeerRunConfig:
    repo_path: str
    mission: str
    peers: int
    communication: str = "p2p"
    workspace_mode: str = "worktree"
    provider: str = "scripted"
    model: str | None = None
    timeout_seconds: float = 300.0
    max_turns: int = 30
    max_output_tokens: int = 4096
    token_budget: int | None = None
    turn_budget: int | None = None
    output_dir: str | None = None
    coordinator_peer: str | None = None
    acceptance_command: list[str] | None = None
    cleanup_worktrees: bool = True

    CONDITIONS: ClassVar[set[str]] = {
        "solo",
        "independent",
        "none",
        "artifact-only",
        "star",
        "p2p",
    }
    WORKSPACE_MODES: ClassVar[set[str]] = {"shared", "worktree"}

    def validate(self) -> None:
        if not isinstance(self.mission, str) or not self.mission.strip():
            raise ValueError("mission must be a non-empty string")
        if not isinstance(self.repo_path, str) or not self.repo_path.strip():
            raise ValueError("repo_path must be a non-empty path")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string")
        if self.peers < 1 or self.peers > 32:
            raise ValueError("peers must be between 1 and 32")
        if self.communication not in self.CONDITIONS:
            raise ValueError(
                "communication must be one of: " + ", ".join(sorted(self.CONDITIONS))
            )
        if self.communication == "solo" and self.peers != 1:
            raise ValueError("solo communication requires exactly one peer")
        if self.communication != "solo" and self.peers < 2:
            raise ValueError(f"{self.communication} communication requires at least two peers")
        if self.workspace_mode not in self.WORKSPACE_MODES:
            raise ValueError("workspace_mode must be shared or worktree")
        if self.timeout_seconds <= 0 or self.timeout_seconds > 86_400:
            raise ValueError("timeout_seconds must be between 0 and 86400")
        if self.max_turns < 1 or self.max_turns > 100_000:
            raise ValueError("max_turns must be between 1 and 100000")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.token_budget is not None and self.token_budget < 1:
            raise ValueError("token_budget must be positive")
        if self.turn_budget is not None and self.turn_budget < 1:
            raise ValueError("turn_budget must be positive")
        if self.communication == "star" and self.coordinator_peer is not None:
            value = self.coordinator_peer.strip()
            if not value:
                raise ValueError("coordinator_peer must be non-empty")
        if self.acceptance_command is not None and (
            not self.acceptance_command
            or not all(isinstance(item, str) and item for item in self.acceptance_command)
        ):
            raise ValueError("acceptance_command must be a non-empty argv list")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PeerRunRecord:
    STATUSES: ClassVar[set[str]] = {
        "created",
        "running",
        "submitted",
        "completed",
        "cancelled",
        "timed_out",
        "budget_exhausted",
        "failed",
    }

    run_id: str
    mission: str
    repo_path: str
    base_revision: str
    peer_count: int
    communication: str
    workspace_mode: str
    provider: str
    model: str | None
    timeout_seconds: float
    max_turns: int
    max_output_tokens: int
    token_budget: int | None
    turn_budget: int | None
    output_dir: str
    coordinator_peer_id: str | None = None
    acceptance_command: list[str] | None = None
    status: str = "created"
    stop_reason: str | None = None
    accepted_submission: dict[str, Any] | None = None
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "total_tokens": 0,
            "turns": 0,
            "model_calls": 0,
            "tool_calls": 0,
        }
    )
    started_at: str | None = None
    completed_at: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.status not in self.STATUSES:
            raise ValueError(f"invalid peer run status: {self.status}")

    def set_status(self, status: str, *, reason: str | None = None) -> None:
        if status not in self.STATUSES:
            raise ValueError(f"invalid peer run status: {status}")
        self.status = status
        self.stop_reason = reason
        self.updated_at = utc_now()
        if status == "running":
            self.started_at = self.started_at or self.updated_at
        if status in {
            "completed",
            "cancelled",
            "timed_out",
            "budget_exhausted",
            "failed",
        }:
            self.completed_at = self.updated_at

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PeerRunRecord":
        return _from_dict(cls, data)


@dataclass
class PeerParticipant:
    STATUSES: ClassVar[set[str]] = {
        "created",
        "running",
        "idle",
        "stopping",
        "stopped",
        "failed",
    }

    peer_id: str
    run_id: str
    name: str
    session_id: str
    workspace_mode: str
    workspace_path: str
    status: str = "created"
    start_monotonic_ns: int | None = None
    started_at: str | None = None
    idle_at: str | None = None
    wake_at: str | None = None
    stopped_at: str | None = None
    error_at: str | None = None
    last_error: str | None = None
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "total_tokens": 0,
            "turns": 0,
            "model_calls": 0,
            "tool_calls": 0,
        }
    )
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.status not in self.STATUSES:
            raise ValueError(f"invalid peer status: {self.status}")

    def set_status(self, status: str, *, error: str | None = None) -> None:
        if status not in self.STATUSES:
            raise ValueError(f"invalid peer status: {status}")
        now = utc_now()
        self.status = status
        self.updated_at = now
        if status == "running":
            self.started_at = self.started_at or now
        elif status == "idle":
            self.idle_at = now
        elif status == "stopped":
            self.stopped_at = now
        elif status == "failed":
            self.error_at = now
            self.last_error = error

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PeerParticipant":
        return _from_dict(cls, data)


@dataclass
class PeerMessage:
    message_id: str
    run_id: str
    sender_id: str
    recipient_id: str
    payload: Any
    payload_size_bytes: int
    summary: str | None = None
    broadcast_id: str | None = None
    idempotency_key: str | None = None
    status: str = "delivered"
    created_at: str = field(default_factory=utc_now)
    delivered_at: str = field(default_factory=utc_now)
    consumed_at: str | None = None
    schema_version: int = 1

    def consume(self) -> None:
        if self.status == "delivered":
            self.status = "consumed"
            self.consumed_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PeerMessage":
        return _from_dict(cls, data)


@dataclass
class PeerBroadcast:
    broadcast_id: str
    run_id: str
    sender_id: str
    recipients: list[str]
    message_ids: list[str]
    payload_size_bytes: int
    idempotency_key: str | None = None
    created_at: str = field(default_factory=utc_now)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PeerBroadcast":
        return _from_dict(cls, data)


@dataclass
class PeerSubmission:
    attempt_id: str
    run_id: str
    peer_id: str
    revision: str
    summary: str
    status: str
    validation: dict[str, Any]
    created_at: str = field(default_factory=utc_now)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PeerSubmission":
        return _from_dict(cls, data)
