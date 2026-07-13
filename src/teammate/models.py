from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_status(status: str, allowed: set[str], kind: str) -> None:
    if status not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"invalid {kind} status {status!r}; expected one of: {choices}")


@dataclass
class Team:
    STATUSES: ClassVar[set[str]] = {"created", "running", "completed", "failed", "cancelled"}
    TRANSITIONS: ClassVar[dict[str, set[str]]] = {
        "created": {"running", "cancelled"},
        "running": {"completed", "failed", "cancelled"},
        "failed": {"running", "cancelled"},
        "completed": set(),
        "cancelled": set(),
    }

    team_id: str
    team_name: str
    lead_agent_id: str
    description: str | None = None
    agent_type: str | None = None
    status: str = "created"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_status(self.status, self.STATUSES, "team")

    def transition_to(self, status: str) -> None:
        _require_status(status, self.STATUSES, "team")
        if status != self.status and status not in self.TRANSITIONS[self.status]:
            raise ValueError(f"cannot transition team from {self.status!r} to {status!r}")
        self.status = status
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Team":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


@dataclass
class AgentRecord:
    STATUSES: ClassVar[set[str]] = {"created", "running", "idle", "completed", "failed", "cancelled"}
    TRANSITIONS: ClassVar[dict[str, set[str]]] = {
        "created": {"running", "cancelled"},
        "running": {"idle", "completed", "failed", "cancelled"},
        "idle": {"running", "completed", "cancelled"},
        "failed": {"running", "cancelled"},
        "completed": set(),
        "cancelled": set(),
    }

    agent_id: str
    team_id: str
    name: str
    role: str
    session_id: str
    model: str | None = None
    instructions: str = ""
    tools: list[str] = field(default_factory=list)
    status: str = "created"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_status(self.status, self.STATUSES, "agent")

    def transition_to(self, status: str) -> None:
        _require_status(status, self.STATUSES, "agent")
        if status != self.status and status not in self.TRANSITIONS[self.status]:
            raise ValueError(f"cannot transition agent from {self.status!r} to {status!r}")
        self.status = status
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentRecord":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


@dataclass
class TeamTask:
    STATUSES: ClassVar[set[str]] = {"pending", "in_progress", "completed", "failed", "cancelled"}
    TRANSITIONS: ClassVar[dict[str, set[str]]] = {
        "pending": {"in_progress", "completed", "cancelled"},
        "in_progress": {"completed", "failed", "cancelled"},
        "failed": {"pending", "in_progress", "cancelled"},
        "completed": set(),
        "cancelled": set(),
    }

    id: str
    subject: str
    description: str
    key: str | None = None
    activeForm: str = ""
    status: str = "pending"
    owner: str | None = None
    blocks: list[str] = field(default_factory=list)
    blockedBy: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_status(self.status, self.STATUSES, "task")

    def transition_to(self, status: str) -> None:
        _require_status(status, self.STATUSES, "task")
        if status != self.status and status not in self.TRANSITIONS[self.status]:
            raise ValueError(f"cannot transition task from {self.status!r} to {status!r}")
        self.status = status
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeamTask":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


@dataclass
class Message:
    STATUSES: ClassVar[set[str]] = {"queued", "delivered", "consumed", "failed"}
    TRANSITIONS: ClassVar[dict[str, set[str]]] = {
        "queued": {"delivered", "failed"},
        "delivered": {"consumed", "failed"},
        "consumed": set(),
        "failed": set(),
    }

    message_id: str
    team_id: str
    sender_id: str
    recipient_id: str
    content: Any
    summary: str | None = None
    status: str = "queued"
    created_at: str = field(default_factory=utc_now)
    delivered_at: str | None = None
    consumed_at: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_status(self.status, self.STATUSES, "message")

    def transition_to(self, status: str) -> None:
        _require_status(status, self.STATUSES, "message")
        if status != self.status and status not in self.TRANSITIONS[self.status]:
            raise ValueError(f"cannot transition message from {self.status!r} to {status!r}")
        now = utc_now()
        self.status = status
        if status == "delivered":
            self.delivered_at = now
        elif status == "consumed":
            self.consumed_at = now

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})
