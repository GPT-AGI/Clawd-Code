from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

from .models import AgentRecord, Message, Team, TeamTask, utc_now


_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _locked_path(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock(lock_path):
        with lock_path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class TeamStore:
    """Filesystem-backed storage for the active team and its shared state."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root).resolve()
        self.clawd_dir = self.workspace_root / ".clawd"
        self.teams_dir = self.clawd_dir / "teams"
        self.active_team_path = self.clawd_dir / "team.json"

    def team_dir(self, team_id: str) -> Path:
        return self.teams_dir / team_id

    def create_team(
        self,
        team_name: str,
        description: str | None = None,
        agent_type: str | None = None,
    ) -> Team:
        if self.load_active_team() is not None:
            raise ValueError("an active team already exists")

        team = Team(
            team_id=uuid.uuid4().hex[:12],
            team_name=team_name,
            description=description,
            agent_type=agent_type,
            lead_agent_id=uuid.uuid4().hex[:12],
        )
        directory = self.team_dir(team.team_id)
        for name in ("agents", "sessions", "messages"):
            (directory / name).mkdir(parents=True, exist_ok=True)

        self._write_json(directory / "team.json", team.to_dict())
        self._write_json(directory / "tasks.json", {})
        (directory / "events.jsonl").touch(exist_ok=True)
        self._write_json(self.active_team_path, team.to_dict())
        self.append_event(team.team_id, "team.created", {"team": team.to_dict()})
        return team

    def load_active_team(self) -> Team | None:
        if not self.active_team_path.exists():
            return None
        data = self._read_json(self.active_team_path)
        if "team_id" not in data:
            data = self._migrate_legacy_team(data)
        return Team.from_dict(data)

    def load_team(self, team_id: str) -> Team | None:
        path = self.team_dir(team_id) / "team.json"
        if not path.exists():
            return None
        return Team.from_dict(self._read_json(path))

    def save_team(self, team: Team) -> Path:
        path = self.team_dir(team.team_id) / "team.json"
        self._write_json(path, team.to_dict())
        active = self.load_active_team()
        if active is not None and active.team_id == team.team_id:
            self._write_json(self.active_team_path, team.to_dict())
        return path

    def load_tasks(self, team_id: str) -> dict[str, dict[str, Any]]:
        path = self.team_dir(team_id) / "tasks.json"
        if not path.exists():
            return {}
        data = self._read_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"invalid task store at {path}")
        return {task_id: TeamTask.from_dict(task).to_dict() for task_id, task in data.items()}

    def save_tasks(self, team_id: str, tasks: dict[str, dict[str, Any]]) -> None:
        serialized = {task_id: TeamTask.from_dict(task).to_dict() for task_id, task in tasks.items()}
        self._write_json(self.team_dir(team_id) / "tasks.json", serialized)

    def mutate_tasks(
        self,
        team_id: str,
        mutator: Callable[[dict[str, TeamTask]], Any],
    ) -> Any:
        """Apply one atomic mutation across the team's task collection."""
        path = self.team_dir(team_id) / "tasks.json"
        with _locked_path(path):
            data = self._read_json(path) if path.exists() else {}
            tasks = {
                task_id: TeamTask.from_dict(task)
                for task_id, task in data.items()
            }
            result = mutator(tasks)
            self._write_json_unlocked(
                path,
                {task_id: task.to_dict() for task_id, task in tasks.items()},
            )
        return result

    def update_task(self, team_id: str, task: TeamTask) -> dict[str, dict[str, Any]]:
        path = self.team_dir(team_id) / "tasks.json"
        with _locked_path(path):
            data = self._read_json(path) if path.exists() else {}
            data[task.id] = task.to_dict()
            self._write_json_unlocked(path, data)
        return self.load_tasks(team_id)

    def claim_task(
        self,
        team_id: str,
        task_id: str,
        *,
        lease_id: str,
        lease_expires_at: str,
        max_retries: int,
    ) -> TeamTask | None:
        path = self.team_dir(team_id) / "tasks.json"
        with _locked_path(path):
            data = self._read_json(path) if path.exists() else {}
            raw = data.get(task_id)
            if not isinstance(raw, dict):
                return None
            task = TeamTask.from_dict(raw)
            if task.status != "pending":
                return None
            task.transition_to("in_progress")
            task.attempt += 1
            task.max_retries = max(task.max_retries, max_retries)
            task.lease_id = lease_id
            task.lease_expires_at = lease_expires_at
            task.started_at = utc_now()
            task.completed_at = None
            task.last_error = None
            data[task_id] = task.to_dict()
            self._write_json_unlocked(path, data)
            return task

    def delete_task(self, team_id: str, task_id: str) -> dict[str, dict[str, Any]]:
        path = self.team_dir(team_id) / "tasks.json"
        with _locked_path(path):
            data = self._read_json(path) if path.exists() else {}
            data.pop(task_id, None)
            self._write_json_unlocked(path, data)
        return self.load_tasks(team_id)

    def save_agent(self, agent: AgentRecord) -> Path:
        path = self.team_dir(agent.team_id) / "agents" / f"{agent.agent_id}.json"
        self._write_json(path, agent.to_dict())
        return path

    def mutate_agent(
        self,
        team_id: str,
        agent_id: str,
        mutator: Callable[[AgentRecord], Any],
    ) -> AgentRecord | None:
        """Atomically load, mutate, and persist one agent record."""
        path = self.team_dir(team_id) / "agents" / f"{agent_id}.json"
        with _locked_path(path):
            if not path.exists():
                return None
            agent = AgentRecord.from_dict(self._read_json(path))
            mutator(agent)
            self._write_json_unlocked(path, agent.to_dict())
        return agent

    def load_agent(self, team_id: str, agent_id: str) -> AgentRecord | None:
        path = self.team_dir(team_id) / "agents" / f"{agent_id}.json"
        if not path.exists():
            return None
        return AgentRecord.from_dict(self._read_json(path))

    def list_agents(self, team_id: str) -> list[AgentRecord]:
        directory = self.team_dir(team_id) / "agents"
        if not directory.exists():
            return []
        return [AgentRecord.from_dict(self._read_json(path)) for path in sorted(directory.glob("*.json"))]

    def find_agent(self, team_id: str, identity: str) -> AgentRecord | None:
        normalized = identity.strip().lower()
        for agent in self.list_agents(team_id):
            if agent.agent_id == identity or agent.name.lower() == normalized:
                return agent
        return None

    def save_message(self, message: Message) -> Path:
        path = self.team_dir(message.team_id) / "messages" / f"{message.message_id}.json"
        self._write_json(path, message.to_dict())
        return path

    def load_message(self, team_id: str, message_id: str) -> Message | None:
        path = self.team_dir(team_id) / "messages" / f"{message_id}.json"
        if not path.exists():
            return None
        return Message.from_dict(self._read_json(path))

    def list_messages(self, team_id: str) -> list[Message]:
        directory = self.team_dir(team_id) / "messages"
        if not directory.exists():
            return []
        messages = [Message.from_dict(self._read_json(path)) for path in directory.glob("*.json")]
        messages.sort(key=lambda message: (message.created_at, message.message_id))
        return messages

    def consume_messages(self, team_id: str, recipient_id: str) -> list[Message]:
        incoming = [
            message
            for message in self.list_messages(team_id)
            if message.recipient_id == recipient_id and message.status == "delivered"
        ]
        for message in incoming:
            message.transition_to("consumed")
            self.save_message(message)
            self.append_event(
                team_id,
                "message.consumed",
                {"message_id": message.message_id, "agent_id": recipient_id},
            )
        return incoming

    def save_session(self, team_id: str, session_id: str, data: dict[str, Any]) -> Path:
        path = self.team_dir(team_id) / "sessions" / f"{session_id}.json"
        self._write_json(path, data)
        return path

    def load_session(self, team_id: str, session_id: str) -> dict[str, Any] | None:
        path = self.team_dir(team_id) / "sessions" / f"{session_id}.json"
        if not path.exists():
            return None
        return self._read_json(path)

    def list_events(self, team_id: str) -> list[dict[str, Any]]:
        path = self.team_dir(team_id) / "events.jsonl"
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
        return events

    def append_event(
        self,
        team_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
        *,
        created_at: str | None = None,
        event_id: str | None = None,
    ) -> None:
        path = self.team_dir(team_id) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "event_id": event_id or uuid.uuid4().hex,
            "team_id": team_id,
            "type": event_type,
            "created_at": created_at or utc_now(),
            "data": data or {},
        }
        with _locked_path(path):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def disband_active_team(self) -> Team | None:
        team = self.load_active_team()
        if team is None:
            return None
        if team.status in {"created", "running", "failed"}:
            team.transition_to("cancelled")
            self.save_team(team)
            self.append_event(team.team_id, "team.cancelled")
        self.active_team_path.unlink(missing_ok=True)
        return team

    def _migrate_legacy_team(self, data: dict[str, Any]) -> dict[str, Any]:
        team_name = str(data.get("team_name") or "legacy-team")
        seed = f"{self.workspace_root}:{team_name}"
        team_id = uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:12]
        lead_agent_id = str(data.get("lead_agent_id") or uuid.uuid5(uuid.NAMESPACE_OID, seed).hex[:12])
        team = Team(
            team_id=team_id,
            team_name=team_name,
            lead_agent_id=lead_agent_id,
            description=data.get("description"),
            agent_type=data.get("agent_type"),
        )
        directory = self.team_dir(team_id)
        for name in ("agents", "sessions", "messages"):
            (directory / name).mkdir(parents=True, exist_ok=True)
        self._write_json(directory / "team.json", team.to_dict())
        if not (directory / "tasks.json").exists():
            self._write_json(directory / "tasks.json", {})
        (directory / "events.jsonl").touch(exist_ok=True)
        self._write_json(self.active_team_path, team.to_dict())
        return team.to_dict()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"expected a JSON object in {path}")
        return data

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        with _locked_path(path):
            TeamStore._write_json_unlocked(path, data)

    @staticmethod
    def _write_json_unlocked(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
