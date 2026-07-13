from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from .models import AgentRecord, Team, TeamTask


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
        return team

    def load_active_team(self) -> Team | None:
        if not self.active_team_path.exists():
            return None
        data = self._read_json(self.active_team_path)
        if "team_id" not in data:
            data = self._migrate_legacy_team(data)
        return Team.from_dict(data)

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

    def save_agent(self, agent: AgentRecord) -> Path:
        path = self.team_dir(agent.team_id) / "agents" / f"{agent.agent_id}.json"
        self._write_json(path, agent.to_dict())
        return path

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

    def disband_active_team(self) -> Team | None:
        team = self.load_active_team()
        if team is None:
            return None
        if team.status in {"created", "running", "failed"}:
            team.transition_to("cancelled")
            self._write_json(self.team_dir(team.team_id) / "team.json", team.to_dict())
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
