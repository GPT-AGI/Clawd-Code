from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .errors import ToolPermissionError
from .permissions import ToolPermissionContext
from .task_manager import TaskManager
from ..teammate.models import TeamTask
from ..teammate.store import TeamStore


@dataclass
class ToolContext:
    workspace_root: Path
    permission_context: ToolPermissionContext = field(default_factory=ToolPermissionContext)
    cwd: Path | None = None
    read_file_fingerprints: dict[Path, tuple[int, int]] = field(default_factory=dict)
    task_manager: TaskManager = field(default_factory=TaskManager)
    mcp_clients: dict[str, Any] = field(default_factory=dict)
    lsp_client: Any | None = None
    todos: list[dict[str, Any]] = field(default_factory=list)
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    plan_mode: bool = False
    worktree_root: Path | None = None
    outbox: list[dict[str, Any]] = field(default_factory=list)
    ask_user: Callable[[list[dict[str, Any]]], dict[str, str]] | None = None
    crons: dict[str, dict[str, Any]] = field(default_factory=dict)
    team: dict[str, Any] | None = None
    actor_id: str | None = None
    current_task_id: str | None = None
    system_prompt_extra: str | None = None
    model_override: str | None = None
    teammate_runtime: Any | None = None
    output_style_name: str | None = None
    output_style_dir: Path | None = None
    workspace_backend: Any | None = None
    execution_workspace_root: str | None = None
    execution_cwd: str | None = None
    remote_file_fingerprints: dict[str, tuple[int, int]] = field(default_factory=dict)
    peer_store: Any | None = None
    peer_run_id: str | None = None
    peer_id: str | None = None
    peer_control: Any | None = None
    # All child contexts in a team share this lock. Protocol-v2 mutation tools use
    # it to make Bash before/after snapshots attributable even while model turns
    # and read-only tools continue concurrently.
    mutation_lock: Any = field(default_factory=threading.RLock, repr=False)
    # Sticky for the lifetime of a worker task. The runtime fails the task even if
    # the model catches the tool error or attempts to reset its task status.
    ownership_violations: list[dict[str, Any]] = field(default_factory=list)
    team_store: TeamStore = field(init=False, repr=False)

    # Permission handler callback: called when a tool needs user consent.
    # Signature: (tool_name: str, message: str, suggestion: str | None)
    #           -> tuple[bool, bool] (allowed: bool, continue_without_caching: bool)
    # If not set, permission errors will be raised as exceptions.
    permission_handler: Callable[[str, str, Optional[str]], tuple[bool, bool]] | None = None

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()
        self.team_store = TeamStore(self.workspace_root)
        if self.cwd is None:
            self.cwd = self.workspace_root
        else:
            self.cwd = Path(self.cwd).resolve()
        if self.workspace_backend is not None:
            self.execution_workspace_root = (
                self.execution_workspace_root
                or str(getattr(self.workspace_backend, "workspace_root", "/workspace"))
            )
            self.execution_cwd = self.execution_cwd or self.execution_workspace_root
        if self.permission_context.workspace_root is None:
            self.permission_context = ToolPermissionContext.from_iterables(
                self.permission_context.deny_names,
                self.permission_context.deny_prefixes,
                workspace_root=self.workspace_root,
                additional_working_directories=self.permission_context.additional_working_directories,
                allow_docs=self.permission_context.allow_docs,
            )
        active_team = self.team_store.load_active_team()
        if active_team is not None:
            self.team = active_team.to_dict()
            self.tasks = self.team_store.load_tasks(active_team.team_id)

    def persist_tasks(self) -> None:
        if self.team is None:
            return
        team_id = self.team.get("team_id")
        if isinstance(team_id, str) and team_id:
            if self.actor_id is not None and self.current_task_id in self.tasks:
                task = TeamTask.from_dict(self.tasks[self.current_task_id])
                self.tasks = self.team_store.update_task(team_id, task)
            else:
                self.team_store.save_tasks(team_id, self.tasks)

    def reload_team_state(self) -> None:
        active_team = self.team_store.load_active_team()
        if active_team is None:
            self.team = None
            self.tasks = {}
            return
        self.team = active_team.to_dict()
        self.tasks = self.team_store.load_tasks(active_team.team_id)

    def mark_file_read(self, path: Path) -> None:
        stat = path.stat()
        self.read_file_fingerprints[path.resolve()] = (int(stat.st_mtime), int(stat.st_size))

    def was_file_read_and_unchanged(self, path: Path) -> bool:
        resolved = path.resolve()
        fingerprint = self.read_file_fingerprints.get(resolved)
        if fingerprint is None:
            return False
        stat = resolved.stat()
        return fingerprint == (int(stat.st_mtime), int(stat.st_size))

    def mark_remote_file_read(self, path: str) -> None:
        if self.workspace_backend is None:
            return
        self.remote_file_fingerprints[path] = self.workspace_backend.stat(path).fingerprint

    def was_remote_file_read_and_unchanged(self, path: str) -> bool:
        if self.workspace_backend is None:
            return False
        fingerprint = self.remote_file_fingerprints.get(path)
        if fingerprint is None:
            return False
        return fingerprint == self.workspace_backend.stat(path).fingerprint

    def resolve_execution_path(self, path: str) -> str:
        if self.workspace_backend is None:
            return str(self.ensure_allowed_path(path))
        return self.workspace_backend.resolve_path(
            path,
            cwd=self.execution_cwd or self.execution_workspace_root or "/workspace",
            local_root=self.workspace_root,
        )

    def ensure_allowed_path(self, path: str | Path) -> Path:
        p = Path(path).expanduser() if isinstance(path, str) else path.expanduser()
        if not p.is_absolute():
            base = self.cwd or self.workspace_root
            p = (base / p).resolve()
        return self.permission_context.ensure_path_allowed(p)

    def ensure_tool_allowed(self, tool_name: str) -> None:
        if self.permission_context.blocks_tool(tool_name):
            raise ToolPermissionError(f"tool is blocked by permission context: {tool_name}")
