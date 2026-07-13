from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .errors import ToolPermissionError
from .permissions import ToolPermissionContext
from .task_manager import TaskManager
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

    def ensure_allowed_path(self, path: str | Path) -> Path:
        p = Path(path).expanduser() if isinstance(path, str) else path.expanduser()
        if not p.is_absolute():
            base = self.cwd or self.workspace_root
            p = (base / p).resolve()
        return self.permission_context.ensure_path_allowed(p)

    def ensure_tool_allowed(self, tool_name: str) -> None:
        if self.permission_context.blocks_tool(tool_name):
            raise ToolPermissionError(f"tool is blocked by permission context: {tool_name}")
