from __future__ import annotations

import re
import subprocess
import threading
from pathlib import Path

from .models import AgentRecord, TeamTask


_INTEGRATION_LOCKS: dict[str, threading.Lock] = {}
_INTEGRATION_LOCKS_GUARD = threading.Lock()


def _integration_lock(repo_root: Path) -> threading.Lock:
    key = str(repo_root.resolve())
    with _INTEGRATION_LOCKS_GUARD:
        return _INTEGRATION_LOCKS.setdefault(key, threading.Lock())


class TeammateWorktreeManager:
    """Create isolated git worktrees and integrate their commits safely."""

    def __init__(self, workspace_root: str | Path):
        self.workspace_root = Path(workspace_root).resolve()
        self.repo_root = self._repo_root()

    def _repo_root(self) -> Path:
        completed = self._git(
            ["rev-parse", "--show-toplevel"], cwd=self.workspace_root, check=False
        )
        if completed.returncode != 0:
            raise ValueError("worktree isolation requires a git repository")
        root = Path(completed.stdout.strip()).resolve()
        if root != self.workspace_root:
            raise ValueError("workspace root must be the git repository root for worktree isolation")
        return root

    def create(self, team_id: str, agent_id: str, name: str) -> Path:
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-") or "agent"
        directory = (
            self.repo_root.parent
            / ".clawd-worktrees"
            / f"{self.repo_root.name}-{team_id}-{safe_name}-{agent_id}"
        )
        if (directory / ".git").exists():
            return directory.resolve()
        directory.parent.mkdir(parents=True, exist_ok=True)
        completed = self._git(
            ["worktree", "add", "--detach", str(directory), "HEAD"],
            cwd=self.repo_root,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "failed to create git worktree")
        return directory.resolve()

    def integrate(self, agent: AgentRecord, task: TeamTask | None = None) -> dict[str, str | bool | None]:
        if agent.workspace_mode != "worktree" or not agent.workspace_path:
            raise ValueError("teammate does not use worktree isolation")
        worktree = Path(agent.workspace_path).resolve()
        if not worktree.is_dir():
            raise ValueError(f"teammate worktree does not exist: {worktree}")

        with _integration_lock(self.repo_root):
            self._git(["add", "-A"], cwd=worktree)
            changed = self._git(["diff", "--cached", "--quiet"], cwd=worktree, check=False)
            if changed.returncode == 0:
                return {"integrated": False, "commit": None, "reason": "no changes"}
            if changed.returncode != 1:
                raise RuntimeError(changed.stderr.strip() or "failed to inspect worktree changes")

            label = task.key or task.id if task is not None else "manual"
            message = f"clawd teammate {agent.name}: {label}"
            committed = self._git(
                [
                    "-c",
                    "user.name=Clawd Teammate",
                    "-c",
                    "user.email=clawd-teammate@local",
                    "commit",
                    "-m",
                    message,
                ],
                cwd=worktree,
                check=False,
            )
            if committed.returncode != 0:
                raise RuntimeError(committed.stderr.strip() or "failed to commit worktree changes")
            commit = self._git(["rev-parse", "HEAD"], cwd=worktree).stdout.strip()
            integrated = self._git(
                [
                    "-c",
                    "user.name=Clawd Teammate",
                    "-c",
                    "user.email=clawd-teammate@local",
                    "cherry-pick",
                    commit,
                ],
                cwd=self.repo_root,
                check=False,
            )
            if integrated.returncode != 0:
                self._git(["cherry-pick", "--abort"], cwd=self.repo_root, check=False)
                raise RuntimeError(
                    integrated.stderr.strip() or "failed to integrate teammate worktree"
                )
            return {"integrated": True, "commit": commit, "reason": None}

    def remove(self, agent: AgentRecord, *, force: bool = False) -> None:
        if agent.workspace_mode != "worktree" or not agent.workspace_path:
            return
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(agent.workspace_path)
        completed = self._git(args, cwd=self.repo_root, check=False)
        if completed.returncode != 0 and Path(agent.workspace_path).exists():
            raise RuntimeError(completed.stderr.strip() or "failed to remove teammate worktree")

    @staticmethod
    def _git(
        args: list[str], *, cwd: Path, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )
        if check and completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
        return completed
