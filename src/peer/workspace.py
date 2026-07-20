from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .models import PeerParticipant


class PeerWorkspaceManager:
    """Prepare peer workspaces without performing hidden integration."""

    def __init__(
        self,
        repo_path: str | Path,
        run_id: str,
        workspace_mode: str,
        *,
        cleanup_worktrees: bool = True,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.run_id = run_id
        self.workspace_mode = workspace_mode
        self.cleanup_worktrees = cleanup_worktrees
        self.repo_root = self._repo_root()
        self.base_revision = self._git(
            ["rev-parse", "HEAD"], cwd=self.repo_root
        ).stdout.strip()
        self.worktree_root = (
            self.repo_root.parent
            / ".clawd-peer-worktrees"
            / f"{self.repo_root.name}-{run_id}"
        )
        self._workspaces: dict[str, Path] = {}

    def _repo_root(self) -> Path:
        completed = self._git(
            ["rev-parse", "--show-toplevel"], cwd=self.repo_path, check=False
        )
        if completed.returncode != 0:
            raise ValueError("peer collaboration requires a Git repository")
        root = Path(completed.stdout.strip()).resolve()
        if root != self.repo_path:
            raise ValueError("repo_path must be the Git repository root")
        return root

    def prepare(self, peer_id: str, name: str) -> Path:
        if self.workspace_mode == "shared":
            self._workspaces[peer_id] = self.repo_root
            return self.repo_root
        if self.workspace_mode != "worktree":
            raise ValueError("workspace_mode must be shared or worktree")
        path = self.worktree_root / f"{name}-{peer_id}"
        path.parent.mkdir(parents=True, exist_ok=True)
        completed = self._git(
            ["worktree", "add", "--detach", str(path), self.base_revision],
            cwd=self.repo_root,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "failed to create peer worktree")
        self._workspaces[peer_id] = path.resolve()
        return path.resolve()

    def workspace_for(self, peer_id: str) -> Path:
        try:
            return self._workspaces[peer_id]
        except KeyError as exc:
            raise ValueError(f"workspace has not been prepared for peer: {peer_id}") from exc

    def validate_revision(self, revision: str) -> dict[str, Any]:
        if not isinstance(revision, str) or not revision.strip():
            return {"valid": False, "reason": "revision must be a non-empty string"}
        requested = revision.strip()
        resolved = self._git(
            ["rev-parse", "--verify", "--end-of-options", f"{requested}^{{commit}}"],
            cwd=self.repo_root,
            check=False,
        )
        if resolved.returncode != 0:
            return {
                "valid": False,
                "reason": "revision is not a commit in the allowed repository",
                "revision": requested,
            }
        commit = resolved.stdout.strip()
        allowed_heads: dict[str, str] = {}
        for peer_id, workspace in self._workspaces.items():
            head = self._git(["rev-parse", "HEAD"], cwd=workspace, check=False)
            if head.returncode == 0:
                allowed_heads[peer_id] = head.stdout.strip()
        main_head = self._git(["rev-parse", "HEAD"], cwd=self.repo_root, check=False)
        if main_head.returncode == 0:
            allowed_heads["shared"] = main_head.stdout.strip()
        reachable_from: list[str] = []
        for identity, head in allowed_heads.items():
            reachable = self._git(
                ["merge-base", "--is-ancestor", commit, head],
                cwd=self.repo_root,
                check=False,
            )
            if reachable.returncode == 0:
                reachable_from.append(identity)
        if not reachable_from:
            return {
                "valid": False,
                "reason": "revision is not reachable from an allowed peer workspace",
                "revision": requested,
                "resolved_revision": commit,
                "allowed_heads": allowed_heads,
            }
        return {
            "valid": True,
            "revision": requested,
            "resolved_revision": commit,
            "reachable_from": sorted(reachable_from),
            "allowed_heads": allowed_heads,
        }

    def attribution(self, participants: list[PeerParticipant]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for participant in participants:
            workspace = Path(participant.workspace_path)
            head_result = self._git(["rev-parse", "HEAD"], cwd=workspace, check=False)
            head = head_result.stdout.strip() if head_result.returncode == 0 else None
            commits: list[str] = []
            if head:
                listed = self._git(
                    ["rev-list", "--reverse", f"{self.base_revision}..{head}"],
                    cwd=self.repo_root,
                    check=False,
                )
                if listed.returncode == 0:
                    commits = [line for line in listed.stdout.splitlines() if line]
            result.append(
                {
                    "peer_id": participant.peer_id,
                    "workspace_path": participant.workspace_path,
                    "workspace_mode": participant.workspace_mode,
                    "base_revision": self.base_revision,
                    "head_revision": head,
                    "commits": commits,
                }
            )
        return result

    def run_acceptance(
        self,
        revision: str,
        command: list[str],
        *,
        timeout_seconds: float = 300.0,
    ) -> dict[str, Any]:
        validation_root = Path(
            tempfile.mkdtemp(prefix=f"clawd-peer-acceptance-{self.run_id}-")
        ).resolve()
        checkout = validation_root / "checkout"
        added = self._git(
            ["worktree", "add", "--detach", str(checkout), revision],
            cwd=self.repo_root,
            check=False,
        )
        if added.returncode != 0:
            shutil.rmtree(validation_root, ignore_errors=True)
            return {
                "command": command,
                "exit_code": 125,
                "stdout": "",
                "stderr": added.stderr.strip() or "failed to prepare acceptance worktree",
            }
        try:
            completed = subprocess.run(
                command,
                cwd=checkout,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return {
                "command": command,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "command": command,
                "exit_code": 124,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "acceptance command timed out",
            }
        finally:
            self._git(
                ["worktree", "remove", "--force", str(checkout)],
                cwd=self.repo_root,
                check=False,
            )
            shutil.rmtree(validation_root, ignore_errors=True)

    def cleanup(self) -> list[str]:
        retained: list[str] = []
        if self.workspace_mode != "worktree" or not self.cleanup_worktrees:
            return retained
        for path in self._workspaces.values():
            completed = self._git(
                ["worktree", "remove", "--force", str(path)],
                cwd=self.repo_root,
                check=False,
            )
            if completed.returncode != 0 and path.exists():
                retained.append(str(path))
        if self.worktree_root.is_dir() and not any(self.worktree_root.iterdir()):
            self.worktree_root.rmdir()
        self._git(["worktree", "prune"], cwd=self.repo_root, check=False)
        return retained

    @staticmethod
    def _git(
        args: list[str], *, cwd: Path, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True
        )
        if check and completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
        return completed
