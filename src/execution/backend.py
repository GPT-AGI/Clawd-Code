from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class CommandOutcome:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class RemoteStat:
    path: str
    exists: bool
    is_file: bool = False
    is_dir: bool = False
    size: int = 0
    mtime_ns: int = 0

    @property
    def fingerprint(self) -> tuple[int, int]:
        return self.mtime_ns, self.size


class WorkspaceBackend(Protocol):
    """Synchronous interface used by Clawd's synchronous tool dispatcher."""

    workspace_root: str
    sandbox_id: str

    def resolve_path(self, path: str, *, cwd: str, local_root: Path) -> str: ...

    def exec(
        self,
        command: str,
        *,
        cwd: str,
        timeout_s: int,
        env: dict[str, str] | None = None,
    ) -> CommandOutcome: ...

    def stat(self, path: str) -> RemoteStat: ...

    def read_text(self, path: str) -> str: ...

    def read_bytes(self, path: str) -> bytes: ...

    def write_text(self, path: str, content: str) -> None: ...

    def run_json_helper(
        self, script: str, payload: dict[str, Any], *, timeout_s: int = 120
    ) -> Any: ...

    def upload_tree(self, local_path: Path, remote_path: str) -> None: ...

    def download_tree(self, remote_path: str, local_path: Path) -> None: ...

    def close(self) -> None: ...
