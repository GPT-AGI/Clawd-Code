from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.execution.backend import CommandOutcome, RemoteStat
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.protocol import ToolCall


class FakeRemoteBackend:
    workspace_root = "/workspace"
    sandbox_id = "fake-sandbox"

    def __init__(self, root: Path) -> None:
        self.root = root

    def _local(self, path: str) -> Path:
        if path == self.workspace_root:
            return self.root
        if not path.startswith(self.workspace_root + "/"):
            raise ValueError(f"outside workspace: {path}")
        return self.root / path[len(self.workspace_root) + 1 :]

    def _remote(self, path: str) -> str:
        local = str(self.root)
        return self.workspace_root + path[len(local) :].replace(os.sep, "/")

    def resolve_path(self, path: str, *, cwd: str, local_root: Path) -> str:
        local_alias = str(local_root.resolve())
        if path.startswith("/") and not path.startswith(self.workspace_root):
            path = str(Path(path).resolve())
        if path == local_alias or path.startswith(local_alias + os.sep):
            path = self.workspace_root + path[len(local_alias) :].replace(os.sep, "/")
        if not path.startswith("/"):
            path = str(Path(cwd) / path)
        normalized = os.path.normpath(path)
        if normalized != self.workspace_root and not normalized.startswith(self.workspace_root + "/"):
            raise ValueError(f"path is outside the remote workspace: {path}")
        return normalized

    def exec(
        self,
        command: str,
        *,
        cwd: str,
        timeout_s: int,
        env: dict[str, str] | None = None,
    ) -> CommandOutcome:
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=self._local(cwd),
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return CommandOutcome(completed.returncode, completed.stdout, completed.stderr)

    def stat(self, path: str) -> RemoteStat:
        local = self._local(path)
        if not local.exists():
            return RemoteStat(path=path, exists=False)
        value = local.stat()
        return RemoteStat(
            path=path,
            exists=True,
            is_file=local.is_file(),
            is_dir=local.is_dir(),
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
        )

    def read_text(self, path: str) -> str:
        return self._local(path).read_text(encoding="utf-8", errors="replace")

    def read_bytes(self, path: str) -> bytes:
        return self._local(path).read_bytes()

    def write_text(self, path: str, content: str) -> None:
        local = self._local(path)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(content, encoding="utf-8")

    def run_json_helper(
        self, script: str, payload: dict[str, Any], *, timeout_s: int = 120
    ) -> Any:
        translated = dict(payload)
        if isinstance(translated.get("root"), str):
            translated["root"] = str(self._local(translated["root"]))
        encoded = base64.b64encode(json.dumps(translated).encode()).decode()
        completed = subprocess.run(
            [sys.executable, "-c", script, encoded],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        )
        output = json.loads(completed.stdout)

        def restore(value: Any) -> Any:
            if isinstance(value, str) and value.startswith(str(self.root)):
                return self._remote(value)
            if isinstance(value, list):
                return [restore(item) for item in value]
            if isinstance(value, dict):
                return {key: restore(item) for key, item in value.items()}
            return value

        return restore(output)

    def upload_tree(self, local_path: Path, remote_path: str) -> None:
        shutil.copytree(local_path, self._local(remote_path), dirs_exist_ok=True)

    def download_tree(self, remote_path: str, local_path: Path) -> None:
        shutil.copytree(self._local(remote_path), local_path, dirs_exist_ok=True)

    def close(self) -> None:
        pass


class TestRemoteWorkspaceTools(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.control = Path(self.temp.name) / "control"
        self.remote = Path(self.temp.name) / "remote"
        self.control.mkdir()
        self.remote.mkdir()
        self.backend = FakeRemoteBackend(self.remote)
        self.context = ToolContext(workspace_root=self.control, workspace_backend=self.backend)
        self.registry = build_default_registry(
            include_user_tools=False, workspace_backend=self.backend
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def call(self, name: str, payload: dict[str, Any]):
        return self.registry.dispatch(ToolCall(name=name, input=payload), self.context)

    def test_remote_read_write_edit_and_host_alias_mapping(self) -> None:
        created = self.call(
            "Write",
            {"file_path": str(self.control / "pkg" / "mod.py"), "content": "VALUE = 1\n"},
        )
        self.assertFalse(created.is_error)
        self.assertEqual((self.remote / "pkg" / "mod.py").read_text(), "VALUE = 1\n")

        read = self.call("Read", {"file_path": "/workspace/pkg/mod.py", "limit": 2000})
        self.assertIn("1\tVALUE = 1", read.output["file"]["content"])
        edited = self.call(
            "Edit",
            {
                "file_path": "/workspace/pkg/mod.py",
                "old_string": "VALUE = 1",
                "new_string": "VALUE = 2",
            },
        )
        self.assertFalse(edited.is_error)
        self.assertEqual((self.remote / "pkg" / "mod.py").read_text(), "VALUE = 2\n")

    def test_remote_bash_glob_and_grep(self) -> None:
        (self.remote / "a.py").write_text("needle = 1\n")
        (self.remote / "b.txt").write_text("nothing\n")
        bash = self.call("Bash", {"command": "pwd"})
        self.assertEqual(bash.output["stdout"].strip(), str(self.remote.resolve()))

        globbed = self.call("Glob", {"pattern": "**/*.py"})
        self.assertEqual(globbed.output["filenames"], ["/workspace/a.py"])
        grepped = self.call(
            "Grep",
            {"pattern": "needle", "output_mode": "content", "-n": True},
        )
        self.assertIn("/workspace/a.py:1:needle = 1", grepped.output["content"])

    def test_remote_paths_cannot_escape_workspace(self) -> None:
        with self.assertRaisesRegex(Exception, "outside the remote workspace"):
            self.registry.dispatch(
                ToolCall(name="Read", input={"file_path": "/etc/passwd"}), self.context
            )


if __name__ == "__main__":
    unittest.main()
