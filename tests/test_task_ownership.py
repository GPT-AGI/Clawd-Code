from __future__ import annotations

import hashlib
import os
import posixpath
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.execution.backend import CommandOutcome, RemoteStat
from src.providers.base import ChatResponse
from src.teammate.models import AgentRecord, TeamTask
from src.teammate.runtime import TeammateRuntime
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.errors import ToolPermissionError
from src.tool_system.remote_tools import (
    RemoteBashTool,
    RemoteFileEditTool,
    RemoteFileWriteTool,
)
from src.tool_system.tools import BashTool, FileEditTool, FileWriteTool
from src.tool_system.tools import TeamCreateTool, TeamPlanTool, TeamRunTool


def _v2_context(
    root: Path,
    task_id: str = "task-one",
    *,
    owner: str = "worker-one",
    owned_files: list[str] | None = None,
    actor_role: str | None = None,
    task_metadata: dict | None = None,
) -> ToolContext:
    context = ToolContext(workspace_root=root)
    team = context.team_store.load_active_team()
    if team is None:
        team = context.team_store.create_team("ownership")
        team.protocol_version = 2
        team.settings["protocol_version"] = 2
        team.settings["quality_gates"] = {
            "strict": True,
            "protocol_version": 2,
        }
        team.set_lifecycle_state("running")
        context.team_store.save_team(team)
    tasks = context.team_store.load_tasks(team.team_id)
    tasks[task_id] = TeamTask(
        id=task_id,
        key=task_id,
        subject=task_id,
        description="ownership test",
        owner=owner,
        owned_files=owned_files or ["owned"],
        metadata=task_metadata or {},
    ).to_dict()
    context.team_store.save_tasks(team.team_id, tasks)
    if actor_role is not None:
        context.team_store.save_agent(
            AgentRecord(
                agent_id=owner,
                team_id=team.team_id,
                name=owner,
                role=actor_role,
                session_id=f"session-{owner}",
            )
        )
    context.reload_team_state()
    context.actor_id = owner
    context.current_task_id = task_id
    return context


class LocalOwnershipToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.context = _v2_context(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_write_and_edit_allow_owned_path(self) -> None:
        target = self.root / "owned" / "module.py"
        written = FileWriteTool().run(
            {"file_path": str(target), "content": "VALUE = 1\n"}, self.context
        )
        self.assertFalse(written.is_error)

        edited = FileEditTool().run(
            {
                "file_path": str(target),
                "old_string": "VALUE = 1",
                "new_string": "VALUE = 2",
            },
            self.context,
        )
        self.assertFalse(edited.is_error)
        self.assertEqual(target.read_text(encoding="utf-8"), "VALUE = 2\n")
        self.assertEqual(self.context.ownership_violations, [])

    def test_write_and_edit_reject_other_tasks_path(self) -> None:
        other = self.root / "other" / "module.py"
        other.parent.mkdir()
        other.write_text("VALUE = 1\n", encoding="utf-8")
        self.context.mark_file_read(other)

        with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
            FileWriteTool().run(
                {"file_path": str(other), "content": "VALUE = 2\n"},
                self.context,
            )
        with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
            FileEditTool().run(
                {
                    "file_path": str(other),
                    "old_string": "VALUE = 1",
                    "new_string": "VALUE = 2",
                },
                self.context,
            )
        self.assertEqual(other.read_text(encoding="utf-8"), "VALUE = 1\n")
        self.assertEqual(len(self.context.ownership_violations), 2)

    def test_workspace_prefixed_owned_path_is_normalized(self) -> None:
        context = _v2_context(
            self.root,
            "workspace-prefix",
            owned_files=["/workspace/pkg"],
        )

        result = FileWriteTool().run(
            {
                "file_path": str(self.root / "pkg" / "module.py"),
                "content": "VALUE = 1\n",
            },
            context,
        )

        self.assertFalse(result.is_error)
        self.assertEqual(context.ownership_violations, [])

    def test_only_explicit_integrator_gets_shared_integration_paths(self) -> None:
        initializer = self.root / "package" / "__init__.py"
        manifest = self.root / "pyproject.toml"
        with self.assertRaisesRegex(ToolPermissionError, "lead/integrator"):
            FileWriteTool().run(
                {"file_path": str(initializer), "content": "VALUE = 1\n"},
                self.context,
            )

        integrator = _v2_context(
            self.root,
            "integration-task",
            owner="integration-worker",
            owned_files=["integration-notes.txt"],
            actor_role="integrator",
        )
        for path, content in (
            (initializer, "VALUE = 2\n"),
            (manifest, "[project]\nname = 'sample'\n"),
        ):
            result = FileWriteTool().run(
                {"file_path": str(path), "content": content}, integrator
            )
            self.assertFalse(result.is_error)

        # Integration authority is narrow; it is not permission to take another
        # worker's ordinary delivery file.
        with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
            FileWriteTool().run(
                {
                    "file_path": str(self.root / "other" / "module.py"),
                    "content": "STOLEN = True\n",
                },
                integrator,
            )

    def test_lead_identity_has_explicit_integration_authority(self) -> None:
        lead = _v2_context(
            self.root,
            "lead-integration",
            owner="worker-two",
            owned_files=["worker-two.py"],
        )
        lead.actor_id = str(lead.team["lead_agent_id"])

        result = FileWriteTool().run(
            {
                "file_path": str(self.root / "package.json"),
                "content": '{"name": "sample"}\n',
            },
            lead,
        )

        self.assertFalse(result.is_error)

    def test_runtime_artifacts_do_not_create_false_bash_conflicts(self) -> None:
        result = BashTool().run(
            {
                "command": " && ".join(
                    [
                        "mkdir -p .cache/tool htmlcov",
                        "printf cache > .cache/tool/result.json",
                        "printf coverage > .coverage.worker-one",
                        "printf xml > coverage.xml",
                        "printf db > application.sqlite",
                        "printf wal > application.sqlite-wal",
                        "printf log > test-run.log",
                        "printf backup > source.py.bak",
                        "printf temp > output.tmp",
                        "printf html > htmlcov/index.html",
                    ]
                )
            },
            self.context,
        )

        self.assertFalse(result.is_error)
        self.assertEqual(self.context.ownership_violations, [])

        with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
            BashTool().run(
                {
                    "command": (
                        "printf updated >> application.sqlite && "
                        "mkdir -p other && printf bad > other/delivery.py"
                    )
                },
                self.context,
            )
        self.assertEqual(
            self.context.ownership_violations[-1]["paths"],
            ["other/delivery.py"],
        )

    def test_runtime_and_test_exemptions_do_not_override_other_task_delivery(self) -> None:
        _v2_context(
            self.root,
            "artifact-owner",
            owner="worker-two",
            owned_files=["fixtures/shared.sqlite", "tests/test_contract.py"],
        )
        self.context.reload_team_state()

        with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
            BashTool().run(
                {
                    "command": (
                        "mkdir -p fixtures && "
                        "printf db > fixtures/shared.sqlite"
                    )
                },
                self.context,
            )
        self.assertEqual(
            self.context.ownership_violations[-1]["paths"],
            ["fixtures/shared.sqlite"],
        )

        with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
            FileWriteTool().run(
                {
                    "file_path": str(self.root / "tests" / "test_contract.py"),
                    "content": "assert True\n",
                },
                self.context,
            )

    def test_new_test_module_is_scratch_but_existing_project_test_is_protected(self) -> None:
        tests = self.root / "tests"
        tests.mkdir()
        existing = tests / "test_existing.py"
        existing.write_text("VALUE = 1\n", encoding="utf-8")
        self.context.mark_file_read(existing)

        with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
            FileEditTool().run(
                {
                    "file_path": str(existing),
                    "old_string": "VALUE = 1",
                    "new_string": "VALUE = 2",
                },
                self.context,
            )
        self.assertEqual(existing.read_text(encoding="utf-8"), "VALUE = 1\n")

        generated = tests / "test_generated_probe.py"
        created = FileWriteTool().run(
            {
                "file_path": str(generated),
                "content": "VALUE = 1\n",
            },
            self.context,
        )
        self.assertFalse(created.is_error)
        edited = FileEditTool().run(
            {
                "file_path": str(generated),
                "old_string": "VALUE = 1",
                "new_string": "VALUE = 2",
            },
            self.context,
        )
        self.assertFalse(edited.is_error)

        created_by_bash = BashTool().run(
            {
                "command": "printf 'VALUE = 3\\n' > tests/test_bash_probe.py",
            },
            self.context,
        )
        self.assertFalse(created_by_bash.is_error)
        updated_by_bash = BashTool().run(
            {
                "command": "printf 'VALUE = 4\\n' > tests/test_bash_probe.py",
            },
            self.context,
        )
        self.assertFalse(updated_by_bash.is_error)

        with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
            BashTool().run(
                {"command": "printf 'VALUE = 9\\n' > tests/test_existing.py"},
                self.context,
            )
        self.assertEqual(existing.read_text(encoding="utf-8"), "VALUE = 1\n")

    def test_bash_audits_legal_and_illegal_workspace_changes(self) -> None:
        legal = BashTool().run(
            {"command": "mkdir -p owned && printf 'ok' > owned/generated.txt"},
            self.context,
        )
        self.assertFalse(legal.is_error)
        self.assertEqual(self.context.ownership_violations, [])

        with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
            BashTool().run(
                {"command": "mkdir -p other && printf 'bad' > other/generated.txt"},
                self.context,
            )
        self.assertEqual(
            self.context.ownership_violations[-1]["paths"],
            ["other/generated.txt"],
        )

    def test_bash_restores_unauthorized_source_overwrite_delete_and_create(self) -> None:
        other = self.root / "other"
        other.mkdir()
        overwritten = other / "overwritten.py"
        deleted = other / "deleted.py"
        created = other / "created.py"
        overwritten.write_text("ORIGINAL = 1\n", encoding="utf-8")
        deleted.write_text("KEEP = True\n", encoding="utf-8")

        with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
            BashTool().run(
                {
                    "command": " && ".join(
                        [
                            "printf 'HACKED = 1\\n' > other/overwritten.py",
                            "rm other/deleted.py",
                            "printf 'NEW = 1\\n' > other/created.py",
                            "mkdir -p owned",
                            "printf 'legal\\n' > owned/retained.txt",
                        ]
                    )
                },
                self.context,
            )

        self.assertEqual(
            overwritten.read_text(encoding="utf-8"), "ORIGINAL = 1\n"
        )
        self.assertEqual(deleted.read_text(encoding="utf-8"), "KEEP = True\n")
        self.assertFalse(created.exists())
        self.assertEqual(
            (self.root / "owned" / "retained.txt").read_text(encoding="utf-8"),
            "legal\n",
        )

    def test_bash_restores_clawd_control_state_before_reporting_tampering(self) -> None:
        team_id = str(self.context.team["team_id"])
        active = self.root / ".clawd" / "team.json"
        team = self.root / ".clawd" / "teams" / team_id / "team.json"
        tasks = self.root / ".clawd" / "teams" / team_id / "tasks.json"
        events = self.root / ".clawd" / "teams" / team_id / "events.jsonl"
        before = {
            active: active.read_bytes(),
            team: team.read_bytes(),
            tasks: tasks.read_bytes(),
            events: events.read_bytes(),
        }

        command = " && ".join(
            [
                "mkdir -p .clawd/task-tests/task-one",
                "printf 'assert True\\n' > .clawd/task-tests/task-one/test_kept.py",
                f"printf 'forged-active\\n' > {active.relative_to(self.root)}",
                f"printf 'forged-team\\n' > {team.relative_to(self.root)}",
                f"printf 'forged-tasks\\n' > {tasks.relative_to(self.root)}",
                f"printf 'forged-event\\n' >> {events.relative_to(self.root)}",
            ]
        )
        with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
            BashTool().run({"command": command}, self.context)

        self.assertEqual(active.read_bytes(), before[active])
        self.assertEqual(team.read_bytes(), before[team])
        self.assertEqual(tasks.read_bytes(), before[tasks])
        # Restore happens before the harness appends its genuine ownership event.
        self.assertTrue(events.read_bytes().startswith(before[events]))
        self.assertNotIn(b"forged-event", events.read_bytes())
        self.assertEqual(
            (
                self.root
                / ".clawd/task-tests/task-one/test_kept.py"
            ).read_text(encoding="utf-8"),
            "assert True\n",
        )
        changed = set(self.context.ownership_violations[-1]["paths"])
        self.assertTrue(
            {
                ".clawd/team.json",
                f".clawd/teams/{team_id}/team.json",
                f".clawd/teams/{team_id}/tasks.json",
                f".clawd/teams/{team_id}/events.jsonl",
            }.issubset(changed)
        )

    def test_harness_event_waits_for_local_bash_audit_without_false_positive(self) -> None:
        team_id = str(self.context.team["team_id"])
        started = self.root / "owned" / "bash-started"

        def append_harness_event() -> None:
            deadline = time.monotonic() + 5
            while not started.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Bash command did not start")
                time.sleep(0.01)
            self.context.team_store.append_event(
                team_id, "test.legitimate_harness_write", {"ok": True}
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            writer = pool.submit(append_harness_event)
            result = BashTool().run(
                {
                    "command": (
                        "mkdir -p owned && touch owned/bash-started && "
                        "sleep 0.15 && printf ok > owned/result.txt"
                    )
                },
                self.context,
            )
            writer.result(timeout=5)

        self.assertFalse(result.is_error)
        self.assertEqual(self.context.ownership_violations, [])
        events = self.context.team_store.list_events(team_id)
        self.assertIn(
            "test.legitimate_harness_write", {event["type"] for event in events}
        )

    def test_disposable_tests_are_isolated_in_task_private_scratch(self) -> None:
        scratch = self.root / ".clawd" / "task-tests" / "task-one"
        written = FileWriteTool().run(
            {
                "file_path": str(scratch / "test_generated.py"),
                "content": "def test_generated():\n    assert True\n",
            },
            self.context,
        )
        self.assertFalse(written.is_error)

        bash_result = BashTool().run(
            {
                "command": (
                    "mkdir -p .clawd/task-tests/task-one && "
                    "printf 'assert 1 == 1\\n' > "
                    ".clawd/task-tests/task-one/test_from_bash.py"
                )
            },
            self.context,
        )
        self.assertFalse(bash_result.is_error)

        with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
            BashTool().run(
                {
                    "command": (
                        "mkdir -p .clawd/task-tests/task-two && "
                        "printf 'assert False\\n' > "
                        ".clawd/task-tests/task-two/test_stolen.py"
                    )
                },
                self.context,
            )
        with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
            FileWriteTool().run(
                {
                    "file_path": str(self.root / "test_undeclared.py"),
                    "content": "assert True\n",
                },
                self.context,
            )

    def test_two_workers_bash_changes_are_attributed_without_false_conflicts(self) -> None:
        first = self.context
        second = _v2_context(
            self.root,
            "task-two",
            owner="worker-two",
            owned_files=["second"],
        )
        # This is what TeammateRuntime._child_context does for real workers.
        second.mutation_lock = first.mutation_lock
        barrier = threading.Barrier(2)

        def write(context: ToolContext, directory: str) -> None:
            barrier.wait(timeout=2)
            BashTool().run(
                {
                    "command": (
                        f"mkdir -p {directory} && sleep 0.05 && "
                        f"printf '{directory}' > {directory}/result.txt"
                    )
                },
                context,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(write, first, "owned"),
                pool.submit(write, second, "second"),
            ]
            for future in futures:
                future.result(timeout=5)

        self.assertEqual(first.ownership_violations, [])
        self.assertEqual(second.ownership_violations, [])
        self.assertEqual((self.root / "owned/result.txt").read_text(), "owned")
        self.assertEqual((self.root / "second/result.txt").read_text(), "second")

    def test_v1_teammate_remains_compatible(self) -> None:
        team = self.context.team_store.load_active_team()
        self.assertIsNotNone(team)
        team.protocol_version = 1
        team.settings["protocol_version"] = 1
        team.settings["quality_gates"] = {"strict": True, "protocol_version": 1}
        self.context.team_store.save_team(team)
        self.context.reload_team_state()

        outside = self.root / "legacy.py"
        result = FileWriteTool().run(
            {"file_path": str(outside), "content": "legacy = True\n"}, self.context
        )
        self.assertFalse(result.is_error)
        self.assertTrue(outside.exists())


class _LocalRemoteBackend:
    workspace_root = "/workspace"
    sandbox_id = "fake-ags"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.exec_commands: list[str] = []

    def _local(self, path: str) -> Path:
        relative = posixpath.relpath(path, self.workspace_root)
        return (self.root / relative).resolve()

    def resolve_path(self, path: str, *, cwd: str, local_root: Path) -> str:
        if path.startswith(str(local_root)):
            path = self.workspace_root + path[len(str(local_root)) :]
        if not path.startswith("/"):
            path = posixpath.join(cwd, path)
        resolved = posixpath.normpath(path)
        if resolved != self.workspace_root and not resolved.startswith(
            self.workspace_root + "/"
        ):
            raise ValueError("outside remote workspace")
        return resolved

    def exec(self, command: str, *, cwd: str, timeout_s: int, env=None) -> CommandOutcome:
        self.exec_commands.append(command)
        command = command.replace(self.workspace_root, str(self.root))
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=self._local(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return CommandOutcome(completed.returncode, completed.stdout, completed.stderr)

    def stat(self, path: str) -> RemoteStat:
        local = self._local(path)
        exists = local.exists()
        stat = local.stat() if exists else None
        return RemoteStat(
            path=path,
            exists=exists,
            is_file=local.is_file(),
            is_dir=local.is_dir(),
            size=stat.st_size if stat else 0,
            mtime_ns=stat.st_mtime_ns if stat else 0,
        )

    def read_text(self, path: str) -> str:
        return self._local(path).read_text(encoding="utf-8")

    def read_bytes(self, path: str) -> bytes:
        return self._local(path).read_bytes()

    def write_text(self, path: str, content: str) -> None:
        local = self._local(path)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(content, encoding="utf-8")

    def run_json_helper(self, script: str, payload: dict, *, timeout_s: int = 120):
        operation = payload.get("operation")
        if operation == "capture_workspace_backup":
            backup = Path(payload["backup"])
            if backup.is_symlink() or backup.is_file():
                backup.unlink(missing_ok=True)
            elif backup.is_dir():
                shutil.rmtree(backup)

            ignored = set(payload["ignored_dirs"])

            def ignore(_directory: str, names: list[str]) -> set[str]:
                return {
                    name
                    for name in names
                    if name in ignored
                    or name.endswith(".egg-info")
                    or (
                        not (Path(_directory) / name).is_symlink()
                        and not (Path(_directory) / name).is_dir()
                        and not (Path(_directory) / name).is_file()
                    )
                }

            if payload["full_workspace"]:
                shutil.copytree(self.root, backup, symlinks=True, ignore=ignore)
            else:
                backup.mkdir(parents=True)
                source = self.root / ".clawd"
                if source.exists():
                    shutil.copytree(source, backup / ".clawd", symlinks=True)
            return {"captured": True}

        if operation == "restore_workspace_backup":
            paths = sorted(
                set(payload["paths"]),
                key=lambda value: value.count("/"),
            )
            for relative in paths:
                target = self.root / relative
                if target.is_symlink() or target.is_file():
                    target.unlink(missing_ok=True)
                elif target.is_dir():
                    shutil.rmtree(target)
            for relative in paths:
                source = Path(payload["backup"]) / relative
                target = self.root / relative
                if not (source.exists() or source.is_symlink()):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.is_symlink():
                    target.symlink_to(os.readlink(source))
                elif source.is_dir():
                    target.mkdir(exist_ok=True)
                else:
                    shutil.copy2(source, target, follow_symlinks=False)
            return {"restored": len(paths)}

        # Ownership snapshots are the only helper used by these tests. Mirror the
        # AGS helper's result while mapping /workspace to this temporary directory.
        output: dict[str, str] = {}
        ignored = set(payload["ignored_dirs"])
        declared_paths = tuple(payload["declared_paths"])
        scan_root = (
            self.root / ".clawd" if payload["control_only"] else self.root
        )

        def declared(relative: str) -> bool:
            return any(
                relative == owned
                or relative.startswith(owned + "/")
                or owned.startswith(relative + "/")
                for owned in declared_paths
            )

        for directory, names, files in os.walk(scan_root):
            names[:] = [
                name
                for name in names
                if (
                    name not in ignored and not name.endswith(".egg-info")
                )
                or declared(
                    (Path(directory) / name)
                    .relative_to(self.root)
                    .as_posix()
                )
            ]
            for name in files:
                path = Path(directory) / name
                relative = path.relative_to(self.root).as_posix()
                ignored_parent = any(
                    part in ignored or part.endswith(".egg-info")
                    for part in Path(relative).parts[:-1]
                )
                ignored_file = (
                    name in payload["ignored_files"]
                    or any(
                        name.startswith(prefix)
                        for prefix in payload["ignored_prefixes"]
                    )
                    or any(
                        name.endswith(suffix)
                        for suffix in payload["ignored_suffixes"]
                    )
                )
                if (
                    ignored_parent or ignored_file
                ) and not declared(relative):
                    continue
                if not path.is_file():
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                output[relative] = f"file:{path.stat().st_mode & 0o777:o}:{digest}"
        return output


class RemoteOwnershipToolsTests(unittest.TestCase):
    def test_remote_bash_restores_unauthorized_source_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            other = root / "other"
            other.mkdir()
            overwritten = other / "overwritten.py"
            deleted = other / "deleted.py"
            created = other / "created.py"
            overwritten.write_text("ORIGINAL = 1\n", encoding="utf-8")
            deleted.write_text("KEEP = True\n", encoding="utf-8")
            context = _v2_context(root, owned_files=["/workspace/owned"])
            context.workspace_backend = _LocalRemoteBackend(root)
            context.execution_workspace_root = "/workspace"
            context.execution_cwd = "/workspace"

            with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
                RemoteBashTool().run(
                    {
                        "command": " && ".join(
                            [
                                "printf 'HACKED = 1\\n' > other/overwritten.py",
                                "rm other/deleted.py",
                                "printf 'NEW = 1\\n' > other/created.py",
                                "mkdir -p owned",
                                "printf 'legal\\n' > owned/retained.txt",
                            ]
                        )
                    },
                    context,
                )

            self.assertEqual(
                overwritten.read_text(encoding="utf-8"), "ORIGINAL = 1\n"
            )
            self.assertEqual(
                deleted.read_text(encoding="utf-8"), "KEEP = True\n"
            )
            self.assertFalse(created.exists())
            self.assertEqual(
                (root / "owned" / "retained.txt").read_text(encoding="utf-8"),
                "legal\n",
            )

    def test_remote_strict_v2_rejects_recursive_delivery_delete_before_exec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            package = root / "pkg"
            package.mkdir()
            (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            context = _v2_context(root, owned_files=["/workspace/pkg"])
            backend = _LocalRemoteBackend(root)
            context.workspace_backend = backend
            context.execution_workspace_root = "/workspace"
            context.execution_cwd = "/workspace"

            commands = [
                "rm -rf /workspace/pkg",
                "sh -c 'rm -rf /workspace/pkg'",
                "find /workspace/pkg -depth -delete",
                "git -C /workspace clean -fdx",
                (
                    "python -c \"import shutil; "
                    "shutil.rmtree('/workspace/pkg')\""
                ),
            ]
            for command in commands:
                with self.subTest(command=command):
                    with self.assertRaisesRegex(
                        ToolPermissionError,
                        "refuses recursive deletion of deliverable path",
                    ):
                        RemoteBashTool().run({"command": command}, context)

            self.assertEqual(backend.exec_commands, [])
            self.assertTrue((package / "module.py").exists())

    def test_remote_strict_guard_ignores_heredoc_source_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            context = _v2_context(root, owned_files=["/workspace/owned"])
            context.workspace_backend = _LocalRemoteBackend(root)
            context.execution_workspace_root = "/workspace"
            context.execution_cwd = "/workspace"

            result = RemoteBashTool().run(
                {
                    "command": (
                        "mkdir -p owned && cat > owned/cleanup.py <<'PY'\n"
                        "rm -rf /workspace/pkg\n"
                        "find /workspace/pkg -delete\n"
                        "shutil.rmtree('/workspace/pkg')\n"
                        "PY"
                    )
                },
                context,
            )

            self.assertFalse(result.is_error)
            self.assertIn(
                "rm -rf ",
                (root / "owned" / "cleanup.py").read_text(encoding="utf-8"),
            )

    def test_remote_strict_lead_cannot_mutate_clawd_control_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            context = _v2_context(root)
            context.actor_id = None
            context.current_task_id = None
            backend = _LocalRemoteBackend(root)
            context.workspace_backend = backend
            context.execution_workspace_root = "/workspace"
            context.execution_cwd = "/workspace"
            state = root / ".clawd" / "lead-state.json"
            state.write_text('{"status": "original"}\n', encoding="utf-8")
            context.mark_remote_file_read("/workspace/.clawd/lead-state.json")

            with self.assertRaisesRegex(
                ToolPermissionError, "protects .clawd control state"
            ):
                RemoteFileWriteTool().run(
                    {
                        "file_path": "/workspace/.clawd/forged.json",
                        "content": "{}\n",
                    },
                    context,
                )
            with self.assertRaisesRegex(
                ToolPermissionError, "protects .clawd control state"
            ):
                RemoteFileEditTool().run(
                    {
                        "file_path": "/workspace/.clawd/lead-state.json",
                        "old_string": "original",
                        "new_string": "forged",
                    },
                    context,
                )
            with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
                RemoteBashTool().run(
                    {
                        "command": (
                            "printf '{\\\"status\\\": \\\"forged\\\"}\\n' "
                            "> /workspace/.clawd/lead-state.json"
                        )
                    },
                    context,
                )

            self.assertFalse((root / ".clawd" / "forged.json").exists())
            self.assertEqual(
                state.read_text(encoding="utf-8"),
                '{"status": "original"}\n',
            )

    def test_remote_strict_v2_allows_pytest_cache_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cache = root / ".pytest_cache"
            cache.mkdir()
            (cache / "state").write_text("generated\n", encoding="utf-8")
            context = _v2_context(root)
            backend = _LocalRemoteBackend(root)
            context.workspace_backend = backend
            context.execution_workspace_root = "/workspace"
            context.execution_cwd = "/workspace"

            result = RemoteBashTool().run(
                {"command": "rm -rf /workspace/.pytest_cache"}, context
            )

            self.assertFalse(result.is_error)
            self.assertFalse(cache.exists())
            self.assertIn(
                "rm -rf /workspace/.pytest_cache", backend.exec_commands
            )

    def test_remote_write_and_bash_use_the_same_v2_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            context = _v2_context(root, owned_files=["/workspace/owned"])
            context.workspace_backend = _LocalRemoteBackend(root)
            context.execution_workspace_root = "/workspace"
            context.execution_cwd = "/workspace"

            result = RemoteFileWriteTool().run(
                {"file_path": "/workspace/owned/remote.py", "content": "OK = 1\n"},
                context,
            )
            self.assertFalse(result.is_error)
            with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
                RemoteFileWriteTool().run(
                    {
                        "file_path": "/workspace/other/remote.py",
                        "content": "BAD = 1\n",
                    },
                    context,
                )
            with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
                RemoteBashTool().run(
                    {"command": "mkdir -p other && printf bad > other/bash.txt"},
                    context,
                )

            generated = RemoteFileWriteTool().run(
                {
                    "file_path": "/workspace/tests/test_remote_probe.py",
                    "content": "VALUE = 1\n",
                },
                context,
            )
            self.assertFalse(generated.is_error)

            existing = root / "tests" / "test_existing.py"
            existing.write_text("VALUE = 1\n", encoding="utf-8")
            context.mark_remote_file_read("/workspace/tests/test_existing.py")
            with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
                RemoteFileWriteTool().run(
                    {
                        "file_path": "/workspace/tests/test_existing.py",
                        "content": "VALUE = 2\n",
                    },
                    context,
                )

            artifacts = RemoteBashTool().run(
                {
                    "command": (
                        "mkdir -p .cache/tool htmlcov && "
                        "printf cache > .cache/tool/result && "
                        "printf coverage > .coverage.remote && "
                        "printf db > test.sqlite3 && "
                        "printf backup > module.py.bak && "
                        "printf html > htmlcov/index.html"
                    )
                },
                context,
            )
            self.assertFalse(artifacts.is_error)

            scratch = "/workspace/.clawd/task-tests/task-one"
            result = RemoteFileWriteTool().run(
                {
                    "file_path": f"{scratch}/test_remote.py",
                    "content": "assert True\n",
                },
                context,
            )
            self.assertFalse(result.is_error)
            result = RemoteBashTool().run(
                {
                    "command": (
                        "mkdir -p .clawd/task-tests/task-one && "
                        "printf ok > .clawd/task-tests/task-one/test_bash.py"
                    )
                },
                context,
            )
            self.assertFalse(result.is_error)
            with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
                RemoteBashTool().run(
                    {
                        "command": (
                            "mkdir -p .clawd/task-tests/task-two && "
                            "printf bad > .clawd/task-tests/task-two/test_stolen.py"
                        )
                    },
                    context,
                )

    def test_remote_bash_restores_clawd_control_state_after_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            context = _v2_context(root)
            context.workspace_backend = _LocalRemoteBackend(root)
            context.execution_workspace_root = "/workspace"
            context.execution_cwd = "/workspace"
            team_id = str(context.team["team_id"])
            active = root / ".clawd" / "team.json"
            tasks = root / ".clawd" / "teams" / team_id / "tasks.json"
            events = root / ".clawd" / "teams" / team_id / "events.jsonl"
            before = {
                active: active.read_bytes(),
                tasks: tasks.read_bytes(),
                events: events.read_bytes(),
            }

            command = " && ".join(
                [
                    "mkdir -p .clawd/task-tests/task-one",
                    "printf 'assert True\\n' > .clawd/task-tests/task-one/test_kept.py",
                    "printf 'forged-active\\n' > .clawd/team.json",
                    f"printf 'forged-tasks\\n' > .clawd/teams/{team_id}/tasks.json",
                    f"printf 'forged-event\\n' >> .clawd/teams/{team_id}/events.jsonl",
                ]
            )
            with self.assertRaisesRegex(ToolPermissionError, "ownership violation"):
                RemoteBashTool().run({"command": command}, context)

            self.assertEqual(active.read_bytes(), before[active])
            self.assertEqual(tasks.read_bytes(), before[tasks])
            self.assertTrue(events.read_bytes().startswith(before[events]))
            self.assertNotIn(b"forged-event", events.read_bytes())
            self.assertEqual(
                (
                    root / ".clawd/task-tests/task-one/test_kept.py"
                ).read_text(encoding="utf-8"),
                "assert True\n",
            )


class _OwnershipProvider:
    model = "test-model"

    def __init__(self, *, infrastructure_error: bool = False) -> None:
        self.infrastructure_error = infrastructure_error
        self._bad_write_sent = False
        self._lock = threading.Lock()

    def chat(self, messages, tools=None, **kwargs):
        if self.infrastructure_error:
            raise ConnectionError("service unavailable during teammate rollout")
        text = "\n".join(str(message.get("content") or "") for message in messages)
        with self._lock:
            if "Task key: one" in text and not self._bad_write_sent:
                self._bad_write_sent = True
                return ChatResponse(
                    content="",
                    model=self.model,
                    usage={"input_tokens": 1, "output_tokens": 1},
                    finish_reason="tool_use",
                    tool_uses=[
                        {
                            "id": "bad-write",
                            "name": "Write",
                            "input": {
                                "file_path": "two.py",
                                "content": "stolen = True\n",
                            },
                        }
                    ],
                )
        return ChatResponse(
            content="worker finished",
            model=self.model,
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="stop",
            tool_uses=None,
        )


class RuntimeOwnershipOutcomeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.registry = build_default_registry(include_user_tools=False)
        self.context = ToolContext(workspace_root=self.root)
        TeamCreateTool().run(
            {"team_name": "strict-ownership", "quality_gates": True}, self.context
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _plan(self) -> None:
        result = TeamPlanTool().run(
            {
                "mode": "replace",
                "contract": {"summary": "disjoint files", "interfaces": []},
                "workers": [
                    {"name": "one", "instructions": "Implement one.py."},
                    {"name": "two", "instructions": "Implement two.py."},
                ],
                "tasks": [
                    {
                        "key": "one",
                        "owner": "one",
                        "instructions": "Implement one.py.",
                        "owned_files": ["one.py"],
                        "acceptance_checks": [
                            "python -c \"from pathlib import Path; assert Path('one.py').exists()\""
                        ],
                    },
                    {
                        "key": "two",
                        "owner": "two",
                        "instructions": "Implement two.py.",
                        "owned_files": ["two.py"],
                        "acceptance_checks": [
                            "python -c \"from pathlib import Path; assert Path('two.py').exists()\""
                        ],
                    },
                ],
                "validation": {
                    "profile": "generic",
                    "install_command": "true",
                    "import_command": "python -c \"import pathlib\"",
                    "integration_command": "python -c \"import pathlib; assert pathlib.Path\"",
                },
            },
            self.context,
        )
        self.assertFalse(result.is_error, result.output)

    def test_runtime_turns_sticky_violation_into_repair_required(self) -> None:
        self._plan()
        self.context.teammate_runtime = TeammateRuntime(
            _OwnershipProvider(), self.registry
        )

        result = TeamRunTool().run({"max_workers": 2}, self.context)

        self.assertEqual(result.output["status"], "repair_required")
        self.assertEqual(result.output["lifecycle_state"], "repair_required")
        tasks = self.context.team_store.load_tasks(self.context.team["team_id"])
        violated = next(task for task in tasks.values() if task.get("key") == "one")
        self.assertEqual(violated["status"], "failed")
        self.assertEqual(
            violated["metadata"]["ownership_audit"]["status"], "failed"
        )
        self.assertFalse((self.root / "two.py").exists())

    def test_v2_transport_failure_pauses_instead_of_failing_candidate(self) -> None:
        self._plan()
        self.context.teammate_runtime = TeammateRuntime(
            _OwnershipProvider(infrastructure_error=True), self.registry
        )

        result = TeamRunTool().run({"max_workers": 2}, self.context)

        self.assertEqual(result.output["status"], "paused")
        self.assertEqual(result.output["lifecycle_state"], "paused")
        self.assertEqual(result.output["failure_domain"], "infrastructure")
        self.assertTrue(result.output["retryable"])
        tasks = self.context.team_store.load_tasks(self.context.team["team_id"])
        self.assertEqual({task["status"] for task in tasks.values()}, {"pending"})
        self.assertTrue(
            all(
                task["metadata"]["infrastructure_failure"]["retryable"]
                for task in tasks.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
