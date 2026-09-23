from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import io
import os
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.execution.ags import AGSSettings, AGSWorkspaceBackend, _build_sandbox_command
from src.execution.backend import CommandOutcome


_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1]
    / "teammate-evals"
    / "nl2repo-pilot"
    / "benchmark.py"
)
_BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "nl2repo_pilot_benchmark_ags_test", _BENCHMARK_PATH
)
assert _BENCHMARK_SPEC is not None and _BENCHMARK_SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_BENCHMARK_SPEC)
sys.modules[_BENCHMARK_SPEC.name] = _BENCHMARK
_BENCHMARK_SPEC.loader.exec_module(_BENCHMARK)


class _FakeRuntime:
    def __init__(self, response: object) -> None:
        self.response = response
        self.command = None

    async def execute(self, command: object) -> object:
        self.command = command
        return self.response


class _FakeCommand:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class TestAGSWorkspaceBackend(unittest.TestCase):
    def settings(self) -> AGSSettings:
        return AGSSettings(secret_id="test-id", secret_key="test-key")

    def test_sandbox_command_removes_only_swerex_runtime_environment(self) -> None:
        command = _build_sandbox_command(
            'python3 -c "import sys; print(sys.executable)"',
            timeout_s=60,
            runtime_mount_path="/nix",
        )

        self.assertEqual(command[:2], ["/bin/bash", "-c"])
        self.assertEqual(
            command[-3:],
            ["/nix/swerex", "60", 'python3 -c "import sys; print(sys.executable)"'],
        )
        wrapper = command[2]
        self.assertIn('"$runtime_root"/*) continue', wrapper)
        self.assertIn("unset VIRTUAL_ENV", wrapper)
        self.assertIn("unset PYTHONHOME", wrapper)
        self.assertIn("--kill-after=5s", wrapper)
        self.assertIn('/bin/bash -c "$user_command"', wrapper)

    def test_submit_cancels_future_and_provides_descriptive_timeout(self) -> None:
        backend = AGSWorkspaceBackend(self.settings())
        backend._loop = object()  # type: ignore[assignment]
        cancelled = False

        class TimedOutFuture:
            def result(self, timeout: float | None = None) -> object:
                raise concurrent.futures.TimeoutError

            def done(self) -> bool:
                return False

            def cancel(self) -> bool:
                nonlocal cancelled
                cancelled = True
                return True

        async def pending() -> None:
            return None

        coroutine = pending()
        try:
            with patch("asyncio.run_coroutine_threadsafe", return_value=TimedOutFuture()):
                with self.assertRaisesRegex(
                    TimeoutError,
                    "sandbox probe timed out after 3s; the pending request was cancelled",
                ):
                    backend._submit(coroutine, timeout=3, operation="sandbox probe")
        finally:
            coroutine.close()

        self.assertTrue(cancelled)

    def test_submit_preserves_timeout_raised_by_completed_coroutine(self) -> None:
        backend = AGSWorkspaceBackend(self.settings())
        backend._loop = object()  # type: ignore[assignment]
        cancelled = False

        class CompletedFuture:
            def result(self, timeout: float | None = None) -> object:
                raise TimeoutError("remote command timeout")

            def done(self) -> bool:
                return True

            def cancel(self) -> bool:
                nonlocal cancelled
                cancelled = True
                return True

        async def completed() -> None:
            return None

        coroutine = completed()
        try:
            with patch("asyncio.run_coroutine_threadsafe", return_value=CompletedFuture()):
                with self.assertRaisesRegex(TimeoutError, "remote command timeout"):
                    backend._submit(coroutine, timeout=3, operation="sandbox probe")
        finally:
            coroutine.close()

        self.assertFalse(cancelled)

    def test_exec_uses_process_group_timeout_and_reports_exit_124(self) -> None:
        response = types.SimpleNamespace(exit_code=124, stdout="partial output", stderr="")
        runtime = _FakeRuntime(response)
        backend = AGSWorkspaceBackend(self.settings())
        backend._started = True
        backend._deployment = types.SimpleNamespace(runtime=runtime)
        backend._loop = object()  # type: ignore[assignment]

        command_module = types.ModuleType("swerex.runtime.abstract")
        command_module.Command = _FakeCommand  # type: ignore[attr-defined]
        runtime_module = types.ModuleType("swerex.runtime")
        runtime_module.abstract = command_module  # type: ignore[attr-defined]
        swerex_module = types.ModuleType("swerex")
        swerex_module.runtime = runtime_module  # type: ignore[attr-defined]

        def submit(coroutine: object, **_: object) -> object:
            return asyncio.run(coroutine)  # type: ignore[arg-type]

        with (
            patch.dict(
                sys.modules,
                {
                    "swerex": swerex_module,
                    "swerex.runtime": runtime_module,
                    "swerex.runtime.abstract": command_module,
                },
            ),
            patch.object(backend, "_submit", side_effect=submit),
        ):
            result = backend.exec("sleep 10", cwd="/workspace", timeout_s=2)

        self.assertEqual(result.exit_code, 124)
        self.assertEqual(result.stdout, "partial output")
        self.assertIn("timed out after 2s", result.stderr)
        self.assertIsNotNone(runtime.command)
        self.assertFalse(runtime.command.shell)
        self.assertEqual(runtime.command.timeout, 12)
        self.assertEqual(runtime.command.command[-1], "sleep 10")

    def test_exec_converts_outer_timeout_to_tool_result(self) -> None:
        backend = AGSWorkspaceBackend(self.settings())
        backend._started = True
        backend._deployment = types.SimpleNamespace(runtime=types.SimpleNamespace())
        backend._loop = object()  # type: ignore[assignment]

        async def never_called(_: object) -> object:
            raise AssertionError

        backend._deployment.runtime.execute = never_called

        def time_out(coroutine: object, **_: object) -> object:
            coroutine.close()  # type: ignore[attr-defined]
            raise TimeoutError(
                "sandbox command (1s limit) timed out after 21s; the pending request was cancelled"
            )

        with patch.object(backend, "_submit", side_effect=time_out):
            result = backend.exec("sleep 10", cwd="/workspace", timeout_s=1)

        self.assertEqual(result.exit_code, 124)
        self.assertIn("pending request was cancelled", result.stderr)

    def test_reset_workspace_preserves_the_ags_mountpoint(self) -> None:
        backend = AGSWorkspaceBackend(self.settings())
        commands: list[tuple[str, str, int]] = []

        def execute(command: str, *, cwd: str, timeout_s: int) -> CommandOutcome:
            commands.append((command, cwd, timeout_s))
            return CommandOutcome(0)

        with patch.object(backend, "exec", side_effect=execute):
            backend.reset_workspace()

        self.assertEqual(len(commands), 1)
        command, cwd, timeout_s = commands[0]
        self.assertNotIn("rm -rf /workspace &&", command)
        self.assertIn("rm -rf -- /workspace/*", command)
        self.assertEqual(cwd, "/")
        self.assertEqual(timeout_s, 120)

    @staticmethod
    def _archive(*members: tuple[tarfile.TarInfo, bytes]) -> tarfile.TarFile:
        packed = io.BytesIO()
        with tarfile.open(fileobj=packed, mode="w") as archive:
            for member, content in members:
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content) if content else None)
        packed.seek(0)
        return tarfile.open(fileobj=packed, mode="r")

    def test_safe_extract_accepts_regular_workspace_prefixed_file(self) -> None:
        member = tarfile.TarInfo("workspace/model.py")
        content = b"class Model:\n    pass\n"
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "download"
            with self._archive((member, content)) as archive:
                AGSWorkspaceBackend._safe_extract(archive, destination)

            self.assertEqual((destination / "workspace/model.py").read_bytes(), content)

    def test_safe_extract_accepts_internal_symbolic_and_hard_links(self) -> None:
        target = tarfile.TarInfo("workspace/computer/model.py")
        symbolic = tarfile.TarInfo("workspace/model.py")
        symbolic.type = tarfile.SYMTYPE
        symbolic.linkname = "computer/model.py"
        hard = tarfile.TarInfo("workspace/model-copy.py")
        hard.type = tarfile.LNKTYPE
        hard.linkname = "workspace/computer/model.py"
        content = b"MODEL = True\n"

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "download"
            with self._archive((target, content), (symbolic, b""), (hard, b"")) as archive:
                AGSWorkspaceBackend._safe_extract(archive, destination)

            workspace = destination / "workspace"
            self.assertFalse((workspace / "model.py").is_symlink())
            self.assertEqual((workspace / "model.py").read_bytes(), content)
            self.assertEqual((workspace / "model-copy.py").read_bytes(), content)
            self.assertEqual(
                os.stat(workspace / "computer/model.py").st_ino,
                os.stat(workspace / "model-copy.py").st_ino,
            )

    def test_safe_extract_materializes_internal_directory_symlink(self) -> None:
        package = tarfile.TarInfo("workspace/implementation")
        package.type = tarfile.DIRTYPE
        module = tarfile.TarInfo("workspace/implementation/module.py")
        alias = tarfile.TarInfo("workspace/package")
        alias.type = tarfile.SYMTYPE
        alias.linkname = "implementation"

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "download"
            with self._archive(
                (package, b""),
                (module, b"VALUE = 1\n"),
                (alias, b""),
            ) as archive:
                AGSWorkspaceBackend._safe_extract(archive, destination)

            materialized = destination / "workspace/package"
            self.assertTrue(materialized.is_dir())
            self.assertFalse(materialized.is_symlink())
            self.assertEqual(
                (materialized / "module.py").read_text(encoding="utf-8"),
                "VALUE = 1\n",
            )

    def test_materialized_workspace_can_be_staged_for_score_context(self) -> None:
        target = tarfile.TarInfo("workspace/computer/model.py")
        symbolic = tarfile.TarInfo("workspace/model.py")
        symbolic.type = tarfile.SYMTYPE
        symbolic.linkname = "computer/model.py"
        hard = tarfile.TarInfo("workspace/model-copy.py")
        hard.type = tarfile.LNKTYPE
        hard.linkname = "workspace/computer/model.py"

        task = {
            "image": "example.invalid/autorccar:1.0",
            "hidden_paths": [],
            "test_commands": ["pytest"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            downloaded = root / "downloaded"
            with self._archive(
                (target, b"MODEL = True\n"),
                (symbolic, b""),
                (hard, b""),
            ) as archive:
                AGSWorkspaceBackend._safe_extract(archive, downloaded)

            metadata = _BENCHMARK.stage_score_context(
                task,
                downloaded / "workspace",
                root / "score",
            )
            staged = root / "score/workspace"
            self.assertEqual(metadata["score_context_stats"]["source"]["file_count"], 3)
            self.assertEqual((staged / "model.py").read_text(), "MODEL = True\n")
            self.assertEqual((staged / "model-copy.py").read_text(), "MODEL = True\n")
            self.assertFalse(any(path.is_symlink() for path in staged.rglob("*")))

    def test_safe_extract_rejects_absolute_and_parent_escape_paths(self) -> None:
        attacks = ("/tmp/absolute.py", "workspace/../../../escape.py")
        for name in attacks:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "download"
                member = tarfile.TarInfo(name)
                with self._archive((member, b"bad")) as archive:
                    with self.assertRaisesRegex(ValueError, "unsafe path"):
                        AGSWorkspaceBackend._safe_extract(archive, destination)
                self.assertFalse((Path(temporary) / "escape.py").exists())

    def test_safe_extract_rejects_links_that_escape_or_are_not_regular(self) -> None:
        attacks: list[tuple[tarfile.TarInfo, bytes]] = []
        symbolic = tarfile.TarInfo("workspace/model.py")
        symbolic.type = tarfile.SYMTYPE
        symbolic.linkname = "../../outside.py"
        attacks.append((symbolic, b""))

        hard = tarfile.TarInfo("workspace/model.py")
        hard.type = tarfile.LNKTYPE
        hard.linkname = "../outside.py"
        attacks.append((hard, b""))

        dangling = tarfile.TarInfo("workspace/dangling.py")
        dangling.type = tarfile.SYMTYPE
        dangling.linkname = "missing.py"
        attacks.append((dangling, b""))

        for member, content in attacks:
            with self.subTest(kind=member.type), tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "download"
                with self._archive((member, content)) as archive:
                    with self.assertRaisesRegex(ValueError, "unsafe (path|entry)"):
                        AGSWorkspaceBackend._safe_extract(archive, destination)
                self.assertFalse((Path(temporary) / "outside.py").exists())

    def test_safe_extract_enforces_archive_and_materialization_limits(self) -> None:
        first = tarfile.TarInfo("workspace/one.py")
        second = tarfile.TarInfo("workspace/two.py")
        with (
            tempfile.TemporaryDirectory() as temporary,
            self._archive((first, b"1"), (second, b"2")) as archive,
            patch("src.execution.ags.AGS_ARCHIVE_MAX_MEMBERS", 1),
        ):
            with self.assertRaisesRegex(ValueError, "member-count limit"):
                AGSWorkspaceBackend._safe_extract(
                    archive, Path(temporary) / "download"
                )

        oversized = tarfile.TarInfo("workspace/large.py")
        with (
            tempfile.TemporaryDirectory() as temporary,
            self._archive((oversized, b"1234")) as archive,
            patch("src.execution.ags.AGS_ARCHIVE_MAX_TOTAL_BYTES", 3),
        ):
            with self.assertRaisesRegex(ValueError, "expanded-size limit"):
                AGSWorkspaceBackend._safe_extract(
                    archive, Path(temporary) / "download"
                )

        target_dir = tarfile.TarInfo("workspace/implementation")
        target_dir.type = tarfile.DIRTYPE
        target = tarfile.TarInfo("workspace/implementation/module.py")
        alias = tarfile.TarInfo("workspace/package")
        alias.type = tarfile.SYMTYPE
        alias.linkname = "implementation"
        with (
            tempfile.TemporaryDirectory() as temporary,
            self._archive(
                (target_dir, b""),
                (target, b"1234"),
                (alias, b""),
            ) as archive,
            patch("src.execution.ags.AGS_ARCHIVE_MAX_TOTAL_BYTES", 7),
        ):
            with self.assertRaisesRegex(ValueError, "expanded-size limit"):
                AGSWorkspaceBackend._safe_extract(
                    archive, Path(temporary) / "download"
                )


if __name__ == "__main__":
    unittest.main()
