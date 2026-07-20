from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import types
import unittest
from unittest.mock import patch

from src.execution.ags import AGSSettings, AGSWorkspaceBackend, _build_sandbox_command
from src.execution.backend import CommandOutcome


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


if __name__ == "__main__":
    unittest.main()
