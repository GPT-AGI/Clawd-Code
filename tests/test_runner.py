from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.cli import _read_run_prompt, main
from src.runner import run_prompt
from src.tool_system.agent_loop import AgentLoopResult


class TestRunner(unittest.TestCase):
    def test_run_prompt_builds_isolated_runtime(self) -> None:
        provider = Mock(model="configured-model")
        provider_class = Mock(return_value=provider)
        expected = AgentLoopResult("done", {"input_tokens": 2, "output_tokens": 1}, 1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with (
                patch("src.runner.get_default_provider", return_value="anthropic"),
                patch(
                    "src.runner.get_provider_config",
                    return_value={
                        "api_key": "test-key",
                        "base_url": "https://example.invalid",
                        "default_model": "configured-model",
                    },
                ),
                patch("src.runner.get_provider_class", return_value=provider_class),
                patch("src.runner.run_agent_loop", return_value=expected) as agent_loop,
            ):
                actual = run_prompt("  inspect this workspace  ", workspace=root)

        self.assertIs(actual, expected)
        provider_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://example.invalid",
            model="configured-model",
        )
        call = agent_loop.call_args.kwargs
        self.assertEqual(call["conversation"].messages[0].content, "inspect this workspace")
        self.assertEqual(call["tool_context"].workspace_root, root)
        self.assertIsNotNone(call["tool_context"].teammate_runtime)
        self.assertEqual(call["max_turns"], 100)

    def test_run_prompt_validates_inputs_before_provider_creation(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt must be non-empty"):
            run_prompt("  ")
        with self.assertRaisesRegex(ValueError, "max_turns"):
            run_prompt("task", max_turns=0)
        with self.assertRaisesRegex(ValueError, "workspace is not a directory"):
            run_prompt("task", workspace="/path/that/does/not/exist")

    def test_anthropic_environment_overrides_persisted_config(self) -> None:
        provider_class = Mock(return_value=Mock(model="env-model"))
        environment = {
            "ANTHROPIC_AUTH_TOKEN": "env-token",
            "ANTHROPIC_BASE_URL": "https://env.example.invalid",
            "ANTHROPIC_MODEL": "env-model",
        }
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.dict(os.environ, environment, clear=False),
                patch(
                    "src.runner.get_provider_config",
                    return_value={
                        "api_key": "saved-token",
                        "base_url": "https://saved.example.invalid",
                        "default_model": "saved-model",
                    },
                ),
                patch("src.runner.get_provider_class", return_value=provider_class),
                patch(
                    "src.runner.run_agent_loop",
                    return_value=AgentLoopResult("done", None, 1),
                ),
            ):
                run_prompt("task", workspace=tmp, provider_name="anthropic")

        provider_class.assert_called_once_with(
            api_key="env-token",
            base_url="https://env.example.invalid",
            model="env-model",
        )

    def test_read_prompt_file_relative_to_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "TASK.md").write_text("Run the task.\n", encoding="utf-8")
            self.assertEqual(_read_run_prompt(None, Path("TASK.md"), root), "Run the task.\n")

    def test_read_prompt_from_stdin(self) -> None:
        stdin = io.StringIO("Piped task\n")
        with patch("src.cli.sys.stdin", stdin):
            self.assertEqual(_read_run_prompt(None, None, Path(".")), "Piped task\n")

    def test_cli_dispatches_run_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "TASK.md"
            task.write_text("Do it.\n", encoding="utf-8")
            argv = [
                "clawd",
                "run",
                "--workspace",
                str(root),
                "--prompt-file",
                "TASK.md",
                "--max-turns",
                "42",
                "--quiet",
            ]
            with patch("src.cli.sys.argv", argv), patch("src.cli.run_once", return_value=0) as once:
                self.assertEqual(main(), 0)

        once.assert_called_once_with(
            "Do it.\n",
            workspace=root,
            provider_name=None,
            model=None,
            max_turns=42,
            stream=False,
            quiet=True,
        )

    def test_cli_dispatches_team_stop_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = [
                "clawd",
                "team",
                "stop",
                "coder",
                "--workspace",
                str(root),
                "--task-policy",
                "cancel",
                "--reason",
                "replace worker",
            ]
            with patch("src.cli.sys.argv", argv), patch(
                "src.cli.handle_team_command", return_value=0
            ) as team_command:
                self.assertEqual(main(), 0)

        args = team_command.call_args.args[0]
        self.assertEqual(args.team_command, "stop")
        self.assertEqual(args.teammate, "coder")
        self.assertEqual(args.task_policy, "cancel")
        self.assertEqual(args.reason, "replace worker")
        self.assertEqual(args.workspace, root)


if __name__ == "__main__":
    unittest.main()
