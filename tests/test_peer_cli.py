from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.cli import main
from src.peer.models import PeerRunConfig


class PeerCliTests(unittest.TestCase):
    def test_peer_cli_routes_all_reproducibility_options(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            prompt = repo / "TASK.md"
            prompt.write_text("same mission\n", encoding="utf-8")
            completed = {
                "status": "completed",
                "acceptance": {"exit_code": 0},
                "run_id": "run",
            }
            argv = [
                "clawd",
                "peer",
                "run",
                "--repo",
                str(repo),
                "--prompt-file",
                "TASK.md",
                "--peers",
                "3",
                "--communication",
                "star",
                "--workspace-mode",
                "worktree",
                "--provider",
                "anthropic",
                "--model",
                "glm-5.2",
                "--timeout-seconds",
                "42",
                "--max-turns",
                "9",
                "--max-output-tokens",
                "2048",
                "--token-budget",
                "50000",
                "--turn-budget",
                "25",
                "--output-dir",
                str(repo / "outputs"),
                "--coordinator-peer",
                "peer-2",
                "--acceptance-command",
                "python -m pytest -q",
            ]
            with (
                patch("sys.argv", argv),
                patch(
                    "src.peer.runner.run_peer_collaboration", return_value=completed
                ) as run,
                redirect_stdout(io.StringIO()),
            ):
                exit_code = main()
            self.assertEqual(exit_code, 0)
            kwargs = run.call_args.kwargs
            self.assertEqual(run.call_args.args, ("same mission\n",))
            self.assertEqual(kwargs["peers"], 3)
            self.assertEqual(kwargs["communication"], "star")
            self.assertEqual(kwargs["workspace_mode"], "worktree")
            self.assertEqual(kwargs["model"], "glm-5.2")
            self.assertEqual(kwargs["coordinator_peer"], "peer-2")
            self.assertEqual(kwargs["acceptance_command"], ["python", "-m", "pytest", "-q"])

    def test_peer_config_rejects_invalid_protocol_combinations(self) -> None:
        base = {
            "repo_path": ".",
            "mission": "mission",
            "peers": 2,
            "communication": "p2p",
            "workspace_mode": "shared",
        }
        invalid = [
            {**base, "peers": 0},
            {**base, "communication": "solo"},
            {**base, "workspace_mode": "unknown"},
            {**base, "timeout_seconds": 0},
            {**base, "token_budget": 0},
            {**base, "mission": ""},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                PeerRunConfig(**values).validate()


if __name__ == "__main__":
    unittest.main()
