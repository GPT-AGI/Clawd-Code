from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
BENCHMARK = ROOT / "teammate-evals" / "solo-vs-team" / "benchmark.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("solo_vs_team_benchmark", BENCHMARK)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load benchmark module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSoloVsTeamBenchmark(unittest.TestCase):
    def test_all_five_fixtures_are_valid_and_fail_before_repair(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(BENCHMARK), "--validate-fixtures"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("5 scenarios valid", completed.stdout)

    def test_lists_the_five_named_scenarios(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(BENCHMARK), "--list"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        self.assertEqual(len(lines), 5)
        self.assertTrue(any(line.startswith("webhook-idempotency:") for line in lines))

    def test_reads_historical_team_metrics_after_team_delete(self) -> None:
        benchmark = load_benchmark_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            team_dir = workspace / ".clawd" / "teams" / "team-1"
            (team_dir / "agents").mkdir(parents=True)
            (team_dir / "messages").mkdir()
            (team_dir / "agents" / "worker.json").write_text("{}", encoding="utf-8")
            (team_dir / "messages" / "handoff.json").write_text("{}", encoding="utf-8")
            (team_dir / "team.json").write_text(
                json.dumps(
                    {
                        "team_id": "team-1",
                        "status": "cancelled",
                        "usage": {"input_tokens": 11, "output_tokens": 7, "turns": 3},
                    }
                ),
                encoding="utf-8",
            )
            (team_dir / "tasks.json").write_text(
                json.dumps({"task-1": {"status": "completed"}}), encoding="utf-8"
            )
            (team_dir / "events.jsonl").write_text(
                '{"type":"team.failed"}\n{"type":"team.cancelled"}\n', encoding="utf-8"
            )

            metrics = benchmark._load_team_metrics(workspace)

        self.assertTrue(metrics["present"])
        self.assertFalse(metrics["active"])
        self.assertEqual(metrics["status"], "cancelled")
        self.assertEqual(metrics["worker_usage"]["input_tokens"], 11)
        self.assertEqual(metrics["failed_events"], 1)
        self.assertEqual(metrics["cancelled_events"], 1)


if __name__ == "__main__":
    unittest.main()
