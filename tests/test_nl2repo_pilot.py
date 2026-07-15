from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tool_system.agent_loop import AgentLoopResult, ToolEvent


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "teammate-evals"
    / "nl2repo-pilot"
    / "benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("nl2repo_pilot_benchmark", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class TestNL2RepoPilot(unittest.TestCase):
    def make_upstream(self, root: Path) -> Path:
        task_dir = root / "test_files" / "sample-task"
        task_dir.mkdir(parents=True)
        (root / "test_files" / "task_difficulty.csv").write_text(
            "task-name,Level\nsample-task,Medium\n", encoding="utf-8"
        )
        (task_dir / "start.md").write_text("Build a sample package.\n", encoding="utf-8")
        (task_dir / "test_case_count.txt").write_text("12\n", encoding="utf-8")
        (task_dir / "test_commands.json").write_text(
            json.dumps(["pip install -e .", "pytest tests"]), encoding="utf-8"
        )
        (task_dir / "test_files.json").write_text(json.dumps(["tests"]), encoding="utf-8")
        return root

    def test_load_task_reads_external_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            upstream = self.make_upstream(Path(tmp))
            task = benchmark.load_task(upstream, "sample-task")

        self.assertEqual(task["difficulty"], "Medium")
        self.assertEqual(task["expected_tests"], 12)
        self.assertEqual(task["hidden_paths"], ["tests"])
        self.assertEqual(
            task["image"],
            "ghcr.io/multimodal-art-projection/nl2repobench/sample-task:1.0",
        )

    def test_prompt_leaves_topology_to_the_lead(self) -> None:
        adaptive = benchmark.build_prompt("adaptive")
        forced = benchmark.build_prompt("forced-team")

        self.assertIn("valid to remain solo", adaptive)
        self.assertIn("use at least one teammate", forced)
        self.assertIn("must be your runtime decision", adaptive)
        self.assertNotIn("planner", adaptive.lower())

    def test_prepare_workspace_creates_only_spec_and_git_metadata(self) -> None:
        task = {"document": "Build it.\n"}
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            digest = benchmark.prepare_workspace(task, workspace)

            self.assertTrue((workspace / ".git").is_dir())
            self.assertEqual((workspace / "start.md").read_text(encoding="utf-8"), "Build it.\n")
            self.assertEqual(digest, benchmark._hash_file(workspace / "start.md"))
            self.assertEqual(
                sorted(path.name for path in workspace.iterdir()),
                [".git", "start.md"],
            )

    def test_stage_score_context_removes_agent_tests_and_packaging(self) -> None:
        task = {
            "image": "example.invalid/sample:1.0",
            "hidden_paths": ["tests"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
            (workspace / "package.py").write_text("VALUE = 1\n", encoding="utf-8")
            (workspace / "tests").mkdir()
            (workspace / "tests" / "test_fake.py").write_text("pass\n", encoding="utf-8")

            metadata = benchmark.stage_score_context(task, workspace, root / "score")
            staged = root / "score" / "workspace"

            self.assertEqual(metadata["package_files_present"], ["pyproject.toml"])
            self.assertEqual(metadata["generated_hidden_paths"], ["tests"])
            self.assertFalse((staged / "pyproject.toml").exists())
            self.assertFalse((staged / "tests").exists())
            self.assertTrue((staged / "package.py").exists())

    def test_parse_pytest_output_uses_hidden_expected_total(self) -> None:
        result = benchmark.parse_pytest_output(
            "================ 9 passed, 2 failed, 1 skipped in 1.2s ================",
            12,
            1,
        )

        self.assertEqual(result["passed"], 9)
        self.assertEqual(result["failed"], 2)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["quality_score"], 75.0)
        self.assertFalse(result["all_passed"])

    def test_unsafe_hidden_test_paths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsafe"):
            benchmark._safe_relative_path("../tests")

    def test_agent_child_streams_and_persists_progress(self) -> None:
        received: dict[str, object] = {}

        def fake_run_prompt(prompt: str, **kwargs: object) -> AgentLoopResult:
            received.update(kwargs)
            on_event = kwargs["on_event"]
            on_text_chunk = kwargs["on_text_chunk"]
            assert callable(on_event)
            assert callable(on_text_chunk)
            on_event(ToolEvent(kind="model_started", model="test-model", turn=1))
            on_text_chunk("working")
            return AgentLoopResult(
                response_text="done",
                usage={"input_tokens": 3, "output_tokens": 2},
                num_turns=1,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_path = root / "PROMPT.md"
            result_path = root / "agent-result.json"
            progress_path = root / "progress.jsonl"
            prompt_path.write_text("Build it.", encoding="utf-8")
            with patch("src.runner.run_prompt", side_effect=fake_run_prompt):
                returncode = benchmark._run_agent_child(
                    root,
                    prompt_path,
                    result_path,
                    "anthropic",
                    "test-model",
                    5,
                    3,
                    8192,
                    True,
                    progress_path,
                )
            progress = [json.loads(line) for line in progress_path.read_text().splitlines()]
            result = json.loads(result_path.read_text())

        self.assertEqual(returncode, 0)
        self.assertTrue(received["stream"])
        self.assertEqual(received["max_output_tokens"], 8192)
        self.assertEqual([event["kind"] for event in progress], ["model_started", "text_chunk"])
        self.assertEqual(result["lead_model_calls"], 0)
        self.assertEqual(result["lead_usage"]["input_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
