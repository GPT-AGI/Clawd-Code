from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from src.execution.backend import CommandOutcome
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
    def make_upstream(self, root: Path, task_name: str = "sample-task") -> Path:
        task_dir = root / "test_files" / task_name
        task_dir.mkdir(parents=True)
        (root / "test_files" / "task_difficulty.csv").write_text(
            f"task-name,Level\n{task_name},Medium\n", encoding="utf-8"
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

    def test_image_references_normalize_mixed_case_task_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            upstream = self.make_upstream(Path(tmp), "more-Itertools")
            task = benchmark.load_task(upstream, "more-Itertools")

        self.assertEqual(
            task["image"],
            "ghcr.io/multimodal-art-projection/nl2repobench/more-itertools:1.0",
        )
        self.assertEqual(
            benchmark.format_ags_image(
                benchmark.AGS_IMAGE_TEMPLATE, "more-Itertools"
            ),
            "swebenchdocker.tencentcloudcr.com/swebench/nl2repo:more-itertools-1.0",
        )

    def test_forced_team_prompt_uses_the_harness_precreated_team(self) -> None:
        prompt = benchmark.build_prompt("forced-team")

        self.assertIn("harness has already created the active strict Team", prompt)
        self.assertIn("Do not call TeamCreate", prompt)
        self.assertIn("must call TeammateCreate", prompt)
        self.assertIn("TaskCreate", prompt)
        self.assertIn("TeamRun", prompt)
        self.assertIn("TeamConfigure", prompt)
        self.assertIn("TeamVerify", prompt)
        self.assertIn("at least two teammates", prompt)

    def test_failure_classifier_distinguishes_dependency_and_team_contract_failures(self) -> None:
        base = {
            "agent_ok": True,
            "integrity_ok": True,
            "protocol_ok": True,
            "hidden": {
                "pytest": {
                    "returncode": 1,
                    "errors": 1,
                    "failed": 0,
                    "all_passed": False,
                }
            },
        }
        dependency = benchmark.classify_failure(
            **base,
            team={"agents": [{}], "peer_messages": 0},
            hidden_log="ModuleNotFoundError: No module named 'ujson'",
        )
        contract = benchmark.classify_failure(
            **{
                **base,
                "hidden": {
                    "pytest": {
                        "returncode": 1,
                        "errors": 0,
                        "failed": 3,
                        "all_passed": False,
                    }
                },
            },
            team={"agents": [{}, {}], "peer_messages": 0},
            hidden_log="AttributeError: public key has no attribute 'BASE'",
        )

        self.assertEqual(dependency, "dependency_environment")
        self.assertEqual(contract, "cross_module_contract")

    def test_rollout32_selection_matches_fixed_bounded_subset(self) -> None:
        metadata = [
            {"id": f"task-{index:02d}", "prompt_bytes": index * 1_000}
            for index in range(1, 71)
        ]

        first = benchmark.select_task_subset(metadata, count=32, seed=20260715)
        second = benchmark.select_task_subset(
            list(reversed(metadata)), count=32, seed=20260715
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        self.assertTrue(all(int(name.removeprefix("task-")) <= 65 for name in first))

    def test_reward_pool_does_not_occupy_rollout_slots(self) -> None:
        tasks = [
            {"id": f"task-{index}", "difficulty": "Easy", "expected_tests": 1}
            for index in range(3)
        ]
        lock = threading.Lock()
        reward_started = threading.Event()
        third_rollout_started = threading.Event()
        release_reward = threading.Event()
        active_rollouts = 0
        max_active_rollouts = 0
        events: list[dict[str, object]] = []

        def rollout(task: dict[str, object], mode: str) -> object:
            nonlocal active_rollouts, max_active_rollouts
            with lock:
                active_rollouts += 1
                max_active_rollouts = max(max_active_rollouts, active_rollouts)
            if task["id"] == "task-1":
                self.assertTrue(third_rollout_started.wait(timeout=2))
            elif task["id"] == "task-2":
                self.assertTrue(reward_started.wait(timeout=2))
                third_rollout_started.set()
                release_reward.set()
            with lock:
                active_rollouts -= 1
            return task

        def reward(artifact: dict[str, object]) -> dict[str, object]:
            if artifact["id"] == "task-0":
                reward_started.set()
                self.assertTrue(release_reward.wait(timeout=2))
            return {"task": artifact["id"], "quality_score": 100.0, "success": True}

        results = benchmark.run_evaluation_pool(
            [(task, "adaptive") for task in tasks],
            rollout,
            reward,
            rollout_concurrency=2,
            reward_concurrency=1,
            on_event=events.append,
        )

        self.assertEqual([result["task"] for result in results], ["task-0", "task-1", "task-2"])
        self.assertEqual(max_active_rollouts, 2)
        started_third = next(
            index
            for index, event in enumerate(events)
            if event["event"] == "rollout.started" and event["task"] == "task-2"
        )
        completed_reward = next(
            index
            for index, event in enumerate(events)
            if event["event"] == "reward.completed" and event["task"] == "task-0"
        )
        self.assertLess(started_third, completed_reward)

    def test_ags_image_preparation_is_retried(self) -> None:
        attempts = 0
        sleeps: list[float] = []

        class FakeBackend:
            def start(self) -> "FakeBackend":
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise RuntimeError(
                        "[TencentCloudSDKException] code:ResourceUnavailable "
                        "message:image is still preparing, please retry later"
                    )
                return self

        backend = benchmark.start_ags_backend_with_retry(
            FakeBackend,
            attempts=4,
            delay_s=0.25,
            sleep_fn=sleeps.append,
        )

        self.assertIsInstance(backend, FakeBackend)
        self.assertEqual(attempts, 3)
        self.assertEqual(sleeps, [0.25, 0.25])

    def test_ags_non_preparation_error_is_not_retried(self) -> None:
        attempts = 0

        class BrokenBackend:
            def start(self) -> "BrokenBackend":
                nonlocal attempts
                attempts += 1
                raise RuntimeError("permission denied")

        with self.assertRaisesRegex(RuntimeError, "permission denied"):
            benchmark.start_ags_backend_with_retry(
                BrokenBackend,
                attempts=4,
                delay_s=0,
            )

        self.assertEqual(attempts, 1)

    def test_prompt_leaves_topology_to_the_lead(self) -> None:
        adaptive = benchmark.build_prompt("adaptive")
        adaptive_v2 = benchmark.build_prompt("adaptive-team-v2")
        forced = benchmark.build_prompt("forced-team")

        self.assertIn("valid to remain solo", adaptive)
        self.assertIn("at least two substantially independent", adaptive_v2)
        self.assertIn("at most two teammates", adaptive_v2)
        self.assertIn("valid to complete", adaptive_v2)
        self.assertIn("at least two teammates", forced)
        self.assertIn("must be your runtime decision", adaptive)
        self.assertNotIn("planner", adaptive.lower())

    def test_adaptive_team_v2_allows_a_valid_solo_route(self) -> None:
        self.assertTrue(
            benchmark._protocol_ok(
                "adaptive-team-v2",
                {"present": False},
            )
        )

    def test_missing_agent_result_is_rejected_before_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "agent-result.json"
            with self.assertRaisesRegex(
                RuntimeError,
                "exited with code 1 without producing agent-result.json",
            ):
                benchmark._require_agent_result(
                    result_path,
                    returncode=1,
                    timed_out=False,
                    stderr="ImportError: circular import",
                )

    def test_explicit_agent_failure_result_remains_scoreable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "agent-result.json"
            result_path.write_text(
                json.dumps({"ok": False, "error": "model failed after editing"}),
                encoding="utf-8",
            )
            result = benchmark._require_agent_result(
                result_path,
                returncode=1,
                timed_out=False,
                stderr="",
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "model failed after editing")

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
            "test_commands": ["pip install -e .", "pytest tests"],
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
            dockerfile = (root / "score" / "Dockerfile").read_text(encoding="utf-8")
            self.assertNotIn("pip install -e .", dockerfile)
            self.assertEqual(metadata["setup_commands"], ["pip install -e ."])
            self.assertEqual(metadata["test_commands"], ["pytest tests"])

    def test_score_commands_continue_to_pytest_when_setup_fails(self) -> None:
        command = benchmark._score_shell_command(
            ["pip install -e ."],
            ["pytest tests"],
        )

        self.assertEqual(command, "(pip install -e .); (pytest tests)")
        self.assertNotIn("&&", command)

    def test_reward_is_skipped_when_forced_team_protocol_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_root = Path(tmp) / "sample" / "forced-team"
            workspace = case_root / "workspace"
            workspace.mkdir(parents=True)
            start = workspace / "start.md"
            start.write_text("spec\n", encoding="utf-8")
            rollout = benchmark.RolloutArtifact(
                task={"id": "sample", "difficulty": "Easy", "expected_tests": 3},
                mode="forced-team",
                case_root=case_root,
                workspace=workspace,
                start_hash=benchmark._hash_file(start),
                agent={"ok": True, "lead_usage": {}, "lead_turns": 1},
                agent_elapsed_s=1.0,
                agent_timed_out=False,
                agent_returncode=0,
            )
            with patch.object(benchmark, "run_hidden_tests") as scorer:
                result = benchmark.score_rollout(
                    rollout,
                    provider="qwen",
                    model="test",
                    score_timeout_s=60,
                    keep_image=False,
                )

        scorer.assert_not_called()
        self.assertTrue(result["reward_skipped"])
        self.assertEqual(result["failure_class"], "team_protocol")
        self.assertFalse(result["protocol_ok"])

    def test_rescore_reuses_workspace_and_preserves_reward_history(self) -> None:
        task = {"id": "sample-task", "expected_tests": 2}
        hidden = {
            "pytest": {
                "expected": 2,
                "passed": 2,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
                "returncode": 0,
                "quality_score": 100.0,
                "all_passed": True,
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_root = root / "sample-task" / "adaptive"
            workspace = case_root / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
            (case_root / "result.json").write_text(
                json.dumps(
                    {
                        "agent_ok": True,
                        "integrity_ok": True,
                        "protocol_ok": True,
                        "quality_score": 0.0,
                        "success": False,
                        "hidden_tests": {"error": "old scorer failure"},
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(benchmark, "run_hidden_tests", return_value=hidden):
                result = benchmark.rescore_existing_case(
                    task,
                    "adaptive",
                    root,
                    score_backend="docker",
                    score_timeout_s=60,
                    keep_image=False,
                )

            persisted = json.loads((case_root / "result.json").read_text())

        self.assertEqual(result["quality_score"], 100.0)
        self.assertTrue(result["success"])
        self.assertEqual(result["reward_history"][0]["quality_score"], 0.0)
        self.assertEqual(persisted["hidden_tests"], hidden)

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

    def test_combined_usage_separates_components_and_marks_coverage(self) -> None:
        complete = benchmark._combined_usage(
            {"input_tokens": 100, "output_tokens": 20},
            {"input_tokens": 50, "output_tokens": 10, "turns": 3},
            used_team=True,
            lead_turns=4,
        )
        missing_lead = benchmark._combined_usage(
            {},
            {"input_tokens": 50, "output_tokens": 10, "turns": 3},
            used_team=True,
            lead_turns=4,
        )
        solo = benchmark._combined_usage(
            {"input_tokens": 100, "output_tokens": 20},
            {},
            used_team=False,
            lead_turns=4,
        )

        self.assertEqual(complete["total_tokens"], 180)
        self.assertEqual(complete["lead_input_tokens"], 100)
        self.assertEqual(complete["worker_output_tokens"], 10)
        self.assertTrue(complete["complete"])
        self.assertFalse(missing_lead["complete"])
        self.assertTrue(solo["complete"])

    def test_score_command_split_keeps_pytest_install_in_build_phase(self) -> None:
        setup, tests = benchmark._split_score_commands([
            "pip install pytest",
            "pip install -e .",
            "python -m pytest tests",
        ])

        self.assertEqual(setup, ["pip install pytest", "pip install -e ."])
        self.assertEqual(tests, ["python -m pytest tests"])

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

    def test_forced_team_agent_child_precreates_active_team(self) -> None:
        observed: dict[str, object] = {}

        def fake_run_prompt(prompt: str, **kwargs: object) -> AgentLoopResult:
            from src.teammate.store import TeamStore

            active = TeamStore(Path(kwargs["workspace"])).load_active_team()
            observed["team"] = active
            return AgentLoopResult("done", {}, 1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_path = root / "PROMPT.md"
            result_path = root / "agent-result.json"
            progress_path = root / "progress.jsonl"
            prompt_path.write_text(benchmark.build_prompt("forced-team"), encoding="utf-8")
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
                    False,
                    progress_path,
                    mode="forced-team",
                )
            progress = [json.loads(line) for line in progress_path.read_text().splitlines()]

        self.assertEqual(returncode, 0)
        self.assertIsNotNone(observed["team"])
        self.assertEqual(observed["team"].team_name, "nl2repo-forced-team")
        self.assertTrue(observed["team"].settings["quality_gates"]["strict"])
        self.assertEqual(progress[0]["kind"], "forced_team_precreated")

    def test_agent_child_preserves_terminal_usage_when_provider_raises(self) -> None:
        def fake_run_prompt(prompt: str, **kwargs: object) -> AgentLoopResult:
            on_event = kwargs["on_event"]
            assert callable(on_event)
            on_event(ToolEvent(
                kind="model_response",
                model="test-model",
                usage={"input_tokens": 7, "output_tokens": 5},
                turn=4,
            ))
            on_event(ToolEvent(
                kind="run_failed",
                model="test-model",
                usage={"input_tokens": 31, "output_tokens": 19},
                turn=4,
                is_error=True,
                error="provider rejected history",
            ))
            raise RuntimeError("provider rejected history")

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
                    False,
                    progress_path,
                )
            result = json.loads(result_path.read_text())

        self.assertEqual(returncode, 1)
        self.assertEqual(result["lead_turns"], 4)
        self.assertEqual(result["lead_usage"], {"input_tokens": 31, "output_tokens": 19})

    def test_agent_child_uses_and_cleans_up_ags_workspace(self) -> None:
        calls: list[object] = []
        received: dict[str, object] = {}

        class FakeAGSBackend:
            workspace_root = "/workspace"
            sandbox_id = "ags-test-1"

            def __init__(self, settings: object) -> None:
                calls.append(("init", settings))

            def start(self):
                calls.append("start")
                return self

            def reset_workspace(self) -> None:
                calls.append("reset")

            def upload_tree(self, local: Path, remote: str) -> None:
                calls.append(("upload", local, remote))

            def download_tree(self, remote: str, local: Path) -> None:
                calls.append(("download", remote, local))
                (local / "generated.py").write_text("VALUE = 1\n", encoding="utf-8")

            def close(self) -> None:
                calls.append("close")

        def fake_run_prompt(prompt: str, **kwargs: object) -> AgentLoopResult:
            received.update(kwargs)
            return AgentLoopResult("done", {"input_tokens": 2, "output_tokens": 1}, 1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "start.md").write_text("Build it.\n", encoding="utf-8")
            prompt_path = root / "PROMPT.md"
            result_path = root / "agent-result.json"
            progress_path = root / "progress.jsonl"
            prompt_path.write_text("Build it.", encoding="utf-8")
            with (
                patch("src.runner.run_prompt", side_effect=fake_run_prompt),
                patch("src.execution.ags.AGSSettings.from_env", return_value="settings"),
                patch("src.execution.ags.AGSWorkspaceBackend", FakeAGSBackend),
            ):
                returncode = benchmark._run_agent_child(
                    workspace,
                    prompt_path,
                    result_path,
                    "anthropic",
                    "test-model",
                    5,
                    3,
                    8192,
                    True,
                    progress_path,
                    execution_backend="ags",
                    ags_image="registry.invalid/nl2repo:sample-1.0",
                )

            result = json.loads(result_path.read_text())
            progress = [json.loads(line)["kind"] for line in progress_path.read_text().splitlines()]
            self.assertTrue((workspace / "generated.py").is_file())

        self.assertEqual(returncode, 0)
        self.assertEqual(result["sandbox_id"], "ags-test-1")
        self.assertIsInstance(received["workspace_backend"], FakeAGSBackend)
        self.assertIn("sandbox_started", progress)
        self.assertIn("sandbox_workspace_downloaded", progress)
        self.assertEqual(calls[-1], "close")

    def test_ags_scorer_uses_fresh_image_without_resetting_workspace(self) -> None:
        calls: list[object] = []

        class FakeAGSBackend:
            workspace_root = "/workspace"
            sandbox_id = "ags-score-1"

            def __init__(self, settings: object) -> None:
                calls.append(("init", settings))

            def start(self):
                calls.append("start")
                return self

            def upload_tree(self, local: Path, remote: str) -> None:
                calls.append(("upload", local, remote))

            def exec(self, command: str, *, cwd: str, timeout_s: int) -> CommandOutcome:
                calls.append(("exec", command, cwd, timeout_s))
                return CommandOutcome(0, "12 passed in 0.1s\n", "")

            def close(self) -> None:
                calls.append("close")

        task = {
            "id": "sample-task",
            "image": "ghcr.invalid/sample:1.0",
            "expected_tests": 12,
            "hidden_paths": ["tests"],
            "test_commands": ["pip install -e .", "pytest tests"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
            with (
                patch("src.execution.ags.AGSSettings.from_env", return_value=type("S", (), {"runtime_timeout": 60.0})()),
                patch("src.execution.ags.AGSWorkspaceBackend", FakeAGSBackend),
            ):
                result = benchmark.run_hidden_tests_ags(
                    task,
                    workspace,
                    root / "case",
                    timeout_s=120,
                    ags_image="registry.invalid/sample:1.0",
                    ags_env_file=None,
                    ags_timeout="3h",
                    ags_cpu="2",
                    ags_memory="4Gi",
                    ags_score_tool_id="sdt-no-egress",
                )

        self.assertEqual(result["pytest"]["passed"], 12)
        self.assertTrue(result["pytest"]["all_passed"])
        self.assertFalse(any(call == "reset" for call in calls))
        executed = next(
            call
            for call in calls
            if isinstance(call, tuple) and call[0] == "exec" and "pytest tests" in call[1]
        )
        self.assertIn("pip install -e .", executed[1])
        self.assertIn("pytest tests", executed[1])

    def test_ags_scorer_fails_closed_when_tool_has_egress(self) -> None:
        calls: list[str] = []

        class PublicAGSBackend:
            workspace_root = "/workspace"
            sandbox_id = "ags-public-score"

            def __init__(self, settings: object) -> None:
                pass

            def start(self):
                return self

            def exec(self, command: str, *, cwd: str, timeout_s: int) -> CommandOutcome:
                calls.append(command)
                return CommandOutcome(86, "", "")

            def upload_tree(self, local: Path, remote: str) -> None:
                raise AssertionError("candidate must not be uploaded before isolation passes")

            def close(self) -> None:
                calls.append("closed")

        task = {
            "id": "sample-task",
            "image": "ghcr.invalid/sample:1.0",
            "expected_tests": 1,
            "hidden_paths": ["tests"],
            "test_commands": ["pytest tests"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
            with (
                patch(
                    "src.execution.ags.AGSSettings.from_env",
                    return_value=type("S", (), {"runtime_timeout": 60.0})(),
                ),
                patch("src.execution.ags.AGSWorkspaceBackend", PublicAGSBackend),
            ):
                result = benchmark.run_hidden_tests_ags(
                    task,
                    workspace,
                    root / "case",
                    timeout_s=120,
                    ags_image="registry.invalid/sample:1.0",
                    ags_env_file=None,
                    ags_timeout="3h",
                    ags_cpu="2",
                    ags_memory="4Gi",
                    ags_score_tool_id="sdt-claimed-no-egress",
                )

            log = (root / "case" / "hidden-tests.log").read_text(encoding="utf-8")

        self.assertEqual(result["pytest"]["returncode"], 1)
        self.assertIn("outbound network access", log)
        self.assertEqual(calls[-1], "closed")

    def test_ags_scorer_preserves_result_when_sandbox_cleanup_times_out(self) -> None:
        class SlowCleanupAGSBackend:
            workspace_root = "/workspace"
            sandbox_id = "ags-slow-cleanup"

            def __init__(self, settings: object) -> None:
                pass

            def start(self):
                return self

            def upload_tree(self, local: Path, remote: str) -> None:
                pass

            def exec(self, command: str, *, cwd: str, timeout_s: int) -> CommandOutcome:
                return CommandOutcome(0, "3 passed in 0.1s\n", "")

            def close(self) -> None:
                raise TimeoutError("AGS sandbox cleanup timed out after 600s")

        task = {
            "id": "sample-task",
            "image": "ghcr.invalid/sample:1.0",
            "expected_tests": 3,
            "hidden_paths": ["tests"],
            "test_commands": ["pytest tests"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
            with (
                patch(
                    "src.execution.ags.AGSSettings.from_env",
                    return_value=type("S", (), {"runtime_timeout": 60.0})(),
                ),
                patch("src.execution.ags.AGSWorkspaceBackend", SlowCleanupAGSBackend),
            ):
                result = benchmark.run_hidden_tests_ags(
                    task,
                    workspace,
                    root / "case",
                    timeout_s=1200,
                    ags_image="registry.invalid/sample:1.0",
                    ags_env_file=None,
                    ags_timeout="3h",
                    ags_cpu="2",
                    ags_memory="4Gi",
                    ags_score_tool_id="sdt-no-egress",
                )

        self.assertTrue(result["pytest"]["all_passed"])
        self.assertIn("cleanup timed out", result["cleanup_error"])


if __name__ == "__main__":
    unittest.main()
