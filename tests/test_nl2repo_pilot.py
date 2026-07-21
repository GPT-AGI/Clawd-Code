from __future__ import annotations

import copy
import importlib.util
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from src.execution.backend import CommandOutcome
from src.teammate.models import TeamTask
from src.tool_system.agent_loop import AgentLoopResult, ToolEvent
from src.tool_system.context import ToolContext
from src.tool_system.tools import TeamCreateTool, TeamPlanTool


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

    def test_legacy_pool_cli_is_disabled_before_upstream_resolution(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH)],
            cwd=MODULE_PATH.parents[2],
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("direct benchmark.py rollout/reward pools are disabled", completed.stderr)
        self.assertNotIn("NL2Repo checkout", completed.stderr)

        child = subprocess.run(
            [sys.executable, str(MODULE_PATH), "_run-one"],
            cwd=MODULE_PATH.parents[2],
            capture_output=True,
            text=True,
            env={
                key: value
                for key, value in os.environ.items()
                if key != benchmark.GLOBAL_POOL_WORKER_ENV
            },
        )
        self.assertNotEqual(child.returncode, 0)
        self.assertIn("direct 'benchmark.py _run-one' is disabled", child.stderr)

    def test_private_child_requires_supervisor_marker_and_live_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "global-pool.lock"
            lock_path.write_text(json.dumps({"pid": 31337}), encoding="utf-8")
            with lock_path.open("r+", encoding="utf-8") as owner:
                fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(SystemExit, "must be launched"):
                    benchmark.enforce_child_launch_policy(
                        ["_run-one"], environ={}, lock_path=lock_path
                    )
                with patch.object(
                    benchmark, "_process_parent_pid", return_value=31337
                ):
                    benchmark.enforce_child_launch_policy(
                        ["_run-one"],
                        environ={
                            benchmark.GLOBAL_POOL_WORKER_ENV:
                                benchmark.GLOBAL_POOL_WORKER_MARKER
                        },
                        lock_path=lock_path,
                        parent_pid=4242,
                    )
                with patch.object(
                    benchmark, "_process_parent_pid", return_value=99999
                ):
                    with self.assertRaisesRegex(SystemExit, "must be launched"):
                        benchmark.enforce_child_launch_policy(
                            ["_run-one"],
                            environ={
                                benchmark.GLOBAL_POOL_WORKER_ENV:
                                    benchmark.GLOBAL_POOL_WORKER_MARKER
                            },
                            lock_path=lock_path,
                            parent_pid=4242,
                        )
                fcntl.flock(owner.fileno(), fcntl.LOCK_UN)

            with self.assertRaisesRegex(SystemExit, "must be launched"):
                benchmark.enforce_child_launch_policy(
                    ["_run-one"],
                    environ={
                        benchmark.GLOBAL_POOL_WORKER_ENV:
                            benchmark.GLOBAL_POOL_WORKER_MARKER
                    },
                    lock_path=lock_path,
                )

    def test_agent_child_watchdog_signals_when_queue_parent_disappears(self) -> None:
        stop_event = threading.Event()
        orphaned = threading.Event()
        thread = benchmark.start_parent_watchdog(
            4242,
            stop_event,
            interval_s=0.001,
            on_orphan=orphaned.set,
            parent_pid_loader=lambda: 9999,
        )
        self.assertTrue(orphaned.wait(1))
        stop_event.set()
        thread.join(timeout=1)

    def test_private_child_parent_must_be_a_registered_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "global-pool.lock"
            lock_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "pid": 100,
                        "runs": ["/tmp/run"],
                        "worker_pids": [4242],
                    }
                ),
                encoding="utf-8",
            )
            with lock_path.open("r+", encoding="utf-8") as owner:
                fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                environment = {
                    benchmark.GLOBAL_POOL_WORKER_ENV:
                        benchmark.GLOBAL_POOL_WORKER_MARKER
                }
                benchmark.enforce_child_launch_policy(
                    ["_run-one"],
                    environ=environment,
                    lock_path=lock_path,
                    parent_pid=4242,
                )
                with self.assertRaisesRegex(SystemExit, "must be launched"):
                    benchmark.enforce_child_launch_policy(
                        ["_run-one"],
                        environ=environment,
                        lock_path=lock_path,
                        parent_pid=9999,
                    )

    def test_metadata_only_cli_actions_do_not_use_legacy_pool(self) -> None:
        for action in ("list", "plan", "validate"):
            args = type(
                "Args",
                (),
                {
                    "list": action == "list",
                    "plan": action == "plan",
                    "validate": action == "validate",
                    "rescore": False,
                },
            )()
            benchmark.enforce_top_level_pool_policy(args)

        rescore = type(
            "Args",
            (),
            {"list": False, "plan": False, "validate": False, "rescore": True},
        )()
        with self.assertRaisesRegex(SystemExit, "global_pool_supervisor.py"):
            benchmark.enforce_top_level_pool_policy(rescore)

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

        self.assertIn("harness has already created the active strict", prompt)
        self.assertIn("protocol-v2 Team", prompt)
        self.assertIn("Do not call TeamCreate", prompt)
        self.assertIn("one atomic TeamPlan", prompt)
        self.assertIn("TeamRun", prompt)
        self.assertIn("exactly two real implementation workers", prompt)
        self.assertIn("acceptance_checks", prompt)
        self.assertIn("verification automatically", prompt)
        self.assertIn("TeamAbort", prompt)

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
        forced = benchmark.build_prompt(
            "forced-team",
            teammate_max_turns=40,
            max_output_tokens=4_096,
            team_timeout_s=1_800,
        )

        self.assertIn("valid to remain solo", adaptive)
        self.assertIn("at least two substantially independent", adaptive_v2)
        self.assertIn("exactly two real", adaptive_v2)
        self.assertIn("TeamPlan -> TeamRun", adaptive_v2)
        self.assertIn("valid to complete", adaptive_v2)
        self.assertIn("exactly two real implementation workers", forced)
        self.assertIn("timeout_s=1800", forced)
        self.assertIn("token_budget=1310720", forced)
        self.assertIn("turn_budget=80", forced)
        self.assertIn("rollout-wide caps", forced)
        self.assertIn("must be your runtime decision", adaptive)
        self.assertNotIn("planner", adaptive.lower())

    def test_adaptive_team_v2_allows_a_valid_solo_route(self) -> None:
        self.assertTrue(
            benchmark._protocol_ok(
                "adaptive-team-v2",
                {"present": False},
            )
        )

    def test_adaptive_team_v2_rejects_a_legacy_incremental_team(self) -> None:
        self.assertFalse(
            benchmark._protocol_ok(
                "adaptive-team-v2",
                {
                    "present": True,
                    "protocol_version": 1,
                    "status": "completed",
                    "agents": [{}, {}],
                    "tasks": 2,
                    "completed_tasks": 2,
                    "quality_gates": {
                        "strict": True,
                        "configured": True,
                        "plan_accepted": True,
                        "validation_status": "passed",
                    },
                },
            )
        )

    def test_protocol_v2_requires_every_task_to_be_harness_accepted(self) -> None:
        team = {
            "present": True,
            "protocol_version": 2,
            "status": "completed",
            "lifecycle_state": "completed",
            "agents": [{}, {}],
            "tasks": 3,
            "completed_tasks": 3,
            "accepted_tasks": 2,
            "attempted_tasks": 3,
            "produced_tasks": 3,
            "plan_revision": 1,
            "plan_hash": "sha256",
            "plan_hash_valid": True,
            "manifest_valid": True,
            "quality_gates": {
                "strict": True,
                "configured": True,
                "plan_accepted": True,
                "validation_status": "passed",
            },
        }

        self.assertFalse(benchmark._protocol_ok("forced-team", team))
        team["accepted_tasks"] = 3
        self.assertTrue(benchmark._protocol_ok("forced-team", team))

    def test_team_metrics_counts_only_accepted_completed_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            team_id = "team-v2"
            team = {
                "team_id": team_id,
                "lead_agent_id": "lead",
                "status": "completed",
                "protocol_version": 2,
                "lifecycle_state": "completed",
                "settings": {
                    "quality_gates": {
                        "strict": True,
                        "validation": {"status": "passed"},
                    }
                },
            }
            team_dir = workspace / ".clawd" / "teams" / team_id
            (team_dir / "agents").mkdir(parents=True)
            (team_dir / "messages").mkdir()
            (workspace / ".clawd").mkdir(exist_ok=True)
            (workspace / ".clawd" / "team.json").write_text(
                json.dumps(team), encoding="utf-8"
            )
            (team_dir / "team.json").write_text(json.dumps(team), encoding="utf-8")
            (team_dir / "tasks.json").write_text(
                json.dumps(
                    {
                        "accepted": {
                            "status": "completed",
                            "lifecycle_state": "accepted",
                        },
                        "produced": {
                            "status": "completed",
                            "lifecycle_state": "produced",
                        },
                    }
                ),
                encoding="utf-8",
            )

            metrics = benchmark._team_metrics(workspace)

        self.assertEqual(metrics["tasks"], 2)
        self.assertEqual(metrics["completed_tasks"], 2)
        self.assertEqual(metrics["accepted_tasks"], 1)

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
            stats = metadata["score_context_stats"]
            self.assertEqual(stats["source"]["file_count"], 3)
            self.assertEqual(stats["copied"]["file_count"], 3)
            self.assertEqual(stats["staged"]["file_count"], 1)
            self.assertEqual(stats["staged"]["directory_count"], 1)
            self.assertGreater(stats["limits"]["max_total_bytes"], 0)

    def test_stage_score_context_rejects_symlink_before_touching_destination(self) -> None:
        task = {
            "image": "example.invalid/sample:1.0",
            "hidden_paths": [],
            "test_commands": ["pytest"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside-secret.txt"
            outside.write_text("secret\n", encoding="utf-8")
            (workspace / "leak.txt").symlink_to(outside)
            destination = root / "score"
            destination.mkdir()
            sentinel = destination / "keep.txt"
            sentinel.write_text("untouched\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "symbolic link"):
                benchmark.stage_score_context(task, workspace, destination)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "untouched\n")

    def test_stage_score_context_ignores_cache_directories(self) -> None:
        task = {
            "image": "example.invalid/sample:1.0",
            "hidden_paths": [],
            "test_commands": ["pytest"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "package.py").write_text("VALUE = 1\n", encoding="utf-8")
            outside = root / "outside"
            outside.mkdir()
            (workspace / ".git").symlink_to(outside, target_is_directory=True)

            metadata = benchmark.stage_score_context(task, workspace, root / "score")

            self.assertEqual(metadata["score_context_stats"]["source"]["file_count"], 1)
            self.assertFalse((root / "score" / "workspace" / ".git").exists())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO unsupported")
    def test_stage_score_context_rejects_non_regular_file(self) -> None:
        task = {
            "image": "example.invalid/sample:1.0",
            "hidden_paths": [],
            "test_commands": ["pytest"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            os.mkfifo(workspace / "unsafe.pipe")

            with self.assertRaisesRegex(ValueError, "non-regular file"):
                benchmark.stage_score_context(task, workspace, root / "score")

    def test_stage_score_context_enforces_size_limits(self) -> None:
        task = {
            "image": "example.invalid/sample:1.0",
            "hidden_paths": [],
            "test_commands": ["pytest"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "one.txt").write_bytes(b"abc")
            (workspace / "two.txt").write_bytes(b"def")

            with patch.object(benchmark, "SCORE_CONTEXT_MAX_FILES", 1):
                with self.assertRaisesRegex(ValueError, "file-count limit"):
                    benchmark.stage_score_context(task, workspace, root / "score-files")
            with patch.object(benchmark, "SCORE_CONTEXT_MAX_FILE_BYTES", 2):
                with self.assertRaisesRegex(ValueError, "file exceeds size limit"):
                    benchmark.stage_score_context(task, workspace, root / "score-file-size")
            with patch.object(benchmark, "SCORE_CONTEXT_MAX_TOTAL_BYTES", 5):
                with self.assertRaisesRegex(ValueError, "total-size limit"):
                    benchmark.stage_score_context(task, workspace, root / "score-total")

    def test_score_commands_continue_to_pytest_when_setup_fails(self) -> None:
        command = benchmark._score_shell_command(
            ["pip install -e ."],
            ["pytest tests"],
        )

        self.assertEqual(command, "(pip install -e .); (pytest tests)")
        self.assertNotIn("&&", command)

    def test_protocol_failure_still_scores_a_complete_workspace(self) -> None:
        hidden = {
            "pytest": {
                "expected": 3,
                "passed": 2,
                "failed": 1,
                "errors": 0,
                "skipped": 0,
                "returncode": 1,
                "quality_score": 66.67,
                "all_passed": False,
            }
        }
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
            with patch.object(benchmark, "run_hidden_tests", return_value=hidden) as scorer:
                result = benchmark.score_rollout(
                    rollout,
                    provider="qwen",
                    model="test",
                    score_timeout_s=60,
                    keep_image=False,
                )

        scorer.assert_called_once()
        self.assertFalse(result["reward_skipped"])
        self.assertEqual(result["failure_class"], "team_protocol")
        self.assertFalse(result["protocol_ok"])
        self.assertEqual(result["result_schema_version"], 2)
        self.assertEqual(result["code_quality_score"], 66.67)
        self.assertEqual(result["quality_score"], 66.67)
        self.assertEqual(result["protocol_status"], "failed")
        self.assertEqual(result["protocol_credit"], 0.0)
        self.assertTrue(result["delivery_valid"])
        self.assertEqual(result["effective_quality_score"], 0.0)
        self.assertEqual(result["reward_outcome"], "scored")
        self.assertTrue(result["reward_score_valid"])
        self.assertEqual(result["failure_domain"], "protocol")
        self.assertFalse(result["is_infrastructure"])
        self.assertFalse(result["retryable"])
        self.assertIsNone(result["timeout_scope"])

    def test_protocol_v2_requires_exact_manifest_and_real_task_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            context = ToolContext(workspace_root=workspace)
            TeamCreateTool().run(
                {"team_name": "metric-v2", "quality_gates": True}, context
            )
            planned = TeamPlanTool().run(
                {
                    "contract": {"summary": "two independent modules", "interfaces": []},
                    "workers": [
                        {"name": "worker-a", "instructions": "Implement module A."},
                        {"name": "worker-b", "instructions": "Implement module B."},
                    ],
                    "tasks": [
                        {
                            "key": "module-a",
                            "owner": "worker-a",
                            "instructions": "Implement a.py.",
                            "owned_files": ["a.py"],
                            "acceptance_checks": ["python -m py_compile a.py"],
                        },
                        {
                            "key": "module-b",
                            "owner": "worker-b",
                            "instructions": "Implement b.py.",
                            "owned_files": ["b.py"],
                            "acceptance_checks": ["python -m py_compile b.py"],
                        },
                    ],
                    "validation": {
                        "profile": "generic",
                        "integration_command": "python -m compileall -q .",
                    },
                    "execution": {"timeout_s": 300},
                },
                context,
            )
            self.assertFalse(planned.is_error, planned.output)
            team = context.team_store.load_active_team()
            assert team is not None
            for raw in context.team_store.load_tasks(team.team_id).values():
                task = TeamTask.from_dict(raw)
                task.transition_to("in_progress")
                task.attempt = 1
                task.transition_to("completed")
                task.set_lifecycle_state("accepted")
                context.team_store.update_task(team.team_id, task)
                context.team_store.append_event(
                    team.team_id, "task.produced", {"task_id": task.id}
                )
            quality = dict(team.settings["quality_gates"])
            quality["plan_accepted"] = True
            quality["validation"] = {"status": "passed"}
            team.settings["quality_gates"] = quality
            # Protocol v2 keeps frozen and effective execution values in one
            # immutable manifest.  A runtime-enforced minimum is valid only when
            # the same manifest records its exact requested/effective adjustment.
            manifest = dict(team.settings["execution_manifest"])
            manifest["status"] = "accepted"
            manifest["effective_execution"] = dict(manifest["execution"])
            manifest["effective_execution"]["timeout_s"] = 900.0
            manifest["runtime_adjustments"] = {
                "timeout_s": {
                    "requested": 300,
                    "effective": 900.0,
                    "reason": "runtime minimum",
                }
            }
            team.settings["execution_manifest"] = manifest
            team.transition_to("running")
            team.transition_to("completed")
            context.team_store.save_team(team)
            context.team_store.append_event(
                team.team_id,
                "team.options_adjusted",
                {
                    "timeout_s": {
                        "requested": 300,
                        "effective": 900.0,
                        "reason": "runtime minimum",
                    }
                },
            )

            metrics = benchmark._team_metrics(workspace)

            self.assertTrue(metrics["plan_hash_valid"])
            self.assertTrue(metrics["execution_manifest_valid"])
            self.assertTrue(metrics["manifest_valid"])
            self.assertEqual(metrics["execution_manifest_mismatches"], [])
            self.assertEqual(metrics["manifest_errors"], [])
            self.assertEqual(metrics["attempted_tasks"], metrics["tasks"])
            self.assertEqual(metrics["produced_tasks"], metrics["tasks"])
            self.assertTrue(benchmark._protocol_ok("forced-team", metrics))

            missing_evidence = dict(metrics)
            missing_evidence["produced_tasks"] -= 1
            self.assertFalse(benchmark._protocol_ok("forced-team", missing_evidence))

            manifest = dict(team.settings["execution_manifest"])
            manifest["effective_execution"] = dict(
                manifest["effective_execution"]
            )
            manifest["effective_execution"]["timeout_s"] = 901
            team.settings["execution_manifest"] = manifest
            context.team_store.save_team(team)
            undocumented_execution = benchmark._team_metrics(workspace)
            self.assertFalse(undocumented_execution["execution_manifest_valid"])
            self.assertFalse(undocumented_execution["manifest_valid"])
            self.assertIn(
                "execution.timeout_s:effective_value_mismatch",
                undocumented_execution["manifest_errors"],
            )
            self.assertEqual(
                undocumented_execution["execution_manifest_mismatches"][0]["field"],
                "timeout_s",
            )
            manifest["effective_execution"]["timeout_s"] = 900.0
            team.settings["execution_manifest"] = manifest
            context.team_store.save_team(team)

            tampered_budget_manifest = copy.deepcopy(manifest)
            tampered_budget_manifest["budget_window"]["hard_ceiling"]["turns"] = 1
            team.settings["execution_manifest"] = tampered_budget_manifest
            context.team_store.save_team(team)
            tampered_budget = benchmark._team_metrics(workspace)
            self.assertFalse(tampered_budget["execution_manifest_valid"])
            self.assertIn(
                "execution.budget_window.hard_ceiling.turns:derived_value_mismatch",
                tampered_budget["manifest_errors"],
            )
            team.settings["execution_manifest"] = manifest
            context.team_store.save_team(team)

            tasks = context.team_store.load_tasks(team.team_id)
            first = next(iter(tasks.values()))
            first["acceptance_checks"] = ["python -m compileall -q ."]
            context.team_store.save_tasks(team.team_id, tasks)
            tampered = benchmark._team_metrics(workspace)
            self.assertFalse(tampered["manifest_valid"])
            self.assertTrue(
                any(
                    reason.startswith("task_spec_mismatch:")
                    for reason in tampered["manifest_errors"]
                )
            )
            self.assertFalse(benchmark._protocol_ok("forced-team", tampered))

    def test_invalid_delivery_skips_reward_and_is_not_code_quality_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_root = Path(tmp) / "sample" / "adaptive"
            workspace = case_root / "workspace"
            workspace.mkdir(parents=True)
            start = workspace / "start.md"
            start.write_text("mutated spec\n", encoding="utf-8")
            rollout = benchmark.RolloutArtifact(
                task={"id": "sample", "difficulty": "Easy", "expected_tests": 1},
                mode="adaptive",
                case_root=case_root,
                workspace=workspace,
                start_hash="different",
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
        self.assertEqual(result["reward_outcome"], "missing_artifact")
        self.assertFalse(result["reward_score_valid"])
        self.assertIsNone(result["code_quality_score"])
        self.assertFalse(result["metric_eligibility"]["code_quality"])

    def test_agent_failure_keeps_valid_code_score_but_zeroes_effective_quality(self) -> None:
        hidden = {
            "pytest": {
                "quality_score": 75.0,
                "returncode": 1,
                "all_passed": False,
            }
        }

        metrics = benchmark._result_metrics_v2(
            agent_ok=False,
            agent_timed_out=False,
            integrity_ok=True,
            protocol_ok=True,
            hidden=hidden,
            failure_class="rollout_failure",
        )

        self.assertFalse(metrics["delivery_valid"])
        self.assertTrue(metrics["reward_score_valid"])
        self.assertEqual(metrics["code_quality_score"], 75.0)
        self.assertEqual(metrics["effective_quality_score"], 0.0)
        self.assertEqual(metrics["failure_domain"], "candidate")

    def test_rollout_infrastructure_is_excluded_from_all_quality_metrics(self) -> None:
        metrics = benchmark._result_metrics_v2(
            agent_ok=False,
            agent_timed_out=False,
            integrity_ok=True,
            protocol_ok=False,
            hidden={
                "skipped": True,
                "pytest": {"quality_score": 0.0, "returncode": 1},
            },
            failure_class="rollout_infrastructure",
            rollout_infrastructure=True,
            rollout_retryable=True,
            rollout_outcome="infra_error",
        )

        self.assertEqual(metrics["rollout_outcome"], "infra_error")
        self.assertEqual(metrics["protocol_status"], "not_evaluated")
        self.assertIsNone(metrics["protocol_credit"])
        self.assertIsNone(metrics["code_quality_score"])
        self.assertIsNone(metrics["effective_quality_score"])
        self.assertFalse(any(metrics["metric_eligibility"].values()))
        self.assertTrue(metrics["is_infrastructure"])
        self.assertTrue(metrics["retryable"])

    def test_rollout_infrastructure_skips_reward_for_stale_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_root = Path(tmp) / "sample" / "adaptive"
            workspace = case_root / "workspace"
            workspace.mkdir(parents=True)
            start = workspace / "start.md"
            start.write_text("spec\n", encoding="utf-8")
            rollout = benchmark.RolloutArtifact(
                task={"id": "sample", "difficulty": "Easy", "expected_tests": 1},
                mode="adaptive",
                case_root=case_root,
                workspace=workspace,
                start_hash=benchmark._hash_file(start),
                agent={
                    "ok": False,
                    "error": "AGS upload failed",
                    "rollout_outcome": "infra_error",
                    "rollout_infrastructure": True,
                    "rollout_retryable": True,
                    "lead_usage": {},
                },
                agent_elapsed_s=1.0,
                agent_timed_out=False,
                agent_returncode=1,
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
        self.assertEqual(result["failure_domain"], "infrastructure")
        self.assertEqual(result["reward_outcome"], "pending")
        self.assertFalse(result["metric_eligibility"]["protocol_yield"])

    def test_hidden_test_timeout_is_candidate_not_infrastructure(self) -> None:
        metrics = benchmark._result_metrics_v2(
            agent_ok=True,
            agent_timed_out=False,
            integrity_ok=True,
            protocol_ok=True,
            hidden={
                "timed_out": True,
                "pytest": {"quality_score": 0.0, "returncode": 124, "all_passed": False},
            },
            failure_class="reward_timeout",
        )

        self.assertEqual(metrics["reward_outcome"], "candidate_timeout")
        self.assertEqual(metrics["timeout_scope"], "reward")
        self.assertFalse(metrics["is_infrastructure"])
        self.assertFalse(metrics["retryable"])
        self.assertEqual(metrics["failure_domain"], "candidate")

    def test_reward_infrastructure_timeout_is_retryable_and_not_scored(self) -> None:
        metrics = benchmark._result_metrics_v2(
            agent_ok=True,
            agent_timed_out=False,
            integrity_ok=True,
            protocol_ok=True,
            hidden={
                "error": "sandbox provisioning timed out",
                "infrastructure_timed_out": True,
                "pytest": {"quality_score": 0.0, "returncode": 124, "all_passed": False},
            },
            failure_class="scorer_infrastructure",
        )

        self.assertEqual(metrics["reward_outcome"], "infra_timeout")
        self.assertFalse(metrics["reward_score_valid"])
        self.assertIsNone(metrics["code_quality_score"])
        self.assertIsNone(metrics["effective_quality_score"])
        self.assertFalse(any(metrics["metric_eligibility"].values()))
        self.assertTrue(metrics["is_infrastructure"])
        self.assertTrue(metrics["retryable"])
        self.assertEqual(metrics["timeout_scope"], "reward")

    def test_reward_infrastructure_excludes_failed_protocol_from_qpe(self) -> None:
        metrics = benchmark._result_metrics_v2(
            agent_ok=True,
            agent_timed_out=False,
            integrity_ok=True,
            protocol_ok=False,
            hidden={
                "error": "scorer sandbox unavailable",
                "pytest": {
                    "quality_score": 0.0,
                    "returncode": 1,
                    "all_passed": False,
                },
            },
            failure_class="scorer_infrastructure",
        )

        # The final Team state remains useful diagnosis, but this scorer attempt
        # is not an observation of any Q/P/E metric.
        self.assertEqual(metrics["protocol_status"], "failed")
        self.assertEqual(metrics["protocol_credit"], 0.0)
        self.assertEqual(metrics["reward_outcome"], "infra_error")
        self.assertIsNone(metrics["code_quality_score"])
        self.assertIsNone(metrics["effective_quality_score"])
        self.assertFalse(any(metrics["metric_eligibility"].values()))
        self.assertEqual(metrics["failure_domain"], "infrastructure")
        self.assertTrue(metrics["retryable"])

    def test_scheduler_failure_domain_depends_on_phase(self) -> None:
        task = {"id": "sample", "difficulty": "Easy", "expected_tests": 1}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout = benchmark._failed_case_result(
                task,
                "adaptive",
                "rollout",
                RuntimeError("agent child failed"),
                output_root=root,
                provider="qwen",
                model="test",
                execution_backend="ags",
                score_backend="ags",
            )
            reward = benchmark._failed_case_result(
                task,
                "forced-team",
                "reward",
                TimeoutError("sandbox unavailable"),
                output_root=root,
                provider="qwen",
                model="test",
                execution_backend="ags",
                score_backend="ags",
            )

        self.assertEqual(rollout["failure_domain"], "candidate")
        self.assertFalse(rollout["is_infrastructure"])
        self.assertEqual(rollout["reward_outcome"], "missing_artifact")
        self.assertNotIn("error", rollout["hidden_tests"])
        self.assertTrue(rollout["hidden_tests"]["skipped"])
        self.assertEqual(reward["failure_domain"], "infrastructure")
        self.assertTrue(reward["is_infrastructure"])
        self.assertEqual(reward["reward_outcome"], "infra_timeout")
        self.assertEqual(reward["timeout_scope"], "reward")
        self.assertIn("error", reward["hidden_tests"])

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
                        # Protocol accounting is deliberately stale; a rescore must
                        # derive it again from the final persisted team manifest.
                        "protocol_ok": False,
                        "protocol_status": "failed",
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
        self.assertTrue(result["protocol_ok"])
        self.assertEqual(result["protocol_status"], "passed")
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
        self.assertTrue(result["ok"])
        self.assertFalse(result["failed"])

    def test_agent_child_persists_explicit_lifecycle_failure(self) -> None:
        def fake_run_prompt(prompt: str, **kwargs: object) -> AgentLoopResult:
            return AgentLoopResult(
                response_text="Team aborted after unrecoverable validation.",
                usage={"input_tokens": 3, "output_tokens": 2},
                num_turns=2,
                failed=True,
                failure_reason="team_aborted",
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
                    False,
                    progress_path,
                )
            result = json.loads(result_path.read_text())

        self.assertEqual(returncode, 0)
        self.assertFalse(result["ok"])
        self.assertTrue(result["failed"])
        self.assertEqual(result["failure_reason"], "team_aborted")

    def test_agent_child_classifies_terminal_team_budget_exhaustion(self) -> None:
        def fake_run_prompt(prompt: str, **kwargs: object) -> AgentLoopResult:
            return AgentLoopResult(
                response_text="Team execution budget exhausted.",
                usage={"input_tokens": 30, "output_tokens": 2},
                num_turns=3,
                failed=True,
                failure_reason="team_budget_exhausted",
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
                    False,
                    progress_path,
                )
            result = json.loads(result_path.read_text())

        self.assertEqual(returncode, 0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["rollout_outcome"], "budget_exhausted")
        self.assertEqual(result["failure_reason"], "team_budget_exhausted")

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
        self.assertEqual(observed["team"].protocol_version, 2)
        self.assertEqual(observed["team"].lifecycle_state, "draft")
        self.assertEqual(observed["team"].settings["protocol_version"], 2)
        self.assertTrue(observed["team"].settings["quality_gates"]["strict"])
        self.assertEqual(
            observed["team"].settings["quality_gates"]["protocol_version"], 2
        )
        self.assertEqual(progress[0]["kind"], "forced_team_precreated")
        self.assertEqual(progress[0]["protocol_version"], 2)

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
