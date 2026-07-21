from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "teammate-evals"
    / "nl2repo-pilot"
    / "evaluation_queue.py"
)
SPEC = importlib.util.spec_from_file_location("nl2repo_evaluation_queue", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
queue = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = queue
SPEC.loader.exec_module(queue)

SUPERVISOR_PATH = MODULE_PATH.with_name("global_pool_supervisor.py")
SUPERVISOR_SPEC = importlib.util.spec_from_file_location(
    "nl2repo_global_pool_supervisor", SUPERVISOR_PATH
)
assert SUPERVISOR_SPEC is not None and SUPERVISOR_SPEC.loader is not None
supervisor = importlib.util.module_from_spec(SUPERVISOR_SPEC)
sys.modules[SUPERVISOR_SPEC.name] = supervisor
with patch.dict(sys.modules, {"evaluation_queue": queue}):
    SUPERVISOR_SPEC.loader.exec_module(supervisor)


class TestEvaluationQueue(unittest.TestCase):
    @staticmethod
    def _rollout_artifact(
        root: Path,
        task: dict[str, object],
        mode: str,
        agent: dict[str, object],
    ) -> SimpleNamespace:
        case_root = root / str(task["id"]) / mode
        workspace = case_root / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "start.md").write_text("spec", encoding="utf-8")
        (case_root / "agent-result.json").write_text(
            json.dumps(agent), encoding="utf-8"
        )
        return SimpleNamespace(
            task=task,
            mode=mode,
            case_root=case_root,
            workspace=workspace,
            start_hash="hash",
            agent=agent,
            agent_elapsed_s=0.01,
            agent_timed_out=False,
            agent_returncode=0,
        )

    def test_reward_retries_only_infrastructure_failures(self) -> None:
        calls = 0
        sleeps: list[float] = []
        retries: list[tuple[int, str]] = []

        def score() -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"hidden_tests": {"error": "Docker build failed"}}
            return {"quality_score": 75.0, "hidden_tests": {"pytest": {}}}

        result = queue.score_with_infrastructure_retries(
            score,
            attempts=3,
            delay_s=0.25,
            sleep_fn=sleeps.append,
            on_retry=lambda attempt, error: retries.append((attempt, error)),
        )

        self.assertEqual(result["quality_score"], 75.0)
        self.assertEqual(calls, 2)
        self.assertEqual(sleeps, [0.25])
        self.assertEqual(retries, [(1, "Docker build failed")])

    def test_reward_retries_transient_exceptions(self) -> None:
        calls = 0

        def score() -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise TimeoutError("temporary scorer timeout")
            return {"hidden_tests": {}}

        result = queue.score_with_infrastructure_retries(
            score, attempts=3, delay_s=0, sleep_fn=lambda _: None
        )

        self.assertEqual(result, {"hidden_tests": {}})
        self.assertEqual(calls, 3)

    def test_remaining_qwen32_selects_the_other_tasks(self) -> None:
        tasks = [{"id": f"task-{index:03d}"} for index in range(104)]
        args = SimpleNamespace(task=None, task_set="remaining-qwen32")
        with (
            patch.object(queue.benchmark, "list_tasks", return_value=tasks),
            patch.object(
                queue.benchmark,
                "select_task_subset",
                return_value=[task["id"] for task in tasks[:32]],
            ),
            patch.object(queue.benchmark, "load_task", return_value={}),
        ):
            selected = queue.resolve_task_names(args, Path("/tmp/upstream"))

        self.assertEqual(len(selected), 72)
        self.assertEqual(selected[0], "task-032")
        self.assertEqual(selected[-1], "task-103")

    def test_enqueue_is_persistent_deduplicated_and_priority_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = queue.QueueStore(root)
            added, skipped = first.enqueue(["one", "two"], "adaptive")
            more, duplicate = first.enqueue(["urgent", "one"], "adaptive", priority=10)
            reopened = queue.QueueStore(root)
            cases = reopened.cases()
            database_exists = (root / queue.QUEUE_DB).is_file()

        self.assertEqual(added, ["one", "two"])
        self.assertEqual(skipped, [])
        self.assertEqual(more, ["urgent"])
        self.assertEqual(duplicate, ["one"])
        self.assertEqual([case["task"] for case in cases], ["urgent", "one", "two"])
        self.assertTrue(database_exists)

    def test_done_persists_the_valid_canonical_reward_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = queue.QueueStore(Path(tmp))
            store.enqueue(["scored"], "adaptive-team-v2")
            case = store.claim("queued", "rollout", 1)[0]
            store.mark_rollout_complete(int(case["id"]))
            case = store.claim("reward_pending", "rewarding", 1)[0]
            store.mark_done(
                int(case["id"]),
                {
                    "code_quality_score": 87.5,
                    "quality_score": None,
                    "reward_outcome": "scored",
                    "reward_score_valid": True,
                    "success": False,
                },
            )
            persisted = store.cases()[0]

        self.assertEqual(persisted["status"], "done")
        self.assertEqual(persisted["quality_score"], 87.5)

    def test_concurrency_configuration_is_persistent_and_capacity_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = queue.QueueStore(Path(tmp))
            initial = store.initialize_concurrency(
                8, 4, max_rollout=64, max_reward=16
            )
            scaled = store.set_concurrency(rollout=32)
            paused = store.set_concurrency(rollout=0, reward=0)
            reopened = queue.QueueStore(Path(tmp)).concurrency()
            with self.assertRaisesRegex(ValueError, "capacity 64"):
                store.set_concurrency(rollout=65)

        self.assertEqual(initial["rollout_concurrency"], 8)
        self.assertEqual(scaled["rollout_concurrency"], 32)
        self.assertEqual(scaled["reward_concurrency"], 4)
        self.assertEqual(paused["rollout_concurrency"], 0)
        self.assertEqual(reopened["reward_concurrency"], 0)

    def test_worker_can_start_paused_for_global_pool_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = queue.QueueStore(Path(tmp))
            configured = store.initialize_concurrency(
                0, 0, max_rollout=64, max_reward=64
            )

        self.assertEqual(configured["rollout_concurrency"], 0)
        self.assertEqual(configured["reward_concurrency"], 0)

    def test_direct_serve_is_rejected_before_creating_queue_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "direct-run"
            with (
                patch.dict(os.environ, {}, clear=True),
                self.assertRaisesRegex(
                    SystemExit,
                    r"direct .*serve.*disabled.*global_pool_supervisor\.py",
                ),
            ):
                queue.main(["--run", str(run_root), "serve"])

            self.assertFalse(run_root.exists())

    def test_direct_scale_is_rejected_before_creating_queue_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "direct-scale"
            with self.assertRaisesRegex(
                SystemExit, r"direct .*scale.*disabled.*global pool supervisor"
            ):
                queue.main(
                    [
                        "--run",
                        str(run_root),
                        "scale",
                        "--rollout-concurrency",
                        "64",
                    ]
                )

            self.assertFalse(run_root.exists())

    def test_scale_has_no_emergency_command_line_bypass(self) -> None:
        with self.assertRaises(SystemExit):
            queue.build_parser().parse_args(
                [
                    "--run",
                    "/tmp/run",
                    "scale",
                    "--rollout-concurrency",
                    "1",
                    "--emergency-allow-manual-scale",
                ]
            )

    def test_serve_policy_requires_live_parent_lock_and_registered_run(self) -> None:
        parser = queue.build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = (root / "run").resolve()
            lock_path = root / "global-pool.lock"
            lock_path.write_text(
                json.dumps({"pid": 4242, "runs": [str(run_root)]}),
                encoding="utf-8",
            )
            managed = parser.parse_args(["--run", str(run_root), "serve"])
            environment = {
                queue.GLOBAL_POOL_WORKER_ENV: queue.GLOBAL_POOL_WORKER_MARKER
            }
            with patch.object(queue, "_global_pool_lock_is_held", return_value=True):
                queue.enforce_serve_launch_policy(
                    managed,
                    environ=environment,
                    lock_path=lock_path,
                    parent_pid=4242,
                )
                with self.assertRaisesRegex(SystemExit, "matching live supervisor"):
                    queue.enforce_serve_launch_policy(
                        managed,
                        environ=environment,
                        lock_path=lock_path,
                        parent_pid=9999,
                    )

                lock_path.write_text(
                    json.dumps({"pid": 4242, "runs": [run_root.name]}),
                    encoding="utf-8",
                )
                queue.enforce_serve_launch_policy(
                    managed,
                    environ=environment,
                    lock_path=lock_path,
                    parent_pid=4242,
                )

    def test_supervisor_marker_alone_cannot_start_worker(self) -> None:
        args = queue.build_parser().parse_args(["--run", "/tmp/run", "serve"])
        with self.assertRaisesRegex(SystemExit, "matching live supervisor"):
            queue.enforce_serve_launch_policy(
                args,
                environ={
                    queue.GLOBAL_POOL_WORKER_ENV: queue.GLOBAL_POOL_WORKER_MARKER
                },
                lock_path=Path("/definitely/missing/global-pool.lock"),
                parent_pid=4242,
            )

    def test_worker_watchdog_terminates_after_persistent_lease_loss(self) -> None:
        stop_event = threading.Event()
        terminated = threading.Event()
        codes: list[int] = []

        def terminate(code: int) -> None:
            codes.append(code)
            terminated.set()

        with patch.object(
            queue, "_is_authorized_global_pool_worker", return_value=False
        ):
            thread = queue.start_supervisor_lease_watchdog(
                Path("/tmp/run"),
                stop_event,
                interval_s=0.001,
                exit_fn=terminate,
            )
            self.assertTrue(terminated.wait(1))
            stop_event.set()
            thread.join(timeout=1)

        self.assertEqual(codes, [75])

    def test_serve_has_no_emergency_command_line_bypass(self) -> None:
        with self.assertRaises(SystemExit):
            queue.build_parser().parse_args(
                [
                    "--run",
                    "/tmp/run",
                    "serve",
                    "--emergency-allow-standalone-serve",
                ]
            )

    def test_supervisor_injects_worker_marker_without_emergency_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = supervisor.build_parser().parse_args(
                [
                    "--run",
                    str(root / "run"),
                    "--ags-env-file",
                    str(root / "ags.env"),
                ]
            )
            command = supervisor.build_worker_command(args, root / "run")
            with patch.dict(os.environ, {"PRESERVED": "yes"}, clear=True):
                environment = supervisor.build_worker_environment()

        self.assertEqual(environment["PRESERVED"], "yes")
        self.assertEqual(
            environment[queue.GLOBAL_POOL_WORKER_ENV],
            queue.GLOBAL_POOL_WORKER_MARKER,
        )
        self.assertNotIn("--emergency-allow-standalone-serve", command)
        with self.assertRaises(SystemExit):
            supervisor.build_parser().parse_args(
                [
                    "--run",
                    str(root / "run"),
                    "--ags-env-file",
                    str(root / "ags.env"),
                    "--stop-when-empty",
                ]
            )

    def test_managed_worker_starts_in_an_isolated_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            process = SimpleNamespace(pid=1234, poll=lambda: None)
            with patch.object(
                supervisor.subprocess, "Popen", return_value=process
            ) as popen:
                worker = supervisor.ManagedWorker(Path(tmp), ["worker-command"])
                worker.start()
                assert worker.log_handle is not None
                worker.log_handle.close()
                worker.log_handle = None

        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_second_global_supervisor_is_rejected_before_starting_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pool_root = Path(tmp) / "pilot-wide-pool"
            run_root = Path(tmp) / "some-other-directory" / "r1"
            env_file = Path(tmp) / "ags.env"
            env_file.write_text("AGS_ENDPOINT=https://example.invalid\n", encoding="utf-8")
            lock = supervisor.GlobalPoolLock(
                pool_root / "global-pool.lock", [run_root.resolve()]
            )
            with patch.object(supervisor, "GLOBAL_POOL_ROOT", pool_root):
                with lock:
                    metadata = json.loads(
                        (pool_root / "global-pool.lock").read_text(encoding="utf-8")
                    )
                    self.assertEqual(metadata["runs"], [str(run_root.resolve())])
                    self.assertEqual(metadata["schema_version"], 2)
                    self.assertEqual(metadata["worker_pids"], [])
                    with self.assertRaisesRegex(
                        SystemExit, "another global pool supervisor already owns"
                    ):
                        supervisor.main(
                            [
                                "--run",
                                str(run_root),
                                "--ags-env-file",
                                str(env_file),
                            ]
                        )

            self.assertFalse((run_root / queue.QUEUE_DB).exists())

    def test_dead_worker_is_frozen_before_restart_and_allocation_is_restored(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = queue.QueueStore(Path(tmp))
            store.initialize_concurrency(8, 4, max_rollout=64, max_reward=64)
            observed_at_restart: list[tuple[int, int]] = []

            class DeadWorker:
                run_root = Path(tmp)

                @staticmethod
                def needs_restart() -> bool:
                    return True

                @staticmethod
                def restart() -> None:
                    configured = store.concurrency()
                    assert configured is not None
                    observed_at_restart.append(
                        (
                            int(configured["rollout_concurrency"]),
                            int(configured["reward_concurrency"]),
                        )
                    )

            restarted = supervisor.restart_worker_safely(DeadWorker(), store)
            corrected = supervisor.reconcile_worker_concurrency(
                [store], ((8, 4),)
            )
            restored = store.concurrency()

        self.assertTrue(restarted)
        self.assertEqual(observed_at_restart, [(0, 0)])
        self.assertEqual(corrected, [0])
        self.assertIsNotNone(restored)
        self.assertEqual(restored["rollout_concurrency"], 8)
        self.assertEqual(restored["reward_concurrency"], 4)

    def test_supervisor_reconciles_out_of_band_scale_even_when_allocation_is_same(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = queue.QueueStore(Path(tmp))
            store.initialize_concurrency(8, 4, max_rollout=64, max_reward=64)
            # Simulate a direct evaluation_queue.py scale while the calculated
            # global allocation tuple itself remains unchanged.
            store.set_concurrency(rollout=32, reward=32)

            corrected = supervisor.reconcile_worker_concurrency(
                [store], ((8, 4),)
            )
            restored = store.concurrency()

        self.assertEqual(corrected, [0])
        self.assertIsNotNone(restored)
        self.assertEqual(restored["rollout_concurrency"], 8)
        self.assertEqual(restored["reward_concurrency"], 4)

    def test_global_supervisor_allows_exactly_one_disabled_pool(self) -> None:
        supervisor.validate_pool_capacities(0, 64, 64)
        supervisor.validate_pool_capacities(32, 0, 64)

        with self.assertRaisesRegex(
            ValueError, "at least one global capacity must be positive"
        ):
            supervisor.validate_pool_capacities(0, 0, 64)
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            supervisor.validate_pool_capacities(-1, 32, 64)

    def test_one_sided_pool_completion_ignores_the_disabled_stage(self) -> None:
        reward_backlog = [
            {"queued": 0, "rollout": 0, "reward_pending": 2, "rewarding": 1}
        ]
        rollout_backlog = [
            {"queued": 2, "rollout": 1, "reward_pending": 0, "rewarding": 0}
        ]

        self.assertFalse(
            supervisor.enabled_pool_has_work(
                reward_backlog, rollout_capacity=32, reward_capacity=0
            )
        )
        self.assertTrue(
            supervisor.enabled_pool_has_work(
                reward_backlog, rollout_capacity=0, reward_capacity=64
            )
        )
        self.assertFalse(
            supervisor.enabled_pool_has_work(
                rollout_backlog, rollout_capacity=0, reward_capacity=64
            )
        )
        self.assertTrue(
            supervisor.enabled_pool_has_work(
                rollout_backlog, rollout_capacity=32, reward_capacity=0
            )
        )

    def test_global_supervisor_rejects_both_pools_disabled_before_queue_creation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            with self.assertRaisesRegex(
                SystemExit, "at least one global capacity must be positive"
            ):
                supervisor.main(
                    [
                        "--run",
                        str(run_root),
                        "--ags-env-file",
                        str(root / "ags.env"),
                        "--rollout-capacity",
                        "0",
                        "--reward-capacity",
                        "0",
                    ]
                )

            self.assertFalse(run_root.exists())

    def test_global_slots_pipeline_into_the_next_run(self) -> None:
        snapshots = [
            {"rollout": 17, "queued": 0},
            {"rollout": 0, "queued": 104},
        ]

        allocated = queue.allocate_global_slots(
            snapshots,
            32,
            active_key="rollout",
            pending_key="queued",
        )

        self.assertEqual(allocated, [17, 15])

    def test_global_slots_fair_share_multiple_pending_runs(self) -> None:
        snapshots = [
            {"rollout": 0, "queued": 104},
            {"rollout": 0, "queued": 104},
            {"rollout": 0, "queued": 104},
        ]

        allocated = queue.allocate_global_slots(
            snapshots,
            32,
            active_key="rollout",
            pending_key="queued",
        )

        self.assertEqual(allocated, [11, 11, 10])

    def test_global_slots_preserve_active_work_before_fair_sharing(self) -> None:
        snapshots = [
            {"rollout": 17, "queued": 100},
            {"rollout": 0, "queued": 100},
        ]

        allocated = queue.allocate_global_slots(
            snapshots,
            32,
            active_key="rollout",
            pending_key="queued",
        )

        self.assertEqual(allocated, [17, 15])

    def test_live_loop_expands_and_shrinks_without_preempting_active_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = queue.QueueStore(root)
            store.initialize_concurrency(2, 1, max_rollout=4, max_reward=2)
            store.enqueue([f"task-{index}" for index in range(6)], "adaptive")
            stop = threading.Event()
            first_release = threading.Event()
            second_release = threading.Event()
            lock = threading.Lock()
            started: list[str] = []
            active = 0
            max_active = 0
            resized: list[dict[str, int]] = []

            def load_task(name: str) -> dict[str, object]:
                return {"id": name}

            def rollout(task: dict[str, object], mode: str) -> SimpleNamespace:
                nonlocal active, max_active
                task_name = str(task["id"])
                with lock:
                    started.append(task_name)
                    active += 1
                    max_active = max(max_active, active)
                    start_number = len(started)
                release = first_release if start_number <= 4 else second_release
                self.assertTrue(release.wait(timeout=4))
                case_root = root / task_name / mode
                workspace = case_root / "workspace"
                workspace.mkdir(parents=True)
                (case_root / "agent-result.json").write_text(
                    json.dumps({"ok": True}), encoding="utf-8"
                )
                with lock:
                    active -= 1
                return SimpleNamespace(
                    task=task,
                    mode=mode,
                    case_root=case_root,
                    workspace=workspace,
                    start_hash="hash",
                    agent={"ok": True},
                    agent_elapsed_s=0.01,
                    agent_timed_out=False,
                    agent_returncode=0,
                )

            def reward(artifact: object) -> dict[str, object]:
                return {"quality_score": 100.0, "success": True}

            thread = threading.Thread(
                target=queue.run_queue_loop,
                args=(store, load_task, rollout, reward),
                kwargs={
                    "rollout_concurrency": 2,
                    "reward_concurrency": 1,
                    "max_rollout_concurrency": 4,
                    "max_reward_concurrency": 2,
                    "concurrency_loader": store.concurrency,
                    "stop_event": stop,
                    "poll_interval_s": 0.01,
                    "on_resize": resized.append,
                },
            )
            thread.start()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and len(started) < 2:
                time.sleep(0.01)
            store.set_concurrency(rollout=4)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and len(started) < 4:
                time.sleep(0.01)
            store.set_concurrency(rollout=1)
            first_release.set()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and len(started) < 5:
                time.sleep(0.01)
            time.sleep(0.08)
            started_while_fifth_blocked = len(started)
            second_release.set()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and store.counts()["done"] < 6:
                time.sleep(0.01)
            stop.set()
            thread.join(timeout=3)

        self.assertFalse(thread.is_alive())
        self.assertEqual(max_active, 4)
        self.assertEqual(started_while_fifth_blocked, 5)
        self.assertEqual(len(started), 6)
        self.assertEqual(
            [item["rollout_concurrency"] for item in resized], [4, 1]
        )

    def test_live_loop_can_restart_while_both_pools_are_paused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = queue.QueueStore(root)
            store.initialize_concurrency(1, 1, max_rollout=2, max_reward=2)
            store.set_concurrency(rollout=0, reward=0)
            stop = threading.Event()
            started = threading.Event()

            def rollout(task: dict[str, object], mode: str) -> object:
                started.set()
                raise AssertionError("paused queue must not start a rollout")

            thread = threading.Thread(
                target=queue.run_queue_loop,
                args=(store, lambda name: {"id": name}, rollout, lambda artifact: {}),
                kwargs={
                    "rollout_concurrency": 0,
                    "reward_concurrency": 0,
                    "max_rollout_concurrency": 2,
                    "max_reward_concurrency": 2,
                    "concurrency_loader": store.concurrency,
                    "stop_event": stop,
                    "poll_interval_s": 0.01,
                },
            )
            thread.start()
            time.sleep(0.05)
            stop.set()
            thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertFalse(started.is_set())

    def test_recover_returns_interrupted_work_to_the_correct_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = queue.QueueStore(Path(tmp))
            store.enqueue(["rollout", "reward"], "adaptive")
            rollout_case = store.claim("queued", "rollout", 1)[0]
            partial = Path(tmp) / "rollout" / "adaptive"
            partial.mkdir(parents=True)
            (partial / "progress.jsonl").write_text("partial\n", encoding="utf-8")
            reward_case = store.claim("queued", "rollout", 1)[0]
            store.mark_rollout_complete(reward_case["id"])
            store.claim("reward_pending", "rewarding", 1)

            recovered = store.recover_interrupted()
            statuses = {case["task"]: case["status"] for case in store.cases()}
            archives = list(
                (Path(tmp) / "_attempts" / "rollout" / "adaptive").glob(
                    "attempt-1-interrupted-*"
                )
            )
            archived_partial = (
                len(archives) == 1 and (archives[0] / "progress.jsonl").is_file()
            )

        self.assertEqual(recovered, {"rollout": 1, "reward": 1})
        self.assertEqual(statuses["rollout"], "queued")
        self.assertEqual(statuses["reward"], "reward_pending")
        self.assertGreaterEqual(rollout_case["id"], 1)
        self.assertEqual(len(archives), 1)
        self.assertTrue(archived_partial)

    def test_completed_inflight_rollout_can_be_salvaged_during_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = queue.QueueStore(root)
            store.enqueue(["finished", "still-running"], "adaptive")
            finished = store.claim("queued", "rollout", 1)[0]
            running = store.claim("queued", "rollout", 1)[0]
            case_root = root / "finished" / "adaptive"
            workspace = case_root / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "start.md").write_text("spec", encoding="utf-8")
            (case_root / "agent-result.json").write_text(
                json.dumps({"ok": True}), encoding="utf-8"
            )

            salvaged = store.salvage_completed_rollouts(
                exclude_case_ids={int(running["id"])}
            )
            statuses = {case["task"]: case["status"] for case in store.cases()}
            artifact = json.loads(
                (case_root / "rollout-artifact.json").read_text(encoding="utf-8")
            )

        self.assertEqual([case["id"] for case in salvaged], [finished["id"]])
        self.assertEqual(statuses["finished"], "reward_pending")
        self.assertEqual(statuses["still-running"], "rollout")
        self.assertTrue(artifact["salvaged"])

    def test_add_cli_can_update_a_queue_owned_by_another_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(queue.benchmark, "resolve_upstream", return_value=root),
                patch.object(queue.benchmark, "load_task", return_value={"id": "new-task"}),
            ):
                returncode = queue.main(
                    ["--run", str(root / "run"), "add", "--task", "new-task"]
                )
            cases = queue.QueueStore(root / "run").cases()

        self.assertEqual(returncode, 0)
        self.assertEqual(
            [(case["task"], case["status"]) for case in cases],
            [("new-task", "queued")],
        )

    def test_retry_archives_the_previous_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = queue.QueueStore(root)
            store.enqueue(["again"], "adaptive")
            case = store.claim("queued", "rollout", 1)[0]
            store.mark_failed(case["id"], "broken")
            case_root = root / "again" / "adaptive"
            case_root.mkdir(parents=True)
            (case_root / "result.json").write_text("{}", encoding="utf-8")

            retried, missing = store.retry(["again"], "adaptive")
            archives = list((root / "_attempts" / "again" / "adaptive").glob("attempt-1-*"))
            archived_result = len(archives) == 1 and (archives[0] / "result.json").is_file()

        self.assertEqual(retried, ["again"])
        self.assertEqual(missing, [])
        self.assertEqual(len(archives), 1)
        self.assertTrue(archived_result)

    def test_retry_does_not_expose_queued_case_before_archive_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = queue.QueueStore(root)
            store.enqueue(["atomic"], "adaptive")
            case = store.claim("queued", "rollout", 1)[0]
            store.mark_failed(case["id"], "try again")
            case_root = root / "atomic" / "adaptive"
            case_root.mkdir(parents=True)
            (case_root / "result.json").write_text("{}", encoding="utf-8")

            archive_started = threading.Event()
            release_archive = threading.Event()
            claim_finished = threading.Event()
            retry_result: list[tuple[list[str], list[str]]] = []
            claim_result: list[list[dict[str, object]]] = []
            errors: list[BaseException] = []
            real_move = queue.shutil.move

            def blocking_move(source: str, destination: str) -> object:
                archive_started.set()
                if not release_archive.wait(timeout=3):
                    raise TimeoutError("test did not release archive")
                return real_move(source, destination)

            def retry_case() -> None:
                try:
                    retry_result.append(store.retry(["atomic"], "adaptive"))
                except BaseException as exc:
                    errors.append(exc)

            def claim_case() -> None:
                try:
                    claim_result.append(store.claim("queued", "rollout", 1))
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    claim_finished.set()

            with patch.object(queue.shutil, "move", side_effect=blocking_move):
                retry_thread = threading.Thread(target=retry_case)
                retry_thread.start()
                self.assertTrue(archive_started.wait(timeout=2))
                claim_thread = threading.Thread(target=claim_case)
                claim_thread.start()
                self.assertFalse(claim_finished.wait(timeout=0.1))
                release_archive.set()
                retry_thread.join(timeout=3)
                claim_thread.join(timeout=3)

            archives = list(
                (root / "_attempts" / "atomic" / "adaptive").glob(
                    "attempt-1-manual-retry-*"
                )
            )
            archived_result = bool(
                archives and (archives[0] / "result.json").is_file()
            )

        self.assertEqual(errors, [])
        self.assertEqual(retry_result, [(["atomic"], [])])
        self.assertEqual(len(claim_result), 1)
        self.assertEqual(len(claim_result[0]), 1)
        self.assertEqual(claim_result[0][0]["attempt"], 2)
        self.assertTrue(archived_result)

    def test_retry_restores_case_root_when_archive_step_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = queue.QueueStore(root)
            store.enqueue(["restore"], "adaptive")
            case = store.claim("queued", "rollout", 1)[0]
            store.mark_failed(case["id"], "try again")
            case_root = root / "restore" / "adaptive"
            case_root.mkdir(parents=True)
            (case_root / "result.json").write_text("old", encoding="utf-8")
            real_move = queue.shutil.move
            moves = 0

            def fail_after_first_move(source: str, destination: str) -> object:
                nonlocal moves
                moves += 1
                result = real_move(source, destination)
                if moves == 1:
                    raise OSError("simulated post-rename failure")
                return result

            with (
                patch.object(
                    queue.shutil, "move", side_effect=fail_after_first_move
                ),
                self.assertRaisesRegex(OSError, "post-rename failure"),
            ):
                store.retry(["restore"], "adaptive")
            restored_case = store.cases()[0]
            restored_result = (case_root / "result.json").read_text(encoding="utf-8")

        self.assertEqual(moves, 2)
        self.assertEqual(restored_case["status"], "failed")
        self.assertEqual(restored_case["attempt"], 1)
        self.assertEqual(restored_result, "old")

    def test_retry_of_reward_failure_reuses_the_completed_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = queue.QueueStore(root)
            store.enqueue(["rescore"], "adaptive")
            case = store.claim("queued", "rollout", 1)[0]
            case_root = root / "rescore" / "adaptive"
            workspace = case_root / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "start.md").write_text("spec", encoding="utf-8")
            (case_root / "agent-result.json").write_text(
                json.dumps({"ok": True, "rollout_outcome": "completed"}),
                encoding="utf-8",
            )
            (case_root / "rollout-artifact.json").write_text(
                json.dumps(
                    {
                        "start_hash": "hash",
                        "agent_elapsed_s": 1.0,
                        "agent_timed_out": False,
                        "agent_returncode": 0,
                    }
                ),
                encoding="utf-8",
            )
            (case_root / "result.json").write_text(
                json.dumps({"reward_outcome": "infra_error"}), encoding="utf-8"
            )
            store.mark_rollout_complete(case["id"])
            store.claim("reward_pending", "rewarding", 1)
            store.mark_failed(case["id"], "scorer unavailable")

            retried, missing = store.retry(["rescore"], "adaptive")
            retried_case = store.cases()[0]
            artifact_preserved = (case_root / "rollout-artifact.json").is_file()
            result_preserved = (case_root / "result.json").is_file()

        self.assertEqual(retried, ["rescore"])
        self.assertEqual(missing, [])
        self.assertEqual(retried_case["status"], "reward_pending")
        self.assertEqual(retried_case["attempt"], 1)
        self.assertTrue(artifact_preserved)
        self.assertTrue(result_preserved)

    def test_incomplete_rollout_cannot_enter_reward_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_root = Path(tmp) / "broken" / "adaptive"
            case_root.mkdir(parents=True)
            artifact = SimpleNamespace(
                task={"id": "broken"},
                mode="adaptive",
                case_root=case_root,
                start_hash="hash",
                agent_elapsed_s=0.1,
                agent_timed_out=False,
                agent_returncode=1,
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "refusing to persist incomplete rollout",
            ):
                queue.persist_artifact(artifact)

        self.assertFalse((case_root / "rollout-artifact.json").exists())

    def test_retryable_rollout_infrastructure_error_requeues_with_attempt_budget(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = queue.QueueStore(root)
            store.enqueue(["retry-me"], "adaptive")
            rollout_calls = 0
            reward_calls = 0
            events: list[dict[str, object]] = []

            def rollout(task: dict[str, object], mode: str) -> SimpleNamespace:
                nonlocal rollout_calls
                rollout_calls += 1
                agent: dict[str, object]
                if rollout_calls == 1:
                    agent = {
                        "ok": False,
                        "rollout_outcome": "infra_error",
                        "rollout_infrastructure": True,
                        "rollout_retryable": True,
                        "workspace_download_error": "temporary transfer failure",
                    }
                else:
                    agent = {"ok": True, "rollout_outcome": "completed"}
                return self._rollout_artifact(root, task, mode, agent)

            def reward(artifact: object) -> dict[str, object]:
                nonlocal reward_calls
                reward_calls += 1
                return {"quality_score": 75.0, "success": False}

            queue.run_queue_loop(
                store,
                lambda name: {"id": name},
                rollout,
                reward,
                rollout_concurrency=1,
                reward_concurrency=1,
                rollout_attempts=2,
                stop_event=threading.Event(),
                poll_interval_s=0.001,
                stop_when_empty=True,
                on_event=events.append,
            )
            case = store.cases()[0]
            archives = list(
                (root / "_attempts" / "retry-me" / "adaptive").glob(
                    "attempt-1-infra-retry-*"
                )
            )
            archived_agent_exists = bool(
                archives and (archives[0] / "agent-result.json").is_file()
            )

        self.assertEqual(case["status"], "done")
        self.assertEqual(case["attempt"], 2)
        self.assertEqual(rollout_calls, 2)
        self.assertEqual(reward_calls, 1)
        self.assertEqual(len(archives), 1)
        self.assertTrue(archived_agent_exists)
        self.assertEqual(
            [event["event"] for event in events].count("rollout.requeued"), 1
        )

    def test_retryable_infrastructure_exception_from_rollout_is_requeued(self) -> None:
        class RetryableInfrastructureError(RuntimeError):
            retryable = True
            failure_domain = "infrastructure"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = queue.QueueStore(root)
            store.enqueue(["exception"], "adaptive")
            rollout_calls = 0

            def rollout(task: dict[str, object], mode: str) -> SimpleNamespace:
                nonlocal rollout_calls
                rollout_calls += 1
                if rollout_calls == 1:
                    raise RetryableInfrastructureError("gateway wrapper")
                return self._rollout_artifact(
                    root,
                    task,
                    mode,
                    {"ok": True, "rollout_outcome": "completed"},
                )

            queue.run_queue_loop(
                store,
                lambda name: {"id": name},
                rollout,
                lambda artifact: {"quality_score": 100.0, "success": True},
                rollout_concurrency=1,
                reward_concurrency=1,
                rollout_attempts=2,
                stop_event=threading.Event(),
                poll_interval_s=0.001,
                stop_when_empty=True,
            )
            case = store.cases()[0]

        self.assertEqual(case["status"], "done")
        self.assertEqual(case["attempt"], 2)
        self.assertEqual(rollout_calls, 2)

    def test_retryable_infrastructure_exception_from_persist_is_requeued(self) -> None:
        class RetryableInfrastructureError(RuntimeError):
            retryable = True
            infrastructure = True

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = queue.QueueStore(root)
            store.enqueue(["persist"], "adaptive")
            persist_calls = 0
            rollout_calls = 0
            real_persist = queue.persist_artifact

            def rollout(task: dict[str, object], mode: str) -> SimpleNamespace:
                nonlocal rollout_calls
                rollout_calls += 1
                return self._rollout_artifact(
                    root,
                    task,
                    mode,
                    {"ok": True, "rollout_outcome": "completed"},
                )

            def persist(artifact: object) -> None:
                nonlocal persist_calls
                persist_calls += 1
                if persist_calls == 1:
                    raise RetryableInfrastructureError("storage unavailable")
                real_persist(artifact)

            with patch.object(queue, "persist_artifact", side_effect=persist):
                queue.run_queue_loop(
                    store,
                    lambda name: {"id": name},
                    rollout,
                    lambda artifact: {"quality_score": 100.0, "success": True},
                    rollout_concurrency=1,
                    reward_concurrency=1,
                    rollout_attempts=2,
                    stop_event=threading.Event(),
                    poll_interval_s=0.001,
                    stop_when_empty=True,
                )
            case = store.cases()[0]

        self.assertEqual(case["status"], "done")
        self.assertEqual(case["attempt"], 2)
        self.assertEqual(rollout_calls, 2)
        self.assertEqual(persist_calls, 2)

    def test_programming_and_missing_file_exceptions_are_not_retryable(self) -> None:
        self.assertFalse(queue.retryable_rollout_exception(TypeError("bad call")))
        self.assertFalse(
            queue.retryable_rollout_exception(
                FileNotFoundError("agent-result.json is missing")
            )
        )
        try:
            raise RuntimeError("wrapped") from TimeoutError("gateway timeout")
        except RuntimeError as error:
            self.assertTrue(queue.retryable_rollout_exception(error))

    def test_harness_error_is_failed_without_entering_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = queue.QueueStore(root)
            store.enqueue(["harness-broken"], "forced-team")
            reward_calls = 0

            def rollout(task: dict[str, object], mode: str) -> SimpleNamespace:
                return self._rollout_artifact(
                    root,
                    task,
                    mode,
                    {
                        "ok": False,
                        "rollout_outcome": "harness_error",
                        # Older agent results incorrectly set this on harness
                        # failures; the explicit outcome must still win.
                        "rollout_infrastructure": True,
                        "rollout_retryable": False,
                        "error": "model returned an empty response",
                    },
                )

            def reward(artifact: object) -> dict[str, object]:
                nonlocal reward_calls
                reward_calls += 1
                return {"quality_score": 100.0}

            queue.run_queue_loop(
                store,
                lambda name: {"id": name},
                rollout,
                reward,
                rollout_concurrency=1,
                reward_concurrency=1,
                rollout_attempts=3,
                stop_event=threading.Event(),
                poll_interval_s=0.001,
                stop_when_empty=True,
            )
            case = store.cases()[0]

        self.assertEqual(case["status"], "failed")
        self.assertEqual(case["attempt"], 1)
        self.assertIn("empty response", case["error"])
        self.assertEqual(reward_calls, 0)

    def test_pending_or_skipped_reward_is_failed_and_keeps_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = queue.QueueStore(root)
            store.enqueue(["unscored"], "adaptive-team-v2")

            def rollout(task: dict[str, object], mode: str) -> SimpleNamespace:
                return self._rollout_artifact(
                    root,
                    task,
                    mode,
                    {"ok": True, "rollout_outcome": "completed"},
                )

            def reward(artifact: object) -> dict[str, object]:
                return {
                    "quality_score": 0.0,
                    "success": False,
                    "reward_outcome": "pending",
                    "reward_score_valid": False,
                    "retryable": True,
                    "hidden_tests": {
                        "skipped": True,
                        "skip_reason": "scorer unavailable",
                    },
                }

            queue.run_queue_loop(
                store,
                lambda name: {"id": name},
                rollout,
                reward,
                rollout_concurrency=1,
                reward_concurrency=1,
                stop_event=threading.Event(),
                poll_interval_s=0.001,
                stop_when_empty=True,
            )
            case = store.cases()[0]
            case_root = root / "unscored" / "adaptive-team-v2"
            artifact_preserved = (case_root / "rollout-artifact.json").is_file()
            was_not_archived = not (root / "_attempts" / "unscored").exists()

        self.assertEqual(case["status"], "failed")
        self.assertIsNone(case["quality_score"])
        self.assertIn("reward_outcome=pending", case["error"])
        self.assertTrue(artifact_preserved)
        self.assertTrue(was_not_archived)

    def test_live_loop_accepts_cases_added_after_workers_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = queue.QueueStore(root)
            store.enqueue(["first", "second"], "adaptive")
            stop = threading.Event()
            release = threading.Event()
            two_started = threading.Event()
            third_started = threading.Event()
            reward_started = threading.Event()
            release_reward = threading.Event()
            lock = threading.Lock()
            active = 0
            max_active = 0
            started: list[str] = []
            reward_calls = 0
            events: list[dict[str, object]] = []

            def load_task(name: str) -> dict[str, object]:
                return {"id": name}

            def rollout(task: dict[str, object], mode: str) -> SimpleNamespace:
                nonlocal active, max_active
                task_name = str(task["id"])
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                    started.append(task_name)
                    if len(started) >= 2:
                        two_started.set()
                    if task_name == "third":
                        third_started.set()
                self.assertTrue(release.wait(timeout=3))
                case_root = root / task_name / mode
                workspace = case_root / "workspace"
                workspace.mkdir(parents=True)
                (workspace / "start.md").write_text("spec", encoding="utf-8")
                (case_root / "agent-result.json").write_text(
                    json.dumps({"ok": True}), encoding="utf-8"
                )
                with lock:
                    active -= 1
                return SimpleNamespace(
                    task=task,
                    mode=mode,
                    case_root=case_root,
                    workspace=workspace,
                    start_hash="hash",
                    agent={"ok": True},
                    agent_elapsed_s=0.01,
                    agent_timed_out=False,
                    agent_returncode=0,
                )

            def reward(artifact: object) -> dict[str, object]:
                nonlocal reward_calls
                with lock:
                    reward_calls += 1
                    call_number = reward_calls
                if call_number == 1:
                    reward_started.set()
                    self.assertTrue(release_reward.wait(timeout=3))
                return {
                    "task": artifact.task["id"],
                    "quality_score": 100.0,
                    "success": True,
                }

            thread = threading.Thread(
                target=queue.run_queue_loop,
                args=(store, load_task, rollout, reward),
                kwargs={
                    "rollout_concurrency": 2,
                    "reward_concurrency": 1,
                    "stop_event": stop,
                    "poll_interval_s": 0.01,
                    "on_event": events.append,
                },
            )
            thread.start()
            self.assertTrue(two_started.wait(timeout=2))
            added, _ = store.enqueue(["third"], "adaptive")
            release.set()
            self.assertTrue(reward_started.wait(timeout=2))
            self.assertTrue(third_started.wait(timeout=2))
            release_reward.set()

            deadline = time.monotonic() + 4
            while time.monotonic() < deadline and store.counts()["done"] < 3:
                time.sleep(0.02)
            stop.set()
            thread.join(timeout=3)

            counts = store.counts()

        self.assertFalse(thread.is_alive())
        self.assertEqual(added, ["third"])
        self.assertEqual(counts["done"], 3)
        self.assertEqual(max_active, 2)
        self.assertIn("third", started)
        third_started = next(
            event for event in events
            if event["event"] == "rollout.started" and event["task"] == "third"
        )
        self.assertIsNotNone(third_started)


if __name__ == "__main__":
    unittest.main()
