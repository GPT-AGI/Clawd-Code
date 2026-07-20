from __future__ import annotations

import importlib.util
import json
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


class TestEvaluationQueue(unittest.TestCase):
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

    def test_global_slots_pipeline_into_the_next_run(self) -> None:
        snapshots = [
            {"rollout": 17, "queued": 0},
            {"rollout": 0, "queued": 104},
            {"rollout": 0, "queued": 104},
        ]

        allocated = queue.allocate_global_slots(
            snapshots,
            32,
            active_key="rollout",
            pending_key="queued",
        )

        self.assertEqual(allocated, [17, 15, 0])

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
