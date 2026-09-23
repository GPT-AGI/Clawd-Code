from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "teammate-evals"
    / "nl2repo-pilot"
    / "dashboard.py"
)
SPEC = importlib.util.spec_from_file_location("nl2repo_dashboard", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
dashboard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dashboard
SPEC.loader.exec_module(dashboard)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


class TestNL2RepoDashboard(unittest.TestCase):
    def test_rollout_infrastructure_preserves_null_protocol_and_effective_metrics(self) -> None:
        metrics = dashboard._normalize_result_metrics(
            {
                "result_schema_version": 2,
                "agent_ok": False,
                "delivery_valid": False,
                "protocol_status": "not_evaluated",
                "protocol_credit": None,
                "code_quality_score": None,
                "effective_quality_score": None,
                "reward_outcome": "pending",
                "reward_score_valid": False,
                "metric_eligibility": {
                    "code_quality": False,
                    "protocol_yield": False,
                    "effective_quality": False,
                },
                "failure_domain": "infrastructure",
                "is_infrastructure": True,
                "retryable": True,
            }
        )

        self.assertIsNone(metrics["protocol_credit"])
        self.assertIsNone(metrics["effective_quality_score"])
        self.assertFalse(any(metrics["metric_eligibility"].values()))

    def test_reward_infrastructure_is_excluded_from_qpe_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "run-metadata.json",
                {
                    "run_id": "infra-precedence",
                    "tasks": ["scored", "scorer-infra"],
                    "modes": ["forced-team"],
                },
            )
            common = {
                "result_schema_version": 2,
                "agent_ok": True,
                "integrity_ok": True,
                "delivery_valid": True,
                "hidden_tests": {
                    "pytest": {
                        "expected": 1,
                        "passed": 1,
                        "failed": 0,
                        "errors": 0,
                        "returncode": 0,
                        "all_passed": True,
                    }
                },
            }
            write_json(
                root / "scored" / "forced-team" / "result.json",
                {
                    **common,
                    "quality_score": 80.0,
                    "code_quality_score": 80.0,
                    "protocol_status": "passed",
                    "protocol_credit": 1.0,
                    "effective_quality_score": 80.0,
                    "reward_outcome": "scored",
                    "reward_score_valid": True,
                    "metric_eligibility": {
                        "code_quality": True,
                        "protocol_yield": True,
                        "effective_quality": True,
                    },
                },
            )
            write_json(
                root / "scorer-infra" / "forced-team" / "result.json",
                {
                    **common,
                    "quality_score": 0.0,
                    "code_quality_score": None,
                    "protocol_status": "failed",
                    "protocol_credit": 0.0,
                    "effective_quality_score": None,
                    "reward_outcome": "infra_error",
                    "reward_score_valid": False,
                    "metric_eligibility": {
                        "code_quality": False,
                        "protocol_yield": False,
                        "effective_quality": False,
                    },
                    "failure_domain": "infrastructure",
                    "is_infrastructure": True,
                    "retryable": True,
                    "hidden_tests": {
                        "error": "scorer sandbox unavailable",
                        "pytest": common["hidden_tests"]["pytest"],
                    },
                },
            )

            state = dashboard.DashboardStore(root).state()

        self.assertEqual(state["summary"]["infrastructure_errors"], 1)
        self.assertEqual(state["summary"]["code_quality"], 80.0)
        self.assertEqual(state["summary"]["coverage"], 0.5)
        self.assertEqual(state["summary"]["protocol_yield"], 1.0)
        self.assertEqual(state["summary"]["protocol_eligible"], 1)
        self.assertEqual(state["summary"]["effective_quality"], 80.0)
        self.assertEqual(state["summary"]["effective_eligible"], 1)
        infra = next(task for task in state["tasks"] if task["task"] == "scorer-infra")
        self.assertIsNone(infra["effective_quality_score"])
        self.assertFalse(any(infra["metric_eligibility"].values()))

    def make_queue_run(
        self,
        root: Path,
        run_id: str,
        statuses: list[str],
        *,
        rollout_slots: int,
        reward_slots: int,
    ) -> Path:
        run = root / run_id
        write_json(
            run / "run-metadata.json",
            {"run_id": run_id, "queue_mode": "continuous", "provider": "qwen"},
        )
        with sqlite3.connect(run / "queue.sqlite3") as connection:
            connection.execute(
                "CREATE TABLE cases (id INTEGER PRIMARY KEY, status TEXT, quality_score REAL)"
            )
            connection.execute(
                "CREATE TABLE worker_config (id INTEGER PRIMARY KEY, "
                "rollout_concurrency INTEGER, reward_concurrency INTEGER, "
                "max_rollout_concurrency INTEGER, max_reward_concurrency INTEGER, "
                "updated_at TEXT)"
            )
            connection.executemany(
                "INSERT INTO cases(status, quality_score) VALUES (?, ?)",
                [(status, 80.0 if status == "done" else None) for status in statuses],
            )
            connection.execute(
                "INSERT INTO worker_config VALUES (1, ?, ?, 64, 64, ?)",
                (rollout_slots, reward_slots, datetime.now(timezone.utc).isoformat()),
            )
        return run

    def test_registry_discovers_and_safely_switches_sibling_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "run-a"
            second = root / "run-b"
            write_json(first / "run-metadata.json", {"run_id": "run-a"})
            write_json(
                second / "run-metadata.json",
                {"run_id": "run-b", "queue_mode": "continuous"},
            )
            registry = dashboard.DashboardRegistry(first)
            listing = registry.listing()
            selected = registry.get("run-b")

            with self.assertRaises(KeyError):
                registry.get("../outside")

        self.assertEqual(listing["default"], "run-a")
        self.assertEqual([run["id"] for run in listing["runs"]], ["run-b", "run-a"])
        self.assertEqual(selected.run_root.name, "run-b")

    def test_global_state_groups_campaign_and_aggregates_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adaptive = self.make_queue_run(
                root,
                "20260717-qwen104-adaptive-team-v2-pool32-r2",
                ["queued", "rollout", "done"],
                rollout_slots=12,
                reward_slots=2,
            )
            self.make_queue_run(
                root,
                "20260717-qwen104-forced-team-fixed-pool32-r2",
                ["queued", "reward_pending", "rewarding", "done"],
                rollout_slots=20,
                reward_slots=6,
            )
            self.make_queue_run(
                root,
                "20260716-qwen104-adaptive-team-v2-pool32-r1",
                ["queued"],
                rollout_slots=32,
                reward_slots=0,
            )
            state = dashboard.DashboardRegistry(adaptive).global_state(adaptive.name)

        self.assertEqual(state["campaign"], "20260717-qwen104")
        self.assertEqual(state["summary"]["runs"], 2)
        self.assertEqual(state["summary"]["total"], 7)
        self.assertEqual(state["summary"]["queued"], 2)
        self.assertEqual(state["pool"]["allocated_rollout"], 32)
        self.assertEqual(state["pool"]["allocated_reward"], 8)
        self.assertEqual(state["summary"]["rollouts_completed"], 4)
        self.assertEqual(state["summary"]["rewards_completed"], 2)

    def make_run(self, root: Path) -> Path:
        started = datetime.now(timezone.utc) - timedelta(minutes=10)
        write_json(
            root / "run-metadata.json",
            {
                "run_id": "test-run",
                "started_at": started.isoformat(),
                "tasks": ["done", "active", "queued"],
                "modes": ["adaptive"],
                "provider": "qwen",
                "model": "test-model",
                "max_turns": 300,
                "rollout_concurrency": 2,
                "reward_concurrency": 1,
            },
        )
        write_jsonl(
            root / "scheduler.jsonl",
            [
                {"event": "rollout.started", "task": "done", "mode": "adaptive", "elapsed_s": 0},
                {"event": "rollout.started", "task": "active", "mode": "adaptive", "elapsed_s": 3},
                {"event": "rollout.completed", "task": "done", "mode": "adaptive", "elapsed_s": 100},
                {"event": "reward.started", "task": "done", "mode": "adaptive", "elapsed_s": 101},
                {"event": "reward.completed", "task": "done", "mode": "adaptive", "elapsed_s": 110, "quality_score": 0},
            ],
        )
        now = datetime.now(timezone.utc).isoformat()
        write_jsonl(
            root / "done" / "adaptive" / "progress.jsonl",
            [
                {"kind": "model_response", "turn": 1, "duration_ms": 2000, "created_at": now},
                {"kind": "tool_use", "turn": 1, "tool_name": "Bash", "created_at": now},
                {"kind": "run_completed", "turn": 1, "created_at": now},
            ],
        )
        write_jsonl(
            root / "active" / "adaptive" / "progress.jsonl",
            [{"kind": "model_response", "turn": 1, "duration_ms": 1200, "created_at": now}],
        )
        write_json(
            root / "done" / "adaptive" / "result.json",
            {
                "quality_score": 81.82,
                "success": False,
                "agent_ok": True,
                "usage": {"lead_turns": 1},
                "hidden_tests": {"pytest": {"passed": 9, "failed": 2, "errors": 0, "expected": 11}},
                "rescored_at": now,
            },
        )
        (root / "done" / "adaptive" / "hidden-tests.log").write_text(
            "nine passed, two failed\n", encoding="utf-8"
        )
        return root

    def test_state_tracks_configured_queue_and_corrected_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = dashboard.DashboardStore(self.make_run(Path(tmp)))
            first = store.state()
            second = store.state()

        self.assertEqual(first["summary"]["total"], 3)
        self.assertEqual(first["summary"]["queued"], 1)
        self.assertEqual(first["summary"]["active"], 1)
        self.assertEqual(first["summary"]["rewards_completed"], 1)
        self.assertEqual(first["summary"]["code_quality"], 81.82)
        self.assertEqual(first["summary"]["coverage"], 1.0)
        self.assertEqual(first["summary"]["protocol_yield"], 1.0)
        self.assertEqual(first["summary"]["effective_quality"], 81.82)
        done = next(task for task in first["tasks"] if task["task"] == "done")
        self.assertEqual(done["quality_score"], 81.82)
        self.assertEqual(done["code_quality_score"], 81.82)
        self.assertEqual(done["protocol_status"], "passed")
        self.assertEqual(done["effective_quality_score"], 81.82)
        self.assertEqual((done["passed"], done["expected"]), (9, 11))
        self.assertTrue(done["rescored"])
        self.assertEqual(second["summary"]["model_calls"], 2)

    def test_v2_summary_separates_code_protocol_and_effective_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "run-metadata.json",
                {
                    "run_id": "v2-run",
                    "tasks": ["protocol-pass", "protocol-fail", "legacy-skip"],
                    "modes": ["forced-team"],
                },
            )
            common = {
                "agent_ok": True,
                "integrity_ok": True,
                "hidden_tests": {
                    "pytest": {
                        "expected": 4,
                        "passed": 3,
                        "failed": 1,
                        "errors": 0,
                        "quality_score": 75.0,
                        "returncode": 1,
                        "all_passed": False,
                    }
                },
            }
            write_json(
                root / "protocol-pass" / "forced-team" / "result.json",
                {
                    **common,
                    "quality_score": 75.0,
                    "result_schema_version": 2,
                    "code_quality_score": 75.0,
                    "protocol_ok": True,
                    "protocol_status": "passed",
                    "protocol_credit": 1.0,
                    "delivery_valid": True,
                    "effective_quality_score": 75.0,
                    "reward_outcome": "scored",
                    "reward_score_valid": True,
                    "metric_eligibility": {
                        "code_quality": True,
                        "protocol_yield": True,
                        "effective_quality": True,
                    },
                },
            )
            write_json(
                root / "protocol-fail" / "forced-team" / "result.json",
                {
                    **common,
                    "quality_score": 75.0,
                    "result_schema_version": 2,
                    "code_quality_score": 75.0,
                    "protocol_ok": False,
                    "protocol_status": "failed",
                    "protocol_credit": 0.0,
                    "delivery_valid": True,
                    "effective_quality_score": 0.0,
                    "reward_outcome": "scored",
                    "reward_score_valid": True,
                    "metric_eligibility": {
                        "code_quality": True,
                        "protocol_yield": True,
                        "effective_quality": True,
                    },
                },
            )
            write_json(
                root / "legacy-skip" / "forced-team" / "result.json",
                {
                    **common,
                    "quality_score": 0.0,
                    "protocol_ok": False,
                    "reward_skipped": True,
                    "hidden_tests": {
                        "skipped": True,
                        "pytest": common["hidden_tests"]["pytest"],
                    },
                },
            )

            state = dashboard.DashboardStore(root).state()

        self.assertEqual(state["summary"]["code_quality"], 75.0)
        self.assertAlmostEqual(state["summary"]["coverage"], 2 / 3)
        self.assertAlmostEqual(state["summary"]["protocol_yield"], 1 / 3)
        self.assertEqual(state["summary"]["effective_quality"], 25.0)
        legacy = next(task for task in state["tasks"] if task["task"] == "legacy-skip")
        self.assertEqual(legacy["reward_outcome"], "protocol_skipped_legacy")
        self.assertIsNone(legacy["code_quality_score"])
        self.assertEqual(legacy["effective_quality_score"], 0.0)

    def test_task_detail_and_infrastructure_error_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_run(Path(tmp))
            result_path = root / "done" / "adaptive" / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["hidden_tests"]["error"] = "score image build failed"
            write_json(result_path, result)
            store = dashboard.DashboardStore(root)
            detail = store.task_detail("done")

        self.assertEqual(detail["task"]["status"], "infra_error")
        self.assertEqual(detail["task"]["infrastructure_error"], "score image build failed")
        self.assertIn("nine passed", detail["hidden_log"])
        self.assertEqual(len(detail["recent_events"]), 3)

    def test_v2_candidate_failure_is_not_reclassified_by_stale_hidden_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_run(Path(tmp))
            result_path = root / "done" / "adaptive" / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result.update(
                {
                    "result_schema_version": 2,
                    "failure_domain": "candidate",
                    "failure_class": "rollout_failure",
                    "is_infrastructure": False,
                    "retryable": False,
                    "reward_outcome": "missing_artifact",
                    "reward_score_valid": False,
                    "metric_eligibility": {
                        "code_quality": False,
                        "protocol_yield": True,
                        "effective_quality": True,
                    },
                }
            )
            result["hidden_tests"]["error"] = "legacy rollout error"
            write_json(result_path, result)

            detail = dashboard.DashboardStore(root).task_detail("done")

        self.assertEqual(detail["task"]["status"], "scored")
        self.assertEqual(detail["task"]["failure_domain"], "candidate")
        self.assertFalse(detail["task"]["is_infrastructure"])
        self.assertIsNone(detail["task"]["infrastructure_error"])

    def test_task_detail_exposes_actor_aware_team_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_run(Path(tmp))
            team_root = (
                root
                / "done"
                / "adaptive"
                / "workspace"
                / ".clawd"
                / "teams"
                / "team-1"
            )
            write_json(
                team_root / "team.json",
                {"team_id": "team-1", "team_name": "forced", "status": "running"},
            )
            write_json(
                team_root / "tasks.json",
                {
                    "task-1": {
                        "subject": "inspect parser",
                        "status": "completed",
                        "owner": "worker-1",
                        "output": "done",
                    }
                },
            )
            now = datetime.now(timezone.utc).isoformat()
            write_jsonl(
                team_root / "events.jsonl",
                [
                    {
                        "type": "agent.created",
                        "created_at": now,
                        "data": {
                            "agent": {
                                "agent_id": "worker-1",
                                "name": "parser_worker",
                                "role": "parser",
                            }
                        },
                    },
                    {
                        "type": "model.started",
                        "created_at": now,
                        "data": {"actor_name": "parser_worker", "turn": 3},
                    },
                    {
                        "type": "model.response",
                        "created_at": now,
                        "data": {
                            "actor_name": "parser_worker",
                            "turn": 3,
                            "content": "checking parser",
                            "duration_ms": 1200,
                        },
                    },
                    {
                        "type": "tool.started",
                        "created_at": now,
                        "data": {
                            "actor_name": "parser_worker",
                            "tool_name": "Bash",
                            "tool_use_id": "call-1",
                            "tool_input": {"command": "pytest"},
                        },
                    },
                    {
                        "type": "tool.failed",
                        "created_at": now,
                        "data": {
                            "actor_name": "parser_worker",
                            "tool_name": "Bash",
                            "tool_use_id": "call-1",
                            "error": "exit 1",
                            "duration_ms": 400,
                        },
                    },
                ],
            )
            detail = dashboard.DashboardStore(root).task_detail("done")

        self.assertEqual(detail["team"]["name"], "forced")
        self.assertEqual(detail["team"]["tasks"][0]["owner"], "parser_worker")
        worker_events = [
            event
            for event in detail["trace_events"]
            if event["actor"] == "parser_worker"
        ]
        self.assertTrue(worker_events)
        self.assertEqual(worker_events[-1]["turn"], 3)
        self.assertEqual(worker_events[-1]["tool_use_id"], "call-1")
        self.assertTrue(worker_events[-1]["is_error"])

    def test_continuous_queue_is_the_authoritative_task_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "run-metadata.json",
                {
                    "run_id": "continuous",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "queue_mode": "continuous",
                    "modes": ["adaptive"],
                    "rollout_concurrency": 8,
                },
            )
            with sqlite3.connect(root / "queue.sqlite3") as connection:
                connection.execute(
                    """
                    CREATE TABLE cases (
                        id INTEGER PRIMARY KEY, task TEXT, mode TEXT, priority INTEGER,
                        status TEXT, attempt INTEGER, enqueued_at TEXT, started_at TEXT,
                        error TEXT
                    )
                    """
                )
                connection.executemany(
                    "INSERT INTO cases VALUES (?, ?, 'adaptive', 0, ?, 0, ?, ?, NULL)",
                    [
                        (1, "waiting", "queued", datetime.now(timezone.utc).isoformat(), None),
                        (2, "working", "rollout", datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat()),
                    ],
                )
            state = dashboard.DashboardStore(root).state()

        statuses = {task["task"]: task["status"] for task in state["tasks"]}
        self.assertEqual(state["summary"]["total"], 2)
        self.assertEqual(state["summary"]["queue_depth"], 1)
        self.assertTrue(state["summary"]["queue_low"])
        self.assertEqual(statuses, {"waiting": "queued", "working": "running"})

    def test_dashboard_reads_dynamic_concurrency_but_rejects_per_run_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "run-metadata.json",
                {
                    "run_id": "scalable",
                    "queue_mode": "continuous",
                    "rollout_concurrency": 8,
                    "reward_concurrency": 4,
                },
            )
            with sqlite3.connect(root / "queue.sqlite3") as connection:
                connection.execute(
                    """
                    CREATE TABLE cases (
                        id INTEGER PRIMARY KEY, task TEXT, mode TEXT, priority INTEGER,
                        status TEXT, attempt INTEGER, enqueued_at TEXT, started_at TEXT,
                        error TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE worker_config (
                        id INTEGER PRIMARY KEY,
                        rollout_concurrency INTEGER,
                        reward_concurrency INTEGER,
                        max_rollout_concurrency INTEGER,
                        max_reward_concurrency INTEGER,
                        updated_at TEXT
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO worker_config VALUES (1, 32, 4, 64, 16, ?)",
                    (datetime.now(timezone.utc).isoformat(),),
                )
            store = dashboard.DashboardStore(root)
            before = store.state()
            with self.assertRaisesRegex(RuntimeError, "global_pool_supervisor.py"):
                store.set_concurrency(rollout=16, reward=0)
            after = store.state()
            with sqlite3.connect(root / "queue.sqlite3") as connection:
                configured = connection.execute(
                    "SELECT rollout_concurrency, reward_concurrency "
                    "FROM worker_config WHERE id=1"
                ).fetchone()

        self.assertEqual(before["run"]["rollout_concurrency"], 32)
        self.assertEqual(before["run"]["max_rollout_concurrency"], 64)
        self.assertEqual(configured, (32, 4))
        self.assertEqual(after["run"]["rollout_concurrency"], 32)
        self.assertEqual(after["run"]["reward_concurrency"], 4)

    def test_concurrency_post_returns_global_pool_conflict(self) -> None:
        handler = object.__new__(dashboard.DashboardHandler)
        handler.path = "/api/concurrency"
        response: dict[str, object] = {}

        def capture(value: object, status: object = dashboard.HTTPStatus.OK) -> None:
            response.update({"value": value, "status": status})

        handler._send_json = capture
        handler.do_POST()

        self.assertEqual(response["status"], dashboard.HTTPStatus.CONFLICT)
        payload = response["value"]
        self.assertIsInstance(payload, dict)
        assert isinstance(payload, dict)
        self.assertEqual(payload["code"], "global_pool_managed")
        self.assertEqual(payload["global_state_endpoint"], "/api/global")
        self.assertIn("global_pool_supervisor.py", payload["error"])

    def test_requeued_case_ignores_scheduler_events_from_previous_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "run-metadata.json",
                {
                    "run_id": "retrying",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "queue_mode": "continuous",
                    "rollout_concurrency": 8,
                },
            )
            write_jsonl(
                root / "scheduler.jsonl",
                [
                    {
                        "event": "rollout.started",
                        "task": "again",
                        "mode": "adaptive",
                        "elapsed_s": 1,
                    },
                    {
                        "event": "rollout.completed",
                        "task": "again",
                        "mode": "adaptive",
                        "elapsed_s": 2,
                    },
                    {
                        "event": "reward.failed",
                        "task": "again",
                        "mode": "adaptive",
                        "elapsed_s": 3,
                    },
                ],
            )
            with sqlite3.connect(root / "queue.sqlite3") as connection:
                connection.execute(
                    """
                    CREATE TABLE cases (
                        id INTEGER PRIMARY KEY, task TEXT, mode TEXT, priority INTEGER,
                        status TEXT, attempt INTEGER, enqueued_at TEXT, started_at TEXT,
                        rollout_finished_at TEXT, error TEXT
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO cases VALUES "
                    "(1, 'again', 'adaptive', 0, 'queued', 2, ?, NULL, NULL, NULL)",
                    (datetime.now(timezone.utc).isoformat(),),
                )
            state = dashboard.DashboardStore(root).state()

        self.assertEqual(state["tasks"][0]["status"], "queued")
        self.assertEqual(state["summary"]["queued"], 1)
        self.assertEqual(state["summary"]["started"], 0)
        self.assertEqual(state["summary"]["rollouts_completed"], 0)
        self.assertEqual(state["summary"]["rewards_completed"], 0)

    def test_continuous_queue_keeps_modes_as_distinct_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "run-metadata.json",
                {
                    "run_id": "dual-mode",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "queue_mode": "continuous",
                    "rollout_concurrency": 8,
                },
            )
            with sqlite3.connect(root / "queue.sqlite3") as connection:
                connection.execute(
                    """
                    CREATE TABLE cases (
                        id INTEGER PRIMARY KEY, task TEXT, mode TEXT, priority INTEGER,
                        status TEXT, attempt INTEGER, enqueued_at TEXT, started_at TEXT,
                        error TEXT
                    )
                    """
                )
                now = datetime.now(timezone.utc).isoformat()
                connection.executemany(
                    "INSERT INTO cases VALUES (?, 'same-task', ?, 0, 'queued', 0, ?, NULL, NULL)",
                    [(1, "adaptive", now), (2, "forced-team", now)],
                )
            store = dashboard.DashboardStore(root)
            state = store.state()
            forced = store.task_detail("same-task", "forced-team")

        self.assertEqual(state["summary"]["total"], 2)
        self.assertEqual(
            [(task["task"], task["mode"]) for task in state["tasks"]],
            [("same-task", "adaptive"), ("same-task", "forced-team")],
        )
        self.assertEqual(forced["task"]["mode"], "forced-team")

    def test_comparison_merges_missing_mode_from_sibling_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            current = runs / "current"
            baseline = runs / "baseline-32"
            write_json(
                current / "run-metadata.json",
                {
                    "run_id": "current",
                    "tasks": ["old-task", "new-task"],
                    "modes": ["adaptive", "forced-team"],
                    "comparison": {
                        "modes": ["adaptive", "forced-team"],
                        "baseline_runs": {"adaptive": "baseline-32"},
                    },
                },
            )

            def result(quality: float, runtime: float, tokens: int) -> dict[str, object]:
                return {
                    "model": "same-model",
                    "quality_score": quality,
                    "agent_elapsed_s": runtime,
                    "calls": {"model": 10, "tools": 9},
                    "usage": {"total_tokens": tokens},
                    "success": False,
                }

            write_json(
                baseline / "old-task" / "adaptive" / "result.json",
                result(10, 100, 0),
            )
            write_json(
                current / "old-task" / "forced-team" / "result.json",
                result(15, 70, 500),
            )
            write_json(
                current / "new-task" / "adaptive" / "result.json",
                result(20, 80, 400),
            )
            write_json(
                current / "new-task" / "forced-team" / "result.json",
                result(18, 60, 600),
            )

            comparison = dashboard.DashboardStore(current).state()["comparison"]

        self.assertIsNotNone(comparison)
        assert comparison is not None
        self.assertEqual(comparison["paired_count"], 2)
        self.assertEqual(comparison["cross_run_count"], 1)
        self.assertEqual(comparison["deployment_mismatch_count"], 0)
        self.assertEqual(
            comparison["mode_summaries"]["adaptive"]["source_runs"],
            {"baseline-32": 1, "current": 1},
        )
        self.assertEqual(
            comparison["mode_summaries"]["adaptive"]["token_coverage"], 1
        )
        self.assertAlmostEqual(
            comparison["paired"]["average_quality_delta"], 1.5
        )
        self.assertEqual(comparison["paired"]["right_faster"], 2)


if __name__ == "__main__":
    unittest.main()
