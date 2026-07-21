from __future__ import annotations

import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.teammate.models import TeamTask
from src.teammate.runtime import TeammateRuntime, TeamRunOptions
from src.teammate.store import TeamStore
from src.providers.base import ChatResponse
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.protocol import ToolCall
from src.tool_system.tools import (
    TeamCreateTool,
    TeamPlanTool,
    TeamReplanTool,
    TeamRunTool,
)


class ConcurrentFinalProvider:
    model = "test-model"

    def __init__(self) -> None:
        self.barrier = threading.Barrier(2)
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def chat(self, messages, tools=None, **kwargs):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.barrier.wait(timeout=3)
        except threading.BrokenBarrierError:
            pass
        finally:
            with self.lock:
                self.active -= 1
        return ChatResponse(
            content="implementation complete",
            model=self.model,
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="stop",
            tool_uses=None,
        )


class TestAtomicTeamPlan(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.context = ToolContext(workspace_root=self.root)
        self.registry = build_default_registry(include_user_tools=False)
        self.context.teammate_runtime = TeammateRuntime(object(), self.registry)
        TeamCreateTool().run(
            {"team_name": "strict-v2", "quality_gates": True}, self.context
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _payload() -> dict:
        return {
            "mode": "replace",
            "expected_revision": 0,
            "contract": {
                "summary": "core freezes a parse contract before parallel work",
                "interfaces": [
                    {
                        "name": "pkg.core.parse",
                        "provider_task": "core",
                        "consumer_tasks": ["api"],
                        "signature": "parse(text: str) -> dict",
                        "mode": "frozen",
                    },
                    {
                        "name": "pkg.core.artifact",
                        "provider_task": "core",
                        "consumer_tasks": ["integration"],
                        "signature": "installed editable package",
                        "mode": "handoff",
                    },
                ],
            },
            "workers": [
                {"name": "core-worker", "instructions": "Implement parsing core."},
                {"name": "api-worker", "instructions": "Implement and integrate API."},
            ],
            "tasks": [
                {
                    "key": "core",
                    "owner": "core-worker",
                    "instructions": "Implement the core package.",
                    "owned_files": ["/workspace/pkg"],
                    "acceptance_checks": ["python -m compileall -q pkg"],
                },
                {
                    "key": "model",
                    "owner": "core-worker",
                    "instructions": "Implement the core result model.",
                    "owned_files": ["./pkg/model.py"],
                    "acceptance_checks": ["python -m py_compile pkg/model.py"],
                },
                {
                    "key": "api",
                    "owner": "api-worker",
                    "instructions": "Implement the public API facade.",
                    "owned_files": ["./api.py"],
                    "acceptance_checks": ["python -m py_compile api.py"],
                },
                {
                    "key": "integration",
                    "owner": "api-worker",
                    "kind": "validation",
                    "instructions": "Validate the installed package contract.",
                    "owned_files": [],
                },
            ],
            "validation": {
                "profile": "python-package",
                "imports": ["json"],
                "commands": [
                    "python -c \"import json; assert json.loads('1') == 1\""
                ],
            },
            "execution": {},
        }

    def test_registry_exposes_atomic_team_plan(self) -> None:
        self.assertIsNotNone(self.registry.get("TeamPlan"))

    def test_materializes_normalized_plan_and_contract_dependencies(self) -> None:
        result = self.registry.dispatch(
            ToolCall(name="TeamPlan", input=self._payload()), self.context
        )

        self.assertFalse(result.is_error, result.output)
        self.assertEqual(result.output["status"], "ready")
        self.assertEqual(result.output["revision"], 1)
        self.assertEqual(result.output["protocol_version"], 2)
        self.assertEqual(result.output["execution"]["max_workers"], 2)
        self.assertEqual(result.output["execution"]["verify_timeout_s"], 900)
        self.assertTrue(result.output["execution"]["auto_verify"])
        self.assertIsNone(result.output["execution"]["token_budget"])
        self.assertIsNone(result.output["execution"]["turn_budget"])

        tasks = {task["key"]: task for task in self.context.tasks.values()}
        self.assertEqual(tasks["core"]["owned_files"], ["pkg"])
        self.assertEqual(tasks["model"]["owned_files"], ["pkg/model.py"])
        self.assertEqual(tasks["api"]["blockedBy"], [])
        self.assertEqual(tasks["api"]["depends_on_interfaces"], ["pkg.core.parse"])
        self.assertEqual(tasks["integration"]["owned_files"], [])
        self.assertEqual(tasks["integration"]["blockedBy"], [tasks["core"]["id"]])
        self.assertEqual(
            {task["metadata"]["contract_hash"] for task in tasks.values()},
            {
                self.context.team_store.load_active_team()
                .settings["quality_gates"]["contract_hash"]
            },
        )
        self.assertEqual(
            tasks["integration"]["acceptance_checks"],
            [result.output["validation"]["integration_command"]],
        )

        stored = self.context.team_store.load_active_team()
        self.assertEqual(stored.protocol_version, 2)
        self.assertEqual(stored.lifecycle_state, "ready")
        self.assertTrue(stored.settings["quality_gates"]["configured"])
        self.assertEqual(stored.settings["team_plan"]["revision"], 1)
        manifest = stored.settings["execution_manifest"]
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(len(manifest["budget_integrity_hash"]), 64)
        self.assertEqual(manifest["status"], "frozen")
        self.assertEqual(manifest["plan_hash"], result.output["plan_hash"])
        self.assertEqual(manifest["execution"], result.output["execution"])
        self.assertEqual(
            manifest["budget_window"]["baseline"],
            {"total_tokens": 0, "turns": 0},
        )
        for key in manifest["execution"]:
            self.assertNotIn(key, stored.settings)
        self.assertEqual(len(self.context.team_store.list_agents(stored.team_id)), 2)
        self.assertEqual(
            self.context.teammate_runtime._strict_plan_errors(
                self.context, stored, require_parallel_start=True
            ),
            [],
        )
        self.assertEqual(
            TeamRunOptions.build(stored.settings, {}).max_workers,
            2,
        )

    def test_explicit_max_workers_is_respected(self) -> None:
        payload = self._payload()
        payload["execution"]["max_workers"] = 1

        result = TeamPlanTool().run(payload, self.context)

        self.assertFalse(result.is_error, result.output)
        self.assertEqual(result.output["execution"]["max_workers"], 1)
        stored = self.context.team_store.load_active_team()
        self.assertEqual(TeamRunOptions.build(stored.settings, {}).max_workers, 1)

    def test_default_worker_count_runs_distinct_ready_owners_concurrently(self) -> None:
        provider = ConcurrentFinalProvider()
        self.context.teammate_runtime = TeammateRuntime(provider, self.registry)
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.root / "pkg" / "model.py").write_text("MODEL = 1\n", encoding="utf-8")
        (self.root / "api.py").write_text("API = 1\n", encoding="utf-8")
        (self.root / "pyproject.toml").write_text(
            "[build-system]\n"
            "requires = ['setuptools']\n"
            "build-backend = 'setuptools.build_meta'\n"
            "[project]\n"
            "name = 'team-plan-concurrency'\n"
            "version = '0.0.1'\n",
            encoding="utf-8",
        )
        planned = TeamPlanTool().run(self._payload(), self.context)
        self.assertFalse(planned.is_error, planned.output)

        result = TeamRunTool().run({}, self.context)

        self.assertFalse(result.is_error, result.output)
        self.assertEqual(result.output["status"], "completed")
        self.assertEqual(provider.max_active, 2)

    def test_cross_owner_overlap_is_structured_and_has_no_side_effects(self) -> None:
        payload = self._payload()
        payload["tasks"][2]["owned_files"] = ["/workspace/pkg/api.py"]

        result = TeamPlanTool().run(payload, self.context)

        self.assertTrue(result.is_error)
        self.assertEqual(result.output["status"], "needs_plan_fix")
        issue = next(
            item for item in result.output["issues"] if item["code"] == "PATH_OVERLAP"
        )
        self.assertEqual(issue["path"], "tasks[2].owned_files")
        self.assertEqual(issue["conflicts_with"], "tasks[0].owned_files")
        self.assertEqual(self.context.tasks, {})
        team = self.context.team_store.load_active_team()
        self.assertNotIn("team_plan", team.settings)
        self.assertEqual(self.context.team_store.list_agents(team.team_id), [])

    def test_rejects_weak_acceptance_and_ceremonial_second_worker(self) -> None:
        payload = self._payload()
        payload["tasks"][0]["acceptance_checks"] = ["test -e pkg"]
        payload["tasks"][2]["owner"] = "core-worker"

        result = TeamPlanTool().run(payload, self.context)
        codes = {issue["code"] for issue in result.output["issues"]}

        self.assertTrue(result.is_error)
        self.assertIn("TRIVIAL_ACCEPTANCE_CHECK", codes)
        self.assertIn("MIN_IMPLEMENTATION_OWNERS", codes)
        self.assertEqual(self.context.tasks, {})

    def test_rejects_metadata_only_implementation_partition(self) -> None:
        payload = self._payload()
        payload["tasks"][2]["owned_files"] = [
            "README.md",
            "docs/api.rst",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]

        result = TeamPlanTool().run(payload, self.context)
        codes = {issue["code"] for issue in result.output["issues"]}

        self.assertTrue(result.is_error)
        self.assertIn("CEREMONIAL_IMPLEMENTATION_TASK", codes)
        self.assertIn("MIN_IMPLEMENTATION_OWNERS", codes)
        self.assertEqual(self.context.tasks, {})

    def test_rejects_import_and_introspection_only_acceptance(self) -> None:
        payload = self._payload()
        payload["tasks"][2]["acceptance_checks"] = [
            (
                "python -c \"import api; assert hasattr(api, 'parse'); "
                "assert callable(api.parse)\""
            )
        ]

        result = TeamPlanTool().run(payload, self.context)
        codes = {issue["code"] for issue in result.output["issues"]}

        self.assertTrue(result.is_error)
        self.assertIn("WEAK_ACCEPTANCE_CHECK", codes)
        self.assertIn("MIN_IMPLEMENTATION_OWNERS", codes)

    def test_json_stringified_complex_fields_use_full_validation(self) -> None:
        payload = self._payload()
        for field in ("contract", "workers", "tasks", "validation"):
            payload[field] = json.dumps(payload[field])

        result = self.registry.dispatch(
            ToolCall(name="TeamPlan", input=payload), self.context
        )

        self.assertFalse(result.is_error, result.output)
        self.assertEqual(result.output["status"], "ready")
        schema = TeamPlanTool().spec().input_schema
        for field in ("contract", "workers", "tasks", "validation"):
            self.assertIn("oneOf", schema["properties"][field])

    def test_json_stringified_fields_do_not_bypass_semantic_validation(self) -> None:
        payload = self._payload()
        payload["tasks"][2]["owned_files"] = ["pkg/api.py"]
        for field in ("contract", "workers", "tasks", "validation"):
            payload[field] = json.dumps(payload[field])

        result = self.registry.dispatch(
            ToolCall(name="TeamPlan", input=payload), self.context
        )

        self.assertTrue(result.is_error)
        self.assertIn(
            "PATH_OVERLAP", {issue["code"] for issue in result.output["issues"]}
        )
        self.assertEqual(self.context.tasks, {})

    def test_invalid_json_string_returns_structured_issue(self) -> None:
        payload = self._payload()
        payload["tasks"] = "[{not-json}]"

        result = self.registry.dispatch(
            ToolCall(name="TeamPlan", input=payload), self.context
        )

        self.assertTrue(result.is_error)
        issue = next(
            issue
            for issue in result.output["issues"]
            if issue["code"] == "INVALID_JSON_STRING"
        )
        self.assertEqual(issue["path"], "tasks")

    def test_rejects_fail_open_and_trivial_python_acceptance(self) -> None:
        weak_commands = [
            "python -m pytest -q || true",
            "python -m pytest -q || :",
            "python -m pytest -q; true",
            "python -m pytest -q; exit 0",
            "python -c ''",
            "python -c 'pass'",
            "python -c 'print(\"looks good\")'",
            "python -c 'exit(0)'",
            "python -c 'raise SystemExit(0)'",
            "python -c 'assert True'",
            "python -c 'assert 1 + 1 == 2'",
        ]
        for command in weak_commands:
            with self.subTest(command=command):
                payload = self._payload()
                payload["tasks"][0]["acceptance_checks"] = [command]

                result = TeamPlanTool().run(payload, self.context)
                codes = {issue["code"] for issue in result.output["issues"]}

                self.assertTrue(result.is_error)
                self.assertIn("TRIVIAL_ACCEPTANCE_CHECK", codes)
                self.assertEqual(self.context.tasks, {})

    def test_identical_hash_is_idempotent_before_revision_check(self) -> None:
        payload = self._payload()
        first = TeamPlanTool().run(payload, self.context)
        first_agents = [agent.agent_id for agent in self.context.team_store.list_agents(
            first.output["team_id"]
        )]

        repeated = TeamPlanTool().run(payload, self.context)

        self.assertFalse(repeated.is_error, repeated.output)
        self.assertTrue(repeated.output["idempotent"])
        self.assertEqual(repeated.output["revision"], 1)
        self.assertEqual(repeated.output["plan_hash"], first.output["plan_hash"])
        self.assertEqual(
            [agent.agent_id for agent in self.context.team_store.list_agents(first.output["team_id"])],
            first_agents,
        )

    def test_revision_and_idempotency_conflicts_are_actionable(self) -> None:
        payload = self._payload()
        payload["idempotency_key"] = "request-1"
        first = TeamPlanTool().run(payload, self.context)
        self.assertFalse(first.is_error, first.output)

        stale = copy.deepcopy(payload)
        stale["idempotency_key"] = "request-2"
        stale["contract"]["summary"] += " revision"
        stale_result = TeamPlanTool().run(stale, self.context)
        self.assertEqual(stale_result.output["issues"][0]["code"], "REVISION_CONFLICT")

        reused = copy.deepcopy(stale)
        reused["expected_revision"] = 1
        reused["idempotency_key"] = "request-1"
        reused_result = TeamPlanTool().run(reused, self.context)
        self.assertEqual(
            reused_result.output["issues"][0]["code"], "IDEMPOTENCY_KEY_REUSE"
        )

    def test_repair_required_accepts_explicit_new_revision(self) -> None:
        first = TeamPlanTool().run(self._payload(), self.context)
        self.assertFalse(first.is_error, first.output)
        team_id = first.output["team_id"]
        original_task_ids = set(self.context.tasks)
        completed: dict[str, dict] = {}
        for task_id, data in self.context.tasks.items():
            task = TeamTask.from_dict(data)
            task.transition_to("completed")
            completed[task_id] = task.to_dict()
        self.context.team_store.save_tasks(team_id, completed)
        team = self.context.team_store.load_team(team_id)
        team.transition_to("running")
        team.set_lifecycle_state("repair_required")
        self.context.team_store.save_team(team)
        self.context.reload_team_state()

        repaired = self._payload()
        repaired["expected_revision"] = 1
        repaired["contract"]["summary"] += " with repaired integration"
        repaired["tasks"][2]["instructions"] += " Repair the public contract."
        result = TeamPlanTool().run(repaired, self.context)

        self.assertFalse(result.is_error, result.output)
        self.assertEqual(result.output["revision"], 2)
        self.assertTrue(original_task_ids.isdisjoint(self.context.tasks))
        self.assertTrue(
            all(task["status"] == "pending" for task in self.context.tasks.values())
        )
        stored = self.context.team_store.load_team(team_id)
        self.assertEqual(stored.lifecycle_state, "ready")
        self.assertEqual(
            stored.settings["quality_gates"]["validation"]["status"], "pending"
        )

    def test_new_revision_removes_omitted_execution_settings(self) -> None:
        initial = self._payload()
        initial["execution"].update(
            {
                "timeout_s": 30,
                "token_budget": 500,
                "turn_budget": 20,
                "max_retries": 2,
                "lease_timeout_s": 60,
            }
        )
        first = TeamPlanTool().run(initial, self.context)
        self.assertFalse(first.is_error, first.output)
        team = self.context.team_store.load_active_team()
        team.settings["max_batches"] = 99
        team.settings["unrelated_setting"] = "preserved"
        self.context.team_store.save_team(team)
        self.context.reload_team_state()

        replacement = self._payload()
        replacement["expected_revision"] = 1
        replacement["contract"]["summary"] += " revision two"
        result = TeamPlanTool().run(replacement, self.context)

        self.assertFalse(result.is_error, result.output)
        stored = self.context.team_store.load_active_team()
        for stale in (
            "max_batches",
            "timeout_s",
            "token_budget",
            "turn_budget",
            "max_retries",
            "lease_timeout_s",
        ):
            self.assertNotIn(stale, stored.settings)
        execution = stored.settings["execution_manifest"]["execution"]
        self.assertEqual(execution["max_workers"], 2)
        self.assertIsNone(execution["timeout_s"])
        self.assertIsNone(execution["token_budget"])
        self.assertIsNone(execution["turn_budget"])
        self.assertEqual(execution["max_retries"], 0)
        self.assertEqual(execution["lease_timeout_s"], 900)
        self.assertEqual(execution["verify_timeout_s"], 900)
        self.assertTrue(execution["auto_verify"])
        self.assertEqual(stored.settings["unrelated_setting"], "preserved")

    def test_run_rejects_override_of_frozen_execution_manifest(self) -> None:
        payload = self._payload()
        payload["execution"]["turn_budget"] = 20
        planned = TeamPlanTool().run(payload, self.context)
        self.assertFalse(planned.is_error, planned.output)

        result = TeamRunTool().run({"turn_budget": 21}, self.context)

        self.assertTrue(result.is_error)
        self.assertEqual(result.output["status"], "blocked")
        mismatch = result.output["execution_manifest_mismatches"][0]
        self.assertEqual(mismatch["field"], "turn_budget")
        self.assertEqual(mismatch["planned"], 20)
        self.assertEqual(mismatch["requested"], 21)
        stored = self.context.team_store.load_active_team()
        self.assertNotIn("turn_budget", stored.settings)
        self.assertEqual(
            stored.settings["execution_manifest"]["execution"]["turn_budget"], 20
        )
        event = self.context.team_store.list_events(stored.team_id)[-1]
        self.assertEqual(event["type"], "team.execution_manifest_mismatch")
        self.assertEqual(event["data"]["mismatches"][0]["reason"], mismatch["reason"])

    def test_run_blocks_tampered_budget_ceiling_before_model_call(self) -> None:
        payload = self._payload()
        payload["execution"]["turn_budget"] = 20
        planned = TeamPlanTool().run(payload, self.context)
        self.assertFalse(planned.is_error, planned.output)
        team = self.context.team_store.load_active_team()
        manifest = dict(team.settings["execution_manifest"])
        window = dict(manifest["budget_window"])
        ceiling = dict(window["hard_ceiling"])
        ceiling["turns"] = 999
        window["hard_ceiling"] = ceiling
        manifest["budget_window"] = window
        team.settings["execution_manifest"] = manifest
        self.context.team_store.save_team(team)
        self.context.reload_team_state()

        result = TeamRunTool().run({}, self.context)

        self.assertTrue(result.is_error)
        self.assertEqual(result.output["status"], "blocked")
        self.assertEqual(result.output["failure_domain"], "harness")
        self.assertTrue(result.output["workspace_preserved"])
        self.assertTrue(
            any(
                "hard_ceiling.turns" in error
                or "budget_integrity_hash" in error
                for error in result.output["budget_manifest_errors"]
            )
        )

    def test_repair_budget_is_incremental_but_cannot_raise_global_cap(self) -> None:
        initial = self._payload()
        initial["execution"].update({"token_budget": 1_000, "turn_budget": 100})
        first = TeamPlanTool().run(initial, self.context)
        self.assertFalse(first.is_error, first.output)
        team = self.context.team_store.load_active_team()
        team.usage = {
            "input_tokens": 500,
            "output_tokens": 0,
            "total_tokens": 500,
            "turns": 50,
        }
        quality = dict(team.settings["quality_gates"])
        quality["plan_accepted"] = True
        team.settings["quality_gates"] = quality
        manifest = dict(team.settings["execution_manifest"])
        manifest["status"] = "accepted"
        team.settings["execution_manifest"] = manifest
        team.transition_to("running")
        self.context.team_store.save_team(team)
        self.context.reload_team_state()

        checkpoint = TeamReplanTool().run(
            {"reason": "repair the contract"}, self.context
        )
        self.assertFalse(checkpoint.is_error, checkpoint.output)
        repair = self._payload()
        repair["expected_revision"] = 1
        repair["contract"]["summary"] += " repaired"
        repair["execution"].update({"token_budget": 200, "turn_budget": 20})
        second = TeamPlanTool().run(repair, self.context)
        self.assertFalse(second.is_error, second.output)

        stored = self.context.team_store.load_active_team()
        budget = stored.settings["execution_manifest"]
        self.assertEqual(
            budget["global_cap"], {"total_tokens": 1_000, "turns": 100}
        )
        self.assertEqual(
            budget["budget_window"]["baseline"],
            {"total_tokens": 500, "turns": 50},
        )
        self.assertEqual(
            budget["budget_window"]["hard_ceiling"],
            {"total_tokens": 700, "turns": 70},
        )
        options = TeamRunOptions.build(stored.settings, {})
        self.assertIsNone(
            TeammateRuntime._budget_error(stored, options, time.monotonic())
        )

        quality = dict(stored.settings["quality_gates"])
        quality["plan_accepted"] = True
        stored.settings["quality_gates"] = quality
        manifest = dict(stored.settings["execution_manifest"])
        manifest["status"] = "accepted"
        stored.settings["execution_manifest"] = manifest
        stored.usage.update({"input_tokens": 650, "total_tokens": 650, "turns": 65})
        stored.set_lifecycle_state("running")
        self.context.team_store.save_team(stored)
        self.context.reload_team_state()
        TeamReplanTool().run({"reason": "second repair"}, self.context)
        later = self._payload()
        later["expected_revision"] = 2
        later["contract"]["summary"] += " second repair"
        later["execution"].update({"token_budget": 500, "turn_budget": 50})
        third = TeamPlanTool().run(later, self.context)
        self.assertFalse(third.is_error, third.output)
        final = self.context.team_store.load_active_team()
        final_budget = final.settings["execution_manifest"]
        self.assertEqual(
            final_budget["global_cap"], {"total_tokens": 1_000, "turns": 100}
        )
        self.assertEqual(
            final_budget["budget_window"]["hard_ceiling"],
            {"total_tokens": 1_000, "turns": 100},
        )
        consumed = final.settings["last_replan_checkpoint"]
        self.assertEqual(consumed["consumed_by_revision"], 3)
        self.assertIn("consumed_at", consumed)

    def test_accepted_plan_requires_replan_and_completed_team_is_terminal(self) -> None:
        first = TeamPlanTool().run(self._payload(), self.context)
        self.assertFalse(first.is_error, first.output)
        team = self.context.team_store.load_active_team()
        quality = dict(team.settings["quality_gates"])
        quality["plan_accepted"] = True
        team.settings["quality_gates"] = quality
        manifest = dict(team.settings["execution_manifest"])
        manifest["status"] = "accepted"
        team.settings["execution_manifest"] = manifest
        team.transition_to("running")
        self.context.team_store.save_team(team)
        self.context.reload_team_state()
        replacement = self._payload()
        replacement["expected_revision"] = 1
        replacement["contract"]["summary"] += " unauthorized"

        rejected = TeamPlanTool().run(replacement, self.context)

        self.assertTrue(rejected.is_error)
        self.assertEqual(rejected.output["issues"][0]["code"], "REPLAN_REQUIRED")

        team = self.context.team_store.load_active_team()
        team.transition_to("completed")
        self.context.team_store.save_team(team)
        self.context.reload_team_state()
        terminal = TeamPlanTool().run(self._payload(), self.context)
        self.assertTrue(terminal.is_error)
        self.assertEqual(terminal.output["issues"][0]["code"], "TEAM_TERMINAL")

    def test_repair_rejects_unchanged_plan_and_keeps_checkpoint_unconsumed(self) -> None:
        payload = self._payload()
        first = TeamPlanTool().run(payload, self.context)
        self.assertFalse(first.is_error, first.output)
        team = self.context.team_store.load_active_team()
        quality = dict(team.settings["quality_gates"])
        quality["plan_accepted"] = True
        team.settings["quality_gates"] = quality
        manifest = dict(team.settings["execution_manifest"])
        manifest["status"] = "accepted"
        team.settings["execution_manifest"] = manifest
        self.context.team_store.save_team(team)
        self.context.reload_team_state()
        checkpoint = TeamReplanTool().run(
            {"reason": "repair validation without losing the workspace"}, self.context
        )
        self.assertFalse(checkpoint.is_error, checkpoint.output)

        unchanged = TeamPlanTool().run(payload, self.context)

        self.assertTrue(unchanged.is_error)
        self.assertEqual(unchanged.output["issues"][0]["code"], "REPLAN_REQUIRED")
        self.assertIn("unchanged plan", unchanged.output["issues"][0]["message"])
        stored = self.context.team_store.load_active_team()
        self.assertEqual(stored.lifecycle_state, "repair_required")
        self.assertNotIn(
            "consumed_by_revision", stored.settings["last_replan_checkpoint"]
        )

    def test_plan_rollback_cannot_overwrite_concurrent_task_writer(self) -> None:
        team_id = str(self.context.team["team_id"])
        plan_paused = threading.Event()
        release_plan = threading.Event()
        writer_started = threading.Event()
        writer_done = threading.Event()
        plan_errors: list[Exception] = []
        writer_errors: list[Exception] = []
        original = TeamStore._write_json_unlocked

        def pause_then_fail(path: Path, data: dict) -> None:
            if threading.current_thread().name == "plan-commit" and path.name == "tasks.json":
                plan_paused.set()
                if not release_plan.wait(timeout=3):
                    raise TimeoutError("test did not release plan transaction")
                raise OSError("simulated plan commit failure")
            original(path, data)

        def commit_plan() -> None:
            try:
                TeamPlanTool().run(self._payload(), self.context)
            except Exception as exc:  # expected simulated storage failure
                plan_errors.append(exc)

        external = TeamTask(
            id="external-task",
            key="external",
            subject="External writer",
            description="Must survive plan rollback",
        ).to_dict()

        def write_external_task() -> None:
            writer_started.set()
            try:
                self.context.team_store.save_tasks(
                    team_id, {"external-task": external}
                )
            except Exception as exc:  # pragma: no cover - diagnostic path
                writer_errors.append(exc)
            finally:
                writer_done.set()

        with patch.object(
            TeamStore, "_write_json_unlocked", side_effect=pause_then_fail
        ):
            plan_thread = threading.Thread(target=commit_plan, name="plan-commit")
            plan_thread.start()
            self.assertTrue(plan_paused.wait(timeout=3))
            writer_thread = threading.Thread(
                target=write_external_task, name="task-writer"
            )
            writer_thread.start()
            self.assertTrue(writer_started.wait(timeout=1))
            self.assertFalse(
                writer_done.wait(timeout=0.1),
                "task writer entered while TeamPlan transaction was paused",
            )
            release_plan.set()
            plan_thread.join(timeout=3)
            writer_thread.join(timeout=3)

        self.assertFalse(plan_thread.is_alive())
        self.assertFalse(writer_thread.is_alive())
        self.assertEqual(writer_errors, [])
        self.assertEqual(len(plan_errors), 1)
        self.assertIsInstance(plan_errors[0], OSError)
        self.assertEqual(
            set(self.context.team_store.load_tasks(team_id)), {"external-task"}
        )
        stored = self.context.team_store.load_team(team_id)
        self.assertNotIn("team_plan", stored.settings)
        self.assertEqual(self.context.team_store.list_agents(team_id), [])

    def test_replan_and_claim_are_serialized_by_one_team_transaction(self) -> None:
        planned = TeamPlanTool().run(self._payload(), self.context)
        self.assertFalse(planned.is_error, planned.output)
        team = self.context.team_store.load_active_team()
        team.transition_to("running")
        team.set_lifecycle_state("running")
        self.context.team_store.save_team(team)
        self.context.reload_team_state()
        task_id = next(iter(self.context.tasks))
        plan_hash = planned.output["plan_hash"]
        replan_paused = threading.Event()
        release_replan = threading.Event()
        claim_started = threading.Event()
        claim_done = threading.Event()
        claims: list[TeamTask | None] = []
        replan_errors: list[Exception] = []
        original = TeamStore._write_json_unlocked

        def pause_team_write(path: Path, data: dict) -> None:
            if (
                threading.current_thread().name == "replan"
                and path.name == "team.json"
                and path.parent.name == team.team_id
            ):
                replan_paused.set()
                if not release_replan.wait(timeout=3):
                    raise TimeoutError("test did not release TeamReplan")
            original(path, data)

        def request_replan() -> None:
            try:
                TeamReplanTool().run(
                    {"reason": "replace the invalid partition"}, self.context
                )
            except Exception as exc:  # pragma: no cover - diagnostic path
                replan_errors.append(exc)

        def claim_task() -> None:
            claim_started.set()
            claims.append(
                self.context.team_store.claim_task(
                    team.team_id,
                    task_id,
                    lease_id="late-claim",
                    lease_expires_at="2999-01-01T00:00:00+00:00",
                    max_retries=0,
                    expected_plan_hash=plan_hash,
                )
            )
            claim_done.set()

        with patch.object(
            TeamStore, "_write_json_unlocked", side_effect=pause_team_write
        ):
            replan_thread = threading.Thread(target=request_replan, name="replan")
            replan_thread.start()
            self.assertTrue(replan_paused.wait(timeout=3))
            claim_thread = threading.Thread(target=claim_task, name="claim")
            claim_thread.start()
            self.assertTrue(claim_started.wait(timeout=1))
            self.assertFalse(
                claim_done.wait(timeout=0.1),
                "task claim entered while TeamReplan transaction was paused",
            )
            release_replan.set()
            replan_thread.join(timeout=3)
            claim_thread.join(timeout=3)

        self.assertEqual(replan_errors, [])
        self.assertEqual(claims, [None])
        stored = self.context.team_store.load_active_team()
        self.assertEqual(stored.lifecycle_state, "repair_required")
        self.assertEqual(
            self.context.team_store.load_tasks(team.team_id)[task_id]["status"],
            "pending",
        )

    def test_stale_worker_outcome_cannot_resurrect_replaced_revision_task(self) -> None:
        first = TeamPlanTool().run(self._payload(), self.context)
        self.assertFalse(first.is_error, first.output)
        team = self.context.team_store.load_active_team()
        old_task_id = next(iter(self.context.tasks))
        old_task = TeamTask.from_dict(self.context.tasks[old_task_id])
        TeamReplanTool().run({"reason": "replace the task graph"}, self.context)
        replacement = self._payload()
        replacement["expected_revision"] = 1
        replacement["contract"]["summary"] += " with a revised boundary"
        second = TeamPlanTool().run(replacement, self.context)
        self.assertFalse(second.is_error, second.output)
        new_task_ids = set(self.context.tasks)
        self.assertNotIn(old_task_id, new_task_ids)

        old_task.transition_to("in_progress")
        old_task.transition_to("completed")
        old_task.output = "late result from revision one"
        self.context.team_store.update_task(
            team.team_id,
            old_task,
            expected_plan_hash=first.output["plan_hash"],
        )

        stored_tasks = self.context.team_store.load_tasks(team.team_id)
        self.assertEqual(set(stored_tasks), new_task_ids)
        self.assertNotIn(old_task_id, stored_tasks)
        self.assertTrue(
            all(task["status"] == "pending" for task in stored_tasks.values())
        )

    def test_produced_only_task_is_not_carried_forward(self) -> None:
        first = TeamPlanTool().run(self._payload(), self.context)
        self.assertFalse(first.is_error, first.output)
        team = self.context.team_store.load_active_team()
        task_id = next(iter(self.context.tasks))
        produced = TeamTask.from_dict(self.context.tasks[task_id])
        produced.transition_to("completed")
        produced.output = "candidate artifact without harness acceptance"
        self.context.team_store.update_task(team.team_id, produced)
        TeamReplanTool().run(
            {
                "reason": "repair validation after an unaccepted delivery",
                "replace_completed_work": True,
            },
            self.context,
        )
        replacement = self._payload()
        replacement["expected_revision"] = 1
        replacement["validation"]["commands"] = [
            "python -c \"import json; assert json.loads('2') == 2\""
        ]

        second = TeamPlanTool().run(replacement, self.context)

        self.assertFalse(second.is_error, second.output)
        self.assertEqual(second.output["carried_forward_tasks"], [])
        self.assertTrue(
            all(task["status"] == "pending" for task in self.context.tasks.values())
        )

    def test_contract_change_invalidates_previously_accepted_tasks(self) -> None:
        first = TeamPlanTool().run(self._payload(), self.context)
        self.assertFalse(first.is_error, first.output)
        team = self.context.team_store.load_active_team()
        for raw in self.context.tasks.values():
            accepted = TeamTask.from_dict(raw)
            accepted.transition_to("completed")
            accepted.set_lifecycle_state("accepted")
            accepted.output = f"accepted artifact for {accepted.key}"
            accepted.metadata = dict(accepted.metadata)
            accepted.metadata["acceptance"] = {
                "status": "passed",
                "checked_at": "2026-01-01T00:00:00+00:00",
                "stages": [
                    {"command": command, "exit_code": 0}
                    for command in accepted.acceptance_checks
                ],
            }
            self.context.team_store.update_task(team.team_id, accepted)
        TeamReplanTool().run(
            {
                "reason": "replace the shared interface contract",
                "replace_completed_work": True,
            },
            self.context,
        )
        replacement = self._payload()
        replacement["expected_revision"] = 1
        replacement["contract"]["summary"] += " with a breaking revision"

        second = TeamPlanTool().run(replacement, self.context)

        self.assertFalse(second.is_error, second.output)
        self.assertEqual(second.output["carried_forward_tasks"], [])
        self.assertTrue(
            all(task["status"] == "pending" for task in self.context.tasks.values())
        )

    def test_team_transaction_lock_is_reentrant_inside_mutator(self) -> None:
        team = self.context.team_store.load_active_team()
        observed: list[str] = []

        def mutate(tasks: dict[str, TeamTask]) -> None:
            self.context.team_store.save_team(team)
            observed.append(team.team_id)

        self.context.team_store.mutate_tasks(team.team_id, mutate)

        self.assertEqual(observed, [team.team_id])
        self.assertEqual(self.context.team_store.load_tasks(team.team_id), {})

    def test_storage_failure_rolls_back_every_materialized_file(self) -> None:
        payload = self._payload()
        original = TeamStore._write_json_unlocked

        def fail_on_session(path: Path, data: dict) -> None:
            if path.parent.name == "sessions":
                raise OSError("simulated session write failure")
            original(path, data)

        with patch.object(TeamStore, "_write_json_unlocked", side_effect=fail_on_session):
            with self.assertRaisesRegex(OSError, "simulated session"):
                TeamPlanTool().run(payload, self.context)

        team = self.context.team_store.load_active_team()
        self.assertNotIn("team_plan", team.settings)
        self.assertEqual(self.context.team_store.load_tasks(team.team_id), {})
        self.assertEqual(self.context.team_store.list_agents(team.team_id), [])
        self.assertEqual(
            list((self.context.team_store.team_dir(team.team_id) / "sessions").glob("*.json")),
            [],
        )


if __name__ == "__main__":
    unittest.main()
