from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent.conversation import Conversation
from src.providers.base import ChatResponse
from src.teammate.models import TeamTask
from src.teammate.runtime import TeammateRuntime
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.errors import ToolInputError
from src.tool_system.tools import (
    TaskCreateTool,
    TeamAbortTool,
    TeamConfigureTool,
    TeamCreateTool,
    TeamDeleteTool,
    TeamPlanTool,
    TeamReplanTool,
    TeamResumeTool,
    TeamRunTool,
    TeamVerifyTool,
    TeammateCreateTool,
)


class FinalProvider:
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        return ChatResponse(
            content="implemented and checked",
            model=self.model,
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="stop",
            tool_uses=None,
        )


class TestTeamQualityGates(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.registry = build_default_registry(include_user_tools=False)
        self.provider = FinalProvider()
        self.context = ToolContext(workspace_root=self.root)
        self.context.teammate_runtime = TeammateRuntime(self.provider, self.registry)
        TeamCreateTool().run(
            {"team_name": "strict", "quality_gates": True}, self.context
        )
        TeamConfigureTool().run(
            {
                "architecture_contract": "samplepkg owns the public VALUE contract",
                "install_command": (
                    "python -m pip install -e . --no-deps --no-build-isolation"
                ),
                "import_command": "python -c \"import samplepkg\"",
                "integration_command": (
                    "python -c \"import samplepkg; assert samplepkg.VALUE == 1\""
                ),
            },
            self.context,
        )
        for name in ("one", "two"):
            TeammateCreateTool().run(
                {
                    "name": name,
                    "role": "implementation",
                    "instructions": f"Implement the {name} partition.",
                    "tools": ["Read", "Write", "Bash"],
                },
                self.context,
            )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _task(
        self,
        key: str,
        owner: str,
        path: str,
        *,
        blocked_by: list[str] | None = None,
        provides: list[str] | None = None,
        depends: list[str] | None = None,
    ) -> str:
        payload = {
            "key": key,
            "subject": key,
            "description": f"Implement {key}",
            "owner": owner,
            "ownedFiles": [path],
            "acceptanceChecks": [f"python -m py_compile {path}"],
            "providesInterfaces": provides or [],
            "dependsOnInterfaces": depends or [],
        }
        if blocked_by:
            payload["blockedBy"] = blocked_by
        return TaskCreateTool().run(payload, self.context).output["task"]["id"]

    def _write_package(self) -> None:
        (self.root / "samplepkg").mkdir()
        (self.root / "samplepkg" / "__init__.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        (self.root / "helper.py").write_text("HELPER = True\n", encoding="utf-8")
        (self.root / "pyproject.toml").write_text(
            "[build-system]\n"
            "requires = ['setuptools']\n"
            "build-backend = 'setuptools.build_meta'\n"
            "[project]\n"
            "name = 'strict-team-fixture'\n"
            "version = '0.0.1'\n",
            encoding="utf-8",
        )

    def _mark_v2_without_plan(self) -> None:
        team = self.context.team_store.load_active_team()
        self.assertIsNotNone(team)
        team.protocol_version = 2
        team.settings["protocol_version"] = 2
        quality = dict(team.settings.get("quality_gates") or {})
        quality["protocol_version"] = 2
        team.settings["quality_gates"] = quality
        team.set_lifecycle_state("ready")
        self.context.team_store.save_team(team)
        self.context.reload_team_state()

    def _submit_v2_plan(
        self,
        tasks: list[dict],
        *,
        integration_command: str = (
            "python -c \"import samplepkg; assert samplepkg.VALUE == 1\""
        ),
        execution: dict | None = None,
        expected_revision: int | None = None,
        contract_summary: str = "Two independent implementation partitions",
    ):
        tasks = [
            {
                "instructions": f"Implement and verify {task.get('key', 'the task')}.",
                **task,
            }
            for task in tasks
        ]
        payload = {
                "mode": "replace",
                "contract": {
                    "summary": contract_summary,
                    "interfaces": [],
                },
                "workers": [
                    {"name": "one", "instructions": "Implement partition one."},
                    {"name": "two", "instructions": "Implement partition two."},
                ],
                "tasks": tasks,
                "validation": {
                    "profile": "generic",
                    "install_command": "true",
                    "import_command": "python -c \"import samplepkg\"",
                    "integration_command": integration_command,
                },
                "execution": execution or {},
            }
        if expected_revision is not None:
            payload["expected_revision"] = expected_revision
        result = TeamPlanTool().run(payload, self.context)
        self.assertFalse(result.is_error, result.output)
        self.assertEqual(result.output["status"], "ready")
        return result

    def test_rejects_single_worker_and_overlapping_ownership_before_model_calls(self) -> None:
        self._task("first", "one", "samplepkg")
        self._task("second", "two", "samplepkg/__init__.py")

        result = TeamRunTool().run({"max_workers": 2}, self.context)

        self.assertTrue(result.is_error)
        self.assertEqual(result.output["status"], "blocked")
        self.assertIn("ownedFiles overlap", result.output["error"])
        self.assertEqual(self.provider.calls, 0)

    def test_completed_tasks_require_clean_validation_before_team_completion(self) -> None:
        self._write_package()
        self._task("package", "one", "samplepkg")
        self._task("helper", "two", "helper.py")

        rollout = TeamRunTool().run({"max_workers": 2}, self.context)

        self.assertFalse(rollout.is_error)
        self.assertEqual(rollout.output["status"], "verification_required")
        self.assertEqual(self.context.team_store.load_active_team().status, "running")

        verified = TeamVerifyTool().run({"timeout_s": 120}, self.context)

        self.assertFalse(verified.is_error, verified.output)
        self.assertEqual(verified.output["status"], "completed")
        self.assertEqual(verified.output["validation"]["status"], "passed")
        self.assertEqual(
            [stage["stage"] for stage in verified.output["validation"]["stages"]],
            ["bootstrap", "install", "import", "integration"],
        )

    def test_interface_dependency_requires_peer_coordination(self) -> None:
        self._task("provider", "one", "provider.py", provides=["public-api"])
        self._task("independent", "two", "independent.py")
        self._task(
            "consumer",
            "two",
            "consumer.py",
            blocked_by=["provider"],
            depends=["public-api"],
        )

        result = TeamRunTool().run({"max_workers": 2}, self.context)

        self.assertTrue(result.is_error)
        self.assertEqual(result.output["status"], "blocked")
        self.assertIn("peer message", result.output["error"])

    def test_v2_team_run_accepts_tasks_and_verifies_automatically(self) -> None:
        self._write_package()
        self._submit_v2_plan(
            [
                {
                    "key": "package",
                    "owner": "one",
                    "owned_files": ["samplepkg/__init__.py"],
                    "acceptance_checks": [
                        "python -m py_compile samplepkg/__init__.py"
                    ],
                },
                {
                    "key": "helper",
                    "owner": "two",
                    "owned_files": ["helper.py"],
                    "acceptance_checks": ["python -m py_compile helper.py"],
                },
            ]
        )

        rollout = TeamRunTool().run({"max_workers": 2}, self.context)

        self.assertFalse(rollout.is_error, rollout.output)
        self.assertEqual(rollout.output["status"], "completed")
        self.assertEqual(rollout.output["lifecycle_state"], "completed")
        self.context.reload_team_state()
        self.assertEqual(
            {task["lifecycle_state"] for task in self.context.tasks.values()},
            {"accepted"},
        )
        events_before = self.context.team_store.list_events(
            str(self.context.team["team_id"])
        )
        verified = TeamVerifyTool().run({"timeout_s": 120}, self.context)
        events_after = self.context.team_store.list_events(
            str(self.context.team["team_id"])
        )
        self.assertFalse(verified.is_error, verified.output)
        self.assertTrue(verified.output["verification_reused"])
        self.assertEqual(len(events_after), len(events_before))

    def test_v2_verification_infrastructure_failure_cleans_up_and_pauses(self) -> None:
        self._submit_v2_plan(
            [
                {
                    "key": "package",
                    "owner": "one",
                    "owned_files": ["samplepkg/__init__.py"],
                    "acceptance_checks": [
                        "python -m py_compile samplepkg/__init__.py"
                    ],
                },
                {
                    "key": "helper",
                    "owner": "two",
                    "owned_files": ["helper.py"],
                    "acceptance_checks": ["python -m py_compile helper.py"],
                },
            ]
        )
        team = self.context.team_store.load_active_team()
        self.assertIsNotNone(team)
        for raw in self.context.team_store.load_tasks(team.team_id).values():
            task = TeamTask.from_dict(raw)
            task.transition_to("in_progress")
            task.attempt = 1
            task.transition_to("completed")
            task.set_lifecycle_state("accepted")
            self.context.team_store.update_task(team.team_id, task)
        quality = dict(team.settings["quality_gates"])
        quality["plan_accepted"] = True
        team.settings["quality_gates"] = quality
        team.transition_to("running")
        self.context.team_store.save_team(team)
        self.context.workspace_backend = object()
        self.context.execution_workspace_root = "/workspace"
        self.context.execution_cwd = "/workspace"
        self.context.reload_team_state()

        with patch.object(
            TeammateRuntime,
            "_run_validation_command",
            side_effect=ConnectionError("service unavailable during verification"),
        ), patch.object(
            TeammateRuntime, "_cleanup_validation_root"
        ) as cleanup:
            verified = TeamVerifyTool().run({"timeout_s": 120}, self.context)

        self.assertTrue(verified.is_error)
        self.assertEqual(verified.output["status"], "paused")
        self.assertEqual(verified.output["failure_domain"], "infrastructure")
        self.assertTrue(verified.output["retryable"])
        cleanup.assert_called_once()
        validation_root = cleanup.call_args.args[1]
        self.assertTrue(
            validation_root.startswith(f"/tmp/clawd-team-verify-{team.team_id}-")
        )
        self.assertNotEqual(validation_root, f"/tmp/clawd-team-verify-{team.team_id}")
        stored = self.context.team_store.load_active_team()
        self.assertEqual(stored.lifecycle_state, "paused")
        self.assertEqual(
            stored.settings["quality_gates"]["validation"]["status"], "paused"
        )

    def test_same_owner_nested_paths_do_not_trigger_write_conflict(self) -> None:
        self._write_package()
        self._task("package", "one", "samplepkg")
        self._task("module", "one", "samplepkg/__init__.py")
        self._task("helper", "two", "helper.py")

        rollout = TeamRunTool().run({"max_workers": 2}, self.context)

        self.assertFalse(rollout.is_error, rollout.output)
        self.assertEqual(rollout.output["status"], "verification_required")

    def test_v2_validation_task_may_have_no_owned_files(self) -> None:
        self._write_package()
        self._submit_v2_plan(
            [
                {
                    "key": "package",
                    "owner": "one",
                    "owned_files": ["samplepkg/__init__.py"],
                    "acceptance_checks": [
                        "python -m py_compile samplepkg/__init__.py"
                    ],
                },
                {
                    "key": "helper",
                    "owner": "two",
                    "owned_files": ["helper.py"],
                    "acceptance_checks": ["python -m py_compile helper.py"],
                },
                {
                    "key": "validation",
                    "kind": "validation",
                    "owner": "two",
                    "owned_files": [],
                    "acceptance_checks": ["python -c \"import samplepkg\""],
                },
            ]
        )

        rollout = TeamRunTool().run({"max_workers": 2}, self.context)

        self.assertFalse(rollout.is_error, rollout.output)
        self.assertEqual(rollout.output["status"], "completed")

    def test_v2_failed_validation_requires_repair_and_cannot_be_deleted(self) -> None:
        self._write_package()
        self._submit_v2_plan(
            [
                {
                    "key": "package",
                    "owner": "one",
                    "owned_files": ["samplepkg/__init__.py"],
                    "acceptance_checks": [
                        "python -m py_compile samplepkg/__init__.py"
                    ],
                },
                {
                    "key": "helper",
                    "owner": "two",
                    "owned_files": ["helper.py"],
                    "acceptance_checks": ["python -m py_compile helper.py"],
                },
            ],
            integration_command="python -c \"raise SystemExit(3)\"",
        )

        rollout = TeamRunTool().run({"max_workers": 2}, self.context)

        self.assertEqual(rollout.output["status"], "repair_required")
        self.assertEqual(
            self.context.team_store.load_active_team().lifecycle_state,
            "repair_required",
        )
        stale_retry = TeamRunTool().run({"max_workers": 2}, self.context)
        self.assertTrue(stale_retry.is_error)
        self.assertIn("new TeamPlan revision", stale_retry.output["error"])
        self.assertEqual(
            self.context.team_store.load_active_team().lifecycle_state,
            "repair_required",
        )
        deleted = TeamDeleteTool().run({}, self.context)
        self.assertTrue(deleted.is_error)
        self.assertFalse(deleted.output["success"])
        self.assertEqual(deleted.output["next_required_action"], "TeamReplan")
        self.assertNotIn("call TeamAbort", deleted.output["message"])
        self.assertIsNotNone(self.context.team_store.load_active_team())

        aborted = TeamAbortTool().run(
            {"reason": "integration contract cannot be repaired"}, self.context
        )
        self.assertEqual(aborted.output["status"], "aborted")
        self.assertEqual(
            self.context.team_store.load_active_team().lifecycle_state, "aborted"
        )
        self.assertTrue(
            any(
                event["type"] == "team.aborted"
                for event in self.context.team_store.list_events(
                    str(self.context.team["team_id"])
                )
            )
        )

    def test_validation_failure_replan_revised_plan_then_completes(self) -> None:
        self._write_package()
        tasks = [
            {
                "key": "package",
                "owner": "one",
                "owned_files": ["samplepkg/__init__.py"],
                "acceptance_checks": [
                    "python -m py_compile samplepkg/__init__.py"
                ],
            },
            {
                "key": "helper",
                "owner": "two",
                "owned_files": ["helper.py"],
                "acceptance_checks": ["python -m py_compile helper.py"],
            },
        ]
        first = self._submit_v2_plan(
            tasks,
            integration_command="python -c \"raise SystemExit(9)\"",
            execution={"turn_budget": 20},
        )

        failed = TeamRunTool().run({}, self.context)

        self.assertTrue(failed.is_error)
        self.assertEqual(failed.output["status"], "repair_required")
        self.assertIn("TeamReplan first", failed.output["next_required_action"])
        calls_after_first_revision = self.provider.calls
        usage_after_first_revision = dict(
            self.context.team_store.load_active_team().usage
        )
        prior_task_ids = set(self.context.tasks)
        prior_owner_ids = {task["owner"] for task in self.context.tasks.values()}
        checkpointed = TeamReplanTool().run(
            {
                "reason": "replace the failing integration contract",
                "replace_completed_work": True,
            },
            self.context,
        )
        self.assertEqual(
            checkpointed.output["checkpoint"]["plan_hash"],
            first.output["plan_hash"],
        )
        second = self._submit_v2_plan(
            tasks,
            expected_revision=1,
            execution={"turn_budget": 20},
        )
        self.assertNotEqual(second.output["plan_hash"], first.output["plan_hash"])
        self.assertEqual(
            {item["key"] for item in second.output["carried_forward_tasks"]},
            {"package", "helper"},
        )
        self.context.reload_team_state()
        self.assertTrue(prior_task_ids.isdisjoint(self.context.tasks))
        self.assertTrue(
            prior_owner_ids.isdisjoint(
                {task["owner"] for task in self.context.tasks.values()}
            )
        )
        self.assertTrue(
            all(
                task["metadata"]["plan_hash"] == second.output["plan_hash"]
                for task in self.context.tasks.values()
            )
        )
        self.assertEqual(
            {
                (task["status"], task["lifecycle_state"])
                for task in self.context.tasks.values()
            },
            {("completed", "produced")},
        )
        self.assertTrue(
            all(
                "acceptance" not in task["metadata"]
                and task["metadata"]["carry_forward"]["requires_acceptance"]
                for task in self.context.tasks.values()
            )
        )

        completed = TeamRunTool().run({}, self.context)

        self.assertFalse(completed.is_error, completed.output)
        self.assertEqual(completed.output["status"], "completed")
        self.assertEqual(self.provider.calls, calls_after_first_revision)
        stored = self.context.team_store.load_active_team()
        self.assertEqual(stored.lifecycle_state, "completed")
        self.assertEqual(stored.usage, usage_after_first_revision)
        self.assertEqual(
            stored.settings["last_replan_checkpoint"]["consumed_by_revision"], 2
        )
        self.context.reload_team_state()
        self.assertTrue(
            all(
                task["lifecycle_state"] == "accepted"
                and task["metadata"]["acceptance"]["status"] == "passed"
                and task["metadata"]["acceptance"]["checked_at"]
                != task["metadata"]["carry_forward"]["accepted_evidence"][
                    "checked_at"
                ]
                for task in self.context.tasks.values()
            )
        )
        self.assertTrue(
            any(
                event["type"] == "team.tasks_carried_forward"
                for event in self.context.team_store.list_events(stored.team_id)
            )
        )

    def test_changed_task_invalidates_its_dependency_closure_only(self) -> None:
        self._write_package()
        (self.root / "consumer.py").write_text(
            "CONSUMER = True\n", encoding="utf-8"
        )
        tasks = [
            {
                "key": "package",
                "owner": "one",
                "owned_files": ["samplepkg/__init__.py"],
                "acceptance_checks": [
                    "python -m py_compile samplepkg/__init__.py"
                ],
            },
            {
                "key": "helper",
                "owner": "two",
                "owned_files": ["helper.py"],
                "acceptance_checks": ["python -m py_compile helper.py"],
            },
            {
                "key": "consumer",
                "owner": "two",
                "blocked_by": ["package"],
                "owned_files": ["consumer.py"],
                "acceptance_checks": ["python -m py_compile consumer.py"],
            },
        ]
        self._submit_v2_plan(
            tasks,
            integration_command="python -c \"raise SystemExit(7)\"",
        )
        failed = TeamRunTool().run({}, self.context)
        self.assertEqual(failed.output["status"], "repair_required")
        calls_after_first_revision = self.provider.calls
        TeamReplanTool().run(
            {
                "reason": "repair package implementation and revalidate consumers",
                "replace_completed_work": True,
            },
            self.context,
        )
        revised_tasks = [dict(task) for task in tasks]
        revised_tasks[0]["instructions"] = "Repair the package implementation."

        second = self._submit_v2_plan(
            revised_tasks,
            expected_revision=1,
        )

        self.assertEqual(
            [item["key"] for item in second.output["carried_forward_tasks"]],
            ["helper"],
        )
        self.context.reload_team_state()
        by_key = {task["key"]: task for task in self.context.tasks.values()}
        self.assertEqual(by_key["helper"]["lifecycle_state"], "produced")
        self.assertEqual(by_key["package"]["status"], "pending")
        self.assertEqual(by_key["consumer"]["status"], "pending")
        self.assertEqual(by_key["consumer"]["blockedBy"], [by_key["package"]["id"]])

        completed = TeamRunTool().run({}, self.context)

        self.assertFalse(completed.is_error, completed.output)
        self.assertEqual(completed.output["status"], "completed")
        self.assertEqual(self.provider.calls - calls_after_first_revision, 2)

    def test_budget_exhaustion_is_terminal_and_agent_loop_fails(self) -> None:
        self._write_package()
        self._submit_v2_plan(
            [
                {
                    "key": "package",
                    "owner": "one",
                    "owned_files": ["samplepkg/__init__.py"],
                    "acceptance_checks": [
                        "python -m py_compile samplepkg/__init__.py"
                    ],
                },
                {
                    "key": "helper",
                    "owner": "two",
                    "owned_files": ["helper.py"],
                    "acceptance_checks": ["python -m py_compile helper.py"],
                },
            ],
            execution={"turn_budget": 1},
        )
        marker = self.root / "budget-terminal-workspace.txt"
        marker.write_text("preserved\n", encoding="utf-8")

        exhausted = TeamRunTool().run({}, self.context)

        self.assertTrue(exhausted.is_error)
        self.assertEqual(exhausted.output["status"], "budget_exhausted")
        self.assertTrue(exhausted.output["terminal"])
        self.assertFalse(exhausted.output["resume_allowed"])
        self.assertFalse(exhausted.output["replan_allowed"])
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserved\n")
        stored = self.context.team_store.load_active_team()
        self.assertEqual(stored.status, "failed")
        self.assertEqual(stored.lifecycle_state, "budget_exhausted")
        calls_after_exhaustion = self.provider.calls

        resumed = TeamResumeTool().run({}, self.context)
        self.assertTrue(resumed.is_error)
        self.assertEqual(resumed.output["status"], "budget_exhausted")
        self.assertEqual(self.provider.calls, calls_after_exhaustion)
        with self.assertRaisesRegex(ToolInputError, "terminal"):
            TeamReplanTool().run(
                {"reason": "try to add budget after the cap"}, self.context
            )

        conversation = Conversation()
        conversation.add_user_message("Finish the remaining team work")
        loop_result = run_agent_loop(
            conversation,
            self.provider,
            self.registry,
            self.context,
            max_turns=2,
        )
        self.assertTrue(loop_result.failed)
        self.assertEqual(loop_result.failure_reason, "team_budget_exhausted")

    def test_team_replan_preserves_workspace_and_records_checkpoint(self) -> None:
        self._write_package()
        self._submit_v2_plan(
            [
                {
                    "key": "package",
                    "owner": "one",
                    "owned_files": ["samplepkg/__init__.py"],
                    "acceptance_checks": [
                        "python -m py_compile samplepkg/__init__.py"
                    ],
                },
                {
                    "key": "helper",
                    "owner": "two",
                    "owned_files": ["helper.py"],
                    "acceptance_checks": ["python -m py_compile helper.py"],
                },
            ]
        )
        team = self.context.team_store.load_active_team()
        self.assertIsNotNone(team)
        team.set_lifecycle_state("repair_required")
        self.context.team_store.save_team(team)
        tasks_before = self.context.team_store.load_tasks(team.team_id)
        marker = self.root / "keep-this-artifact.txt"
        marker.write_text("preserve me\n", encoding="utf-8")

        replanned = TeamReplanTool().run(
            {"reason": "replace an invalid ownership split"}, self.context
        )

        self.assertFalse(replanned.is_error, replanned.output)
        self.assertEqual(replanned.output["status"], "replan_required")
        self.assertEqual(replanned.output["workspace_action"], "none")
        self.assertTrue(replanned.output["workspace_preserved"])
        self.assertTrue(replanned.output["artifacts_preserved"])
        self.assertEqual(replanned.output["checkpoint"]["plan_revision"], 1)
        self.assertEqual(
            self.context.team_store.load_tasks(team.team_id), tasks_before
        )
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me\n")
        stored = self.context.team_store.load_active_team()
        self.assertEqual(stored.lifecycle_state, "repair_required")
        self.assertEqual(
            stored.settings["last_replan_checkpoint"]["checkpoint_id"],
            replanned.output["checkpoint"]["checkpoint_id"],
        )
        self.assertTrue(
            any(
                event["type"] == "team.replan_requested"
                for event in self.context.team_store.list_events(team.team_id)
            )
        )
        self.assertIsNotNone(self.registry.get("TeamReplan"))
        self.assertIs(self.registry.get("TeamReset"), self.registry.get("TeamReplan"))

    def test_team_replan_requires_opt_in_to_replace_produced_work(self) -> None:
        self._write_package()
        self._submit_v2_plan(
            [
                {
                    "key": "package",
                    "owner": "one",
                    "owned_files": ["samplepkg/__init__.py"],
                    "acceptance_checks": [
                        "python -m py_compile samplepkg/__init__.py"
                    ],
                },
                {
                    "key": "helper",
                    "owner": "two",
                    "owned_files": ["helper.py"],
                    "acceptance_checks": ["python -m py_compile helper.py"],
                },
            ]
        )
        team = self.context.team_store.load_active_team()
        self.assertIsNotNone(team)
        tasks = self.context.team_store.load_tasks(team.team_id)
        task_id = next(iter(tasks))
        produced = TeamTask.from_dict(tasks[task_id])
        produced.transition_to("in_progress")
        produced.transition_to("completed")
        produced.output = "implemented package API"
        self.context.team_store.update_task(team.team_id, produced)
        team.set_lifecycle_state("repair_required")
        self.context.team_store.save_team(team)

        with self.assertRaisesRegex(ToolInputError, "replace_completed_work=true"):
            TeamReplanTool().run({"reason": "start over"}, self.context)

        self.assertEqual(
            self.context.team_store.load_tasks(team.team_id)[task_id]["output"],
            "implemented package API",
        )
        replanned = TeamReplanTool().run(
            {
                "reason": "the accepted interface split is incorrect",
                "replace_completed_work": True,
            },
            self.context,
        )
        self.assertIn(produced.key, replanned.output["checkpoint"]["artifact_tasks"])
        self.assertEqual(
            self.context.team_store.load_tasks(team.team_id)[task_id]["output"],
            "implemented package API",
        )

    def test_team_replan_rejects_completed_team(self) -> None:
        self._write_package()
        self._submit_v2_plan(
            [
                {
                    "key": "package",
                    "owner": "one",
                    "owned_files": ["samplepkg/__init__.py"],
                    "acceptance_checks": [
                        "python -m py_compile samplepkg/__init__.py"
                    ],
                },
                {
                    "key": "helper",
                    "owner": "two",
                    "owned_files": ["helper.py"],
                    "acceptance_checks": ["python -m py_compile helper.py"],
                },
            ]
        )
        completed = TeamRunTool().run({"max_workers": 2}, self.context)
        self.assertEqual(completed.output["status"], "completed")

        with self.assertRaisesRegex(ToolInputError, "completed team"):
            TeamReplanTool().run(
                {
                    "reason": "accidental restart",
                    "replace_completed_work": True,
                },
                self.context,
            )

        stored = self.context.team_store.load_active_team()
        self.assertEqual(stored.lifecycle_state, "completed")

    def test_team_replan_requires_workers_to_stop_without_suggesting_abort(self) -> None:
        self._submit_v2_plan(
            [
                {
                    "key": "package",
                    "owner": "one",
                    "owned_files": ["samplepkg/__init__.py"],
                    "acceptance_checks": [
                        "python -m py_compile samplepkg/__init__.py"
                    ],
                },
                {
                    "key": "helper",
                    "owner": "two",
                    "owned_files": ["helper.py"],
                    "acceptance_checks": ["python -m py_compile helper.py"],
                },
            ]
        )
        team = self.context.team_store.load_active_team()
        self.assertIsNotNone(team)
        tasks = self.context.team_store.load_tasks(team.team_id)
        task_id = next(iter(tasks))
        active = TeamTask.from_dict(tasks[task_id])
        active.transition_to("in_progress")
        self.context.team_store.update_task(team.team_id, active)

        with self.assertRaises(ToolInputError) as raised:
            TeamReplanTool().run({"reason": "restart workers"}, self.context)

        message = str(raised.exception)
        self.assertIn("TeamCancel", message)
        self.assertIn("TeamReplan", message)
        self.assertIn("Do not use TeamAbort", message)

    def test_aborted_v2_team_finishes_agent_loop_as_failure(self) -> None:
        self._write_package()
        self._submit_v2_plan(
            [
                {
                    "key": "package",
                    "owner": "one",
                    "owned_files": ["samplepkg/__init__.py"],
                    "acceptance_checks": [
                        "python -m py_compile samplepkg/__init__.py"
                    ],
                },
                {
                    "key": "helper",
                    "owner": "two",
                    "owned_files": ["helper.py"],
                    "acceptance_checks": ["python -m py_compile helper.py"],
                },
            ]
        )
        TeamAbortTool().run({"reason": "unrecoverable plan"}, self.context)
        replan = TeamPlanTool().run(
            {
                "mode": "replace",
                "contract": {"summary": "replacement", "interfaces": []},
                "workers": [
                    {"name": "one", "instructions": "Retry partition one."},
                    {"name": "two", "instructions": "Retry partition two."},
                ],
                "tasks": [
                    {
                        "key": "package",
                        "owner": "one",
                        "instructions": "Retry package.",
                        "owned_files": ["samplepkg/__init__.py"],
                        "acceptance_checks": [
                            "python -m py_compile samplepkg/__init__.py"
                        ],
                    },
                    {
                        "key": "helper",
                        "owner": "two",
                        "instructions": "Retry helper.",
                        "owned_files": ["helper.py"],
                        "acceptance_checks": ["python -m py_compile helper.py"],
                    },
                ],
                "validation": {
                    "profile": "generic",
                    "install_command": "true",
                    "import_command": "python -c \"import samplepkg\"",
                    "integration_command": "python -c \"import samplepkg\"",
                },
            },
            self.context,
        )
        resumed = TeamResumeTool().run({}, self.context)
        self.assertTrue(replan.is_error)
        self.assertIn("aborted", str(replan.output))
        self.assertTrue(resumed.is_error)
        self.assertEqual(resumed.output["status"], "aborted")
        self.assertEqual(
            self.context.team_store.load_active_team().lifecycle_state, "aborted"
        )
        conversation = Conversation()
        conversation.add_user_message("Implement the task")

        result = run_agent_loop(
            conversation,
            self.provider,
            self.registry,
            self.context,
            max_turns=2,
        )

        self.assertTrue(result.failed)
        self.assertEqual(result.failure_reason, "team_aborted")

    def test_v2_rejects_legacy_incremental_plan(self) -> None:
        self._write_package()
        self._task("package", "one", "samplepkg/__init__.py")
        self._task("helper", "two", "helper.py")
        self._mark_v2_without_plan()

        rollout = TeamRunTool().run({"max_workers": 2}, self.context)

        self.assertTrue(rollout.is_error)
        self.assertEqual(rollout.output["status"], "blocked")
        self.assertIn("atomic TeamPlan", rollout.output["error"])
        self.assertEqual(self.provider.calls, 0)

    def test_v2_completed_state_drift_fails_closed_at_max_turns(self) -> None:
        self._write_package()
        self._submit_v2_plan(
            [
                {
                    "key": "package",
                    "owner": "one",
                    "owned_files": ["samplepkg/__init__.py"],
                    "acceptance_checks": [
                        "python -m py_compile samplepkg/__init__.py"
                    ],
                },
                {
                    "key": "helper",
                    "owner": "two",
                    "owned_files": ["helper.py"],
                    "acceptance_checks": ["python -m py_compile helper.py"],
                },
            ]
        )
        completed = TeamRunTool().run({"max_workers": 2}, self.context)
        self.assertEqual(completed.output["status"], "completed")
        task_id = next(iter(self.context.tasks))
        self.context.tasks[task_id]["lifecycle_state"] = "produced"
        self.context.persist_tasks()
        conversation = Conversation()
        conversation.add_user_message("Finish now")

        result = run_agent_loop(
            conversation,
            self.provider,
            self.registry,
            self.context,
            max_turns=1,
        )

        self.assertTrue(result.failed)
        self.assertEqual(result.response_text, "[Max tool turns reached]")
        self.assertEqual(result.failure_reason, "team_lifecycle_failure")

    def test_v2_infrastructure_interruption_pauses_instead_of_completing(self) -> None:
        self._write_package()
        self._submit_v2_plan(
            [
                {
                    "key": "package",
                    "owner": "one",
                    "owned_files": ["samplepkg/__init__.py"],
                    "acceptance_checks": [
                        "python -m py_compile samplepkg/__init__.py"
                    ],
                },
                {
                    "key": "helper",
                    "owner": "two",
                    "owned_files": ["helper.py"],
                    "acceptance_checks": ["python -m py_compile helper.py"],
                },
            ]
        )
        with patch.object(
            self.context.teammate_runtime,
            "_run_validation_command",
            side_effect=TimeoutError("pending request was cancelled"),
        ):
            rollout = TeamRunTool().run({"max_workers": 2}, self.context)

        self.assertFalse(rollout.is_error)
        self.assertEqual(rollout.output["status"], "paused")
        self.assertEqual(rollout.output["lifecycle_state"], "paused")
        self.assertEqual(
            self.context.team_store.load_active_team().lifecycle_state, "paused"
        )


if __name__ == "__main__":
    unittest.main()
