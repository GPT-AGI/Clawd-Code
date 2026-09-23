from __future__ import annotations

import threading
import time
import tempfile
import unittest
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.providers.base import ChatResponse
from src.teammate.control import reassign_task
from src.teammate.models import TeamTask
from src.teammate.runtime import TeammateRuntime
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.tools import (
    TaskCreateTool,
    TaskRetryTool,
    TaskUpdateTool,
    TeamCancelTool,
    TeamCreateTool,
    TeamResumeTool,
    TeamRunTool,
    TeammateCreateTool,
    TeammateResumeTool,
    TeammateStopTool,
)
from src.tool_system.errors import ToolInputError


class FinalProvider:
    model = "test-model"

    def __init__(self, *, delay: float = 0.0, tokens: int = 2) -> None:
        self.delay = delay
        self.tokens = tokens
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return ChatResponse(
            content="done",
            model=self.model,
            usage={"input_tokens": self.tokens - 1, "output_tokens": 1},
            finish_reason="stop",
            tool_uses=None,
        )


class FailOnceProvider(FinalProvider):
    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient provider failure")
        return ChatResponse(
            content="recovered",
            model=self.model,
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="stop",
            tool_uses=None,
        )


class ParallelProvider(FinalProvider):
    def __init__(self) -> None:
        super().__init__(delay=0.08)
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def chat(self, messages, tools=None, **kwargs):
        with self.lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return ChatResponse(
                content="parallel done",
                model=self.model,
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="stop",
                tool_uses=None,
            )
        finally:
            with self.lock:
                self.active -= 1


class WorktreeProvider(FinalProvider):
    def __init__(self) -> None:
        super().__init__()
        self.responses = [
            ChatResponse(
                content="",
                model=self.model,
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="tool_use",
                tool_uses=[
                    {
                        "id": "write-isolated",
                        "name": "Write",
                        "input": {
                            "file_path": "isolated.txt",
                            "content": "created in teammate worktree\n",
                        },
                    }
                ],
            ),
            ChatResponse(
                content="implemented",
                model=self.model,
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="stop",
                tool_uses=None,
            ),
        ]

    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


class ToolThenFinalProvider(FinalProvider):
    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return ChatResponse(
                content="",
                model=self.model,
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="tool_use",
                tool_uses=[
                    {
                        "id": "read-base",
                        "name": "Read",
                        "input": {"file_path": "input.txt"},
                    }
                ],
            )
        return ChatResponse(
            content="done",
            model=self.model,
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="stop",
            tool_uses=None,
        )


class BlockingProvider(FinalProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=2)
        return ChatResponse(
            content="finished after cancellation",
            model=self.model,
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="stop",
            tool_uses=None,
        )


class SelectiveBlockingProvider(FinalProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.lock = threading.Lock()

    def chat(self, messages, tools=None, **kwargs):
        with self.lock:
            self.calls += 1
        if "stop-me" in str(messages):
            self.started.set()
            self.release.wait(timeout=2)
            content = "stopped worker returned"
        else:
            content = "survivor completed"
        return ChatResponse(
            content=content,
            model=self.model,
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="stop",
            tool_uses=None,
        )


class SequenceProvider(FinalProvider):
    def __init__(self, responses: list[ChatResponse]) -> None:
        super().__init__()
        self.responses = responses

    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


def _tool_response(call_id: str, name: str, tool_input: dict) -> ChatResponse:
    return ChatResponse(
        content="",
        model="test-model",
        usage={"input_tokens": 1, "output_tokens": 1},
        finish_reason="tool_use",
        tool_uses=[{"id": call_id, "name": name, "input": tool_input}],
    )


def _final_response(content: str) -> ChatResponse:
    return ChatResponse(
        content=content,
        model="test-model",
        usage={"input_tokens": 1, "output_tokens": 1},
        finish_reason="stop",
        tool_uses=None,
    )


class TestTeammateResilience(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.registry = build_default_registry(include_user_tools=False)
        self.context = ToolContext(workspace_root=self.root)
        self.team = TeamCreateTool().run({"team_name": "resilience"}, self.context).output

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _agent(self, name: str) -> dict:
        return TeammateCreateTool().run(
            {
                "name": name,
                "role": "worker",
                "instructions": "Complete the assigned task.",
                "tools": ["Read"],
            },
            self.context,
        ).output

    def _task(self, key: str, owner: str) -> str:
        return TaskCreateTool().run(
            {
                "key": key,
                "subject": key,
                "description": f"Complete {key}",
                "owner": owner,
            },
            self.context,
        ).output["task"]["id"]

    def _runtime(self, provider) -> None:
        self.context.teammate_runtime = TeammateRuntime(provider, self.registry)

    def test_completed_team_reopens_for_a_late_task(self) -> None:
        self._agent("worker")
        first_task = self._task("first", "worker")
        provider = FinalProvider()
        self._runtime(provider)

        first = TeamRunTool().run({}, self.context)
        self.assertEqual(first.output["status"], "completed")
        self.assertEqual(self.context.tasks[first_task]["status"], "completed")

        late_task = self._task("late", "worker")
        second = TeamRunTool().run({}, self.context)

        self.assertFalse(second.is_error)
        self.assertEqual(second.output["status"], "completed")
        self.assertEqual(second.output["executed_task_ids"], [late_task])
        self.assertEqual(self.context.tasks[late_task]["status"], "completed")
        self.assertEqual(provider.calls, 2)
        events = self.context.team_store.list_events(self.team["team_id"])
        reopened = [event for event in events if event["type"] == "team.reopened"]
        self.assertEqual(len(reopened), 1)
        self.assertEqual(reopened[0]["data"]["unfinished_task_ids"], [late_task])

    def test_teammate_model_must_match_runtime_allowlist(self) -> None:
        self.context.teammate_runtime = TeammateRuntime(
            FinalProvider(),
            self.registry,
            allowed_models={"served-model"},
        )

        with self.assertRaisesRegex(ToolInputError, "unsupported teammate model"):
            TeammateCreateTool().run(
                {
                    "name": "wrong-model",
                    "role": "worker",
                    "instructions": "Complete the task.",
                    "tools": ["Read"],
                    "model": "claude-3-7-sonnet-20250219",
                },
                self.context,
            )

        created = TeammateCreateTool().run(
            {
                "name": "right-model",
                "role": "worker",
                "instructions": "Complete the task.",
                "tools": ["Read"],
                "model": "served-model",
            },
            self.context,
        )
        self.assertEqual(created.output["name"], "right-model")

    def test_runtime_enforces_minimum_team_timeout(self) -> None:
        self._agent("worker")
        self._task("bounded", "worker")
        self.context.teammate_runtime = TeammateRuntime(
            FinalProvider(),
            self.registry,
            minimum_timeout_s=900,
        )

        result = TeamRunTool().run({"timeout_s": 120}, self.context)

        self.assertEqual(result.output["status"], "completed")
        team = self.context.team_store.load_team(self.team["team_id"])
        self.assertEqual(team.settings["timeout_s"], 900)
        events = self.context.team_store.list_events(self.team["team_id"])
        adjusted = [event for event in events if event["type"] == "team.options_adjusted"]
        self.assertEqual(adjusted[0]["data"]["timeout_s"]["requested"], 120)
        self.assertEqual(adjusted[0]["data"]["timeout_s"]["effective"], 900)

    def test_recovers_expired_in_progress_lease(self) -> None:
        agent = self._agent("worker")
        task_id = self._task("recover", "worker")
        task = TeamTask.from_dict(self.context.tasks[task_id])
        task.transition_to("in_progress")
        task.attempt = 1
        task.lease_id = "dead-run"
        task.lease_expires_at = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()
        self.context.team_store.update_task(self.team["team_id"], task)
        stored_agent = self.context.team_store.load_agent(self.team["team_id"], agent["agent_id"])
        stored_agent.transition_to("running")
        self.context.team_store.save_agent(stored_agent)

        self._runtime(FinalProvider())
        result = TeamResumeTool().run({}, self.context)

        self.assertFalse(result.is_error)
        restored = self.context.team_store.load_tasks(self.team["team_id"])[task_id]
        self.assertEqual(restored["status"], "completed")
        self.assertEqual(restored["attempt"], 2)
        self.assertIsNone(restored["lease_id"])
        events = self.context.team_store.list_events(self.team["team_id"])
        self.assertIn("task.recovered", {event["type"] for event in events})

    def test_automatically_retries_transient_failure(self) -> None:
        self._agent("worker")
        task_id = self._task("retry", "worker")
        provider = FailOnceProvider()
        self._runtime(provider)

        result = TeamRunTool().run({"max_retries": 1}, self.context)

        self.assertFalse(result.is_error)
        task = self.context.team_store.load_tasks(self.team["team_id"])[task_id]
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["attempt"], 2)
        self.assertEqual(provider.calls, 2)
        events = self.context.team_store.list_events(self.team["team_id"])
        self.assertIn("task.retry_scheduled", {event["type"] for event in events})

    def test_manual_task_retry_and_team_resume(self) -> None:
        self._agent("worker")
        task_id = self._task("manual", "worker")
        task = TeamTask.from_dict(self.context.tasks[task_id])
        task.transition_to("in_progress")
        task.transition_to("failed")
        task.output = "failed before restart"
        self.context.team_store.update_task(self.team["team_id"], task)
        team = self.context.team_store.load_team(self.team["team_id"])
        team.transition_to("running")
        team.transition_to("failed")
        self.context.team_store.save_team(team)
        self.context.reload_team_state()

        retried = TaskRetryTool().run({"taskId": "manual"}, self.context)
        self.assertEqual(retried.output["status"], "pending")
        self._runtime(FinalProvider())
        resumed = TeamResumeTool().run({}, self.context)

        self.assertFalse(resumed.is_error)
        self.assertEqual(
            self.context.team_store.load_tasks(self.team["team_id"])[task_id]["status"],
            "completed",
        )

    def test_cancelled_team_can_be_resumed(self) -> None:
        self._agent("worker")
        self._task("cancel", "worker")
        cancelled = TeamCancelTool().run({"reason": "operator request"}, self.context)
        self.assertEqual(cancelled.output["status"], "cancelled")

        self._runtime(FinalProvider())
        resumed = TeamResumeTool().run({}, self.context)

        self.assertFalse(resumed.is_error)
        self.assertEqual(resumed.output["status"], "completed")

    def test_token_budget_overrun_after_completion_is_reported(self) -> None:
        self._agent("worker")
        self._task("budget", "worker")
        self._runtime(FinalProvider(tokens=10))

        result = TeamRunTool().run({"token_budget": 5}, self.context)

        self.assertFalse(result.is_error)
        self.assertIn("token budget", result.output["budget_warning"])
        team = self.context.team_store.load_team(self.team["team_id"])
        self.assertEqual(team.status, "completed")
        self.assertEqual(team.usage["total_tokens"], 10)
        events = self.context.team_store.list_events(self.team["team_id"])
        self.assertTrue(
            any(event["type"] == "team.budget_exceeded_after_completion" for event in events)
        )

    def test_timeout_overrun_preserves_completed_result_with_warning(self) -> None:
        self._agent("worker")
        self._task("timeout", "worker")
        self._runtime(FinalProvider(delay=0.04))

        result = self.context.teammate_runtime.run_team(self.context, timeout_s=0.01)

        self.assertEqual(result["status"], "completed")
        self.assertIn("timeout", result["budget_warning"])

    def test_turn_budget_limits_model_round_trips(self) -> None:
        (self.root / "input.txt").write_text("input\n", encoding="utf-8")
        self._agent("worker")
        self._task("turn-budget", "worker")
        self._runtime(ToolThenFinalProvider())

        result = TeamRunTool().run({"turn_budget": 1}, self.context)

        self.assertTrue(result.is_error)
        self.assertIn("Max tool turns", result.output["error"])
        self.assertEqual(result.output["usage"]["turns"], 1)

    def test_cooperative_cancel_is_observed_after_active_model_call(self) -> None:
        self._agent("worker")
        task_id = self._task("cancel-active", "worker")
        provider = BlockingProvider()
        self._runtime(provider)
        result_box: list = []

        thread = threading.Thread(
            target=lambda: result_box.append(TeamRunTool().run({}, self.context)),
            daemon=True,
        )
        thread.start()
        self.assertTrue(provider.started.wait(timeout=1))
        cancelling_context = ToolContext(workspace_root=self.root)
        TeamCancelTool().run({"reason": "stop active run"}, cancelling_context)
        provider.release.set()
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result_box[0].output["status"], "cancelled")
        task = self.context.team_store.load_tasks(self.team["team_id"])[task_id]
        self.assertEqual(task["status"], "cancelled")

    def test_lead_stops_one_worker_without_cancelling_team(self) -> None:
        stopped = self._agent("stopped-worker")
        self._agent("survivor")
        stopped_task = self._task("stop-me", "stopped-worker")
        survivor_task = self._task("keep-going", "survivor")
        provider = SelectiveBlockingProvider()
        self._runtime(provider)
        holder: dict[str, object] = {}

        def run_team() -> None:
            holder["result"] = TeamRunTool().run(
                {"max_workers": 2}, self.context
            ).output

        thread = threading.Thread(target=run_team)
        thread.start()
        self.assertTrue(provider.started.wait(timeout=1))
        stop_context = ToolContext(workspace_root=self.root)
        stopped_result = TeammateStopTool().run(
            {
                "teammate": stopped["agent_id"],
                "task_policy": "requeue",
                "reason": "lead replaced this worker",
            },
            stop_context,
        ).output
        self.assertEqual(stopped_result["status"], "stopping")
        provider.release.set()
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())

        tasks = self.context.team_store.load_tasks(self.team["team_id"])
        self.assertEqual(tasks[stopped_task]["status"], "pending")
        self.assertIsNone(tasks[stopped_task]["owner"])
        self.assertEqual(tasks[survivor_task]["status"], "completed")
        agent = self.context.team_store.load_agent(
            self.team["team_id"], stopped["agent_id"]
        )
        self.assertEqual(agent.status, "cancelled")
        self.assertIsNotNone(agent.stopped_at)
        self.assertEqual(holder["result"]["status"], "blocked")
        self.assertEqual(
            self.context.team_store.load_active_team().status,
            "running",
        )
        event_types = [
            event["type"]
            for event in self.context.team_store.list_events(self.team["team_id"])
        ]
        self.assertIn("agent.stop_requested", event_types)
        self.assertIn("agent.stopped", event_types)
        self.assertIn("run.cancelled", event_types)
        self.assertIn("task.requeued", event_types)
        self.assertEqual(event_types.count("agent.stopped"), 1)
        self.assertEqual(event_types.count("task.requeued"), 1)

    def test_stop_cancel_policy_and_worker_permissions(self) -> None:
        worker = self._agent("worker")
        task_id = self._task("cancel-me", "worker")
        child_context = ToolContext(
            workspace_root=self.root,
            actor_id=worker["agent_id"],
        )
        with self.assertRaisesRegex(ToolInputError, "only the lead"):
            TeammateStopTool().run({"teammate": "worker"}, child_context)

        result = TeammateStopTool().run(
            {"teammate": "worker", "task_policy": "cancel"},
            self.context,
        ).output
        self.assertEqual(result["cancelled_task_ids"], [task_id])
        task = self.context.team_store.load_tasks(self.team["team_id"])[task_id]
        self.assertEqual(task["status"], "cancelled")
        self.assertEqual(
            self.context.team_store.load_active_team().status,
            "created",
        )

    def test_stopped_worker_can_resume_and_receive_requeued_task(self) -> None:
        self._agent("worker")
        replacement = self._agent("replacement")
        task_id = self._task("handoff", "worker")
        TeammateStopTool().run(
            {"teammate": "worker", "task_policy": "requeue"},
            self.context,
        )
        resumed = TeammateResumeTool().run(
            {"teammate": "worker"}, self.context
        ).output
        self.assertEqual(resumed["status"], "idle")
        reassigned = reassign_task(
            self.context.team_store,
            task_id,
            replacement["agent_id"],
        )
        self.assertEqual(reassigned["owner"], replacement["agent_id"])
        task = self.context.team_store.load_tasks(self.team["team_id"])[task_id]
        self.assertEqual(task["status"], "pending")
        self.assertEqual(task["owner"], replacement["agent_id"])

    def test_ready_tasks_run_in_parallel_without_lost_updates(self) -> None:
        self._agent("one")
        self._agent("two")
        one = self._task("one", "one")
        two = self._task("two", "two")
        provider = ParallelProvider()
        self._runtime(provider)

        result = TeamRunTool().run({"max_workers": 2}, self.context)

        self.assertFalse(result.is_error)
        self.assertGreaterEqual(provider.max_active, 2)
        tasks = self.context.team_store.load_tasks(self.team["team_id"])
        self.assertEqual(tasks[one]["status"], "completed")
        self.assertEqual(tasks[two]["status"], "completed")
        self.assertEqual(result.output["usage"]["turns"], 2)

    def test_same_teammate_tasks_are_serialized(self) -> None:
        self._agent("one")
        self._task("first", "one")
        self._task("second", "one")
        provider = ParallelProvider()
        self._runtime(provider)

        result = TeamRunTool().run({"max_workers": 2}, self.context)

        self.assertFalse(result.is_error)
        self.assertEqual(provider.max_active, 1)
        self.assertEqual(provider.calls, 2)

    def test_active_lease_blocks_second_runner_without_failing_team(self) -> None:
        self._agent("worker")
        task_id = self._task("leased", "worker")
        task = self.context.team_store.claim_task(
            self.team["team_id"],
            task_id,
            lease_id="active-run",
            lease_expires_at=(
                datetime.now(timezone.utc) + timedelta(minutes=5)
            ).isoformat(),
            max_retries=0,
        )
        self.assertIsNotNone(task)
        team = self.context.team_store.load_team(self.team["team_id"])
        team.transition_to("running")
        self.context.team_store.save_team(team)
        self._runtime(FinalProvider())

        result = TeamRunTool().run({}, self.context)

        self.assertTrue(result.is_error)
        self.assertEqual(result.output["status"], "blocked")
        self.assertEqual(
            self.context.team_store.load_team(self.team["team_id"]).status,
            "running",
        )

    def test_task_claim_is_atomic_across_store_instances(self) -> None:
        self._agent("worker")
        task_id = self._task("claim", "worker")
        claimed: list[TeamTask | None] = []
        barrier = threading.Barrier(2)

        def claim(label: str) -> None:
            context = ToolContext(workspace_root=self.root)
            barrier.wait(timeout=1)
            claimed.append(
                context.team_store.claim_task(
                    self.team["team_id"],
                    task_id,
                    lease_id=label,
                    lease_expires_at=(
                        datetime.now(timezone.utc) + timedelta(minutes=5)
                    ).isoformat(),
                    max_retries=0,
                )
            )

        threads = [
            threading.Thread(target=claim, args=("one",)),
            threading.Thread(target=claim, args=("two",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        self.assertEqual(sum(item is not None for item in claimed), 1)

    def test_reviewer_rejection_can_drive_repair_and_re_review(self) -> None:
        self._agent("coder")
        self._agent("reviewer")
        implementation = self._task("implementation", "coder")
        review = TaskCreateTool().run(
            {
                "key": "review",
                "subject": "review",
                "description": "Review implementation",
                "owner": "reviewer",
                "blockedBy": ["implementation"],
            },
            self.context,
        ).output["task"]["id"]
        provider = SequenceProvider(
            [
                _final_response("initial implementation"),
                _tool_response(
                    "reject-message",
                    "SendMessage",
                    {"to": "coder", "summary": "changes requested", "message": "Fix edge case."},
                ),
                _tool_response(
                    "fail-review",
                    "TaskUpdate",
                    {"taskId": review, "status": "failed", "output": "edge case remains"},
                ),
                _final_response("review rejected"),
                _tool_response(
                    "repair-message",
                    "SendMessage",
                    {"to": "reviewer", "summary": "repair", "message": "Edge case fixed."},
                ),
                _final_response("repair complete"),
                _tool_response(
                    "approval-message",
                    "SendMessage",
                    {"to": "lead", "summary": "approved", "message": "Repair approved."},
                ),
                _final_response("review approved"),
            ]
        )
        self._runtime(provider)

        first = TeamRunTool().run({}, self.context)
        self.assertTrue(first.is_error)
        self.assertEqual(
            self.context.team_store.load_tasks(self.team["team_id"])[review]["status"],
            "failed",
        )

        self.context.reload_team_state()
        repair = TaskCreateTool().run(
            {
                "key": "repair",
                "subject": "repair",
                "description": "Address reviewer feedback",
                "owner": "coder",
                "blockedBy": [implementation],
            },
            self.context,
        ).output["task"]["id"]
        TaskUpdateTool().run(
            {"taskId": review, "addBlockedBy": [repair]}, self.context
        )
        TaskRetryTool().run({"taskId": review}, self.context)

        resumed = TeamResumeTool().run({}, self.context)

        self.assertFalse(resumed.is_error, resumed.output)
        tasks = self.context.team_store.load_tasks(self.team["team_id"])
        self.assertEqual(tasks[repair]["status"], "completed")
        self.assertEqual(tasks[review]["status"], "completed")
        messages = self.context.team_store.list_messages(self.team["team_id"])
        self.assertEqual(len(messages), 3)


class TestTeammateWorktree(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        (self.root / "base.txt").write_text("base\n", encoding="utf-8")
        subprocess.run(["git", "add", "base.txt"], cwd=self.root, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "base",
            ],
            cwd=self.root,
            check=True,
        )
        self.registry = build_default_registry(include_user_tools=False)
        self.context = ToolContext(workspace_root=self.root)
        self.team = TeamCreateTool().run({"team_name": "worktree"}, self.context).output
        self.agent = None

    def tearDown(self) -> None:
        if self.agent is not None:
            from src.teammate.worktree import TeammateWorktreeManager

            stored = self.context.team_store.load_agent(
                self.team["team_id"], self.agent["agent_id"]
            )
            if stored is not None:
                TeammateWorktreeManager(self.root).remove(stored, force=True)
        self.tmp.cleanup()

    def test_auto_integrates_isolated_teammate_changes(self) -> None:
        self.context.teammate_runtime = TeammateRuntime(WorktreeProvider(), self.registry)
        self.agent = TeammateCreateTool().run(
            {
                "name": "coder",
                "role": "implementation",
                "instructions": "Create the requested file.",
                "tools": ["Write"],
                "workspace_mode": "worktree",
                "auto_integrate": True,
            },
            self.context,
        ).output
        TaskCreateTool().run(
            {
                "key": "isolated",
                "subject": "isolated",
                "description": "Create isolated.txt",
                "owner": "coder",
            },
            self.context,
        )

        result = TeamRunTool().run({}, self.context)

        self.assertFalse(result.is_error, result.output)
        self.assertEqual(
            (self.root / "isolated.txt").read_text(encoding="utf-8"),
            "created in teammate worktree\n",
        )
        log = subprocess.run(
            ["git", "log", "-1", "--pretty=%s"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(log, "clawd teammate coder: isolated")
        events = self.context.team_store.list_events(self.team["team_id"])
        self.assertIn("worktree.integrated", {event["type"] for event in events})


if __name__ == "__main__":
    unittest.main()
