from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..agent.conversation import Conversation
from ..tool_system.agent_loop import ToolEvent, run_agent_loop
from ..tool_system.context import ToolContext
from ..tool_system.permissions import ToolPermissionContext
from ..tool_system.registry import ToolRegistry
from .models import AgentRecord, Message, Team, TeamTask, utc_now
from .store import TeamStore
from .worktree import TeammateWorktreeManager


_MANDATORY_TEAMMATE_TOOLS = (
    "SendMessage",
    "TaskGet",
    "TaskList",
    "TaskUpdate",
    "StructuredOutput",
)
_FORBIDDEN_TEAMMATE_TOOLS = {
    "Agent",
    "TeamCreate",
    "TeammateCreate",
    "TeamRun",
    "TeamResume",
    "TeamCancel",
    "TeamIntegrate",
    "TeamDelete",
    "TaskRetry",
}


@dataclass(frozen=True)
class TeamRunOptions:
    max_workers: int = 1
    timeout_s: float | None = None
    token_budget: int | None = None
    turn_budget: int | None = None
    max_retries: int = 0
    lease_timeout_s: int = 900

    @classmethod
    def build(cls, persisted: dict[str, Any], overrides: dict[str, Any]) -> "TeamRunOptions":
        values: dict[str, Any] = {}
        for name in cls.__dataclass_fields__:
            if overrides.get(name) is not None:
                values[name] = overrides[name]
            elif persisted.get(name) is not None:
                values[name] = persisted[name]
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_workers": self.max_workers,
            "timeout_s": self.timeout_s,
            "token_budget": self.token_budget,
            "turn_budget": self.turn_budget,
            "max_retries": self.max_retries,
            "lease_timeout_s": self.lease_timeout_s,
        }


@dataclass(frozen=True)
class TaskOutcome:
    status: str
    task_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0
    error: str | None = None


class TeammateRuntime:
    """Persistent teammate scheduler with recovery, budgets, and optional parallelism."""

    def __init__(self, provider: Any, registry: ToolRegistry, *, max_turns: int = 30):
        self.provider = provider
        self.registry = registry
        self.max_turns = max_turns

    def validate_tools(self, names: list[str]) -> list[str]:
        canonical: list[str] = []
        seen: set[str] = set()
        for name in names:
            tool = self.registry.get(name)
            if tool is None:
                raise ValueError(f"unknown tool in teammate allowlist: {name}")
            spec_name = tool.spec().name
            if spec_name in _FORBIDDEN_TEAMMATE_TOOLS:
                raise ValueError(f"teammates cannot use team-management tool: {spec_name}")
            key = spec_name.lower()
            if key not in seen:
                canonical.append(spec_name)
                seen.add(key)
        return canonical

    def run_team(
        self,
        lead_context: ToolContext,
        *,
        resume: bool = False,
        retry_failed: bool = False,
        retry_cancelled: bool = False,
        max_workers: int | None = None,
        timeout_s: float | None = None,
        token_budget: int | None = None,
        turn_budget: int | None = None,
        max_retries: int | None = None,
        lease_timeout_s: int | None = None,
    ) -> dict[str, Any]:
        team = lead_context.team_store.load_active_team()
        if team is None:
            return {"status": "failed", "error": "no active team"}
        if team.status == "completed":
            lead_context.reload_team_state()
            return self._result(lead_context, team, [])
        if team.status == "cancelled" and not resume:
            return {"status": "cancelled", "error": "team is cancelled", "team_id": team.team_id}

        options = TeamRunOptions.build(
            team.settings,
            {
                "max_workers": max_workers,
                "timeout_s": timeout_s,
                "token_budget": token_budget,
                "turn_budget": turn_budget,
                "max_retries": max_retries,
                "lease_timeout_s": lease_timeout_s,
            },
        )
        team.settings.update(options.to_dict())
        team.usage = self._normalized_usage(team.usage)
        team.started_at = team.started_at or utc_now()
        if resume:
            team.cancel_requested_at = None

        try:
            if team.status in {"created", "failed", "cancelled"}:
                team.transition_to("running")
                lead_context.team_store.save_team(team)
                lead_context.team_store.append_event(
                    team.team_id,
                    "team.resumed" if resume else "team.running",
                    {"settings": options.to_dict()},
                )
            else:
                lead_context.team_store.save_team(team)

            self._recover_tasks(
                lead_context,
                team,
                options,
                retry_failed=retry_failed,
                retry_cancelled=retry_cancelled,
            )
            executed: list[str] = []
            run_started = time.monotonic()

            while True:
                current = lead_context.team_store.load_team(team.team_id) or team
                if current.status == "cancelled" or current.cancel_requested_at:
                    lead_context.reload_team_state()
                    return self._cancelled_result(lead_context, current, executed)

                budget_error = self._budget_error(current, options, run_started)
                if budget_error:
                    return self._fail_team(lead_context, current, budget_error, executed)

                lead_context.reload_team_state()
                tasks = lead_context.tasks
                if not tasks:
                    return self._fail_team(lead_context, current, "team has no tasks", executed)
                if all(task.get("status") == "completed" for task in tasks.values()):
                    completed = self._complete_team(lead_context, current)
                    return self._result(lead_context, completed, executed)

                failed = [task for task in tasks.values() if task.get("status") == "failed"]
                if failed:
                    names = ", ".join(str(task.get("key") or task.get("id")) for task in failed)
                    return self._fail_team(lead_context, current, f"failed tasks: {names}", executed)

                ready = [
                    task
                    for task in tasks.values()
                    if task.get("status") == "pending" and self._dependencies_completed(task, tasks)
                ]
                if not ready:
                    active = [
                        task for task in tasks.values() if task.get("status") == "in_progress"
                    ]
                    pending = [
                        str(task.get("key") or task.get("id"))
                        for task in tasks.values()
                        if task.get("status") in {"pending", "in_progress"}
                    ]
                    reason = "no runnable tasks; active leases or dependencies remain"
                    if pending:
                        reason += f" ({', '.join(pending)})"
                    if active:
                        return self._blocked_result(lead_context, current, reason, executed)
                    return self._fail_team(
                        lead_context, current, reason, executed, status="blocked"
                    )

                ready.sort(
                    key=lambda task: (
                        str(task.get("created_at") or ""),
                        str(task.get("id") or ""),
                    )
                )
                batch, turn_limits = self._build_batch(ready, current, options)
                if not batch:
                    return self._fail_team(
                        lead_context, current, "turn budget exhausted", executed
                    )
                outcomes = self._run_batch(
                    lead_context, current, batch, options, turn_limits
                )
                self._record_usage(lead_context.team_store, current.team_id, outcomes)

                terminal_failures: list[TaskOutcome] = []
                for outcome in outcomes:
                    executed.append(outcome.task_id)
                    if outcome.status == "failed":
                        task_data = lead_context.team_store.load_tasks(current.team_id).get(
                            outcome.task_id
                        )
                        if task_data and self._schedule_retry(
                            lead_context.team_store,
                            current,
                            TeamTask.from_dict(task_data),
                            options,
                        ):
                            continue
                        terminal_failures.append(outcome)
                    elif outcome.status == "cancelled":
                        latest = lead_context.team_store.load_team(current.team_id) or current
                        return self._cancelled_result(lead_context, latest, executed)

                if terminal_failures:
                    first = terminal_failures[0]
                    return self._fail_team(
                        lead_context,
                        current,
                        first.error or f"task failed: {first.task_id}",
                        executed,
                    )
        except Exception as exc:
            return self._fail_team(
                lead_context, team, str(exc), locals().get("executed", [])
            )

    @staticmethod
    def _dependencies_completed(
        task: dict[str, Any], tasks: dict[str, dict[str, Any]]
    ) -> bool:
        dependencies = list(task.get("blockedBy") or [])
        return all(
            dependency in tasks and tasks[dependency].get("status") == "completed"
            for dependency in dependencies
        )

    @staticmethod
    def _build_batch(
        ready: list[dict[str, Any]], team: Team, options: TeamRunOptions
    ) -> tuple[list[dict[str, Any]], list[int]]:
        workers = min(options.max_workers, len(ready))
        usage = TeammateRuntime._normalized_usage(team.usage)
        remaining_turns = None
        if options.turn_budget is not None:
            remaining_turns = max(0, options.turn_budget - usage["turns"])
            workers = min(workers, remaining_turns)
        if workers <= 0:
            return [], []
        batch: list[dict[str, Any]] = []
        owners: set[str] = set()
        for task in ready:
            owner_key = str(task.get("owner") or f"task:{task.get('id')}")
            if owner_key in owners:
                continue
            owners.add(owner_key)
            batch.append(task)
            if len(batch) >= workers:
                break
        workers = len(batch)
        if workers == 0:
            return [], []
        if remaining_turns is None:
            return batch, [0] * len(batch)
        base = max(1, remaining_turns // len(batch))
        limits = [base] * len(batch)
        for index in range(remaining_turns - base * len(batch)):
            limits[index] += 1
        return batch, limits

    def _run_batch(
        self,
        lead_context: ToolContext,
        team: Team,
        batch: list[dict[str, Any]],
        options: TeamRunOptions,
        turn_limits: list[int],
    ) -> list[TaskOutcome]:
        tasks = [TeamTask.from_dict(task) for task in batch]
        effective_limits = [
            self.max_turns if limit <= 0 else min(self.max_turns, limit)
            for limit in turn_limits
        ]
        if len(tasks) == 1:
            return [
                self._run_task(
                    lead_context, team, tasks[0], options, effective_limits[0]
                )
            ]
        with ThreadPoolExecutor(
            max_workers=len(tasks), thread_name_prefix=f"clawd-{team.team_id}"
        ) as pool:
            futures = [
                pool.submit(
                    self._run_task,
                    lead_context,
                    team,
                    task,
                    options,
                    limit,
                )
                for task, limit in zip(tasks, effective_limits)
            ]
            return [future.result() for future in futures]

    def _run_task(
        self,
        lead_context: ToolContext,
        team: Team,
        task: TeamTask,
        options: TeamRunOptions,
        task_max_turns: int,
    ) -> TaskOutcome:
        store = lead_context.team_store
        input_tokens = 0
        output_tokens = 0
        turns = 0
        if not task.owner:
            self._set_task_failed(store, team, task, "task has no owner")
            return TaskOutcome("failed", task.id, error="task has no owner")
        agent = store.load_agent(team.team_id, task.owner)
        if agent is None:
            error = f"unknown task owner: {task.owner}"
            self._set_task_failed(store, team, task, error)
            return TaskOutcome("failed", task.id, error=error)

        lease_id = uuid.uuid4().hex
        try:
            claimed = store.claim_task(
                team.team_id,
                task.id,
                lease_id=lease_id,
                lease_expires_at=self._lease_expiry(options.lease_timeout_s),
                max_retries=options.max_retries,
            )
            if claimed is None:
                return TaskOutcome("leased", task.id)
            task = claimed
            self._transition_agent(store, agent, "running")
            store.append_event(
                team.team_id,
                "task.started",
                {
                    "task_id": task.id,
                    "task_key": task.key,
                    "agent_id": agent.agent_id,
                    "attempt": task.attempt,
                    "lease_id": lease_id,
                },
            )

            conversation = self._load_conversation(store, agent)
            incoming = self._consume_messages(store, team, agent)
            conversation.add_user_message(self._task_prompt(team, agent, task, incoming))
            self._save_session(store, agent, conversation)

            child_context = self._child_context(lead_context, team, agent, task)

            def heartbeat(event: ToolEvent) -> None:
                if event.kind in {
                    "model_started",
                    "model_response",
                    "tool_use",
                    "tool_result",
                    "tool_error",
                }:
                    self._refresh_lease(
                        store,
                        team.team_id,
                        task.id,
                        lease_id,
                        options.lease_timeout_s,
                    )

            result = run_agent_loop(
                conversation=conversation,
                provider=self.provider,
                tool_registry=self._child_registry(agent),
                tool_context=child_context,
                max_turns=task_max_turns,
                stream=False,
                verbose=False,
                on_event=heartbeat,
            )
            self._save_session(store, agent, conversation)
            usage = result.usage or {}
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            turns = result.num_turns
            outcome_usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "turns": turns,
            }

            persisted_data = store.load_tasks(team.team_id).get(task.id)
            persisted = TeamTask.from_dict(persisted_data or task.to_dict())
            latest_team = store.load_team(team.team_id) or team
            if latest_team.status == "cancelled" or latest_team.cancel_requested_at:
                if persisted.status == "in_progress":
                    persisted.transition_to("cancelled")
                self._clear_lease(persisted)
                persisted.last_error = "team cancelled"
                self._save_task(store, team.team_id, persisted)
                self._transition_agent(store, agent, "cancelled")
                store.append_event(
                    team.team_id,
                    "task.cancelled",
                    {"task_id": task.id, "task_key": task.key, "agent_id": agent.agent_id},
                )
                return TaskOutcome("cancelled", task.id, **outcome_usage)

            if result.response_text == "[Max tool turns reached]":
                self._set_task_failed(store, team, persisted, result.response_text)
                self._transition_agent(store, agent, "failed")
                return TaskOutcome(
                    "failed", task.id, error=result.response_text, **outcome_usage
                )
            if persisted.status == "failed":
                if not persisted.output:
                    persisted.output = result.response_text
                    persisted.updated_at = utc_now()
                    self._save_task(store, team.team_id, persisted)
                self._transition_agent(store, agent, "failed")
                return TaskOutcome(
                    "failed", task.id, error=persisted.output, **outcome_usage
                )
            if persisted.status == "cancelled":
                self._transition_agent(store, agent, "cancelled")
                return TaskOutcome("cancelled", task.id, **outcome_usage)
            if persisted.status == "in_progress":
                persisted.output = result.response_text
                persisted.transition_to("completed")
            elif persisted.status == "completed" and not persisted.output:
                persisted.output = result.response_text
                persisted.updated_at = utc_now()
            if agent.workspace_mode == "worktree" and agent.auto_integrate:
                integration = TeammateWorktreeManager(
                    lead_context.workspace_root
                ).integrate(agent, persisted)
                store.append_event(
                    team.team_id,
                    "worktree.integrated",
                    {"agent_id": agent.agent_id, "task_id": task.id, **integration},
                )
            self._clear_lease(persisted)
            persisted.completed_at = utc_now()
            self._save_task(store, team.team_id, persisted)
            self._transition_agent(store, agent, "idle")
            store.append_event(
                team.team_id,
                "task.completed",
                {
                    "task_id": task.id,
                    "task_key": task.key,
                    "agent_id": agent.agent_id,
                    "attempt": persisted.attempt,
                },
            )
            return TaskOutcome("completed", task.id, **outcome_usage)
        except Exception as exc:
            current_data = store.load_tasks(team.team_id).get(task.id)
            current = TeamTask.from_dict(current_data or task.to_dict())
            if current.status in {"pending", "in_progress"}:
                self._set_task_failed(store, team, current, str(exc))
            latest_agent = store.load_agent(team.team_id, agent.agent_id) or agent
            if latest_agent.status in {"created", "running", "idle"}:
                self._transition_agent(store, latest_agent, "failed")
            return TaskOutcome(
                "failed",
                task.id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                turns=turns,
                error=str(exc),
            )

    def _recover_tasks(
        self,
        lead_context: ToolContext,
        team: Team,
        options: TeamRunOptions,
        *,
        retry_failed: bool,
        retry_cancelled: bool,
    ) -> None:
        store = lead_context.team_store
        for task_data in store.load_tasks(team.team_id).values():
            task = TeamTask.from_dict(task_data)
            reason: str | None = None
            if task.status == "in_progress" and self._lease_expired(task.lease_expires_at):
                reason = "expired or missing task lease"
            elif task.status == "failed" and (
                retry_failed or task.attempt <= max(task.max_retries, options.max_retries)
            ):
                reason = "retrying failed task"
            elif task.status == "cancelled" and retry_cancelled:
                reason = "resuming cancelled task"
            if reason is None:
                continue
            previous = task.status
            task.transition_to("pending")
            self._clear_lease(task)
            task.completed_at = None
            task.max_retries = max(task.max_retries, options.max_retries)
            task.last_error = reason
            self._save_task(store, team.team_id, task)
            self._reset_agent_for_retry(store, team.team_id, task.owner)
            store.append_event(
                team.team_id,
                "task.recovered",
                {
                    "task_id": task.id,
                    "from": previous,
                    "reason": reason,
                    "attempt": task.attempt,
                },
            )
        lead_context.reload_team_state()

    def _schedule_retry(
        self,
        store: TeamStore,
        team: Team,
        task: TeamTask,
        options: TeamRunOptions,
    ) -> bool:
        retry_limit = max(task.max_retries, options.max_retries)
        if task.status != "failed" or task.attempt > retry_limit:
            return False
        task.transition_to("pending")
        self._clear_lease(task)
        task.completed_at = None
        self._save_task(store, team.team_id, task)
        self._reset_agent_for_retry(store, team.team_id, task.owner)
        store.append_event(
            team.team_id,
            "task.retry_scheduled",
            {"task_id": task.id, "attempt": task.attempt, "max_retries": retry_limit},
        )
        return True

    def _child_registry(self, agent: AgentRecord) -> ToolRegistry:
        names = self.validate_tools([*agent.tools, *_MANDATORY_TEAMMATE_TOOLS])
        return ToolRegistry(self.registry.get(name) for name in names)

    @staticmethod
    def _child_context(
        lead_context: ToolContext, team: Team, agent: AgentRecord, task: TeamTask
    ) -> ToolContext:
        workspace_root = lead_context.workspace_root
        if agent.workspace_mode == "worktree" and agent.workspace_path:
            workspace_root = Path(agent.workspace_path)
        permissions = ToolPermissionContext.from_iterables(
            workspace_root=workspace_root,
            allow_docs=lead_context.permission_context.allow_docs,
        )
        context = ToolContext(
            workspace_root=workspace_root,
            permission_context=permissions,
            cwd=workspace_root,
            actor_id=agent.agent_id,
            current_task_id=task.id,
            model_override=agent.model,
            system_prompt_extra=(
                "## Teammate Identity\n"
                f"You are teammate `{agent.name}` with role `{agent.role}` in team `{team.team_name}`.\n"
                f"Role instructions: {agent.instructions}\n"
                f"Your current task is `{task.key or task.id}`. Work only on this task. "
                "Use SendMessage for every handoff; do not claim another teammate's work. "
                "If the task cannot be completed, set your current task to failed with TaskUpdate and explain why."
            ),
        )
        if workspace_root != lead_context.workspace_root:
            context.team_store = lead_context.team_store
            context.team = team.to_dict()
            context.tasks = lead_context.team_store.load_tasks(team.team_id)
        context.permission_handler = lead_context.permission_handler
        return context

    @staticmethod
    def _task_prompt(
        team: Team, agent: AgentRecord, task: TeamTask, incoming: list[Message]
    ) -> str:
        lines = [
            f"Team: {team.team_name}",
            f"Task ID: {task.id}",
            f"Task key: {task.key or task.id}",
            f"Attempt: {task.attempt}",
            f"Subject: {task.subject}",
            f"Description: {task.description}",
        ]
        if incoming:
            lines.append("Incoming teammate messages:")
            for message in incoming:
                lines.append(
                    f"- from {message.sender_id}: {message.summary or ''}\n  {message.content}"
                )
        else:
            lines.append("Incoming teammate messages: none")
        lines.append("Complete the task using only your available tools and report a concrete result.")
        return "\n".join(lines)

    @staticmethod
    def _load_conversation(store: TeamStore, agent: AgentRecord) -> Conversation:
        data = store.load_session(agent.team_id, agent.session_id)
        if data is None:
            return Conversation()
        conversation = data.get("conversation")
        return Conversation.from_dict(conversation) if isinstance(conversation, dict) else Conversation()

    def _save_session(
        self, store: TeamStore, agent: AgentRecord, conversation: Conversation
    ) -> None:
        store.save_session(
            agent.team_id,
            agent.session_id,
            {
                "session_id": agent.session_id,
                "team_id": agent.team_id,
                "agent_id": agent.agent_id,
                "model": agent.model or getattr(self.provider, "model", None),
                "conversation": conversation.to_dict(),
                "updated_at": utc_now(),
            },
        )

    @staticmethod
    def _consume_messages(
        store: TeamStore, team: Team, agent: AgentRecord
    ) -> list[Message]:
        incoming = [
            message
            for message in store.list_messages(team.team_id)
            if message.recipient_id == agent.agent_id and message.status == "delivered"
        ]
        for message in incoming:
            message.transition_to("consumed")
            store.save_message(message)
            store.append_event(
                team.team_id,
                "message.consumed",
                {"message_id": message.message_id, "agent_id": agent.agent_id},
            )
        return incoming

    @staticmethod
    def _save_task(store: TeamStore, team_id: str, task: TeamTask) -> None:
        store.update_task(team_id, task)

    @staticmethod
    def _set_task_failed(
        store: TeamStore, team: Team, task: TeamTask, error: str
    ) -> None:
        if task.status == "pending":
            task.transition_to("in_progress")
        if task.status == "in_progress":
            task.transition_to("failed")
        TeammateRuntime._clear_lease(task)
        task.output = error
        task.last_error = error
        task.completed_at = utc_now()
        task.updated_at = utc_now()
        TeammateRuntime._save_task(store, team.team_id, task)
        store.append_event(
            team.team_id,
            "task.failed",
            {
                "task_id": task.id,
                "task_key": task.key,
                "attempt": task.attempt,
                "error": error,
            },
        )

    @staticmethod
    def _transition_agent(store: TeamStore, agent: AgentRecord, status: str) -> None:
        if agent.status != status:
            agent.transition_to(status)
            store.save_agent(agent)
            store.append_event(
                agent.team_id,
                f"agent.{status}",
                {"agent_id": agent.agent_id, "name": agent.name},
            )

    @staticmethod
    def _reset_agent_for_retry(
        store: TeamStore, team_id: str, agent_id: str | None
    ) -> None:
        if not agent_id:
            return
        agent = store.load_agent(team_id, agent_id)
        if agent is None:
            return
        if agent.status in {"failed", "cancelled"}:
            TeammateRuntime._transition_agent(store, agent, "running")
        if agent.status == "running":
            TeammateRuntime._transition_agent(store, agent, "idle")

    @staticmethod
    def _refresh_lease(
        store: TeamStore,
        team_id: str,
        task_id: str,
        lease_id: str,
        lease_timeout_s: int,
    ) -> None:
        task_data = store.load_tasks(team_id).get(task_id)
        if task_data is None:
            return
        task = TeamTask.from_dict(task_data)
        if task.status != "in_progress" or task.lease_id != lease_id:
            return
        task.lease_expires_at = TeammateRuntime._lease_expiry(lease_timeout_s)
        TeammateRuntime._save_task(store, team_id, task)

    @staticmethod
    def _lease_expiry(seconds: int) -> str:
        return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()

    @staticmethod
    def _lease_expired(value: str | None) -> bool:
        if not value:
            return True
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return True
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed <= datetime.now(timezone.utc)

    @staticmethod
    def _clear_lease(task: TeamTask) -> None:
        task.lease_id = None
        task.lease_expires_at = None

    @staticmethod
    def _normalized_usage(usage: dict[str, Any]) -> dict[str, int]:
        return {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "turns": int(usage.get("turns", 0) or 0),
        }

    @staticmethod
    def _record_usage(
        store: TeamStore, team_id: str, outcomes: list[TaskOutcome]
    ) -> None:
        team = store.load_team(team_id)
        if team is None:
            return
        usage = TeammateRuntime._normalized_usage(team.usage)
        for outcome in outcomes:
            usage["input_tokens"] += outcome.input_tokens
            usage["output_tokens"] += outcome.output_tokens
            usage["turns"] += outcome.turns
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        team.usage = usage
        store.save_team(team)

    @staticmethod
    def _budget_error(
        team: Team, options: TeamRunOptions, run_started: float
    ) -> str | None:
        usage = TeammateRuntime._normalized_usage(team.usage)
        if options.timeout_s is not None and time.monotonic() - run_started >= options.timeout_s:
            return f"team timeout exceeded ({options.timeout_s}s)"
        if options.token_budget is not None and usage["total_tokens"] >= options.token_budget:
            return f"team token budget exhausted ({options.token_budget})"
        if options.turn_budget is not None and usage["turns"] >= options.turn_budget:
            return f"team turn budget exhausted ({options.turn_budget})"
        return None

    def _complete_team(self, lead_context: ToolContext, team: Team) -> Team:
        store = lead_context.team_store
        for agent in store.list_agents(team.team_id):
            if agent.status == "created":
                self._transition_agent(store, agent, "running")
                self._transition_agent(store, agent, "completed")
            elif agent.status in {"running", "idle"}:
                self._transition_agent(store, agent, "completed")
            elif agent.status == "failed":
                self._transition_agent(store, agent, "running")
                self._transition_agent(store, agent, "completed")
        current = store.load_team(team.team_id) or team
        if current.status != "completed":
            current.transition_to("completed")
            current.completed_at = utc_now()
            store.save_team(current)
            store.append_event(current.team_id, "team.completed", {"usage": current.usage})
        lead_context.reload_team_state()
        return current

    @staticmethod
    def _fail_team(
        lead_context: ToolContext,
        team: Team,
        error: str,
        executed: list[str],
        *,
        status: str = "failed",
    ) -> dict[str, Any]:
        current = lead_context.team_store.load_team(team.team_id) or team
        if current.status == "running":
            current.transition_to("failed")
            lead_context.team_store.save_team(current)
            lead_context.team_store.append_event(
                current.team_id, "team.failed", {"error": error, "usage": current.usage}
            )
        lead_context.reload_team_state()
        return {
            "status": status,
            "team_id": current.team_id,
            "error": error,
            "executed_task_ids": executed,
            "usage": current.usage,
        }

    @staticmethod
    def _cancelled_result(
        lead_context: ToolContext, team: Team, executed: list[str]
    ) -> dict[str, Any]:
        return {
            "status": "cancelled",
            "team_id": team.team_id,
            "error": "team cancellation requested",
            "executed_task_ids": executed,
            "usage": team.usage,
        }

    @staticmethod
    def _blocked_result(
        lead_context: ToolContext, team: Team, error: str, executed: list[str]
    ) -> dict[str, Any]:
        lead_context.reload_team_state()
        return {
            "status": "blocked",
            "team_id": team.team_id,
            "error": error,
            "executed_task_ids": executed,
            "usage": team.usage,
        }

    @staticmethod
    def _result(
        lead_context: ToolContext, team: Team, executed: list[str]
    ) -> dict[str, Any]:
        agents = lead_context.team_store.list_agents(team.team_id)
        names = {agent.agent_id: agent.name for agent in agents}
        names[team.lead_agent_id] = "lead"
        messages = [
            {
                "message_id": message.message_id,
                "from": names.get(message.sender_id, message.sender_id),
                "to": names.get(message.recipient_id, message.recipient_id),
                "summary": message.summary,
                "status": message.status,
            }
            for message in lead_context.team_store.list_messages(team.team_id)
        ]
        return {
            "status": team.status,
            "team_id": team.team_id,
            "executed_task_ids": executed,
            "tasks": list(lead_context.tasks.values()),
            "messages": messages,
            "usage": team.usage,
        }
