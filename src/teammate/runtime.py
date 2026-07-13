from __future__ import annotations

from typing import Any

from ..agent.conversation import Conversation
from ..tool_system.agent_loop import run_agent_loop
from ..tool_system.context import ToolContext
from ..tool_system.permissions import ToolPermissionContext
from ..tool_system.registry import ToolRegistry
from .models import AgentRecord, Message, Team, TeamTask, utc_now


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
    "TeamDelete",
}


class TeammateRuntime:
    """Deterministic, synchronous scheduler for persistent teammate tasks."""

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

    def run_team(self, lead_context: ToolContext) -> dict[str, Any]:
        team = lead_context.team_store.load_active_team()
        if team is None:
            return {"status": "failed", "error": "no active team"}
        if team.status == "completed":
            lead_context.reload_team_state()
            return self._result(lead_context, team, [])
        if team.status == "cancelled":
            return {"status": "failed", "error": "team is cancelled", "team_id": team.team_id}

        try:
            if team.status in {"created", "failed"}:
                team.transition_to("running")
                lead_context.team_store.save_team(team)
                lead_context.team_store.append_event(team.team_id, "team.running")
            executed: list[str] = []

            while True:
                lead_context.reload_team_state()
                tasks = lead_context.tasks
                if not tasks:
                    return self._fail_team(lead_context, team, "team has no tasks", executed)
                if all(task.get("status") == "completed" for task in tasks.values()):
                    team = self._complete_team(lead_context, team)
                    return self._result(lead_context, team, executed)

                failed = [task for task in tasks.values() if task.get("status") == "failed"]
                if failed:
                    names = ", ".join(str(task.get("key") or task.get("id")) for task in failed)
                    return self._fail_team(lead_context, team, f"failed tasks: {names}", executed)

                ready = [
                    task
                    for task in tasks.values()
                    if task.get("status") == "pending" and self._dependencies_completed(task, tasks)
                ]
                if not ready:
                    pending = [
                        str(task.get("key") or task.get("id"))
                        for task in tasks.values()
                        if task.get("status") in {"pending", "in_progress"}
                    ]
                    reason = "no runnable tasks; check dependencies and task states"
                    if pending:
                        reason += f" ({', '.join(pending)})"
                    return self._fail_team(lead_context, team, reason, executed, status="blocked")

                ready.sort(key=lambda task: (str(task.get("created_at") or ""), str(task.get("id") or "")))
                for task_data in ready:
                    task = TeamTask.from_dict(task_data)
                    outcome = self._run_task(lead_context, team, task)
                    executed.append(task.id)
                    if outcome == "failed":
                        return self._fail_team(
                            lead_context,
                            team,
                            f"task failed: {task.key or task.id}",
                            executed,
                        )
        except Exception as exc:
            return self._fail_team(lead_context, team, str(exc), locals().get("executed", []))

    @staticmethod
    def _dependencies_completed(
        task: dict[str, Any], tasks: dict[str, dict[str, Any]]
    ) -> bool:
        dependencies = list(task.get("blockedBy") or [])
        return all(
            dependency in tasks and tasks[dependency].get("status") == "completed"
            for dependency in dependencies
        )

    def _run_task(self, lead_context: ToolContext, team: Team, task: TeamTask) -> str:
        if not task.owner:
            self._set_task_failed(lead_context, team, task, "task has no owner")
            return "failed"
        agent = lead_context.team_store.load_agent(team.team_id, task.owner)
        if agent is None:
            self._set_task_failed(lead_context, team, task, f"unknown task owner: {task.owner}")
            return "failed"

        try:
            task.transition_to("in_progress")
            self._save_task(lead_context, task)
            self._transition_agent(lead_context, agent, "running")
            lead_context.team_store.append_event(
                team.team_id,
                "task.started",
                {"task_id": task.id, "task_key": task.key, "agent_id": agent.agent_id},
            )

            conversation = self._load_conversation(lead_context, agent)
            incoming = self._consume_messages(lead_context, team, agent)
            conversation.add_user_message(self._task_prompt(team, agent, task, incoming))
            self._save_session(lead_context, agent, conversation)

            child_context = self._child_context(lead_context, team, agent, task)
            result = run_agent_loop(
                conversation=conversation,
                provider=self.provider,
                tool_registry=self._child_registry(agent),
                tool_context=child_context,
                max_turns=self.max_turns,
                stream=False,
                verbose=False,
            )
            self._save_session(lead_context, agent, conversation)
            lead_context.reload_team_state()
            persisted = TeamTask.from_dict(lead_context.tasks[task.id])
            if result.response_text == "[Max tool turns reached]":
                self._set_task_failed(lead_context, team, persisted, result.response_text)
                self._transition_agent(lead_context, agent, "failed")
                return "failed"
            if persisted.status == "failed":
                if not persisted.output:
                    persisted.output = result.response_text
                    persisted.updated_at = utc_now()
                    self._save_task(lead_context, persisted)
                self._transition_agent(lead_context, agent, "failed")
                return "failed"
            if persisted.status == "cancelled":
                self._transition_agent(lead_context, agent, "cancelled")
                lead_context.team_store.append_event(
                    team.team_id,
                    "task.cancelled",
                    {"task_id": task.id, "task_key": task.key, "agent_id": agent.agent_id},
                )
                return "failed"
            if persisted.status == "in_progress":
                persisted.output = result.response_text
                persisted.transition_to("completed")
                self._save_task(lead_context, persisted)
            elif persisted.status == "completed" and not persisted.output:
                persisted.output = result.response_text
                persisted.updated_at = utc_now()
                self._save_task(lead_context, persisted)
            self._transition_agent(lead_context, agent, "idle")
            lead_context.team_store.append_event(
                team.team_id,
                "task.completed",
                {"task_id": task.id, "task_key": task.key, "agent_id": agent.agent_id},
            )
            return "completed"
        except Exception as exc:
            lead_context.reload_team_state()
            current = TeamTask.from_dict(lead_context.tasks.get(task.id, task.to_dict()))
            if current.status in {"pending", "in_progress"}:
                self._set_task_failed(lead_context, team, current, str(exc))
            latest_agent = lead_context.team_store.load_agent(team.team_id, agent.agent_id) or agent
            if latest_agent.status in {"created", "running", "idle"}:
                self._transition_agent(lead_context, latest_agent, "failed")
            return "failed"

    def _child_registry(self, agent: AgentRecord) -> ToolRegistry:
        names = self.validate_tools([*agent.tools, *_MANDATORY_TEAMMATE_TOOLS])
        return ToolRegistry(self.registry.get(name) for name in names)

    @staticmethod
    def _child_context(
        lead_context: ToolContext, team: Team, agent: AgentRecord, task: TeamTask
    ) -> ToolContext:
        permissions = ToolPermissionContext.from_iterables(
            workspace_root=lead_context.workspace_root,
            additional_working_directories=lead_context.permission_context.additional_working_directories,
            allow_docs=lead_context.permission_context.allow_docs,
        )
        context = ToolContext(
            workspace_root=lead_context.workspace_root,
            permission_context=permissions,
            cwd=lead_context.cwd,
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
    def _load_conversation(lead_context: ToolContext, agent: AgentRecord) -> Conversation:
        data = lead_context.team_store.load_session(agent.team_id, agent.session_id)
        if data is None:
            return Conversation()
        conversation = data.get("conversation")
        return Conversation.from_dict(conversation) if isinstance(conversation, dict) else Conversation()

    def _save_session(
        self, lead_context: ToolContext, agent: AgentRecord, conversation: Conversation
    ) -> None:
        lead_context.team_store.save_session(
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
        lead_context: ToolContext, team: Team, agent: AgentRecord
    ) -> list[Message]:
        incoming = [
            message
            for message in lead_context.team_store.list_messages(team.team_id)
            if message.recipient_id == agent.agent_id and message.status == "delivered"
        ]
        for message in incoming:
            message.transition_to("consumed")
            lead_context.team_store.save_message(message)
            lead_context.team_store.append_event(
                team.team_id,
                "message.consumed",
                {"message_id": message.message_id, "agent_id": agent.agent_id},
            )
        return incoming

    @staticmethod
    def _save_task(lead_context: ToolContext, task: TeamTask) -> None:
        lead_context.tasks[task.id] = task.to_dict()
        lead_context.persist_tasks()

    @staticmethod
    def _set_task_failed(
        lead_context: ToolContext, team: Team, task: TeamTask, error: str
    ) -> None:
        if task.status == "pending":
            task.transition_to("in_progress")
        if task.status == "in_progress":
            task.transition_to("failed")
        task.output = error
        task.updated_at = utc_now()
        TeammateRuntime._save_task(lead_context, task)
        lead_context.team_store.append_event(
            team.team_id,
            "task.failed",
            {"task_id": task.id, "task_key": task.key, "error": error},
        )

    @staticmethod
    def _transition_agent(
        lead_context: ToolContext, agent: AgentRecord, status: str
    ) -> None:
        if agent.status != status:
            agent.transition_to(status)
            lead_context.team_store.save_agent(agent)
            lead_context.team_store.append_event(
                agent.team_id,
                f"agent.{status}",
                {"agent_id": agent.agent_id, "name": agent.name},
            )

    def _complete_team(self, lead_context: ToolContext, team: Team) -> Team:
        for agent in lead_context.team_store.list_agents(team.team_id):
            if agent.status == "created":
                self._transition_agent(lead_context, agent, "running")
                self._transition_agent(lead_context, agent, "completed")
            elif agent.status in {"running", "idle"}:
                self._transition_agent(lead_context, agent, "completed")
            elif agent.status == "failed":
                self._transition_agent(lead_context, agent, "running")
                self._transition_agent(lead_context, agent, "completed")
        current = lead_context.team_store.load_team(team.team_id) or team
        if current.status != "completed":
            current.transition_to("completed")
            lead_context.team_store.save_team(current)
            lead_context.team_store.append_event(current.team_id, "team.completed")
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
            lead_context.team_store.append_event(current.team_id, "team.failed", {"error": error})
        lead_context.reload_team_state()
        return {
            "status": status,
            "team_id": current.team_id,
            "error": error,
            "executed_task_ids": executed,
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
        }
