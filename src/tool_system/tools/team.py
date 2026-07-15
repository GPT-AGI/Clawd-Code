from __future__ import annotations

import uuid
from typing import Any

from ...teammate.control import cancel_team, resume_teammate, stop_teammate
from ...teammate.models import AgentRecord
from ...teammate.worktree import TeammateWorktreeManager
from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


class TeamCreateTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamCreate",
            description=(
                "Create a team only when the lead decides delegation is worth its cost. "
                "The lead chooses the team shape; no fixed roles or topology are required. "
                "After creation, add teammates and owned tasks, then call TeamRun; creating "
                "a team does not start any worker."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "team_name": {"type": "string"},
                    "description": {"type": "string"},
                    "agent_type": {"type": "string"},
                },
                "required": ["team_name"],
            },
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        team_name = tool_input.get("team_name")
        if not isinstance(team_name, str) or not team_name.strip():
            raise ToolInputError("team_name must be a non-empty string")
        description = tool_input.get("description")
        if description is not None and not isinstance(description, str):
            raise ToolInputError("description must be a string when provided")
        agent_type = tool_input.get("agent_type")
        if agent_type is not None and not isinstance(agent_type, str):
            raise ToolInputError("agent_type must be a string when provided")

        try:
            team = context.team_store.create_team(team_name.strip(), description, agent_type)
        except ValueError as exc:
            raise ToolInputError(str(exc)) from exc
        context.team = team.to_dict()
        context.tasks = {}
        return ToolResult(
            name="TeamCreate",
            output={
                "team_id": team.team_id,
                "team_name": team.team_name,
                "team_file_path": str(context.team_store.team_dir(team.team_id) / "team.json"),
                "lead_agent_id": team.lead_agent_id,
                "team_started": False,
                "next_required_actions": [
                    {
                        "tool": "TeammateCreate",
                        "instruction": "Create each task-specific worker with its role and tool allowlist.",
                    },
                    {
                        "tool": "TaskCreate",
                        "instruction": "Create at least one task owned by a teammate.",
                    },
                    {
                        "tool": "TeamRun",
                        "instruction": "Run the owned tasks; team and teammate creation alone do not start workers.",
                    },
                ],
            },
        )


class TeammateCreateTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeammateCreate",
            description=(
                "Create one persistent teammate with a lead-defined role, model, tool allowlist, "
                "and workspace mode. Roles are task-specific rather than predefined. The new "
                "teammate remains idle until it owns a TaskCreate task and the lead calls TeamRun."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "instructions": {"type": "string"},
                    "tools": {"type": "array", "items": {"type": "string"}},
                    "model": {"type": "string"},
                    "workspace_mode": {"type": "string", "enum": ["shared", "worktree"]},
                    "auto_integrate": {"type": "boolean"},
                },
                "required": ["name", "role", "instructions", "tools"],
            },
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.team is None:
            raise ToolInputError(
                "no active team: call TeamCreate first, then retry TeammateCreate; "
                "afterward create an owned task and call TeamRun"
            )
        name = tool_input.get("name")
        role = tool_input.get("role")
        instructions = tool_input.get("instructions")
        tools = tool_input.get("tools")
        model = tool_input.get("model")
        workspace_mode = tool_input.get("workspace_mode", "shared")
        auto_integrate = bool(tool_input.get("auto_integrate", False))
        for field_name, value in (("name", name), ("role", role), ("instructions", instructions)):
            if not isinstance(value, str) or not value.strip():
                raise ToolInputError(f"{field_name} must be a non-empty string")
        if not isinstance(tools, list) or not tools or not all(isinstance(item, str) and item.strip() for item in tools):
            raise ToolInputError("tools must be a non-empty array of tool names")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ToolInputError("model must be a non-empty string when provided")
        if workspace_mode not in {"shared", "worktree"}:
            raise ToolInputError("workspace_mode must be shared or worktree")
        if auto_integrate and workspace_mode != "worktree":
            raise ToolInputError("auto_integrate requires workspace_mode=worktree")

        team_id = str(context.team["team_id"])
        if context.team_store.find_agent(team_id, name.strip()) is not None:
            raise ToolInputError(f"teammate name already exists: {name.strip()}")
        normalized_tools = list(dict.fromkeys(item.strip() for item in tools))
        validate_tools = getattr(context.teammate_runtime, "validate_tools", None)
        if callable(validate_tools):
            try:
                normalized_tools = validate_tools(normalized_tools)
            except ValueError as exc:
                raise ToolInputError(str(exc)) from exc
        agent = AgentRecord(
            agent_id=uuid.uuid4().hex[:12],
            team_id=team_id,
            name=name.strip(),
            role=role.strip(),
            session_id=uuid.uuid4().hex,
            model=model.strip() if isinstance(model, str) else None,
            instructions=instructions.strip(),
            tools=normalized_tools,
            workspace_mode=workspace_mode,
            auto_integrate=auto_integrate,
        )
        if workspace_mode == "worktree":
            try:
                agent.workspace_path = str(
                    TeammateWorktreeManager(context.workspace_root).create(
                        team_id, agent.agent_id, agent.name
                    )
                )
            except (ValueError, RuntimeError) as exc:
                raise ToolInputError(str(exc)) from exc
        path = context.team_store.save_agent(agent)
        context.team_store.save_session(
            team_id,
            agent.session_id,
            {
                "session_id": agent.session_id,
                "team_id": team_id,
                "agent_id": agent.agent_id,
                "model": agent.model,
                "conversation": {"messages": [], "max_history": 100},
            },
        )
        context.team_store.append_event(team_id, "agent.created", {"agent": agent.to_dict()})
        return ToolResult(
            name="TeammateCreate",
            output={
                "agent_id": agent.agent_id,
                "name": agent.name,
                "role": agent.role,
                "session_id": agent.session_id,
                "workspace_mode": agent.workspace_mode,
                "workspace_path": agent.workspace_path,
                "auto_integrate": agent.auto_integrate,
                "agent_file_path": str(path),
                "worker_started": False,
                "next_required_actions": [
                    {
                        "tool": "TaskCreate",
                        "instruction": f"Create a task with owner={agent.name!r}.",
                    },
                    {
                        "tool": "TeamRun",
                        "instruction": "Call TeamRun after owned tasks exist; TeammateCreate does not execute the worker.",
                    },
                ],
            },
        )


_RUN_PROPERTIES: dict[str, dict[str, Any]] = {
    "max_workers": {"type": "integer"},
    "max_batches": {"type": "integer"},
    "timeout_s": {"type": "number"},
    "token_budget": {"type": "integer"},
    "turn_budget": {"type": "integer"},
    "max_retries": {"type": "integer"},
    "lease_timeout_s": {"type": "integer"},
}


def _run_options(tool_input: dict[str, Any]) -> dict[str, Any]:
    options = {key: tool_input[key] for key in _RUN_PROPERTIES if key in tool_input}
    bounds = {
        "max_workers": (1, 16),
        "max_batches": (1, 10_000),
        "timeout_s": (1, 86_400),
        "token_budget": (1, 100_000_000),
        "turn_budget": (1, 100_000),
        "max_retries": (0, 10),
        "lease_timeout_s": (5, 86_400),
    }
    for name, value in options.items():
        minimum, maximum = bounds[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolInputError(f"{name} must be numeric")
        if value < minimum or value > maximum:
            raise ToolInputError(f"{name} must be between {minimum} and {maximum}")
        if name != "timeout_s" and not isinstance(value, int):
            raise ToolInputError(f"{name} must be an integer")
    return options


class TeamRunTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamRun",
            description=(
                "Run ready teammate tasks with optional parallelism, retries, leases, budgets, "
                "and a batch limit so the lead can inspect and adapt the team between batches."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": _RUN_PROPERTIES,
            },
            is_read_only=False,
            max_result_size_chars=200_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.team is None:
            raise ToolInputError("no active team")
        if context.teammate_runtime is None:
            raise ToolInputError("teammate runtime is not configured")
        output = context.teammate_runtime.run_team(context, **_run_options(tool_input))
        return ToolResult(
            name="TeamRun",
            output=output,
            is_error=output.get("status") in {"failed", "blocked", "cancelled"},
        )


class TeamResumeTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamResume",
            description="Resume a failed or cancelled team, recovering expired leases and optionally retrying failed tasks.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    **_RUN_PROPERTIES,
                    "retry_failed": {"type": "boolean"},
                    "retry_cancelled": {"type": "boolean"},
                },
            },
            is_read_only=False,
            max_result_size_chars=200_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.team is None:
            raise ToolInputError("no active team")
        if context.teammate_runtime is None:
            raise ToolInputError("teammate runtime is not configured")
        output = context.teammate_runtime.run_team(
            context,
            resume=True,
            retry_failed=bool(tool_input.get("retry_failed", True)),
            retry_cancelled=bool(tool_input.get("retry_cancelled", True)),
            **_run_options(tool_input),
        )
        return ToolResult(
            name="TeamResume",
            output=output,
            is_error=output.get("status") in {"failed", "blocked", "cancelled"},
        )


class TeamCancelTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamCancel",
            description="Request cooperative cancellation of the active team without deleting its state.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"reason": {"type": "string"}},
            },
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        reason = tool_input.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ToolInputError("reason must be a string when provided")
        try:
            output = cancel_team(context.team_store, reason)
        except ValueError as exc:
            raise ToolInputError(str(exc)) from exc
        context.reload_team_state()
        return ToolResult(name="TeamCancel", output=output)


class TeammateStopTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeammateStop",
            description="Stop one teammate without cancelling the team; unfinished tasks may be requeued or cancelled.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "teammate": {"type": "string"},
                    "reason": {"type": "string"},
                    "task_policy": {
                        "type": "string",
                        "enum": ["requeue", "cancel"],
                    },
                },
                "required": ["teammate"],
            },
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.actor_id is not None:
            raise ToolInputError("only the lead may stop teammates")
        identity = tool_input.get("teammate")
        reason = tool_input.get("reason")
        policy = tool_input.get("task_policy", "requeue")
        if not isinstance(identity, str) or not identity.strip():
            raise ToolInputError("teammate must be a non-empty name or ID")
        if reason is not None and not isinstance(reason, str):
            raise ToolInputError("reason must be a string when provided")
        if not isinstance(policy, str):
            raise ToolInputError("task_policy must be requeue or cancel")
        try:
            output = stop_teammate(
                context.team_store,
                identity.strip(),
                task_policy=policy,
                reason=reason,
            )
        except ValueError as exc:
            raise ToolInputError(str(exc)) from exc
        context.reload_team_state()
        return ToolResult(name="TeammateStop", output=output)


class TeammateResumeTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeammateResume",
            description="Resume a fully stopped teammate so the lead may assign new work.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"teammate": {"type": "string"}},
                "required": ["teammate"],
            },
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.actor_id is not None:
            raise ToolInputError("only the lead may resume teammates")
        identity = tool_input.get("teammate")
        if not isinstance(identity, str) or not identity.strip():
            raise ToolInputError("teammate must be a non-empty name or ID")
        try:
            output = resume_teammate(context.team_store, identity.strip())
        except ValueError as exc:
            raise ToolInputError(str(exc)) from exc
        context.reload_team_state()
        return ToolResult(name="TeammateResume", output=output)


class TeamIntegrateTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamIntegrate",
            description="Commit and cherry-pick an isolated teammate worktree into the lead workspace.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"teammate": {"type": "string"}},
                "required": ["teammate"],
            },
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.actor_id is not None:
            raise ToolInputError("only the lead may integrate teammate worktrees")
        if context.team is None:
            raise ToolInputError("no active team")
        identity = tool_input.get("teammate")
        if not isinstance(identity, str) or not identity.strip():
            raise ToolInputError("teammate must be a non-empty name or ID")
        agent = context.team_store.find_agent(str(context.team["team_id"]), identity.strip())
        if agent is None:
            raise ToolInputError(f"unknown teammate: {identity}")
        try:
            result = TeammateWorktreeManager(context.workspace_root).integrate(agent)
        except (ValueError, RuntimeError) as exc:
            raise ToolInputError(str(exc)) from exc
        context.team_store.append_event(
            agent.team_id,
            "worktree.integrated",
            {"agent_id": agent.agent_id, **result},
        )
        return ToolResult(name="TeamIntegrate", output=result)


class TeamDeleteTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamDelete",
            description="Disband the current team context.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.team is None:
            return ToolResult(name="TeamDelete", output={"success": False, "message": "No active team"})
        team_name = context.team.get("team_name")
        retained_worktrees: list[str] = []
        for agent in context.team_store.list_agents(str(context.team["team_id"])):
            if agent.workspace_mode != "worktree" or not agent.workspace_path:
                continue
            try:
                TeammateWorktreeManager(context.workspace_root).remove(agent)
            except (ValueError, RuntimeError):
                retained_worktrees.append(agent.workspace_path)
        context.team_store.disband_active_team()
        context.team = None
        context.tasks = {}
        return ToolResult(
            name="TeamDelete",
            output={
                "success": True,
                "message": "Team deleted",
                "team_name": team_name,
                "retained_worktrees": retained_worktrees,
            },
        )
