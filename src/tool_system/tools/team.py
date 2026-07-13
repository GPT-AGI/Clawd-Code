from __future__ import annotations

import uuid
from typing import Any

from ...teammate.models import AgentRecord, utc_now
from ...teammate.worktree import TeammateWorktreeManager
from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


class TeamCreateTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamCreate",
            description="Create a lightweight team context for multi-agent workflows.",
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
            },
        )


class TeammateCreateTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeammateCreate",
            description="Create a persistent teammate with an independent session and an explicit tool allowlist.",
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
            raise ToolInputError("create a team before creating teammates")
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
            },
        )


_RUN_PROPERTIES: dict[str, dict[str, Any]] = {
    "max_workers": {"type": "integer"},
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
            description="Run ready teammate tasks with optional parallelism, retries, leases, timeout, and usage budgets.",
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
        team = context.team_store.load_active_team()
        if team is None:
            raise ToolInputError("no active team")
        if team.status == "completed":
            raise ToolInputError("completed teams cannot be cancelled")
        reason = tool_input.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ToolInputError("reason must be a string when provided")
        team.cancel_requested_at = utc_now()
        if team.status != "cancelled":
            team.transition_to("cancelled")
        context.team_store.save_team(team)
        context.team_store.append_event(
            team.team_id,
            "team.cancel_requested",
            {"reason": reason or "cancelled by user"},
        )
        context.reload_team_state()
        return ToolResult(
            name="TeamCancel",
            output={"status": "cancelled", "team_id": team.team_id},
        )


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
