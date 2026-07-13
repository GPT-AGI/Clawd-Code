from __future__ import annotations

import uuid
from typing import Any

from ...teammate.models import AgentRecord
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
        for field_name, value in (("name", name), ("role", role), ("instructions", instructions)):
            if not isinstance(value, str) or not value.strip():
                raise ToolInputError(f"{field_name} must be a non-empty string")
        if not isinstance(tools, list) or not tools or not all(isinstance(item, str) and item.strip() for item in tools):
            raise ToolInputError("tools must be a non-empty array of tool names")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ToolInputError("model must be a non-empty string when provided")

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
        )
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
                "agent_file_path": str(path),
            },
        )


class TeamRunTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamRun",
            description="Run ready teammate tasks synchronously until the team completes, fails, or is blocked.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            is_read_only=False,
            max_result_size_chars=200_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.team is None:
            raise ToolInputError("no active team")
        if context.teammate_runtime is None:
            raise ToolInputError("teammate runtime is not configured")
        output = context.teammate_runtime.run_team(context)
        return ToolResult(
            name="TeamRun",
            output=output,
            is_error=output.get("status") in {"failed", "blocked"},
        )


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
        context.team_store.disband_active_team()
        context.team = None
        context.tasks = {}
        return ToolResult(name="TeamDelete", output={"success": True, "message": "Team deleted", "team_name": team_name})
