from __future__ import annotations

import uuid
from typing import Any

from ...teammate.control import cancel_team, resume_teammate, stop_teammate
from ...teammate.models import AgentRecord, utc_now
from ...teammate.worktree import TeammateWorktreeManager
from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


def _team_protocol_version(team: dict[str, Any] | None) -> int:
    value = team or {}
    settings = value.get("settings") if isinstance(value.get("settings"), dict) else {}
    quality = (
        settings.get("quality_gates")
        if isinstance(settings.get("quality_gates"), dict)
        else {}
    )
    versions = [1]
    for raw in (
        value.get("protocol_version"),
        settings.get("protocol_version"),
        quality.get("protocol_version"),
    ):
        try:
            versions.append(int(raw))
        except (TypeError, ValueError):
            continue
    return max(versions)


def _reject_v2_incremental_mutation(context: ToolContext, tool_name: str) -> None:
    if _team_protocol_version(context.team) >= 2:
        raise ToolInputError(
            f"{tool_name} cannot mutate a protocol v2 team; submit a complete "
            "TeamPlan replacement instead. If the current non-terminal plan must "
            "be restarted, call TeamReplan first; TeamAbort is terminal and must "
            "not be used as a restart operation"
        )


def _strict_v2(team: Any) -> bool:
    settings = team.settings if isinstance(team.settings, dict) else {}
    quality = (
        settings.get("quality_gates")
        if isinstance(settings.get("quality_gates"), dict)
        else {}
    )
    versions = [1]
    for raw in (
        getattr(team, "protocol_version", 1),
        settings.get("protocol_version"),
        quality.get("protocol_version"),
    ):
        try:
            versions.append(int(raw))
        except (TypeError, ValueError):
            continue
    return bool(quality.get("strict")) and max(versions) >= 2


def _active_team_work(context: ToolContext, team_id: str) -> tuple[list[str], list[str]]:
    active_tasks = [
        str(task.get("key") or task_id)
        for task_id, task in context.team_store.load_tasks(team_id).items()
        if task.get("status") == "in_progress"
    ]
    running_agents = [
        agent.name
        for agent in context.team_store.list_agents(team_id)
        if agent.status in {"running", "stopping"}
    ]
    return active_tasks, running_agents


class TeamCreateTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamCreate",
            description=(
                "Create a team only when the lead decides delegation is worth its cost. "
                "The lead chooses the team shape; no fixed roles or topology are required. "
                "For quality-gated work, follow creation with one atomic TeamPlan and TeamRun. "
                "Creating a team does not start any worker."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "team_name": {"type": "string"},
                    "description": {"type": "string"},
                    "agent_type": {"type": "string"},
                    "quality_gates": {"type": "boolean"},
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
            active = context.team_store.load_active_team()
            if active is not None and _strict_v2(active):
                lifecycle = str(active.lifecycle_state or active.status)
                if lifecycle in {"completed", "aborted", "budget_exhausted"}:
                    raise ToolInputError(
                        f"active strict team is terminal (lifecycle={lifecycle}) and "
                        "cannot be replaced inside this rollout; preserve it for scoring "
                        "and start a new top-level rollout"
                    ) from exc
                raise ToolInputError(
                    "an active recoverable strict team already exists; call TeamReplan "
                    "to request a fresh complete TeamPlan while preserving the workspace. "
                    "Do not call TeamAbort or delete files merely to restart"
                ) from exc
            raise ToolInputError(str(exc)) from exc
        if bool(tool_input.get("quality_gates", False)):
            team.settings["quality_gates"] = {
                "strict": True,
                "configured": False,
                "validation": {"status": "pending"},
            }
            context.team_store.save_team(team)
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
                "quality_gates": bool(tool_input.get("quality_gates", False)),
                "next_required_actions": (
                    [
                        {
                            "tool": "TeamPlan",
                            "instruction": (
                                "Atomically define the contract, real workers, owned tasks, "
                                "acceptance checks, validation, and execution settings."
                            ),
                        },
                        {
                            "tool": "TeamRun",
                            "instruction": (
                                "Run the committed plan; protocol v2 performs acceptance "
                                "and final verification automatically."
                            ),
                        },
                    ]
                    if bool(tool_input.get("quality_gates", False))
                    else [
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
                            "instruction": "Run the owned tasks; team creation alone does not start workers.",
                        },
                    ]
                ),
            },
        )


class TeamConfigureTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamConfigure",
            description=(
                "Configure strict team architecture and final validation gates before TeamRun. "
                "The install and import checks run in a fresh system-site-packages virtual "
                "environment; the integration check runs only after every teammate task finishes."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "architecture_contract": {"type": "string"},
                    "install_command": {"type": "string"},
                    "import_command": {"type": "string"},
                    "integration_command": {"type": "string"},
                },
                "required": [
                    "architecture_contract",
                    "install_command",
                    "import_command",
                    "integration_command",
                ],
            },
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.actor_id is not None:
            raise ToolInputError("only the lead may configure team quality gates")
        if context.team is None:
            raise ToolInputError("no active team")
        _reject_v2_incremental_mutation(context, "TeamConfigure")
        values: dict[str, str] = {}
        for name in (
            "architecture_contract",
            "install_command",
            "import_command",
            "integration_command",
        ):
            value = tool_input.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ToolInputError(f"{name} must be a non-empty string")
            values[name] = value.strip()
        team_id = str(context.team["team_id"])
        team = context.team_store.load_team(team_id)
        if team is None:
            raise ToolInputError("active team state is unavailable")
        quality = dict(team.settings.get("quality_gates") or {})
        quality.update(
            {
                "strict": True,
                "configured": True,
                **values,
                "validation": {"status": "pending"},
            }
        )
        team.settings["quality_gates"] = quality
        context.team_store.save_team(team)
        context.team_store.append_event(
            team_id,
            "team.quality_configured",
            {
                "architecture_contract": values["architecture_contract"],
                "validation_stages": ["install", "import", "integration"],
            },
        )
        context.reload_team_state()
        return ToolResult(
            name="TeamConfigure",
            output={
                "team_id": team_id,
                "strict": True,
                "configured": True,
                "validation_stages": ["install", "import", "integration"],
                "next_required_actions": [
                    {
                        "tool": "TaskCreate",
                        "instruction": (
                            "Create at least two independently ready teammate tasks with "
                            "non-overlapping ownedFiles and acceptanceChecks."
                        ),
                    },
                    {
                        "tool": "TeamRun",
                        "instruction": "Run the validated plan, then call TeamVerify.",
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
        _reject_v2_incremental_mutation(context, "TeammateCreate")
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
        validate_model = getattr(context.teammate_runtime, "validate_model", None)
        if callable(validate_model):
            try:
                model = validate_model(model)
            except ValueError as exc:
                raise ToolInputError(str(exc)) from exc
        if workspace_mode not in {"shared", "worktree"}:
            raise ToolInputError("workspace_mode must be shared or worktree")
        if workspace_mode == "worktree" and context.workspace_backend is not None:
            raise ToolInputError(
                "worktree teammates are not supported by the remote sandbox backend; "
                "use workspace_mode=shared"
            )
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
            model=model,
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
                "conversation": {"messages": [], "max_history": 300},
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


def _add_recovery_guidance(output: dict[str, Any]) -> dict[str, Any]:
    status = str(output.get("status") or "")
    guided = dict(output)
    if status == "repair_required":
        guided["recovery_guidance"] = (
            "Stop active workers and call TeamReplan first so the current workspace and "
            "artifacts are checkpointed, then submit one complete replacement TeamPlan "
            "revision and call TeamRun. Do not use TeamAbort for restart."
        )
    elif status in {"failed", "blocked", "cancelled"}:
        guided["recovery_guidance"] = (
            "This non-terminal team may be recoverable. After active workers stop, call "
            "TeamReplan for a fresh complete plan while preserving the workspace. "
            "TeamAbort is only for an intentional terminal failed outcome."
        )
    elif status == "paused":
        guided["recovery_guidance"] = (
            "Call TeamResume for a transient pause. If the plan itself must change, call "
            "TeamReplan; both preserve the workspace. Do not use TeamAbort for restart."
        )
    elif status == "aborted":
        guided["recovery_guidance"] = (
            "This team is terminal and cannot be resumed or replanned."
        )
    elif status == "budget_exhausted":
        guided["recovery_guidance"] = (
            "This rollout exhausted its frozen execution budget and is terminal. "
            "The workspace is preserved for scoring, but TeamResume and TeamReplan "
            "cannot add budget; start a new top-level rollout if more work is required."
        )
    return guided


class TeamRunTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamRun",
            description=(
                "Run ready teammate tasks with optional parallelism, retries, leases, budgets, "
                "and a batch limit so the lead can inspect and adapt the team between batches. "
                "For protocol v2, this also runs task acceptance and final Team verification "
                "automatically, returning completed or repair_required."
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
        output = _add_recovery_guidance(
            context.teammate_runtime.run_team(context, **_run_options(tool_input))
        )
        return ToolResult(
            name="TeamRun",
            output=output,
            is_error=output.get("status")
            in {
                "failed",
                "blocked",
                "cancelled",
                "aborted",
                "budget_exhausted",
                "repair_required",
            },
        )


class TeamVerifyTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamVerify",
            description=(
                "Run the configured clean-install, import-smoke, and integration checks. "
                "A strict team cannot become completed until all three checks pass. Protocol "
                "v2 normally invokes this automatically through TeamRun; repeated calls after "
                "success are idempotent."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"timeout_s": {"type": "integer"}},
            },
            is_read_only=False,
            max_result_size_chars=200_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.actor_id is not None:
            raise ToolInputError("only the lead may verify a team")
        if context.team is None:
            raise ToolInputError("no active team")
        if context.teammate_runtime is None:
            raise ToolInputError("teammate runtime is not configured")
        timeout_s = tool_input.get("timeout_s", 300)
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, int):
            raise ToolInputError("timeout_s must be an integer")
        if timeout_s < 1 or timeout_s > 900:
            raise ToolInputError("timeout_s must be between 1 and 900")
        output = context.teammate_runtime.verify_team(context, timeout_s=timeout_s)
        return ToolResult(
            name="TeamVerify",
            output=output,
            is_error=output.get("status") != "completed",
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
        output = _add_recovery_guidance(
            context.teammate_runtime.run_team(
                context,
                resume=True,
                retry_failed=bool(tool_input.get("retry_failed", True)),
                retry_cancelled=bool(tool_input.get("retry_cancelled", True)),
                **_run_options(tool_input),
            )
        )
        return ToolResult(
            name="TeamResume",
            output=output,
            is_error=output.get("status")
            in {
                "failed",
                "blocked",
                "cancelled",
                "aborted",
                "budget_exhausted",
                "repair_required",
            },
        )


class TeamCancelTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamCancel",
            description=(
                "Request cooperative cancellation of active workers without deleting team "
                "state or workspace artifacts. For a strict-team restart, wait for workers "
                "to stop and then call TeamReplan; TeamAbort is terminal."
            ),
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
        if _team_protocol_version(context.team) >= 2:
            output = {
                **output,
                "terminal": False,
                "artifacts_preserved": True,
                "next_required_action": (
                    "Wait for active workers to stop, then call TeamReplan for a "
                    "recoverable restart or TeamResume to continue the same plan."
                ),
            }
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


class TeamReplanTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamReplan",
            description=(
                "Request a recoverable replacement of a non-terminal strict protocol v2 "
                "plan. This preserves the current workspace, artifacts, usage, and history; "
                "it never deletes files or terminates the rollout. Use it before a fresh "
                "TeamPlan when ownership, contracts, or task partitioning must change. "
                "TeamAbort is terminal and must not be used for restart/replan."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "reason": {"type": "string"},
                    "replace_completed_work": {"type": "boolean"},
                },
                "required": ["reason"],
            },
            aliases=("TeamReset",),
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.actor_id is not None:
            raise ToolInputError("only the lead may request a team replan")
        reason = tool_input.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ToolInputError("reason must be a non-empty string")
        replace_completed_work = tool_input.get("replace_completed_work", False)
        if not isinstance(replace_completed_work, bool):
            raise ToolInputError("replace_completed_work must be a boolean")

        team = context.team_store.load_active_team()
        if team is None:
            raise ToolInputError("no active team")
        if not _strict_v2(team):
            raise ToolInputError(
                "TeamReplan is only available to strict protocol v2 teams"
            )
        try:
            team, checkpoint = context.team_store.request_team_replan(
                team.team_id,
                reason=reason.strip(),
                replace_completed_work=replace_completed_work,
            )
        except ValueError as exc:
            raise ToolInputError(str(exc)) from exc
        context.reload_team_state()
        return ToolResult(
            name="TeamReplan",
            output={
                "status": "replan_required",
                "team_id": team.team_id,
                "lifecycle_state": "repair_required",
                "reason": reason.strip(),
                "artifacts_preserved": True,
                "workspace_preserved": True,
                "workspace_action": "none",
                "checkpoint": checkpoint,
                "next_required_action": {
                    "tool": "TeamPlan",
                    "instruction": (
                        "Submit one complete replacement plan against the checkpointed "
                        "revision. Reuse the existing workspace; do not delete or recreate it."
                    ),
                },
            },
        )


class TeamAbortTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamAbort",
            description=(
                "Explicitly terminate an unrecoverable strict protocol v2 team while "
                "preserving its active state and artifacts for scoring and diagnosis. "
                "This is a terminal failure outcome: it cannot be resumed or replanned. "
                "For restart or a fresh plan, use TeamReplan instead."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.actor_id is not None:
            raise ToolInputError("only the lead may abort a team")
        reason = tool_input.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ToolInputError("reason must be a non-empty string")
        team = context.team_store.load_active_team()
        if team is None:
            raise ToolInputError("no active team")
        if not _strict_v2(team):
            raise ToolInputError("TeamAbort is only available to strict protocol v2 teams")
        if team.lifecycle_state == "completed":
            raise ToolInputError(
                "a completed team cannot be aborted or reopened; preserve it for scoring"
            )
        if team.lifecycle_state == "budget_exhausted":
            return ToolResult(
                name="TeamAbort",
                output={
                    "status": "budget_exhausted",
                    "team_id": team.team_id,
                    "lifecycle_state": "budget_exhausted",
                    "artifacts_preserved": True,
                    "terminal": True,
                    "already_terminal": True,
                },
                is_error=True,
            )
        if team.lifecycle_state == "aborted":
            return ToolResult(
                name="TeamAbort",
                output={
                    "status": "aborted",
                    "team_id": team.team_id,
                    "lifecycle_state": "aborted",
                    "artifacts_preserved": True,
                    "terminal": True,
                    "already_aborted": True,
                },
            )
        active_tasks, running_agents = _active_team_work(context, team.team_id)
        if active_tasks or running_agents:
            details = []
            if active_tasks:
                details.append("active tasks: " + ", ".join(active_tasks))
            if running_agents:
                details.append("running teammates: " + ", ".join(running_agents))
            raise ToolInputError(
                "cannot abort while workers are active. For recovery, call TeamCancel, "
                "wait for cooperative shutdown, then call TeamReplan. Call TeamAbort "
                "after shutdown only when a terminal failed outcome is intended ("
                + "; ".join(details)
                + ")"
            )
        if team.status != "cancelled":
            team.transition_to("cancelled")
        team.set_lifecycle_state("aborted")
        team.cancel_requested_at = team.cancel_requested_at or utc_now()
        context.team_store.save_team(team)
        context.team_store.append_event(
            team.team_id,
            "team.aborted",
            {"reason": reason.strip(), "usage": team.usage},
        )
        context.reload_team_state()
        return ToolResult(
            name="TeamAbort",
            output={
                "status": "aborted",
                "team_id": team.team_id,
                "lifecycle_state": "aborted",
                "reason": reason.strip(),
                "artifacts_preserved": True,
                "terminal": True,
                "replan_allowed": False,
            },
        )


class TeamDeleteTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamDelete",
            description=(
                "Disband a legacy team context. Strict protocol v2 teams retain their state "
                "for scoring. Recoverable strict teams use TeamReplan; TeamAbort is only "
                "for an intentional terminal failed outcome."
            ),
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.team is None:
            return ToolResult(name="TeamDelete", output={"success": False, "message": "No active team"})
        team_settings = context.team.get("settings") or {}
        quality = dict(team_settings.get("quality_gates") or {})
        versions = [1]
        for raw_version in (
            context.team.get("protocol_version"),
            team_settings.get("protocol_version"),
            quality.get("protocol_version"),
        ):
            try:
                versions.append(int(raw_version))
            except (TypeError, ValueError):
                continue
        protocol_version = max(versions)
        lifecycle_state = str(context.team.get("lifecycle_state") or "draft")
        if (
            bool(quality.get("strict"))
            and protocol_version >= 2
        ):
            terminal = lifecycle_state in {
                "completed",
                "aborted",
                "budget_exhausted",
            }
            return ToolResult(
                name="TeamDelete",
                output={
                    "success": False,
                    "status": "blocked",
                    "message": (
                        "TeamDelete is disabled for strict protocol v2 because its active "
                        "state, workspace, and artifacts must remain available for scoring. "
                        + (
                            "This team is terminal and cannot be reopened."
                            if terminal
                            else "Call TeamReplan for a recoverable fresh plan; do not use "
                            "TeamAbort as a restart operation."
                        )
                    ),
                    "team_id": context.team.get("team_id"),
                    "lifecycle_state": lifecycle_state,
                    "artifacts_preserved": True,
                    "next_required_action": None if terminal else "TeamReplan",
                },
                is_error=True,
            )
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
