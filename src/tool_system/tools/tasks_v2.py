from __future__ import annotations

import uuid
import re
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec
from ...teammate.models import TeamTask, utc_now


_TASK_STATUSES = TeamTask.STATUSES


def _new_task_id() -> str:
    return uuid.uuid4().hex[:12]


def _task_key(subject: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", subject.strip().lower()).strip("-")
    return normalized or _new_task_id()


def _resolve_task_id(tasks: dict[str, dict[str, Any]], identity: str) -> str | None:
    if identity in tasks:
        return identity
    normalized = identity.strip().lower()
    matches = [
        task_id
        for task_id, task in tasks.items()
        if str(task.get("key") or "").lower() == normalized
    ]
    return matches[0] if len(matches) == 1 else None


def _resolve_owner(context: ToolContext, identity: str) -> str:
    if context.team is None:
        return identity
    team_id = str(context.team["team_id"])
    lead_id = str(context.team["lead_agent_id"])
    if identity.strip().lower() == "lead" or identity == lead_id:
        return lead_id
    agent = context.team_store.find_agent(team_id, identity)
    if agent is None:
        raise ToolInputError(f"unknown task owner: {identity}")
    return agent.agent_id


def _resolve_dependencies(context: ToolContext, identities: Any, field_name: str) -> list[str]:
    if identities is None:
        return []
    if not isinstance(identities, list) or not all(isinstance(item, str) and item.strip() for item in identities):
        raise ToolInputError(f"{field_name} must be an array of task IDs or keys")
    resolved: list[str] = []
    for identity in identities:
        task_id = _resolve_task_id(context.tasks, identity)
        if task_id is None:
            raise ToolInputError(f"unknown task dependency: {identity}")
        if task_id not in resolved:
            resolved.append(task_id)
    return resolved


class TaskCreateTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TaskCreate",
            description="Create a task. For teammate work, provide a stable key, owner name, and blockedBy task keys.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "key": {"type": "string"},
                    "subject": {"type": "string"},
                    "description": {"type": "string"},
                    "activeForm": {"type": "string"},
                    "owner": {"type": "string"},
                    "blockedBy": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                },
                "required": ["subject", "description"],
            },
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        subject = tool_input.get("subject")
        description = tool_input.get("description")
        active_form = tool_input.get("activeForm") or ""
        requested_key = tool_input.get("key")
        owner = tool_input.get("owner")
        metadata = tool_input.get("metadata") or {}
        if not isinstance(subject, str) or not subject.strip():
            raise ToolInputError("subject must be a non-empty string")
        if not isinstance(description, str) or not description.strip():
            raise ToolInputError("description must be a non-empty string")
        if not isinstance(active_form, str):
            raise ToolInputError("activeForm must be a string when provided")
        if requested_key is not None and (not isinstance(requested_key, str) or not requested_key.strip()):
            raise ToolInputError("key must be a non-empty string when provided")
        if owner is not None and (not isinstance(owner, str) or not owner.strip()):
            raise ToolInputError("owner must be a non-empty string when provided")
        if not isinstance(metadata, dict):
            raise ToolInputError("metadata must be an object when provided")

        key = requested_key.strip() if isinstance(requested_key, str) else _task_key(subject)
        if any(str(task.get("key") or "").lower() == key.lower() for task in context.tasks.values()):
            raise ToolInputError(f"task key already exists: {key}")
        dependencies = _resolve_dependencies(context, tool_input.get("blockedBy"), "blockedBy")
        task_id = _new_task_id()
        context.tasks[task_id] = TeamTask(
            id=task_id,
            subject=subject.strip(),
            description=description,
            key=key,
            activeForm=active_form,
            owner=_resolve_owner(context, owner.strip()) if isinstance(owner, str) else None,
            blockedBy=dependencies,
            metadata=dict(metadata),
        ).to_dict()
        for dependency_id in dependencies:
            blocks = list(context.tasks[dependency_id].get("blocks") or [])
            if task_id not in blocks:
                blocks.append(task_id)
                context.tasks[dependency_id]["blocks"] = blocks
                context.tasks[dependency_id]["updated_at"] = utc_now()
        context.persist_tasks()
        if context.team is not None:
            context.team_store.append_event(
                str(context.team["team_id"]), "task.created", {"task": context.tasks[task_id]}
            )
        return ToolResult(
            name="TaskCreate",
            output={
                "task": {
                    "id": task_id,
                    "key": key,
                    "subject": subject,
                    "owner": context.tasks[task_id].get("owner"),
                    "blockedBy": dependencies,
                }
            },
        )


class TaskGetTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TaskGet",
            description="Retrieve a task by internal ID or stable key.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"taskId": {"type": "string"}},
                "required": ["taskId"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        identity = tool_input.get("taskId")
        if not isinstance(identity, str) or not identity.strip():
            raise ToolInputError("taskId must be a non-empty task ID or key")
        task_id = _resolve_task_id(context.tasks, identity)
        task = context.tasks.get(task_id) if task_id is not None else None
        if task is None:
            return ToolResult(name="TaskGet", output={"task": None})
        return ToolResult(
            name="TaskGet",
            output={
                "task": {
                    "id": task["id"],
                    "key": task.get("key"),
                    "subject": task["subject"],
                    "description": task["description"],
                    "status": task["status"],
                    "blocks": list(task.get("blocks") or []),
                    "blockedBy": list(task.get("blockedBy") or []),
                    "owner": task.get("owner"),
                    "output": task.get("output") or "",
                }
            },
        )


class TaskListTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TaskList",
            description="List all tasks.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        tasks = []
        for t in context.tasks.values():
            tasks.append(
                {
                    "id": t["id"],
                    "key": t.get("key"),
                    "subject": t["subject"],
                    "status": t["status"],
                    **({"owner": t["owner"]} if t.get("owner") else {}),
                    "blockedBy": list(t.get("blockedBy") or []),
                }
            )
        tasks.sort(key=lambda x: x["id"])
        return ToolResult(name="TaskList", output={"tasks": tasks})


class TaskUpdateTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TaskUpdate",
            description=(
                "Update a task by internal ID or stable key. Canonical statuses are pending, "
                "in_progress, completed, failed, and cancelled; done is accepted as completed."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "taskId": {"type": "string"},
                    "subject": {"type": "string"},
                    "description": {"type": "string"},
                    "activeForm": {"type": "string"},
                    "status": {"type": "string"},
                    "addBlocks": {"type": "array", "items": {"type": "string"}},
                    "addBlockedBy": {"type": "array", "items": {"type": "string"}},
                    "owner": {"type": "string"},
                    "output": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["taskId"],
            },
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        identity = tool_input.get("taskId")
        if not isinstance(identity, str) or not identity.strip():
            raise ToolInputError("taskId must be a non-empty task ID or key")
        task_id = _resolve_task_id(context.tasks, identity)
        task = context.tasks.get(task_id) if task_id is not None else None
        if task is None:
            return ToolResult(
                name="TaskUpdate",
                output={"success": False, "taskId": identity, "updatedFields": [], "error": "Task not found"},
            )
        if context.actor_id is not None and context.current_task_id is not None and task_id != context.current_task_id:
            raise ToolInputError("teammates may only update their current task")

        updated_fields: list[str] = []
        status_change: dict[str, str] | None = None
        requested_status = tool_input.get("status")
        if requested_status == "done":
            requested_status = "completed"
        if context.actor_id is not None:
            structural = {"owner", "addBlocks", "addBlockedBy"}
            if structural.intersection(tool_input):
                raise ToolInputError("teammates cannot change task ownership or dependencies")
            if requested_status == "deleted":
                raise ToolInputError("teammates cannot delete tasks")
        if requested_status is not None:
            if not isinstance(requested_status, str) or (
                requested_status not in _TASK_STATUSES and requested_status != "deleted"
            ):
                raise ToolInputError(
                    "status must be pending|in_progress|completed|failed|cancelled|deleted when provided"
                )
            if requested_status == "deleted":
                context.tasks.pop(task_id, None)
                context.persist_tasks()
                return ToolResult(
                    name="TaskUpdate",
                    output={"success": True, "taskId": task_id, "updatedFields": ["deleted"]},
                )
            if requested_status != task.get("status"):
                task_state = TeamTask.from_dict(task)
                try:
                    task_state.transition_to(requested_status)
                except ValueError as exc:
                    raise ToolInputError(str(exc)) from exc
                status_change = {"from": str(task.get("status")), "to": requested_status}

        for field in ("subject", "description", "activeForm", "output"):
            if field in tool_input and tool_input[field] is not None:
                v = tool_input[field]
                if not isinstance(v, str):
                    raise ToolInputError(f"{field} must be a string when provided")
                if v != task.get(field):
                    task[field] = v
                    updated_fields.append(field)

        if "owner" in tool_input and tool_input["owner"] is not None:
            value = tool_input["owner"]
            if not isinstance(value, str) or not value.strip():
                raise ToolInputError("owner must be a non-empty string when provided")
            resolved_owner = _resolve_owner(context, value.strip())
            if resolved_owner != task.get("owner"):
                task["owner"] = resolved_owner
                updated_fields.append("owner")

        if status_change is not None:
            task["status"] = requested_status
            updated_fields.append("status")

        for rel_field, input_key in (("blocks", "addBlocks"), ("blockedBy", "addBlockedBy")):
            if input_key in tool_input and tool_input[input_key] is not None:
                ids = _resolve_dependencies(context, tool_input[input_key], input_key)
                if task_id in ids:
                    raise ToolInputError("a task cannot depend on or block itself")
                cur = list(task.get(rel_field) or [])
                for x in ids:
                    if x not in cur:
                        cur.append(x)
                if cur != task.get(rel_field):
                    task[rel_field] = cur
                    updated_fields.append(rel_field)
                reciprocal = "blocks" if rel_field == "blockedBy" else "blockedBy"
                for related_id in ids:
                    related = context.tasks[related_id]
                    values = list(related.get(reciprocal) or [])
                    if task_id not in values:
                        values.append(task_id)
                        related[reciprocal] = values
                        related["updated_at"] = utc_now()

        if "metadata" in tool_input and tool_input["metadata"] is not None:
            md = tool_input["metadata"]
            if not isinstance(md, dict):
                raise ToolInputError("metadata must be an object when provided")
            existing = dict(task.get("metadata") or {})
            for k, v in md.items():
                if v is None:
                    existing.pop(k, None)
                else:
                    existing[k] = v
            task["metadata"] = existing
            updated_fields.append("metadata")

        if updated_fields:
            task["updated_at"] = utc_now()
            context.persist_tasks()

        out: dict[str, Any] = {"success": True, "taskId": task_id, "updatedFields": updated_fields}
        if status_change is not None:
            out["statusChange"] = status_change
        return ToolResult(name="TaskUpdate", output=out)


class TaskOutputTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TaskOutput",
            description="Get output for a task by internal ID or stable key (best-effort).",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "task_id": {"type": "string"},
                    "block": {"type": "boolean"},
                    "timeout": {"type": "number"},
                },
                "required": ["task_id"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
            aliases=("AgentOutputTool", "BashOutputTool"),
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        identity = tool_input.get("task_id")
        if not isinstance(identity, str) or not identity.strip():
            raise ToolInputError("task_id must be a non-empty task ID or key")

        task_id = _resolve_task_id(context.tasks, identity)
        task = context.tasks.get(task_id) if task_id is not None else None
        if task is None:
            return ToolResult(name="TaskOutput", output={"retrieval_status": "success", "task": None})

        output = str(task.get("output") or "")
        retrieval_status = "success" if output else "not_ready"
        return ToolResult(
            name="TaskOutput",
            output={
                "retrieval_status": retrieval_status,
                "task": {
                    "task_id": task_id,
                    "task_type": "task_list",
                    "status": task.get("status"),
                    "description": task.get("description"),
                    "output": output,
                },
            },
        )


class TaskRetryTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TaskRetry",
            description="Reset a failed or cancelled task to pending for an explicit retry.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "taskId": {"type": "string"},
                    "clearOutput": {"type": "boolean"},
                },
                "required": ["taskId"],
            },
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.actor_id is not None:
            raise ToolInputError("only the lead may retry tasks")
        identity = tool_input.get("taskId")
        if not isinstance(identity, str) or not identity.strip():
            raise ToolInputError("taskId must be a non-empty task ID or key")
        task_id = _resolve_task_id(context.tasks, identity)
        if task_id is None:
            raise ToolInputError(f"unknown task: {identity}")
        task = TeamTask.from_dict(context.tasks[task_id])
        if task.status not in {"failed", "cancelled"}:
            raise ToolInputError("only failed or cancelled tasks can be retried")
        previous = task.status
        task.transition_to("pending")
        task.lease_id = None
        task.lease_expires_at = None
        task.completed_at = None
        if bool(tool_input.get("clearOutput", True)):
            task.output = ""
        context.tasks[task.id] = task.to_dict()
        context.persist_tasks()
        if context.team is not None:
            context.team_store.append_event(
                str(context.team["team_id"]),
                "task.retry_requested",
                {"task_id": task.id, "from": previous, "attempt": task.attempt},
            )
        return ToolResult(
            name="TaskRetry",
            output={"success": True, "taskId": task.id, "status": "pending"},
        )
