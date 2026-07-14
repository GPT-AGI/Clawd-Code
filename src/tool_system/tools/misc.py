from __future__ import annotations

import platform
import time
import uuid
from typing import Any

from ...teammate.models import Message
from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..protocol import ToolResult
from ..registry import ToolSpec


class SendMessageTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="SendMessage",
            description=(
                "Send and persist a direct message to any teammate or to the lead. "
                "Peer-to-peer messages do not need to pass through the lead."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "to": {"type": "string"},
                    "summary": {"type": "string"},
                    "message": {},
                },
                "required": ["to", "message"],
            },
            is_read_only=False,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        to = tool_input.get("to")
        message = tool_input.get("message")
        summary = tool_input.get("summary")
        if not isinstance(to, str) or not to.strip():
            raise ToolInputError("to must be a non-empty string")
        if summary is not None and not isinstance(summary, str):
            raise ToolInputError("summary must be a string when provided")
        if context.team is None:
            raise ToolInputError("SendMessage requires an active team")

        team_id = str(context.team["team_id"])
        lead_id = str(context.team["lead_agent_id"])
        recipient_name = to.strip()
        if recipient_name.lower() == "lead" or recipient_name == lead_id:
            recipient_id = lead_id
        else:
            recipient = context.team_store.find_agent(team_id, recipient_name)
            if recipient is None:
                raise ToolInputError(f"unknown message recipient: {recipient_name}")
            recipient_id = recipient.agent_id
        sender_id = context.actor_id or lead_id
        if sender_id != lead_id and context.team_store.load_agent(team_id, sender_id) is None:
            raise ToolInputError(f"unknown message sender: {sender_id}")

        persisted = Message(
            message_id=uuid.uuid4().hex,
            team_id=team_id,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=message,
            summary=summary,
        )
        persisted.transition_to("delivered")
        path = context.team_store.save_message(persisted)
        context.team_store.append_event(team_id, "message.delivered", {"message": persisted.to_dict()})
        context.outbox.append(
            {
                "tool": "SendMessage",
                "message_id": persisted.message_id,
                "from": sender_id,
                "to": recipient_id,
                "summary": summary,
                "message": message,
            }
        )
        return ToolResult(
            name="SendMessage",
            output={
                "success": True,
                "message_id": persisted.message_id,
                "sender_id": sender_id,
                "recipient_id": recipient_id,
                "status": persisted.status,
                "message_file_path": str(path),
            },
        )


class ReadMessagesTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ReadMessages",
            description=(
                "Read and consume newly delivered team messages for the current agent. "
                "Use wait_s to wait briefly for a peer response during parallel work."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"wait_s": {"type": "number"}},
            },
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.team is None:
            raise ToolInputError("ReadMessages requires an active team")
        wait_s = tool_input.get("wait_s", 0)
        if isinstance(wait_s, bool) or not isinstance(wait_s, (int, float)):
            raise ToolInputError("wait_s must be numeric")
        if wait_s < 0 or wait_s > 60:
            raise ToolInputError("wait_s must be between 0 and 60")

        team_id = str(context.team["team_id"])
        lead_id = str(context.team["lead_agent_id"])
        recipient_id = context.actor_id or lead_id
        if recipient_id != lead_id and context.team_store.load_agent(team_id, recipient_id) is None:
            raise ToolInputError(f"unknown message recipient: {recipient_id}")

        deadline = time.monotonic() + float(wait_s)
        incoming = context.team_store.consume_messages(team_id, recipient_id)
        while not incoming and time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            incoming = context.team_store.consume_messages(team_id, recipient_id)

        names = {
            agent.agent_id: agent.name for agent in context.team_store.list_agents(team_id)
        }
        names[lead_id] = "lead"
        return ToolResult(
            name="ReadMessages",
            output={
                "messages": [
                    {
                        "message_id": message.message_id,
                        "from": names.get(message.sender_id, message.sender_id),
                        "sender_id": message.sender_id,
                        "summary": message.summary,
                        "message": message.content,
                        "status": message.status,
                    }
                    for message in incoming
                ]
            },
        )


class RemoteTriggerTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="RemoteTrigger",
            description="Trigger a remote action (not implemented).",
            input_schema={"type": "object", "additionalProperties": True},
            is_read_only=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(name="RemoteTrigger", output={"error": "RemoteTrigger is not implemented"}, is_error=True)


class PowerShellTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="PowerShell",
            description="Run a PowerShell command (Windows only).",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            is_destructive=True,
            max_result_size_chars=200_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if platform.system().lower() != "windows":
            return ToolResult(name="PowerShell", output={"error": "PowerShell is only supported on Windows"}, is_error=True)
        raise ToolPermissionError("PowerShell execution is not enabled in this build")


class NotebookEditTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="NotebookEdit",
            description="Edit a Jupyter notebook (not implemented).",
            input_schema={"type": "object", "additionalProperties": True},
            is_destructive=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(name="NotebookEdit", output={"error": "NotebookEdit is not implemented"}, is_error=True)


class REPLTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="REPL",
            description="Interact with the REPL UI (not implemented).",
            input_schema={"type": "object", "additionalProperties": True},
            is_read_only=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(name="REPL", output={"error": "REPL tool is not implemented"}, is_error=True)


class TestingPermissionTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TestingPermission",
            description="Test-only tool (always succeeds).",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(name="TestingPermission", output="TestingPermission executed successfully", content_type="text")
