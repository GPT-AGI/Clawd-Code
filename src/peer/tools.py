from __future__ import annotations

from typing import Any

from ..tool_system.context import ToolContext
from ..tool_system.errors import ToolInputError
from ..tool_system.protocol import ToolResult
from ..tool_system.registry import ToolSpec
from .policy import PolicyRejected


def _peer_context(context: ToolContext) -> tuple[Any, str, str, Any]:
    if (
        context.peer_store is None
        or context.peer_run_id is None
        or context.peer_id is None
        or context.peer_control is None
    ):
        raise ToolInputError("peer tool requires an active peer run context")
    return context.peer_store, context.peer_run_id, context.peer_id, context.peer_control


class PeerListTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="PeerList",
            description="List the equal participants and coarse lifecycle status in this peer run.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            is_read_only=True,
            strict=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        store, run_id, peer_id, control = _peer_context(context)
        peers = [
            {
                "peer_id": peer.peer_id,
                "name": peer.name,
                "status": peer.status,
                "session_id": peer.session_id,
            }
            for peer in store.list_participants(run_id)
        ]
        return ToolResult(
            name="PeerList",
            output={
                "run_id": run_id,
                "self_peer_id": peer_id,
                "communication": control.policy.condition,
                "peers": peers,
            },
        )


class PeerSendMessageTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="SendMessage",
            description=(
                "Send a durable direct message to an allowed peer. The transport enforces "
                "the run's communication graph; idempotency_key makes retries return the same delivery."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "to": {"type": "string"},
                    "message": {},
                    "summary": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["to", "message"],
            },
            is_read_only=False,
            strict=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        store, run_id, peer_id, control = _peer_context(context)
        recipient = tool_input.get("to")
        if not isinstance(recipient, str) or not recipient.strip():
            raise ToolInputError("to must be a non-empty peer ID or name")
        try:
            message = store.send_message(
                run_id,
                peer_id,
                recipient.strip(),
                tool_input.get("message"),
                policy=control.policy,
                summary=tool_input.get("summary"),
                idempotency_key=tool_input.get("idempotency_key"),
            )
        except (ValueError, PolicyRejected) as exc:
            raise ToolInputError(str(exc)) from exc
        return ToolResult(
            name="SendMessage",
            output={
                "message_id": message.message_id,
                "sender_id": message.sender_id,
                "recipient_id": message.recipient_id,
                "status": message.status,
                "created_at": message.created_at,
                "delivered_at": message.delivered_at,
            },
        )


class PeerReadMessagesTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ReadMessages",
            description=(
                "Wait without busy polling, then atomically consume unread messages. "
                "Each delivered message is returned for execution exactly once."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "wait_seconds": {"type": "number"},
                    "include_consumed": {"type": "boolean"},
                },
            },
            is_read_only=False,
            strict=True,
            max_result_size_chars=200_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        store, run_id, peer_id, control = _peer_context(context)
        wait_seconds = tool_input.get("wait_seconds", 0)
        if isinstance(wait_seconds, bool) or not isinstance(wait_seconds, (int, float)):
            raise ToolInputError("wait_seconds must be numeric")
        if wait_seconds < 0 or wait_seconds > 300:
            raise ToolInputError("wait_seconds must be between 0 and 300")
        if wait_seconds and not store.has_unread(run_id, peer_id):
            store.wait_for_unread(
                run_id,
                peer_id,
                float(wait_seconds),
                stop_event=control.stop_event,
            )
        messages = store.consume_messages(run_id, peer_id)
        if tool_input.get("include_consumed"):
            seen = {message.message_id for message in messages}
            messages.extend(
                message
                for message in store.list_messages(
                    run_id, recipient_id=peer_id, status="consumed"
                )
                if message.message_id not in seen
            )
        names = {
            peer.peer_id: peer.name for peer in store.list_participants(run_id)
        }
        return ToolResult(
            name="ReadMessages",
            output={
                "messages": [
                    {
                        "message_id": message.message_id,
                        "sender_id": message.sender_id,
                        "from": names.get(message.sender_id, message.sender_id),
                        "summary": message.summary,
                        "message": message.payload,
                        "broadcast_id": message.broadcast_id,
                        "status": message.status,
                        "created_at": message.created_at,
                        "delivered_at": message.delivered_at,
                        "consumed_at": message.consumed_at,
                    }
                    for message in messages
                ]
            },
        )


class PeerBroadcastTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Broadcast",
            description=(
                "Send one durable message to every other peer allowed by the communication graph. "
                "Supply idempotency_key when retrying a broadcast."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "message": {},
                    "summary": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["message"],
            },
            is_read_only=False,
            strict=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        store, run_id, peer_id, control = _peer_context(context)
        try:
            broadcast = store.broadcast(
                run_id,
                peer_id,
                tool_input.get("message"),
                policy=control.policy,
                summary=tool_input.get("summary"),
                idempotency_key=tool_input.get("idempotency_key"),
            )
        except (ValueError, PolicyRejected) as exc:
            raise ToolInputError(str(exc)) from exc
        return ToolResult(name="Broadcast", output=broadcast.to_dict())


class PeerSubmitTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="PeerSubmit",
            description=(
                "Atomically submit a final Git revision for the run. Any peer may submit; "
                "the first valid revision wins and stops the run."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "revision": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["revision", "summary"],
            },
            is_read_only=False,
            strict=True,
            max_result_size_chars=200_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        _, _, peer_id, control = _peer_context(context)
        revision = tool_input.get("revision")
        summary = tool_input.get("summary")
        if not isinstance(revision, str) or not revision.strip():
            raise ToolInputError("revision must be a non-empty Git revision")
        if not isinstance(summary, str) or not summary.strip():
            raise ToolInputError("summary must be a non-empty string")
        output = control.submit(peer_id, revision.strip(), summary.strip())
        return ToolResult(
            name="PeerSubmit",
            output=output,
            is_error=output["status"] == "rejected",
        )
