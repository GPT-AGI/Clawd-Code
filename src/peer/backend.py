from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..agent.conversation import Conversation
from ..tool_system.agent_loop import AgentLoopResult, ToolEvent, run_agent_loop
from ..tool_system.context import ToolContext
from ..tool_system.registry import ToolRegistry


@dataclass(frozen=True)
class PeerSessionSpec:
    run_id: str
    peer_id: str
    peer_name: str
    session_id: str
    workspace_path: str
    model: str | None


@dataclass
class PeerSessionHandle:
    spec: PeerSessionSpec
    conversation: Conversation
    provider: Any = None
    boundary_index: int = 0


@dataclass(frozen=True)
class PeerBoundaryResult:
    response_text: str = ""
    usage: dict[str, Any] | None = None
    num_turns: int = 0
    cancelled: bool = False


class PeerSessionBackend(Protocol):
    name: str

    def create_session(
        self, spec: PeerSessionSpec, persisted: dict[str, Any] | None = None
    ) -> PeerSessionHandle: ...

    def run_boundary(
        self,
        session: PeerSessionHandle,
        prompt: str,
        registry: ToolRegistry,
        context: ToolContext,
        *,
        max_turns: int,
        max_output_tokens: int,
        should_stop: Callable[[], bool],
        on_event: Callable[[ToolEvent], None] | None = None,
    ) -> PeerBoundaryResult: ...

    def serialize_session(self, session: PeerSessionHandle) -> dict[str, Any]: ...

    def close_session(self, session: PeerSessionHandle) -> None: ...


class AgentLoopPeerBackend:
    """Adapter from persistent peers to Clawd's current agent loop."""

    name = "clawd-agent-loop"

    def __init__(self, provider_factory: Callable[[PeerSessionSpec], Any]) -> None:
        self.provider_factory = provider_factory

    def create_session(
        self, spec: PeerSessionSpec, persisted: dict[str, Any] | None = None
    ) -> PeerSessionHandle:
        conversation_data = (persisted or {}).get("conversation")
        conversation = (
            Conversation.from_dict(conversation_data)
            if isinstance(conversation_data, dict)
            else Conversation(max_history=500)
        )
        return PeerSessionHandle(
            spec=spec,
            conversation=conversation,
            provider=self.provider_factory(spec),
        )

    def run_boundary(
        self,
        session: PeerSessionHandle,
        prompt: str,
        registry: ToolRegistry,
        context: ToolContext,
        *,
        max_turns: int,
        max_output_tokens: int,
        should_stop: Callable[[], bool],
        on_event: Callable[[ToolEvent], None] | None = None,
    ) -> PeerBoundaryResult:
        session.boundary_index += 1
        session.conversation.add_user_message(prompt)
        result: AgentLoopResult = run_agent_loop(
            conversation=session.conversation,
            provider=session.provider,
            tool_registry=registry,
            tool_context=context,
            max_turns=max_turns,
            max_output_tokens=max_output_tokens,
            stream=False,
            verbose=False,
            on_event=on_event,
            should_stop=should_stop,
        )
        return PeerBoundaryResult(
            response_text=result.response_text,
            usage=result.usage,
            num_turns=result.num_turns,
            cancelled=result.cancelled,
        )

    def serialize_session(self, session: PeerSessionHandle) -> dict[str, Any]:
        return {
            "session_id": session.spec.session_id,
            "run_id": session.spec.run_id,
            "peer_id": session.spec.peer_id,
            "model": session.spec.model,
            "boundary_index": session.boundary_index,
            "conversation": session.conversation.to_dict(),
        }

    def close_session(self, session: PeerSessionHandle) -> None:
        return None


ScriptHandler = Callable[
    [PeerSessionHandle, str, ToolRegistry, ToolContext],
    PeerBoundaryResult | None,
]


class ScriptedPeerBackend:
    """Deterministic injectable backend used by protocol and smoke tests."""

    name = "scripted"

    def __init__(self, handler: ScriptHandler) -> None:
        self.handler = handler

    def create_session(
        self, spec: PeerSessionSpec, persisted: dict[str, Any] | None = None
    ) -> PeerSessionHandle:
        boundary = int((persisted or {}).get("boundary_index", 0) or 0)
        return PeerSessionHandle(
            spec=spec,
            conversation=Conversation(max_history=500),
            boundary_index=boundary,
        )

    def run_boundary(
        self,
        session: PeerSessionHandle,
        prompt: str,
        registry: ToolRegistry,
        context: ToolContext,
        *,
        max_turns: int,
        max_output_tokens: int,
        should_stop: Callable[[], bool],
        on_event: Callable[[ToolEvent], None] | None = None,
    ) -> PeerBoundaryResult:
        if should_stop():
            return PeerBoundaryResult(response_text="[Run stopped]", cancelled=True)
        session.boundary_index += 1
        session.conversation.add_user_message(prompt)
        result = self.handler(session, prompt, registry, context)
        return result or PeerBoundaryResult(response_text="idle", num_turns=1)

    def serialize_session(self, session: PeerSessionHandle) -> dict[str, Any]:
        return {
            "session_id": session.spec.session_id,
            "run_id": session.spec.run_id,
            "peer_id": session.spec.peer_id,
            "boundary_index": session.boundary_index,
            "conversation": session.conversation.to_dict(),
        }

    def close_session(self, session: PeerSessionHandle) -> None:
        return None
