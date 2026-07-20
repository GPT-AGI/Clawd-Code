from __future__ import annotations

import hashlib
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ..tool_system.context import ToolContext
from ..tool_system.permissions import ToolPermissionContext
from ..tool_system.protocol import ToolCall, ToolResult
from ..tool_system.registry import ToolRegistry
from ..tool_system.tools.tool_search import ToolSearchTool
from .backend import PeerBoundaryResult, PeerSessionBackend, PeerSessionSpec
from .control import PeerRunControl
from .models import PeerParticipant, PeerRunConfig, PeerRunRecord, utc_now
from .policy import CommunicationPolicy
from .store import PeerSignalBus, PeerStore
from .tools import (
    PeerBroadcastTool,
    PeerListTool,
    PeerReadMessagesTool,
    PeerSendMessageTool,
    PeerSubmitTool,
)
from .workspace import PeerWorkspaceManager


_FORBIDDEN_PEER_TOOLS = {
    "Agent",
    "AskUserQuestion",
    "EnterWorktree",
    "ExitWorktree",
    "ReadMessages",
    "RemoteTrigger",
    "SendMessage",
    "SendUserMessage",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskRetry",
    "TaskStop",
    "TaskUpdate",
    "ToolSearch",
    "TeamCancel",
    "TeamCreate",
    "TeamDelete",
    "TeamIntegrate",
    "TeamResume",
    "TeamRun",
    "TeammateCreate",
    "TeammateResume",
    "TeammateStop",
}


class GuardedPeerRegistry(ToolRegistry):
    def __init__(self, tools: list[Any], control: PeerRunControl):
        super().__init__(tools)
        self.control = control

    def dispatch(self, call: ToolCall, context: ToolContext) -> ToolResult:
        if self.control.should_stop():
            return ToolResult(
                name=call.name,
                output={"error": "peer run has stopped; tool calls are no longer allowed"},
                is_error=True,
                tool_use_id=call.tool_use_id,
            )
        if call.name.casefold() in {"sendmessage", "readmessages", "broadcast"} and self.get(call.name) is None:
            peer_id = context.peer_id or "unknown"
            reason = (
                f"{call.name} is unavailable under "
                f"{self.control.policy.condition} communication policy"
            )
            self.control.store.record_policy_rejection(
                self.control.run_id,
                peer_id,
                None,
                call.name,
                reason,
            )
            return ToolResult(
                name=call.name,
                output={"error": reason},
                is_error=True,
                tool_use_id=call.tool_use_id,
            )
        return super().dispatch(call, context)


class PeerRuntime:
    """Non-LLM supervisor for fixed-size peer-native collaboration runs."""

    def __init__(self, backend: PeerSessionBackend, base_registry: ToolRegistry) -> None:
        self.backend = backend
        self.base_registry = base_registry
        self._controls: dict[str, PeerRunControl] = {}
        self._controls_lock = threading.Lock()

    def run(self, config: PeerRunConfig, *, run_id: str | None = None) -> dict[str, Any]:
        config.validate()
        started = time.monotonic()
        selected_run_id = run_id or uuid.uuid4().hex[:16]
        repo_path = Path(config.repo_path).expanduser().resolve()
        if not repo_path.is_dir():
            raise ValueError(f"repo_path is not a directory: {repo_path}")
        output_base = (
            Path(config.output_dir).expanduser().resolve()
            if config.output_dir
            else repo_path / ".clawd" / "peer-runs"
        )
        self._exclude_control_state(repo_path, output_base)
        signal_bus = PeerSignalBus()
        store = PeerStore(output_base, signal_bus=signal_bus)
        workspace = PeerWorkspaceManager(
            repo_path,
            selected_run_id,
            config.workspace_mode,
            cleanup_worktrees=config.cleanup_worktrees,
        )
        peer_ids = tuple(
            f"{selected_run_id[:8]}-p{index}" for index in range(1, config.peers + 1)
        )
        coordinator = self._resolve_coordinator(config, peer_ids)
        policy = CommunicationPolicy(config.communication, peer_ids, coordinator)
        run = PeerRunRecord(
            run_id=selected_run_id,
            mission=config.mission,
            repo_path=str(repo_path),
            base_revision=workspace.base_revision,
            peer_count=config.peers,
            communication=config.communication,
            workspace_mode=config.workspace_mode,
            provider=config.provider,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            max_turns=config.max_turns,
            max_output_tokens=config.max_output_tokens,
            token_budget=config.token_budget,
            turn_budget=config.turn_budget,
            output_dir=str(output_base),
            coordinator_peer_id=coordinator,
            acceptance_command=config.acceptance_command,
        )
        manifest = self._manifest(config, run, peer_ids)
        store.create_run(run, manifest)
        participants: list[PeerParticipant] = []
        try:
            for index, peer_id in enumerate(peer_ids, start=1):
                name = f"peer-{index}"
                workspace_path = workspace.prepare(peer_id, name)
                participant = PeerParticipant(
                    peer_id=peer_id,
                    run_id=selected_run_id,
                    name=name,
                    session_id=uuid.uuid4().hex,
                    workspace_mode=config.workspace_mode,
                    workspace_path=str(workspace_path),
                )
                participants.append(participant)
                store.save_participant(participant)
                store.save_session(
                    selected_run_id,
                    participant.session_id,
                    {
                        "session_id": participant.session_id,
                        "run_id": selected_run_id,
                        "peer_id": peer_id,
                        "boundary_index": 0,
                        "conversation": {"messages": [], "max_history": 500},
                    },
                )
                store.append_event(
                    selected_run_id,
                    "peer.created",
                    {
                        "peer_id": peer_id,
                        "name": name,
                        "session_id": participant.session_id,
                        "workspace_path": str(workspace_path),
                    },
                )
        except Exception as exc:
            run.set_status("failed", reason=f"workspace setup failed: {exc}")
            store.save_run(run)
            store.append_event(
                selected_run_id,
                "peer_run.failed",
                {"phase": "workspace_setup", "error": str(exc)},
            )
            workspace.cleanup()
            raise

        control = PeerRunControl(
            selected_run_id,
            store,
            policy,
            workspace,
            timeout_seconds=config.timeout_seconds,
            token_budget=config.token_budget,
            turn_budget=config.turn_budget,
        )
        with self._controls_lock:
            self._controls[selected_run_id] = control
        run.set_status("running")
        store.save_run(run)
        store.append_event(
            selected_run_id,
            "peer_run.started",
            {
                "peer_count": config.peers,
                "communication": config.communication,
                "workspace_mode": config.workspace_mode,
            },
        )

        barrier = threading.Barrier(config.peers)
        threads = [
            threading.Thread(
                target=self._run_peer,
                name=f"clawd-peer-{participant.name}",
                daemon=True,
                args=(participant, config, store, control, barrier),
            )
            for participant in participants
        ]
        for thread in threads:
            thread.start()

        if not control.stop_event.wait(config.timeout_seconds):
            control.request_stop("timeout")
        shutdown_grace_seconds = min(
            60.0, max(10.0, config.timeout_seconds * 0.2)
        )
        shutdown_deadline = time.monotonic() + shutdown_grace_seconds
        for thread in threads:
            thread.join(timeout=max(0.0, shutdown_deadline - time.monotonic()))
        orphan_threads = [thread.name for thread in threads if thread.is_alive()]
        if orphan_threads:
            store.append_event(
                selected_run_id,
                "peer_run.orphan_threads",
                {"threads": orphan_threads},
            )

        current = store.load_run(selected_run_id)
        if current is None:
            raise RuntimeError("peer run state disappeared")
        reason = control.reason or "all_peers_stopped"
        acceptance: dict[str, Any] | None = None
        if current.accepted_submission is not None and config.acceptance_command:
            acceptance = workspace.run_acceptance(
                str(current.accepted_submission["revision"]),
                config.acceptance_command,
                timeout_seconds=min(600.0, config.timeout_seconds),
            )
            store.append_event(
                selected_run_id,
                "acceptance.completed",
                {
                    "command": acceptance["command"],
                    "exit_code": acceptance["exit_code"],
                    "stdout_size": len(str(acceptance.get("stdout") or "").encode()),
                    "stderr_size": len(str(acceptance.get("stderr") or "").encode()),
                },
            )
        current_participants = store.list_participants(selected_run_id)
        attribution = workspace.attribution(current_participants)
        retained_worktrees = workspace.cleanup()
        terminal_status = self._terminal_status(current, reason, orphan_threads)
        current.set_status(terminal_status, reason=reason)
        store.save_run(current)
        wall_time = time.monotonic() - started
        result = {
            "schema_version": 1,
            "run_id": selected_run_id,
            "status": current.status,
            "stop_reason": reason,
            "accepted_submission": current.accepted_submission,
            "run": current.to_dict(),
            "participants": [peer.to_dict() for peer in store.list_participants(selected_run_id)],
            "messages": [message.to_dict() for message in store.list_messages(selected_run_id)],
            "broadcasts": [item.to_dict() for item in store.list_broadcasts(selected_run_id)],
            "submissions": [item.to_dict() for item in store.list_submissions(selected_run_id)],
            "workspace_attribution": attribution,
            "acceptance": acceptance,
            "usage": current.usage,
            "wall_time_seconds": round(wall_time, 6),
            "orphan_threads": orphan_threads,
            "retained_worktrees": retained_worktrees,
            "manifest_path": str(store.run_dir(selected_run_id) / "manifest.json"),
            "events_path": str(store.run_dir(selected_run_id) / "events.jsonl"),
        }
        result_path = store.save_result(selected_run_id, result)
        result["result_path"] = str(result_path)
        store.save_result(selected_run_id, result)
        store.append_event(
            selected_run_id,
            f"peer_run.{terminal_status}",
            {
                "stop_reason": reason,
                "accepted_submission": current.accepted_submission,
                "wall_time_seconds": result["wall_time_seconds"],
                "usage": current.usage,
                "acceptance_exit_code": (
                    acceptance.get("exit_code") if acceptance is not None else None
                ),
            },
        )
        with self._controls_lock:
            self._controls.pop(selected_run_id, None)
        return result

    def cancel(self, run_id: str, reason: str = "cancelled") -> bool:
        with self._controls_lock:
            control = self._controls.get(run_id)
        if control is None:
            return False
        control.request_stop(reason)
        return True

    def _run_peer(
        self,
        participant: PeerParticipant,
        config: PeerRunConfig,
        store: PeerStore,
        control: PeerRunControl,
        barrier: threading.Barrier,
    ) -> None:
        session = None
        try:
            barrier.wait(timeout=min(30.0, config.timeout_seconds))
            participant, _ = store.mutate_participant(
                participant.run_id,
                participant.peer_id,
                lambda peer: self._set_peer_running(peer),
            )
            store.append_event(
                participant.run_id,
                "peer.started",
                {
                    "peer_id": participant.peer_id,
                    "name": participant.name,
                    "session_id": participant.session_id,
                },
            )
            spec = PeerSessionSpec(
                run_id=participant.run_id,
                peer_id=participant.peer_id,
                peer_name=participant.name,
                session_id=participant.session_id,
                workspace_path=participant.workspace_path,
                model=config.model,
            )
            session = self.backend.create_session(
                spec, store.load_session(participant.run_id, participant.session_id)
            )
            context = self._peer_context(participant, store, control)
            registry = self._peer_registry(control)
            prompt = config.mission
            first_boundary = True
            while not control.should_stop():
                latest = store.load_participant(participant.run_id, participant.peer_id)
                if latest is None:
                    raise RuntimeError("peer participant state disappeared")
                remaining_turns = config.max_turns - int(latest.usage.get("turns", 0))
                if remaining_turns <= 0:
                    raise RuntimeError("peer max_turns exhausted without a submission")
                if not first_boundary:
                    participant, _ = store.mutate_participant(
                        participant.run_id,
                        participant.peer_id,
                        lambda peer: peer.set_status("idle"),
                    )
                    store.append_event(
                        participant.run_id,
                        "peer.idle",
                        {"peer_id": participant.peer_id},
                    )
                    woke = store.wait_for_unread(
                        participant.run_id,
                        participant.peer_id,
                        control.remaining_seconds(),
                        stop_event=control.stop_event,
                    )
                    if control.should_stop():
                        break
                    if not woke:
                        continue
                    incoming = store.consume_messages(
                        participant.run_id, participant.peer_id
                    )
                    participant, _ = store.mutate_participant(
                        participant.run_id,
                        participant.peer_id,
                        lambda peer: self._set_peer_woken(peer),
                    )
                    store.append_event(
                        participant.run_id,
                        "peer.woken",
                        {
                            "peer_id": participant.peer_id,
                            "message_ids": [message.message_id for message in incoming],
                        },
                    )
                    prompt = self._incoming_prompt(incoming)
                first_boundary = False
                counters = {"model_calls": 0, "tool_calls": 0}

                def count_event(event: Any) -> None:
                    if event.kind == "model_response":
                        counters["model_calls"] += 1
                    elif event.kind == "tool_use":
                        counters["tool_calls"] += 1

                result: PeerBoundaryResult = self.backend.run_boundary(
                    session,
                    prompt,
                    registry,
                    context,
                    max_turns=remaining_turns,
                    max_output_tokens=config.max_output_tokens,
                    should_stop=control.should_stop,
                    on_event=count_event,
                )
                store.save_session(
                    participant.run_id,
                    participant.session_id,
                    self.backend.serialize_session(session),
                )
                usage = result.usage or {}
                control.record_usage(
                    participant.peer_id,
                    {
                        "input_tokens": int(usage.get("input_tokens", 0) or 0),
                        "output_tokens": int(usage.get("output_tokens", 0) or 0),
                        "cache_creation_input_tokens": int(
                            usage.get("cache_creation_input_tokens", 0) or 0
                        ),
                        "cache_read_input_tokens": int(
                            usage.get("cache_read_input_tokens", 0) or 0
                        ),
                        "turns": int(result.num_turns),
                        "model_calls": counters["model_calls"],
                        "tool_calls": counters["tool_calls"],
                    },
                )
                if result.cancelled or control.should_stop():
                    break
                if result.response_text == "[Max tool turns reached]":
                    raise RuntimeError(result.response_text)
        except threading.BrokenBarrierError as exc:
            self._fail_peer(store, participant, f"peer start barrier failed: {exc}")
            control.request_stop("startup_failed")
        except Exception as exc:
            self._fail_peer(store, participant, str(exc))
            statuses = [peer.status for peer in store.list_participants(participant.run_id)]
            if statuses and all(status == "failed" for status in statuses):
                control.request_stop("all_peers_failed")
        finally:
            if session is not None:
                try:
                    self.backend.close_session(session)
                except Exception:
                    pass
            latest = store.load_participant(participant.run_id, participant.peer_id)
            if latest is not None and latest.status != "failed":
                latest, _ = store.mutate_participant(
                    participant.run_id,
                    participant.peer_id,
                    lambda peer: peer.set_status("stopped"),
                )
                store.append_event(
                    participant.run_id,
                    "peer.stopped",
                    {
                        "peer_id": participant.peer_id,
                        "reason": control.reason,
                    },
                )

    def _peer_registry(self, control: PeerRunControl) -> GuardedPeerRegistry:
        tools: list[Any] = []
        for spec in self.base_registry.list_specs():
            if spec.name in _FORBIDDEN_PEER_TOOLS:
                continue
            tool = self.base_registry.get(spec.name)
            if tool is not None:
                tools.append(tool)
        tools.extend([PeerListTool(), PeerSubmitTool()])
        if control.policy.exposes_message_tools():
            tools.extend(
                [PeerSendMessageTool(), PeerReadMessagesTool(), PeerBroadcastTool()]
            )
        registry = GuardedPeerRegistry(tools, control)
        registry.register(ToolSearchTool(registry))
        return registry

    @staticmethod
    def _peer_context(
        participant: PeerParticipant, store: PeerStore, control: PeerRunControl
    ) -> ToolContext:
        workspace = Path(participant.workspace_path)
        context = ToolContext(
            workspace_root=workspace,
            cwd=workspace,
            permission_context=ToolPermissionContext.from_iterables(
                workspace_root=workspace
            ),
            actor_id=participant.peer_id,
            model_override=control.store.load_run(participant.run_id).model,  # type: ignore[union-attr]
            peer_store=store,
            peer_run_id=participant.run_id,
            peer_id=participant.peer_id,
            peer_control=control,
            system_prompt_extra=PeerRuntime.peer_system_context(participant),
        )
        return context

    @staticmethod
    def peer_system_context(participant: PeerParticipant) -> str:
        return (
            "## Peer Collaboration Context\n"
            "You are an equal coding peer in a peer-native collaboration run. "
            "No participant has supervisory authority over another participant. "
            "You may inspect the repository, decide what useful work to pursue, and coordinate "
            "through the communication tools exposed by the run protocol.\n"
            f"Your stable identity is `{participant.name}` (`{participant.peer_id}`). "
            "Use PeerList to inspect the common roster. Any participant may call PeerSubmit "
            "with a verifiable final Git revision. Remain available for later peer messages "
            "after a local work interval ends."
        )

    @staticmethod
    def _incoming_prompt(messages: list[Any]) -> str:
        lines = ["New peer messages were delivered while this session was available:"]
        for message in messages:
            lines.append(
                f"- message_id={message.message_id} from={message.sender_id} "
                f"summary={message.summary or ''}\n  payload={message.payload!r}"
            )
        lines.append(
            "Continue working from the persistent session state. You may respond, adapt the "
            "repository, become available again, or submit a final revision."
        )
        return "\n".join(lines)

    @staticmethod
    def _set_peer_running(peer: PeerParticipant) -> None:
        peer.start_monotonic_ns = peer.start_monotonic_ns or time.monotonic_ns()
        peer.set_status("running")

    @staticmethod
    def _set_peer_woken(peer: PeerParticipant) -> None:
        peer.wake_at = utc_now()
        peer.set_status("running")

    @staticmethod
    def _fail_peer(store: PeerStore, participant: PeerParticipant, error: str) -> None:
        try:
            failed, _ = store.mutate_participant(
                participant.run_id,
                participant.peer_id,
                lambda peer: peer.set_status("failed", error=error),
            )
            store.append_event(
                participant.run_id,
                "peer.failed",
                {"peer_id": failed.peer_id, "error": error},
            )
        except Exception:
            return

    @staticmethod
    def _resolve_coordinator(
        config: PeerRunConfig, peer_ids: tuple[str, ...]
    ) -> str | None:
        if config.communication != "star":
            return None
        requested = (config.coordinator_peer or "peer-1").strip()
        if requested in peer_ids:
            return requested
        if requested.startswith("peer-"):
            try:
                index = int(requested.split("-", 1)[1]) - 1
            except ValueError:
                index = -1
            if 0 <= index < len(peer_ids):
                return peer_ids[index]
        raise ValueError(f"unknown star coordinator peer: {requested}")

    def _manifest(
        self,
        config: PeerRunConfig,
        run: PeerRunRecord,
        peer_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        noncommunication_tools = [
            spec.name
            for spec in self.base_registry.list_specs()
            if spec.name not in _FORBIDDEN_PEER_TOOLS
        ]
        noncommunication_tools.append("ToolSearch")
        return {
            "schema_version": 1,
            "run_id": run.run_id,
            "created_at": run.created_at,
            "config": config.to_dict(),
            "repo_path": run.repo_path,
            "repo_revision": run.base_revision,
            "mission_sha256": hashlib.sha256(config.mission.encode()).hexdigest(),
            "mission": config.mission,
            "peer_ids": list(peer_ids),
            "coordinator_peer_id": run.coordinator_peer_id,
            "backend": self.backend.name,
            "noncommunication_tools": noncommunication_tools,
            "communication_tools": (
                ["PeerList", "PeerSubmit", "SendMessage", "ReadMessages", "Broadcast"]
                if config.communication in {"star", "p2p"}
                else ["PeerList", "PeerSubmit"]
            ),
            "process_isolation": False,
        }

    @staticmethod
    def _terminal_status(
        run: PeerRunRecord, reason: str, orphan_threads: list[str]
    ) -> str:
        if orphan_threads:
            return "failed"
        if run.accepted_submission is not None:
            return "completed"
        if reason == "timeout":
            return "timed_out"
        if reason == "budget_exhausted":
            return "budget_exhausted"
        if reason in {"cancelled", "user_cancelled"}:
            return "cancelled"
        return "failed"

    @staticmethod
    def _exclude_control_state(repo_path: Path, output_base: Path) -> None:
        try:
            relative = output_base.relative_to(repo_path).as_posix().rstrip("/") + "/"
        except ValueError:
            return
        exclude = repo_path / ".git" / "info" / "exclude"
        if not exclude.parent.is_dir():
            return
        existing = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
        lines = {line.strip() for line in existing.splitlines()}
        if relative in lines:
            return
        with exclude.open("a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write(relative + "\n")
