from __future__ import annotations

import json
import sys
import time
import urllib.parse
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .store import TeamStore
from .trace import redact_trace_value


_TRACE_PREFIXES = ("run.", "model.", "tool.")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def list_team_summaries(workspace_root: str | Path) -> list[dict[str, Any]]:
    store = TeamStore(Path(workspace_root))
    summaries: list[dict[str, Any]] = []
    if not store.teams_dir.exists():
        return summaries
    for directory in store.teams_dir.iterdir():
        if not directory.is_dir():
            continue
        team = _read_json(directory / "team.json")
        if not team.get("team_id"):
            continue
        summaries.append({
            "team_id": team.get("team_id"),
            "team_name": team.get("team_name") or team.get("team_id"),
            "status": team.get("status") or "unknown",
            "created_at": team.get("created_at"),
            "updated_at": team.get("updated_at"),
        })
    summaries.sort(key=lambda team: str(team.get("updated_at") or team.get("created_at") or ""), reverse=True)
    return summaries


def _select_team_id(store: TeamStore, requested_team_id: str | None) -> str | None:
    teams = list_team_summaries(store.workspace_root)
    known_ids = {str(team["team_id"]) for team in teams}
    if requested_team_id:
        if requested_team_id not in known_ids:
            raise ValueError(f"unknown team: {requested_team_id}")
        return requested_team_id
    active = store.load_active_team()
    if active is not None and active.team_id in known_ids:
        return active.team_id
    return str(teams[0]["team_id"]) if teams else None


def _parse_timestamp(value: Any) -> float:
    if not isinstance(value, str) or not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _task_for_agent(tasks: dict[str, Any], agent_id: str) -> str | None:
    for task_id, task in tasks.items():
        if isinstance(task, dict) and task.get("owner") == agent_id:
            return task_id
    return None


def _decode_tool_output(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def _reconstruct_session_events(
    store: TeamStore,
    team_id: str,
    tasks: dict[str, Any],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    directory = store.team_dir(team_id) / "sessions"
    if not directory.exists():
        return events
    agents = {agent.agent_id: agent for agent in store.list_agents(team_id)}
    for path in sorted(directory.glob("*.json")):
        session = _read_json(path)
        agent_id = str(session.get("agent_id") or "")
        if not agent_id:
            continue
        agent = agents.get(agent_id)
        actor_name = agent.name if agent is not None else agent_id
        task_id = _task_for_agent(tasks, agent_id)
        conversation = session.get("conversation")
        messages = conversation.get("messages") if isinstance(conversation, dict) else []
        if not isinstance(messages, list):
            continue
        turn = 0
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            created_at = str(message.get("timestamp") or session.get("updated_at") or "")
            if role == "assistant":
                turn += 1
                text_parts: list[str] = []
                blocks = content if isinstance(content, list) else []
                if isinstance(content, str) and content:
                    text_parts.append(content)
                for block in blocks:
                    if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                        text_parts.append(str(block["text"]))
                if text_parts or blocks:
                    events.append(_session_event(
                        team_id,
                        path.stem,
                        message_index,
                        0,
                        "model.response",
                        created_at,
                        {
                            "actor_id": agent_id,
                            "actor_name": actor_name,
                            "task_id": task_id,
                            "turn": turn,
                            "model": session.get("model"),
                            "content": "\n".join(text_parts),
                            "source": "session-reconstruction",
                        },
                    ))
                for block_index, block in enumerate(blocks, start=1):
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    events.append(_session_event(
                        team_id,
                        path.stem,
                        message_index,
                        block_index,
                        "tool.started",
                        created_at,
                        {
                            "actor_id": agent_id,
                            "actor_name": actor_name,
                            "task_id": task_id,
                            "turn": turn,
                            "tool_name": block.get("name"),
                            "tool_input": block.get("input") or {},
                            "tool_use_id": block.get("id"),
                            "source": "session-reconstruction",
                        },
                    ))
            elif role == "user" and isinstance(content, list):
                for block_index, block in enumerate(content):
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    is_error = bool(block.get("is_error"))
                    events.append(_session_event(
                        team_id,
                        path.stem,
                        message_index,
                        block_index,
                        "tool.failed" if is_error else "tool.completed",
                        created_at,
                        {
                            "actor_id": agent_id,
                            "actor_name": actor_name,
                            "task_id": task_id,
                            "tool_output": _decode_tool_output(block.get("content")),
                            "tool_use_id": block.get("tool_use_id"),
                            "is_error": is_error,
                            "source": "session-reconstruction",
                        },
                    ))
    return events


def _session_event(
    team_id: str,
    session_id: str,
    message_index: int,
    block_index: int,
    event_type: str,
    created_at: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    return {
        "event_id": f"session-{session_id}-{message_index:04d}-{block_index:03d}",
        "team_id": team_id,
        "type": event_type,
        "created_at": created_at,
        "data": redact_trace_value(data),
        "reconstructed": True,
    }


def _event_actor_id(event: dict[str, Any], lead_id: str) -> str:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    if data.get("actor_id"):
        return str(data["actor_id"])
    if data.get("agent_id"):
        return str(data["agent_id"])
    agent = data.get("agent")
    if isinstance(agent, dict) and agent.get("agent_id"):
        return str(agent["agent_id"])
    message = data.get("message")
    if isinstance(message, dict) and message.get("sender_id"):
        return str(message["sender_id"])
    task = data.get("task")
    if isinstance(task, dict) and task.get("owner"):
        return str(task["owner"])
    return lead_id


def build_trace_snapshot(
    workspace_root: str | Path,
    team_id: str | None = None,
) -> dict[str, Any]:
    store = TeamStore(Path(workspace_root))
    selected_id = _select_team_id(store, team_id)
    teams = list_team_summaries(store.workspace_root)
    if selected_id is None:
        return {
            "workspace": str(store.workspace_root),
            "teams": teams,
            "team": None,
            "agents": [],
            "tasks": [],
            "messages": [],
            "events": [],
            "stats": {"event_count": 0, "tool_count": 0, "message_count": 0, "error_count": 0},
            "historical_reconstruction": False,
        }

    team = store.load_team(selected_id)
    if team is None:
        raise ValueError(f"team state is missing: {selected_id}")
    tasks_by_id = store.load_tasks(selected_id)
    agents = [agent.to_dict() for agent in store.list_agents(selected_id)]
    agents.insert(0, {
        "agent_id": team.lead_agent_id,
        "team_id": team.team_id,
        "name": "lead",
        "role": "lead",
        "session_id": None,
        "model": None,
        "instructions": "",
        "tools": [],
        "status": team.status,
        "created_at": team.created_at,
        "updated_at": team.updated_at,
        "schema_version": 1,
    })
    names = {str(agent["agent_id"]): str(agent.get("name") or agent["agent_id"]) for agent in agents}
    messages = []
    for message in store.list_messages(selected_id):
        item = message.to_dict()
        item["sender_name"] = names.get(message.sender_id, message.sender_id)
        item["recipient_name"] = names.get(message.recipient_id, message.recipient_id)
        messages.append(redact_trace_value(item))

    persisted_events = store.list_events(selected_id)
    has_native_trace = any(str(event.get("type") or "").startswith(_TRACE_PREFIXES) for event in persisted_events)
    events = list(persisted_events)
    if not has_native_trace:
        events.extend(_reconstruct_session_events(store, selected_id, tasks_by_id))
    for event in events:
        event["actor_id"] = _event_actor_id(event, team.lead_agent_id)
        event["actor_name"] = names.get(event["actor_id"], event["actor_id"])
    events.sort(key=lambda event: (
        _parse_timestamp(event.get("created_at")),
        0 if event.get("type") == "model.response" else 1,
        str(event.get("event_id") or ""),
    ))
    for sequence, event in enumerate(events):
        event["sequence"] = sequence

    model_events = [event for event in events if event.get("type") == "model.response"]
    input_tokens = sum(int((event.get("data") or {}).get("usage", {}).get("input_tokens", 0)) for event in model_events)
    output_tokens = sum(int((event.get("data") or {}).get("usage", {}).get("output_tokens", 0)) for event in model_events)
    first_time = _parse_timestamp(events[0].get("created_at")) if events else 0
    last_time = _parse_timestamp(events[-1].get("created_at")) if events else 0
    safe_events = redact_trace_value(events)
    return {
        "workspace": str(store.workspace_root),
        "teams": teams,
        "team": redact_trace_value(team.to_dict()),
        "agents": redact_trace_value(agents),
        "tasks": redact_trace_value(list(tasks_by_id.values())),
        "messages": messages,
        "events": safe_events,
        "stats": {
            "event_count": len(events),
            "tool_count": sum(1 for event in events if event.get("type") == "tool.started"),
            "message_count": len(messages),
            "error_count": sum(1 for event in events if event.get("type") in {"tool.failed", "model.failed", "task.failed", "team.failed", "run.failed"}),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_ms": max(0, round((last_time - first_time) * 1000)) if first_time and last_time else 0,
        },
        "historical_reconstruction": not has_native_trace,
    }


def _team_fingerprint(workspace_root: Path, team_id: str | None) -> tuple[int, int]:
    if not team_id:
        root = workspace_root / ".clawd"
    else:
        root = workspace_root / ".clawd" / "teams" / team_id
    total_size = 0
    latest_ns = 0
    if root.exists():
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            total_size += stat.st_size
            latest_ns = max(latest_ns, stat.st_mtime_ns)
    return latest_ns, total_size


class TraceViewerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        workspace_root: Path,
        team_id: str | None = None,
    ):
        self.workspace_root = workspace_root.resolve()
        self.default_team_id = team_id
        super().__init__(address, TraceViewerHandler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


class TraceViewerHandler(BaseHTTPRequestHandler):
    server: TraceViewerServer

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        team_id = query.get("team", [self.server.default_team_id])[0]
        if parsed.path == "/":
            self._send_html()
            return
        if parsed.path == "/api/state":
            try:
                snapshot = build_trace_snapshot(self.server.workspace_root, team_id)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(snapshot)
            return
        if parsed.path == "/api/stream":
            self._stream_changes(team_id)
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _send_html(self) -> None:
        path = Path(__file__).with_name("trace_viewer.html")
        try:
            body = path.read_bytes()
        except OSError:
            self._send_json({"error": "trace viewer asset is missing"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream_changes(self, team_id: str | None) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last = None
        try:
            for tick in range(900):
                current = _team_fingerprint(self.server.workspace_root, team_id)
                if current != last:
                    payload = json.dumps({"type": "refresh", "fingerprint": current})
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    last = current
                elif tick % 15 == 0:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                time.sleep(1)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_trace_server(
    workspace_root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    team_id: str | None = None,
) -> TraceViewerServer:
    return TraceViewerServer((host, port), Path(workspace_root), team_id)


def serve_trace_viewer(
    workspace_root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    team_id: str | None = None,
    open_browser: bool = False,
) -> int:
    server = create_trace_server(workspace_root, host=host, port=port, team_id=team_id)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}"
    print(f"Teammate Trace Viewer: {url}")
    print(f"Workspace: {server.workspace_root}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
