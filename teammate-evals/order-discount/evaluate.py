from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def verify_collaboration(workspace: Path) -> list[str]:
    errors: list[str] = []
    acceptance = _read_json(workspace / "acceptance.json")
    active_path = workspace / ".clawd" / "team.json"
    if not active_path.exists():
        return ["missing active team state: .clawd/team.json"]
    team = _read_json(active_path)
    team_id = str(team.get("team_id") or "")
    team_dir = workspace / ".clawd" / "teams" / team_id
    if not team_id or not team_dir.is_dir():
        return [f"missing team directory for team_id={team_id!r}"]

    agent_records = [
        _read_json(path) for path in sorted((team_dir / "agents").glob("*.json"))
    ]
    agents_by_name = {
        str(agent.get("name") or "").lower(): agent for agent in agent_records
    }
    required_agent_names = {
        str(name).lower() for name in acceptance.get("required_agents", [])
    }
    extra_agents = sorted(set(agents_by_name) - required_agent_names)
    if extra_agents:
        errors.append(f"unexpected teammates: {', '.join(extra_agents)}")
    agent_names_by_id = {
        str(agent.get("agent_id")): str(agent.get("name")) for agent in agent_records
    }
    agent_names_by_id[str(team.get("lead_agent_id"))] = "lead"
    for required_name in acceptance.get("required_agents", []):
        agent = agents_by_name.get(str(required_name).lower())
        if agent is None:
            errors.append(f"missing teammate: {required_name}")
            continue
        session_id = str(agent.get("session_id") or "")
        session_path = team_dir / "sessions" / f"{session_id}.json"
        if not session_path.exists():
            errors.append(f"missing session for teammate: {required_name}")
            continue
        conversation = _read_json(session_path).get("conversation")
        messages = conversation.get("messages") if isinstance(conversation, dict) else None
        if not isinstance(messages, list) or not messages:
            errors.append(f"empty session for teammate: {required_name}")

    tasks_data = _read_json(team_dir / "tasks.json")
    tasks = [task for task in tasks_data.values() if isinstance(task, dict)]
    tasks_by_key = {
        _normalize(str(task.get("key") or task.get("subject") or "")): task
        for task in tasks
    }
    task_keys_by_id = {
        str(task.get("id")): _normalize(str(task.get("key") or task.get("subject") or ""))
        for task in tasks
    }
    for required in acceptance.get("required_tasks", []):
        required_key = _normalize(str(required.get("name") or ""))
        task = tasks_by_key.get(required_key)
        if task is None:
            errors.append(f"missing task: {required_key}")
            continue
        owner_name = agent_names_by_id.get(str(task.get("owner")), str(task.get("owner")))
        if owner_name != required.get("owner"):
            errors.append(
                f"task {required_key} owner is {owner_name!r}, expected {required.get('owner')!r}"
            )
        actual_dependencies = sorted(
            task_keys_by_id.get(str(task_id), str(task_id))
            for task_id in task.get("blockedBy") or []
        )
        expected_dependencies = sorted(str(item) for item in required.get("blocked_by") or [])
        if actual_dependencies != expected_dependencies:
            errors.append(
                f"task {required_key} dependencies are {actual_dependencies}, expected {expected_dependencies}"
            )
        if task.get("status") != "completed":
            errors.append(f"task {required_key} status is {task.get('status')!r}, expected 'completed'")

    message_pairs: set[tuple[str, str]] = set()
    for path in sorted((team_dir / "messages").glob("*.json")):
        message = _read_json(path)
        sender = agent_names_by_id.get(str(message.get("sender_id")), str(message.get("sender_id")))
        recipient = agent_names_by_id.get(
            str(message.get("recipient_id")), str(message.get("recipient_id"))
        )
        message_pairs.add((sender, recipient))
    for required in acceptance.get("required_messages", []):
        pair = (str(required.get("from")), str(required.get("to")))
        if pair not in message_pairs:
            errors.append(f"missing message handoff: {pair[0]} -> {pair[1]}")

    expected_status = acceptance.get("required_final_status")
    if team.get("status") != expected_status:
        errors.append(
            f"team status is {team.get('status')!r}, expected {expected_status!r}"
        )
    events_path = team_dir / "events.jsonl"
    if not events_path.exists() or not events_path.read_text(encoding="utf-8").strip():
        errors.append("missing team event log")
    return errors


def run_business_tests(workspace: Path) -> subprocess.CompletedProcess[str]:
    acceptance = _read_json(workspace / "acceptance.json")
    command = shlex.split(str(acceptance["required_test_command"]))
    if command and command[0] in {"python", "python3"}:
        command[0] = sys.executable
    return subprocess.run(command, cwd=workspace, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the order-discount teammate workflow.")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--collaboration-only", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()

    errors = verify_collaboration(workspace)
    if errors:
        print("Collaboration evidence: FAILED")
        for error in errors:
            print(f"- {error}")
    else:
        print("Collaboration evidence: PASSED")

    tests_ok = True
    if not args.collaboration_only:
        completed = run_business_tests(workspace)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        tests_ok = completed.returncode == 0
        print(f"Business acceptance: {'PASSED' if tests_ok else 'FAILED'}")
    return 0 if not errors and tests_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
