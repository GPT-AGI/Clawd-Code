from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
UPSTREAM_URL = "https://github.com/multimodal-art-projection/NL2RepoBench.git"
UPSTREAM_REF = "781a1da1ee41fb8edb0bed22f586d69111610edf"
IMAGE_ROOT = "ghcr.io/multimodal-art-projection/nl2repobench"
PILOT_TASKS = ("jsonlines", "tinydb", "aiofiles", "flask-restful", "fastapi-users")
PACKAGE_FILES = {
    "setup.py",
    "pyproject.toml",
    "setup.cfg",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "tox.ini",
    "pytest.ini",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
    "environment.yml",
    "conda-env.yaml",
    "manifest.in",
    "MANIFEST.in",
}
TASK_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
PYTEST_COMMAND_RE = re.compile(
    r"^\s*(?:[A-Za-z_]\w*=[^\s]+\s+)*"
    r"(?:pytest|python(?:\d+(?:\.\d+)?)?\s+-m\s+pytest|xvfb-run\b.*\bpytest)"
    r"(?:\s|$)"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_checked(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def default_cache_root() -> Path:
    configured = os.environ.get("XDG_CACHE_HOME")
    base = Path(configured).expanduser() if configured else Path.home() / ".cache"
    return base / "clawd-code" / "nl2repo-bench"


def resolve_upstream(
    upstream_root: Path | None,
    *,
    cache_root: Path | None = None,
    upstream_url: str = UPSTREAM_URL,
    upstream_ref: str = UPSTREAM_REF,
) -> Path:
    explicit = upstream_root or (
        Path(os.environ["NL2REPO_BENCH_ROOT"]).expanduser()
        if os.environ.get("NL2REPO_BENCH_ROOT")
        else None
    )
    if explicit is not None:
        root = explicit.resolve()
        _validate_upstream_root(root)
        return root

    destination = (cache_root or default_cache_root()) / upstream_ref[:12]
    if destination.exists():
        _validate_upstream_root(destination)
        head = _run_checked(["git", "rev-parse", "HEAD"], cwd=destination).stdout.strip()
        if head != upstream_ref:
            raise ValueError(
                f"cached NL2Repo checkout is {head}, expected pinned commit {upstream_ref}: "
                f"{destination}"
            )
        return destination.resolve()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ValueError(f"temporary clone path already exists: {temporary}")
    try:
        _run_checked(
            ["git", "clone", "--filter=blob:none", "--no-checkout", upstream_url, str(temporary)]
        )
        _run_checked(["git", "fetch", "--depth", "1", "origin", upstream_ref], cwd=temporary)
        _run_checked(["git", "checkout", "--detach", upstream_ref], cwd=temporary)
        temporary.rename(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    _validate_upstream_root(destination)
    return destination.resolve()


def _validate_upstream_root(root: Path) -> None:
    if not (root / "test_files" / "task_difficulty.csv").is_file():
        raise ValueError(f"not an NL2Repo-Bench checkout: {root}")


def _difficulty_map(upstream_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with (upstream_root / "test_files" / "task_difficulty.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("task-name") or "").strip()
            level = str(row.get("Level") or "").strip()
            if name:
                result[name.casefold()] = level
    return result


def list_tasks(upstream_root: Path) -> list[dict[str, Any]]:
    difficulty = _difficulty_map(upstream_root)
    tasks: list[dict[str, Any]] = []
    for task_dir in sorted((upstream_root / "test_files").iterdir()):
        if not task_dir.is_dir() or not (task_dir / "start.md").is_file():
            continue
        count_path = task_dir / "test_case_count.txt"
        tasks.append(
            {
                "id": task_dir.name,
                "difficulty": difficulty.get(task_dir.name.casefold(), ""),
                "expected_tests": int(count_path.read_text(encoding="utf-8").strip()),
                "prompt_bytes": (task_dir / "start.md").stat().st_size,
            }
        )
    return tasks


def load_task(upstream_root: Path, task_name: str) -> dict[str, Any]:
    if not TASK_NAME_RE.fullmatch(task_name):
        raise ValueError(f"invalid task name: {task_name!r}")
    task_dir = upstream_root / "test_files" / task_name
    required = ("start.md", "test_case_count.txt", "test_commands.json", "test_files.json")
    missing = [name for name in required if not (task_dir / name).is_file()]
    if missing:
        raise ValueError(f"invalid NL2Repo task {task_name}: missing {', '.join(missing)}")
    commands = _read_json(task_dir / "test_commands.json")
    hidden_paths = _read_json(task_dir / "test_files.json")
    if not isinstance(commands, list) or not commands or not all(
        isinstance(item, str) and item.strip() for item in commands
    ):
        raise ValueError(f"invalid test commands for {task_name}")
    if not isinstance(hidden_paths, list) or not all(isinstance(item, str) for item in hidden_paths):
        raise ValueError(f"invalid hidden test paths for {task_name}")
    return {
        "id": task_name,
        "difficulty": _difficulty_map(upstream_root).get(task_name.casefold(), ""),
        "task_dir": str(task_dir),
        "document": (task_dir / "start.md").read_text(encoding="utf-8"),
        "expected_tests": int((task_dir / "test_case_count.txt").read_text(encoding="utf-8").strip()),
        "test_commands": commands,
        "hidden_paths": hidden_paths,
        "image": f"{IMAGE_ROOT}/{task_name}:1.0",
    }


def build_prompt(mode: str) -> str:
    common = """Build the complete Python repository described in start.md in this workspace.
Begin by reading the whole specification and inspecting the initially empty repository.
The official upstream tests are hidden and will be run only after you finish. You may
create your own focused tests, but do not fetch, install, copy, or inspect the target
project's implementation from GitHub, PyPI, caches, or another machine. Implement it
from the provided specification. Do not stop to ask for confirmation. Continue through
architecture, implementation, integration, and local validation before summarizing.
"""
    if mode == "solo":
        return common + """
Execution protocol: work directly as one agent. Do not create a teammate team or call
Team*/Teammate* tools. Plan, implement, test, and review the repository yourself.
"""
    if mode == "adaptive":
        return common + """
Execution protocol: act as the lead and decide whether collaboration is worth its cost.
It is valid to remain solo. If you delegate, choose the number of agents, task-specific
roles, models, tool permissions, workspaces, dependencies, and concurrency from the
repository itself. Agents may communicate directly when useful. Observe progress,
intervene when needed, integrate the work, and personally perform final validation.
The team topology must be your runtime decision, not a predefined role pipeline.
"""
    if mode == "forced-team":
        return common + """
Diagnostic execution protocol: use at least one teammate, but choose the number of
agents, task-specific roles, models, tool permissions, workspaces, dependencies,
concurrency, and communication topology from the repository itself. Do not use a
predefined role pipeline. Observe progress, intervene when needed, integrate the work,
and personally perform final validation.
"""
    raise ValueError(f"unknown mode: {mode}")


def prepare_workspace(task: dict[str, Any], workspace: Path) -> str:
    if workspace.exists():
        raise ValueError(f"workspace already exists: {workspace}")
    workspace.mkdir(parents=True)
    task_path = workspace / "start.md"
    task_path.write_text(str(task["document"]), encoding="utf-8")
    commands = [
        ["git", "init", "-q"],
        ["git", "config", "user.name", "NL2Repo Benchmark"],
        ["git", "config", "user.email", "benchmark@localhost"],
        ["git", "add", "start.md"],
        ["git", "commit", "-q", "-m", "add benchmark specification"],
    ]
    for command in commands:
        _run_checked(command, cwd=workspace)
    return _hash_file(task_path)


def _run_agent_child(
    workspace: Path,
    prompt_path: Path,
    result_path: Path,
    provider: str,
    model: str,
    max_turns: int,
    teammate_max_turns: int,
    max_output_tokens: int,
    stream: bool,
    progress_path: Path,
) -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.runner import run_prompt

    events: list[dict[str, Any]] = []

    def capture(event: Any) -> None:
        payload = dataclasses.asdict(event)
        events.append(payload)
        _append_jsonl(progress_path, payload)

    def capture_text(content: str) -> None:
        _append_jsonl(
            progress_path,
            {
                "kind": "text_chunk",
                "content": content,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    try:
        result = run_prompt(
            prompt_path.read_text(encoding="utf-8"),
            workspace=workspace,
            provider_name=provider,
            model=model,
            max_turns=max_turns,
            teammate_max_turns=teammate_max_turns,
            max_output_tokens=max_output_tokens,
            stream=stream,
            on_event=capture,
            on_text_chunk=capture_text,
        )
        payload = {
            "ok": result.response_text != "[Max tool turns reached]",
            "response_text": result.response_text,
            "lead_usage": result.usage or {},
            "lead_turns": result.num_turns,
            "lead_model_calls": sum(event["kind"] == "model_response" for event in events),
            "lead_tool_calls": sum(event["kind"] == "tool_use" for event in events),
        }
    except Exception as exc:
        terminal = next(
            (
                event
                for event in reversed(events)
                if event.get("kind") in {"run_completed", "run_failed", "run_cancelled"}
            ),
            {},
        )
        lead_turns = int(terminal.get("turn") or 0)
        if lead_turns == 0:
            lead_turns = max(
                (int(event.get("turn") or 0) for event in events if event.get("kind") == "model_response"),
                default=0,
            )
        payload = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "lead_usage": terminal.get("usage") or {},
            "lead_turns": lead_turns,
            "lead_model_calls": sum(event["kind"] == "model_response" for event in events),
            "lead_tool_calls": sum(event["kind"] == "tool_use" for event in events),
        }
    _write_json(result_path, payload)
    return 0 if payload["ok"] else 1


def _team_metrics(workspace: Path) -> dict[str, Any]:
    active_path = workspace / ".clawd" / "team.json"
    active = active_path.exists()
    historical = list((workspace / ".clawd" / "teams").glob("*/team.json"))
    team_path = active_path if active else (
        max(historical, key=lambda path: path.stat().st_mtime) if historical else None
    )
    if team_path is None:
        return {
            "present": False,
            "active": False,
            "status": None,
            "agents": [],
            "tasks": 0,
            "completed_tasks": 0,
            "messages": 0,
            "peer_messages": 0,
            "worker_usage": {},
            "trace_model_calls": 0,
            "trace_tool_calls": 0,
            "interventions": {},
        }
    team = _read_json(team_path)
    team_id = str(team.get("team_id") or team_path.parent.name)
    team_dir = workspace / ".clawd" / "teams" / team_id
    lead_id = str(team.get("lead_agent_id") or "")
    agents: list[dict[str, Any]] = []
    for path in sorted((team_dir / "agents").glob("*.json")):
        raw = _read_json(path)
        agents.append(
            {
                "id": raw.get("agent_id"),
                "name": raw.get("name"),
                "role": raw.get("role"),
                "status": raw.get("status"),
                "model": raw.get("model"),
                "tools": raw.get("tools") or [],
                "workspace_mode": raw.get("workspace_mode"),
            }
        )
    tasks = _read_json(team_dir / "tasks.json") if (team_dir / "tasks.json").exists() else {}
    messages = [_read_json(path) for path in sorted((team_dir / "messages").glob("*.json"))]
    event_types: list[str] = []
    events_path = team_dir / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                event_types.append(str(json.loads(line).get("type") or ""))
            except (json.JSONDecodeError, AttributeError):
                continue
    intervention_types = (
        "agent.stop_requested",
        "agent.resumed",
        "task.reassigned",
        "task.retry_requested",
        "team.resumed",
    )
    return {
        "present": True,
        "active": active,
        "team_id": team_id,
        "status": team.get("status"),
        "agents": agents,
        "tasks": len(tasks),
        "completed_tasks": sum(
            1 for task in tasks.values() if isinstance(task, dict) and task.get("status") == "completed"
        ),
        "messages": len(messages),
        "peer_messages": sum(
            str(message.get("sender_id") or "") != lead_id
            and str(message.get("recipient_id") or "") != lead_id
            for message in messages
        ),
        "worker_usage": team.get("usage") or {},
        "trace_model_calls": event_types.count("model.response"),
        "trace_tool_calls": event_types.count("tool.started"),
        "interventions": {name: event_types.count(name) for name in intervention_types},
    }


def _protocol_ok(mode: str, team: dict[str, Any]) -> bool:
    if mode == "solo":
        return not team["present"]
    if mode == "adaptive" and not team["present"]:
        return True
    return bool(
        team["present"]
        and team["status"] == "completed"
        and len(team["agents"]) >= 1
        and team["tasks"] >= 1
        and team["completed_tasks"] == team["tasks"]
    )


def parse_pytest_output(output: str, expected_tests: int, returncode: int) -> dict[str, Any]:
    def last_count(label: str) -> int:
        matches = re.findall(rf"(?<!\w)(\d+)\s+{label}\b", output, flags=re.IGNORECASE)
        return int(matches[-1]) if matches else 0

    passed = last_count("passed")
    failed = last_count("failed")
    errors = last_count("errors?")
    skipped = last_count("skipped")
    quality = 100.0 * min(passed, expected_tests) / expected_tests if expected_tests else 0.0
    return {
        "expected": expected_tests,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "returncode": returncode,
        "quality_score": round(quality, 2),
        "all_passed": bool(returncode == 0 and passed >= expected_tests),
    }


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe upstream test path: {value!r}")
    return path


def _split_score_commands(commands: list[str]) -> tuple[list[str], list[str]]:
    """Split ordered setup commands from the pytest invocation."""
    for index, command in enumerate(commands):
        if PYTEST_COMMAND_RE.search(command):
            return commands[:index], commands[index:]
    return [], commands


def stage_score_context(task: dict[str, Any], workspace: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        workspace,
        destination / "workspace",
        ignore=shutil.ignore_patterns(".git", ".clawd", "__pycache__", ".pytest_cache"),
    )
    staged_workspace = destination / "workspace"
    package_files = sorted(
        path.relative_to(staged_workspace).as_posix()
        for path in staged_workspace.rglob("*")
        if path.is_file() and path.name in PACKAGE_FILES
    )
    generated_hidden_paths = [
        value
        for value in task["hidden_paths"]
        if (staged_workspace / _safe_relative_path(value)).exists()
    ]
    for path in list(staged_workspace.rglob("*")):
        if path.is_file() and path.name in PACKAGE_FILES:
            path.unlink()
    for value in task["hidden_paths"]:
        target = staged_workspace / _safe_relative_path(value)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    setup_commands, test_commands = _split_score_commands(list(task.get("test_commands", [])))
    dockerfile_lines = [
        f"FROM --platform=linux/amd64 {task['image']}",
        "COPY workspace /workspace",
        "WORKDIR /workspace",
        "ENV PYTHONPATH=/workspace:$PYTHONPATH",
    ]
    dockerfile_lines.extend(
        f"RUN {json.dumps(['/bin/bash', '-lc', command])}" for command in setup_commands
    )
    dockerfile_lines.extend(["CMD [\"tail\", \"-f\", \"/dev/null\"]", ""])
    dockerfile = destination / "Dockerfile"
    dockerfile.write_text("\n".join(dockerfile_lines), encoding="utf-8")
    return {
        "package_files_present": package_files,
        "generated_hidden_paths": generated_hidden_paths,
        "dockerfile": str(dockerfile),
        "setup_commands": setup_commands,
        "test_commands": test_commands,
    }


def run_hidden_tests(
    task: dict[str, Any],
    workspace: Path,
    case_root: Path,
    *,
    timeout_s: float,
    keep_image: bool = False,
) -> dict[str, Any]:
    context = case_root / "score-context"
    metadata = stage_score_context(task, workspace, context)
    tag = f"clawd-nl2repo-{task['id'].lower()}-{uuid.uuid4().hex[:10]}"
    build_started = time.monotonic()
    build = subprocess.run(
        ["docker", "build", "--platform", "linux/amd64", "-t", tag, "."],
        cwd=context,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    build_elapsed = time.monotonic() - build_started
    (case_root / "docker-build.log").write_text(
        f"{build.stdout}\n{build.stderr}".strip(), encoding="utf-8"
    )
    if build.returncode != 0:
        return {
            **metadata,
            "image": task["image"],
            "build_returncode": build.returncode,
            "build_elapsed_s": round(build_elapsed, 3),
            "error": "Docker score image build failed",
            "pytest": parse_pytest_output("", int(task["expected_tests"]), 1),
        }
    test_started = time.monotonic()
    test_command = " && ".join(f"({command})" for command in metadata["test_commands"])
    try:
        completed = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                "linux/amd64",
                "--network",
                "none",
                tag,
                "/bin/bash",
                "-lc",
                test_command,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        test_output = f"{completed.stdout}\n{completed.stderr}".strip()
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        test_output = f"{exc.stdout or ''}\n{exc.stderr or ''}".strip()
        returncode = 124
        timed_out = True
    finally:
        if not keep_image:
            subprocess.run(["docker", "image", "rm", "-f", tag], capture_output=True, text=True)
    test_elapsed = time.monotonic() - test_started
    (case_root / "hidden-tests.log").write_text(test_output, encoding="utf-8")
    result = {
        **metadata,
        "image": task["image"],
        "built_image": tag if keep_image else None,
        "build_returncode": build.returncode,
        "build_elapsed_s": round(build_elapsed, 3),
        "test_elapsed_s": round(test_elapsed, 3),
        "timed_out": timed_out,
        "pytest": parse_pytest_output(test_output, int(task["expected_tests"]), returncode),
    }
    shutil.rmtree(context, ignore_errors=True)
    return result


def run_case(
    task: dict[str, Any],
    mode: str,
    output_root: Path,
    *,
    provider: str,
    model: str,
    max_turns: int,
    teammate_max_turns: int,
    max_output_tokens: int,
    agent_timeout_s: float,
    score_timeout_s: float,
    keep_image: bool,
    stream: bool,
) -> dict[str, Any]:
    case_root = output_root / task["id"] / mode
    case_root.mkdir(parents=True, exist_ok=True)
    workspace = case_root / "workspace"
    start_hash = prepare_workspace(task, workspace)
    prompt_path = case_root / "PROMPT.md"
    prompt_path.write_text(build_prompt(mode), encoding="utf-8")
    result_path = case_root / "agent-result.json"
    progress_path = case_root / "progress.jsonl"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_run-one",
        "--workspace",
        str(workspace),
        "--prompt-file",
        str(prompt_path),
        "--result-file",
        str(result_path),
        "--provider",
        provider,
        "--model",
        model,
        "--max-turns",
        str(max_turns),
        "--teammate-max-turns",
        str(teammate_max_turns),
        "--max-output-tokens",
        str(max_output_tokens),
        "--progress-file",
        str(progress_path),
    ]
    if stream:
        command.append("--stream")
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=agent_timeout_s,
            env=os.environ.copy(),
        )
        agent_returncode = completed.returncode
        stdout, stderr = completed.stdout, completed.stderr
        agent_timed_out = False
    except subprocess.TimeoutExpired as exc:
        agent_returncode = 124
        stdout, stderr = exc.stdout or "", exc.stderr or ""
        agent_timed_out = True
    agent_elapsed = time.monotonic() - started
    (case_root / "stdout.log").write_text(str(stdout), encoding="utf-8")
    (case_root / "stderr.log").write_text(str(stderr), encoding="utf-8")
    agent = _read_json(result_path) if result_path.exists() else {
        "ok": False,
        "error": "agent timed out" if agent_timed_out else "missing agent result",
        "lead_usage": {},
        "lead_turns": 0,
        "lead_model_calls": 0,
        "lead_tool_calls": 0,
    }
    hidden = run_hidden_tests(
        task,
        workspace,
        case_root,
        timeout_s=score_timeout_s,
        keep_image=keep_image,
    )
    team = _team_metrics(workspace)
    protocol_ok = _protocol_ok(mode, team)
    integrity_ok = (workspace / "start.md").is_file() and _hash_file(workspace / "start.md") == start_hash
    lead_usage = agent.get("lead_usage") or {}
    worker_usage = team.get("worker_usage") or {}
    input_tokens = int(lead_usage.get("input_tokens", 0) or 0) + int(
        worker_usage.get("input_tokens", 0) or 0
    )
    output_tokens = int(lead_usage.get("output_tokens", 0) or 0) + int(
        worker_usage.get("output_tokens", 0) or 0
    )
    pytest_result = hidden["pytest"]
    result = {
        "task": task["id"],
        "difficulty": task["difficulty"],
        "mode": mode,
        "provider": provider,
        "model": model,
        "agent_elapsed_s": round(agent_elapsed, 3),
        "agent_timed_out": agent_timed_out,
        "agent_returncode": agent_returncode,
        "agent_ok": bool(agent.get("ok")),
        "agent_error": agent.get("error"),
        "integrity_ok": integrity_ok,
        "protocol_ok": protocol_ok,
        "used_team": team["present"],
        "quality_score": pytest_result["quality_score"] if integrity_ok else 0.0,
        "success": bool(agent.get("ok") and integrity_ok and protocol_ok and pytest_result["all_passed"]),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "lead_turns": int(agent.get("lead_turns", 0) or 0),
            "worker_turns": int(worker_usage.get("turns", 0) or 0),
        },
        "calls": {
            "model": team["trace_model_calls"] if team["present"] else agent.get("lead_model_calls", 0),
            "tools": team["trace_tool_calls"] if team["present"] else agent.get("lead_tool_calls", 0),
        },
        "team": team,
        "hidden_tests": hidden,
        "workspace": str(workspace),
    }
    _write_json(case_root / "result.json", result)
    return result


def render_report(results: list[dict[str, Any]], run_id: str, upstream_ref: str) -> str:
    lines = [
        "# NL2Repo Pilot Benchmark",
        "",
        f"Run: `{run_id}`",
        f"Upstream: `{upstream_ref}`",
        "",
        "| Task | Difficulty | Mode | Quality | Passed | Seconds | Tokens | Agents | Peer messages | Protocol |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for result in results:
        tests = result["hidden_tests"]["pytest"]
        lines.append(
            "| "
            + " | ".join(
                [
                    result["task"],
                    result["difficulty"] or "-",
                    result["mode"],
                    str(result["quality_score"]),
                    f"{tests['passed']}/{tests['expected']}",
                    str(result["agent_elapsed_s"]),
                    str(result["usage"]["total_tokens"]),
                    str(len(result["team"]["agents"])),
                    str(result["team"]["peer_messages"]),
                    "yes" if result["protocol_ok"] else "no",
                ]
            )
            + " |"
        )
    success = sum(bool(result["success"]) for result in results)
    lines.extend(
        [
            "",
            f"Strict successful runs: **{success}/{len(results)}**",
            "",
            "Quality is the percentage of hidden upstream pytest cases passed. Strict success",
            "also requires an intact specification, a valid execution protocol, and a completed",
            "agent run. The upstream data is referenced externally and is not vendored here.",
        ]
    )
    return "\n".join(lines) + "\n"


def validate_tasks(tasks: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    try:
        _run_checked(["docker", "info"])
    except Exception as exc:
        errors.append(f"Docker is unavailable: {exc}")
        return errors
    for task in tasks:
        if int(task["expected_tests"]) < 1:
            errors.append(f"{task['id']}: expected test count must be positive")
        completed = subprocess.run(
            ["docker", "manifest", "inspect", task["image"]],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            errors.append(f"{task['id']}: test image unavailable: {task['image']}")
    return errors


def _child_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("_command")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--teammate-max-turns", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--progress-file", type=Path, required=True)
    parser.add_argument("--stream", action="store_true")
    return parser


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_run-one":
        args = _child_parser().parse_args()
        return _run_agent_child(
            args.workspace.resolve(),
            args.prompt_file.resolve(),
            args.result_file.resolve(),
            args.provider,
            args.model,
            args.max_turns,
            args.teammate_max_turns,
            args.max_output_tokens,
            args.stream,
            args.progress_file.resolve(),
        )

    parser = argparse.ArgumentParser(description="Run Clawd against pinned NL2Repo-Bench tasks.")
    parser.add_argument("--list", action="store_true", help="List all upstream tasks")
    parser.add_argument("--validate", action="store_true", help="Validate task metadata and images")
    parser.add_argument("--task", action="append", help="Task ID; repeat to select several")
    parser.add_argument(
        "--mode",
        choices=("solo", "adaptive", "forced-team", "both", "all"),
        default="both",
    )
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--teammate-max-turns", type=int, default=80)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--agent-timeout", type=float, default=7200.0)
    parser.add_argument("--score-timeout", type=float, default=1200.0)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-image", action="store_true")
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use structured streaming for model calls (default: enabled)",
    )
    args = parser.parse_args()

    upstream_root = resolve_upstream(args.upstream_root, cache_root=args.cache_root)
    if args.list:
        for task in list_tasks(upstream_root):
            marker = "*" if task["id"] in PILOT_TASKS else " "
            print(
                f"{marker} {task['id']:<28} {task['difficulty']:<6} "
                f"tests={task['expected_tests']:<4} prompt_bytes={task['prompt_bytes']}"
            )
        return 0

    task_names = args.task or list(PILOT_TASKS)
    tasks = [load_task(upstream_root, name) for name in task_names]
    if args.validate:
        errors = validate_tasks(tasks)
        if errors:
            for error in errors:
                print(f"- {error}")
            return 1
        print(f"{len(tasks)} NL2Repo tasks and Docker images are available")
        return 0

    if args.mode == "both":
        modes = ("solo", "adaptive")
    elif args.mode == "all":
        modes = ("solo", "adaptive", "forced-team")
    else:
        modes = (args.mode,)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (args.output or ROOT / "runs" / run_id).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for task in tasks:
        for mode in modes:
            print(f"[{task['id']}] running {mode}...", flush=True)
            result = run_case(
                task,
                mode,
                output_root,
                provider=args.provider,
                model=args.model,
                max_turns=args.max_turns,
                teammate_max_turns=args.teammate_max_turns,
                max_output_tokens=args.max_output_tokens,
                agent_timeout_s=args.agent_timeout,
                score_timeout_s=args.score_timeout,
                keep_image=args.keep_image,
                stream=args.stream,
            )
            results.append(result)
            print(
                f"  quality={result['quality_score']:.2f} success={result['success']} "
                f"time={result['agent_elapsed_s']:.1f}s tokens={result['usage']['total_tokens']}",
                flush=True,
            )
    aggregate = {
        "run_id": run_id,
        "upstream_url": UPSTREAM_URL,
        "upstream_ref": UPSTREAM_REF,
        "provider": args.provider,
        "model": args.model,
        "results": results,
    }
    _write_json(output_root / "results.json", aggregate)
    report = render_report(results, run_id, UPSTREAM_REF)
    (output_root / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"\n{report}\nArtifacts: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
