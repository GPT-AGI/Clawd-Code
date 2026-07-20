from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import random
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
UPSTREAM_URL = "https://github.com/multimodal-art-projection/NL2RepoBench.git"
UPSTREAM_REF = "781a1da1ee41fb8edb0bed22f586d69111610edf"
IMAGE_ROOT = "ghcr.io/multimodal-art-projection/nl2repobench"
AGS_IMAGE_TEMPLATE = "swebenchdocker.tencentcloudcr.com/swebench/nl2repo:{task}-1.0"
AGS_SCORE_SETUP_CONCURRENCY = max(
    1, int(os.environ.get("AGS_SCORE_SETUP_CONCURRENCY", "8"))
)
AGS_SCORE_SETUP_SLOTS = threading.BoundedSemaphore(AGS_SCORE_SETUP_CONCURRENCY)
PILOT_TASKS = ("jsonlines", "tinydb", "aiofiles", "flask-restful", "fastapi-users")
ROLLOUT32_SEED = 20260715
ROLLOUT32_SIZE = 32
ROLLOUT32_MAX_PROMPT_BYTES = 64 * 1024
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


def format_ags_image(template: str, task_name: str) -> str:
    """Format a registry-safe image reference without changing the task ID."""
    return template.format(task=task_name.casefold())


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


def select_task_subset(
    task_metadata: list[dict[str, Any]],
    *,
    count: int = ROLLOUT32_SIZE,
    seed: int = ROLLOUT32_SEED,
    max_prompt_bytes: int = ROLLOUT32_MAX_PROMPT_BYTES,
) -> list[str]:
    """Select the same deterministic, bounded-context subset as the latency probe."""
    eligible = sorted(
        (
            task
            for task in task_metadata
            if int(task.get("prompt_bytes", 0)) <= max_prompt_bytes
        ),
        key=lambda task: str(task["id"]),
    )
    if len(eligible) < count:
        raise ValueError(
            f"only {len(eligible)} tasks are <= {max_prompt_bytes} bytes; "
            f"cannot select {count}"
        )
    selected = random.Random(seed).sample(eligible, count)
    return sorted(str(task["id"]) for task in selected)


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
        "image": f"{IMAGE_ROOT}/{task_name.casefold()}:1.0",
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
    if mode == "adaptive-team-v2":
        return common + """
Adaptive Team v2 protocol: act as the lead and make an explicit collaboration routing
decision after reading the complete specification. Create a Team when the work contains
at least two substantially independent implementation, investigation, or validation
streams whose parallel progress is likely to exceed coordination cost. Remain solo for
small, tightly coupled work where delegation would only duplicate context.

When using a Team, start with at most two teammates. Give each teammate a concrete,
non-overlapping, independently verifiable task and omit the model override so every
teammate inherits the configured endpoint model. Keep ownership of architecture,
integration, and final end-to-end validation as the lead. Reserve substantial lead
budget for integration and tests; inspect teammate output before relying on it, and do
not repeatedly resume an unproductive task. The topology and Team decision must follow
from this repository rather than a predefined role pipeline. It is valid to complete
the rollout without creating a Team when the routing criteria are not met.

If you use a Team, create it with quality_gates=true. Then call TeamConfigure with a
written architecture contract plus clean-install, import-smoke, and integration
commands. Create at least two initially runnable tasks owned by distinct teammates.
Every task must declare concrete non-overlapping ownedFiles and acceptanceChecks;
declare providesInterfaces/dependsOnInterfaces and blockedBy whenever work crosses a
module boundary. After TeamRun completes worker tasks, call TeamVerify. Do not finish
the rollout while the Team status is verification_required or validation has failed.
"""
    if mode == "forced-team":
        return common + """
Diagnostic execution protocol: the harness has already created the active strict Team.
Do not call TeamCreate. After reading start.md, call TeamConfigure with a concrete
architecture contract and clean-install, import-smoke, and integration commands. You
must call TeammateCreate at least twice to create at least two teammates, then create
at least two initially runnable tasks owned by
distinct teammates. Every TaskCreate must declare concrete non-overlapping ownedFiles
and acceptanceChecks. Declare providesInterfaces/dependsOnInterfaces plus blockedBy
for cross-module dependencies; owners on an interface edge must exchange at least one
substantive peer message. Execute the plan with TeamRun and then call TeamVerify. The
Team is not complete while verification is pending or failed.
Choose the number of agents,
task-specific roles, tool permissions, workspaces, dependencies, concurrency,
and communication topology from the repository itself. Do not use a predefined role
pipeline. All teammates must inherit the configured endpoint model; omit the model
override in TeammateCreate. Observe progress, intervene when needed, integrate the work, and personally
perform final validation. A rollout without two completed teammate tasks and passed
TeamVerify validation is invalid.
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


def _install_remote_workspace(downloaded: Path, workspace: Path) -> None:
    """Mirror agent-visible files locally while retaining local orchestration state."""
    for item in workspace.iterdir():
        if item.name == ".clawd":
            continue
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()
    for item in downloaded.iterdir():
        target = workspace / item.name
        if target.name == ".clawd":
            continue
        shutil.move(str(item), target)


def _is_ags_image_preparing_error(error: Exception) -> bool:
    message = str(error).casefold()
    return "resourceunavailable" in message and "image is still preparing" in message


def start_ags_backend_with_retry(
    factory: Callable[[], Any],
    *,
    attempts: int | None = None,
    delay_s: float | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> Any:
    """Wait for lazily prepared AGS task images without failing the rollout."""
    retry_attempts = attempts or int(os.environ.get("AGS_IMAGE_PREPARE_ATTEMPTS", "20"))
    retry_delay = (
        delay_s
        if delay_s is not None
        else float(os.environ.get("AGS_IMAGE_PREPARE_RETRY_SEC", "30"))
    )
    if retry_attempts < 1 or retry_delay < 0:
        raise ValueError("AGS image retry attempts must be positive and delay non-negative")
    sleeper = sleep_fn or time.sleep
    for attempt in range(1, retry_attempts + 1):
        try:
            return factory().start()
        except Exception as exc:
            if not _is_ags_image_preparing_error(exc) or attempt == retry_attempts:
                raise
            if on_retry is not None:
                on_retry(attempt, exc)
            sleeper(retry_delay)
    raise AssertionError("unreachable")


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
    *,
    teammate_min_timeout_s: float | None = None,
    mode: str = "adaptive",
    execution_backend: str = "local",
    ags_image: str | None = None,
    ags_env_file: Path | None = None,
    ags_timeout: str = "3h",
    ags_cpu: str = "2",
    ags_memory: str = "4Gi",
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

    backend: Any | None = None
    sandbox_id = ""
    payload: dict[str, Any]
    try:
        if mode == "forced-team":
            from src.teammate.store import TeamStore

            store = TeamStore(workspace)
            team = store.load_active_team()
            if team is None:
                team = store.create_team(
                    "nl2repo-forced-team",
                    description=(
                        "Harness-created team for the forced-team NL2Repo evaluation protocol"
                    ),
                    agent_type="adaptive",
                )
            quality = dict(team.settings.get("quality_gates") or {})
            quality.update(
                {
                    "strict": True,
                    "configured": bool(quality.get("configured", False)),
                    "validation": quality.get("validation") or {"status": "pending"},
                }
            )
            team.settings["quality_gates"] = quality
            store.save_team(team)
            _append_jsonl(
                progress_path,
                {
                    "kind": "forced_team_precreated",
                    "team_id": team.team_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        if execution_backend == "ags":
            if not ags_image:
                raise ValueError("AGS execution requires an image")
            from src.execution.ags import AGSSettings, AGSWorkspaceBackend

            settings = AGSSettings.from_env(
                image=ags_image,
                env_file=ags_env_file,
                timeout=ags_timeout,
                cpu=ags_cpu,
                memory=ags_memory,
            )
            def record_image_retry(attempt: int, error: Exception) -> None:
                _append_jsonl(
                    progress_path,
                    {
                        "kind": "sandbox_image_waiting",
                        "backend": "ags",
                        "image": ags_image,
                        "attempt": attempt,
                        "retry_in_s": float(
                            os.environ.get("AGS_IMAGE_PREPARE_RETRY_SEC", "30")
                        ),
                        "error_type": type(error).__name__,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

            backend = start_ags_backend_with_retry(
                lambda: AGSWorkspaceBackend(settings),
                on_retry=record_image_retry,
            )
            sandbox_id = backend.sandbox_id
            _append_jsonl(
                progress_path,
                {
                    "kind": "sandbox_started",
                    "backend": "ags",
                    "sandbox_id": sandbox_id,
                    "image": ags_image,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            backend.reset_workspace()
            backend.upload_tree(workspace, backend.workspace_root)
        result = run_prompt(
            prompt_path.read_text(encoding="utf-8"),
            workspace=workspace,
            provider_name=provider,
            model=model,
            max_turns=max_turns,
            teammate_max_turns=teammate_max_turns,
            teammate_min_timeout_s=teammate_min_timeout_s,
            max_output_tokens=max_output_tokens,
            stream=stream,
            on_event=capture,
            on_text_chunk=capture_text,
            workspace_backend=backend,
        )
        payload = {
            "ok": result.response_text != "[Max tool turns reached]",
            "response_text": result.response_text,
            "lead_usage": result.usage or {},
            "lead_turns": result.num_turns,
            "lead_model_calls": sum(event["kind"] == "model_response" for event in events),
            "lead_tool_calls": sum(event["kind"] == "tool_use" for event in events),
            "execution_backend": execution_backend,
            "sandbox_id": sandbox_id or None,
        }
    except BaseException as exc:
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
            "execution_backend": execution_backend,
            "sandbox_id": sandbox_id or None,
        }
    finally:
        if backend is not None:
            try:
                with tempfile.TemporaryDirectory(
                    prefix="clawd-ags-download-", dir=workspace.parent
                ) as temporary:
                    downloaded = Path(temporary)
                    backend.download_tree(backend.workspace_root, downloaded)
                    _install_remote_workspace(downloaded, workspace)
                _append_jsonl(
                    progress_path,
                    {
                        "kind": "sandbox_workspace_downloaded",
                        "backend": "ags",
                        "sandbox_id": sandbox_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except BaseException as exc:
                payload["ok"] = False
                payload["workspace_download_error"] = str(exc)
            finally:
                try:
                    backend.close()
                except BaseException as exc:
                    payload["sandbox_close_error"] = str(exc)
                _append_jsonl(
                    progress_path,
                    {
                        "kind": "sandbox_stopped",
                        "backend": "ags",
                        "sandbox_id": sandbox_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
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
            "quality_gates": {},
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
    quality_gates = dict((team.get("settings") or {}).get("quality_gates") or {})
    validation = dict(quality_gates.get("validation") or {})
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
        "quality_gates": {
            "strict": bool(quality_gates.get("strict")),
            "configured": bool(quality_gates.get("configured")),
            "plan_accepted": bool(quality_gates.get("plan_accepted")),
            "validation_status": validation.get("status"),
        },
    }


def _protocol_ok(mode: str, team: dict[str, Any]) -> bool:
    if mode == "solo":
        return not team["present"]
    if mode in {"adaptive", "adaptive-team-v2"} and not team["present"]:
        return True
    quality = team.get("quality_gates") or {}
    strict_ok = not quality.get("strict") or (
        quality.get("configured")
        and quality.get("plan_accepted")
        and quality.get("validation_status") == "passed"
    )
    minimum = 2 if quality.get("strict") else 1
    return bool(
        team["present"]
        and team["status"] == "completed"
        and len(team["agents"]) >= minimum
        and team["tasks"] >= minimum
        and team["completed_tasks"] == team["tasks"]
        and strict_ok
    )


def _combined_usage(
    lead_usage: dict[str, Any],
    worker_usage: dict[str, Any],
    *,
    used_team: bool,
    lead_turns: int,
) -> dict[str, Any]:
    """Combine auditable lead/worker token counts and expose coverage explicitly."""
    lead_input = int(lead_usage.get("input_tokens", 0) or 0)
    lead_output = int(lead_usage.get("output_tokens", 0) or 0)
    worker_input = int(worker_usage.get("input_tokens", 0) or 0)
    worker_output = int(worker_usage.get("output_tokens", 0) or 0)
    lead_recorded = lead_input > 0 or lead_output > 0
    worker_recorded = not used_team or worker_input > 0 or worker_output > 0
    return {
        "input_tokens": lead_input + worker_input,
        "output_tokens": lead_output + worker_output,
        "total_tokens": lead_input + lead_output + worker_input + worker_output,
        "lead_input_tokens": lead_input,
        "lead_output_tokens": lead_output,
        "worker_input_tokens": worker_input,
        "worker_output_tokens": worker_output,
        "lead_turns": lead_turns,
        "worker_turns": int(worker_usage.get("turns", 0) or 0),
        "lead_recorded": lead_recorded,
        "worker_recorded": worker_recorded,
        "complete": lead_recorded and worker_recorded,
    }


def classify_failure(
    *,
    agent_ok: bool,
    integrity_ok: bool,
    protocol_ok: bool,
    team: dict[str, Any],
    hidden: dict[str, Any],
    hidden_log: str,
) -> str | None:
    """Assign a stable, dashboard-friendly failure class without changing reward."""
    pytest_result = hidden.get("pytest") if isinstance(hidden.get("pytest"), dict) else {}
    if hidden.get("error"):
        return "scorer_infrastructure"
    if hidden.get("timed_out") or pytest_result.get("returncode") == 124:
        return "reward_timeout"
    if not agent_ok:
        return "rollout_failure"
    if not integrity_ok:
        return "spec_integrity"
    if not protocol_ok:
        quality = team.get("quality_gates") or {}
        if quality.get("strict") and quality.get("validation_status") != "passed":
            return "team_validation"
        return "team_protocol"
    if bool(pytest_result.get("all_passed")):
        return None

    lowered = hidden_log.casefold()
    if any(
        marker in lowered
        for marker in (
            "modulenotfounderror",
            "no module named",
            "could not install packages",
            "distributionnotfound",
        )
    ):
        return "dependency_environment"
    if int(pytest_result.get("errors", 0) or 0) > 0:
        return "collection_error"
    if any(
        marker in lowered
        for marker in (
            "unexpected keyword argument",
            "has no attribute",
            "missing 1 required positional argument",
            "got an unexpected keyword",
        )
    ):
        if len(team.get("agents") or []) > 1 and int(team.get("peer_messages", 0) or 0) == 0:
            return "cross_module_contract"
        return "api_contract"
    if int(pytest_result.get("failed", 0) or 0) > 0:
        return "functional_test_failure"
    return "unknown_failure"


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


def _score_shell_command(setup_commands: list[str], test_commands: list[str]) -> str:
    """Match upstream: run every command and use the final command's exit status."""
    commands = [*setup_commands, *test_commands]
    return "; ".join(f"({command})" for command in commands) or "true"


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
        "CMD [\"tail\", \"-f\", \"/dev/null\"]",
        "",
    ]
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
    test_command = _score_shell_command(
        metadata["setup_commands"], metadata["test_commands"]
    )
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


def run_hidden_tests_ags(
    task: dict[str, Any],
    workspace: Path,
    case_root: Path,
    *,
    timeout_s: float,
    ags_image: str,
    ags_env_file: Path | None,
    ags_timeout: str,
    ags_cpu: str,
    ags_memory: str,
    ags_score_tool_id: str | None = None,
) -> dict[str, Any]:
    """Score in a fresh AGS instance so hidden tests never enter the agent sandbox."""
    from src.execution.ags import AGSSettings, AGSWorkspaceBackend

    context = case_root / "score-context"
    metadata = stage_score_context(task, workspace, context)
    settings = AGSSettings.from_env(
        image=ags_image,
        env_file=ags_env_file,
        timeout=ags_timeout,
        cpu=ags_cpu,
        memory=ags_memory,
    )
    score_tool_id = ags_score_tool_id or os.environ.get("AGS_SCORE_TOOL_ID", "").strip()
    if not score_tool_id:
        raise RuntimeError(
            "AGS scoring requires a dedicated no-egress tool; configure AGS_SCORE_TOOL_ID "
            "or keep --score-backend docker"
        )
    settings.tool_id = score_tool_id
    settings.runtime_timeout = max(settings.runtime_timeout, timeout_s + 30)
    settings.network_mode = "SANDBOX"
    started = time.monotonic()
    backend: Any | None = None
    sandbox_id = ""
    test_output = ""
    returncode = 1
    timed_out = False
    test_elapsed = 0.0
    cleanup_error: str | None = None
    try:
        # Creating many sandboxes and uploading all workspaces in one burst can
        # saturate the AGS runtime gateway even though the service can execute
        # many tests concurrently. Throttle setup only; release the slot before
        # the long-running hidden tests so the reward pool can still reach 64.
        with AGS_SCORE_SETUP_SLOTS:
            backend = start_ags_backend_with_retry(lambda: AGSWorkspaceBackend(settings))
            sandbox_id = backend.sandbox_id
            startup_elapsed = time.monotonic() - started
            network_probe = backend.exec(
                "python3 - <<'PY'\n"
                "import urllib.request\n"
                "try:\n"
                " urllib.request.urlopen('https://example.com', timeout=5)\n"
                "except Exception:\n"
                " raise SystemExit(0)\n"
                "raise SystemExit(86)\n"
                "PY",
                cwd=backend.workspace_root,
                timeout_s=15,
            )
            if network_probe.exit_code != 0:
                raise RuntimeError(
                    "AGS score sandbox has outbound network access; use a SandboxTool whose "
                    "NetworkConfiguration.NetworkMode is SANDBOX"
                )
            # Do not reset /workspace: the fresh task image owns the official package
            # metadata and hidden tests. Uploading the stripped candidate overlays only
            # implementation files, matching Docker COPY semantics.
            backend.upload_tree(context / "workspace", backend.workspace_root)
            print(
                f"[{task['id']}] ags-score.upload.completed · sandbox={sandbox_id}",
                flush=True,
            )
        test_command = "export PYTHONPATH=/workspace:${PYTHONPATH:-}; " + _score_shell_command(
            metadata["setup_commands"], metadata["test_commands"]
        )
        test_started = time.monotonic()
        print(
            f"[{task['id']}] ags-score.tests.started · timeout={int(timeout_s)}s",
            flush=True,
        )
        completed = backend.exec(
            test_command,
            cwd=backend.workspace_root,
            timeout_s=max(1, int(timeout_s)),
        )
        test_elapsed = time.monotonic() - test_started
        print(
            f"[{task['id']}] ags-score.tests.completed · "
            f"exit={completed.exit_code} elapsed={test_elapsed:.1f}s",
            flush=True,
        )
        test_output = f"{completed.stdout}\n{completed.stderr}".strip()
        returncode = completed.exit_code
    except TimeoutError as exc:
        startup_elapsed = time.monotonic() - started
        test_output = str(exc)
        returncode = 124
        timed_out = True
    except Exception as exc:
        startup_elapsed = time.monotonic() - started
        test_output = f"{exc}\n{traceback.format_exc()}"
        returncode = 1
    finally:
        try:
            if backend is not None:
                try:
                    backend.close()
                except Exception as exc:
                    # Sandbox cleanup is infrastructure housekeeping. It must
                    # not overwrite a completed hidden-test result and cause
                    # the whole reward to be retried or marked failed.
                    cleanup_error = f"{type(exc).__name__}: {exc}"
        finally:
            shutil.rmtree(context, ignore_errors=True)
    (case_root / "hidden-tests.log").write_text(test_output, encoding="utf-8")
    return {
        **metadata,
        "backend": "ags",
        "image": ags_image,
        "sandbox_id": sandbox_id or None,
        "startup_elapsed_s": round(startup_elapsed, 3),
        "test_elapsed_s": round(test_elapsed, 3),
        "timed_out": timed_out,
        "cleanup_error": cleanup_error,
        "pytest": parse_pytest_output(test_output, int(task["expected_tests"]), returncode),
    }


@dataclasses.dataclass
class RolloutArtifact:
    task: dict[str, Any]
    mode: str
    case_root: Path
    workspace: Path
    start_hash: str
    agent: dict[str, Any]
    agent_elapsed_s: float
    agent_timed_out: bool
    agent_returncode: int


def _require_agent_result(
    result_path: Path,
    *,
    returncode: int,
    timed_out: bool,
    stderr: str,
) -> dict[str, Any]:
    """Reject harness-level child failures before they can reach reward scoring."""
    if not result_path.is_file():
        reason = "timed out" if timed_out else f"exited with code {returncode}"
        stderr_tail = " ".join(str(stderr).split())[-1_200:]
        detail = f"; stderr: {stderr_tail}" if stderr_tail else ""
        raise RuntimeError(
            f"agent subprocess {reason} without producing {result_path.name}{detail}"
        )
    try:
        agent = _read_json(result_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"agent subprocess produced an invalid {result_path.name}: {exc}") from exc
    if not isinstance(agent, dict):
        raise RuntimeError(
            f"agent subprocess produced a non-object {result_path.name}: "
            f"{type(agent).__name__}"
        )
    return agent


def run_rollout(
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
    stream: bool,
    teammate_min_timeout_s: float | None = None,
    execution_backend: str = "local",
    ags_image: str | None = None,
    ags_env_file: Path | None = None,
    ags_timeout: str = "3h",
    ags_cpu: str = "2",
    ags_memory: str = "4Gi",
) -> RolloutArtifact:
    """Run only the agent phase and release its rollout slot before scoring."""
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
        "--teammate-min-timeout",
        str(teammate_min_timeout_s or 0),
        "--max-output-tokens",
        str(max_output_tokens),
        "--progress-file",
        str(progress_path),
        "--mode",
        mode,
    ]
    if stream:
        command.append("--stream")
    command.extend(["--execution-backend", execution_backend])
    if execution_backend == "ags":
        if not ags_image:
            raise ValueError("AGS execution requires an image")
        command.extend(
            [
                "--ags-image",
                ags_image,
                "--ags-timeout",
                ags_timeout,
                "--ags-cpu",
                ags_cpu,
                "--ags-memory",
                ags_memory,
            ]
        )
        if ags_env_file is not None:
            command.extend(["--ags-env-file", str(ags_env_file)])
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    try:
        stdout, stderr = process.communicate(timeout=agent_timeout_s)
        agent_returncode = process.returncode
        agent_timed_out = False
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        agent_returncode = 124
        agent_timed_out = True
    agent_elapsed = time.monotonic() - started
    (case_root / "stdout.log").write_text(str(stdout), encoding="utf-8")
    (case_root / "stderr.log").write_text(str(stderr), encoding="utf-8")
    agent = _require_agent_result(
        result_path,
        returncode=agent_returncode,
        timed_out=agent_timed_out,
        stderr=stderr,
    )
    return RolloutArtifact(
        task=task,
        mode=mode,
        case_root=case_root,
        workspace=workspace,
        start_hash=start_hash,
        agent=agent,
        agent_elapsed_s=agent_elapsed,
        agent_timed_out=agent_timed_out,
        agent_returncode=agent_returncode,
    )


def score_rollout(
    rollout: RolloutArtifact,
    *,
    provider: str,
    model: str,
    score_timeout_s: float,
    keep_image: bool,
    execution_backend: str = "local",
    score_backend: str = "docker",
    ags_image: str | None = None,
    ags_env_file: Path | None = None,
    ags_timeout: str = "3h",
    ags_cpu: str = "2",
    ags_memory: str = "4Gi",
    ags_score_tool_id: str | None = None,
) -> dict[str, Any]:
    """Score a completed rollout in the independent reward pool."""
    task = rollout.task
    mode = rollout.mode
    case_root = rollout.case_root
    workspace = rollout.workspace
    agent = rollout.agent
    team = _team_metrics(workspace)
    protocol_ok = _protocol_ok(mode, team)
    integrity_ok = (
        (workspace / "start.md").is_file()
        and _hash_file(workspace / "start.md") == rollout.start_hash
    )
    rollout_gate_errors: list[str] = []
    if not bool(agent.get("ok")):
        rollout_gate_errors.append("agent rollout did not finish successfully")
    if not integrity_ok:
        rollout_gate_errors.append("start.md integrity check failed")
    if not protocol_ok:
        rollout_gate_errors.append("team protocol or strict validation is incomplete")

    if rollout_gate_errors:
        reason = "; ".join(rollout_gate_errors)
        (case_root / "hidden-tests.log").write_text(
            "Reward skipped: " + reason + "\n", encoding="utf-8"
        )
        hidden = {
            "skipped": True,
            "skip_reason": reason,
            "pytest": parse_pytest_output("", int(task["expected_tests"]), 1),
        }
    elif score_backend == "ags":
        if not ags_image:
            raise ValueError("AGS scoring requires an image")
        hidden = run_hidden_tests_ags(
            task,
            workspace,
            case_root,
            timeout_s=score_timeout_s,
            ags_image=ags_image,
            ags_env_file=ags_env_file,
            ags_timeout=ags_timeout,
            ags_cpu=ags_cpu,
            ags_memory=ags_memory,
            ags_score_tool_id=ags_score_tool_id,
        )
    else:
        hidden = run_hidden_tests(
            task,
            workspace,
            case_root,
            timeout_s=score_timeout_s,
            keep_image=keep_image,
        )
    lead_usage = agent.get("lead_usage") or {}
    worker_usage = team.get("worker_usage") or {}
    usage = _combined_usage(
        lead_usage,
        worker_usage,
        used_team=bool(team["present"]),
        lead_turns=int(agent.get("lead_turns", 0) or 0),
    )
    pytest_result = hidden["pytest"]
    try:
        hidden_log = (case_root / "hidden-tests.log").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        hidden_log = ""
    failure_class = classify_failure(
        agent_ok=bool(agent.get("ok")),
        integrity_ok=integrity_ok,
        protocol_ok=protocol_ok,
        team=team,
        hidden=hidden,
        hidden_log=hidden_log,
    )
    result = {
        "task": task["id"],
        "difficulty": task["difficulty"],
        "mode": mode,
        "provider": provider,
        "model": model,
        "execution_backend": execution_backend,
        "score_backend": score_backend,
        "agent_elapsed_s": round(rollout.agent_elapsed_s, 3),
        "agent_timed_out": rollout.agent_timed_out,
        "agent_returncode": rollout.agent_returncode,
        "agent_ok": bool(agent.get("ok")),
        "agent_error": agent.get("error"),
        "integrity_ok": integrity_ok,
        "protocol_ok": protocol_ok,
        "used_team": team["present"],
        "quality_score": pytest_result["quality_score"] if integrity_ok else 0.0,
        "success": bool(agent.get("ok") and integrity_ok and protocol_ok and pytest_result["all_passed"]),
        "failure_class": failure_class,
        "usage": usage,
        "calls": {
            "model": team["trace_model_calls"] if team["present"] else agent.get("lead_model_calls", 0),
            "tools": team["trace_tool_calls"] if team["present"] else agent.get("lead_tool_calls", 0),
        },
        "team": team,
        "hidden_tests": hidden,
        "reward_skipped": bool(hidden.get("skipped")),
        "workspace": str(workspace),
    }
    _write_json(case_root / "result.json", result)
    return result


def rescore_existing_case(
    task: dict[str, Any],
    mode: str,
    output_root: Path,
    *,
    score_backend: str,
    score_timeout_s: float,
    keep_image: bool,
    ags_image: str | None = None,
    ags_env_file: Path | None = None,
    ags_timeout: str = "3h",
    ags_cpu: str = "2",
    ags_memory: str = "4Gi",
    ags_score_tool_id: str | None = None,
) -> dict[str, Any]:
    """Re-run only hidden tests for a persisted rollout workspace."""
    case_root = output_root / task["id"] / mode
    workspace = case_root / "workspace"
    result_path = case_root / "result.json"
    if not workspace.is_dir() or not result_path.is_file():
        raise FileNotFoundError(f"completed case not found: {case_root}")
    result = _read_json(result_path)
    previous_reward = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "quality_score": result.get("quality_score"),
        "success": result.get("success"),
        "hidden_tests": result.get("hidden_tests"),
    }
    if score_backend == "ags":
        if not ags_image:
            raise ValueError("AGS scoring requires an image")
        hidden = run_hidden_tests_ags(
            task,
            workspace,
            case_root,
            timeout_s=score_timeout_s,
            ags_image=ags_image,
            ags_env_file=ags_env_file,
            ags_timeout=ags_timeout,
            ags_cpu=ags_cpu,
            ags_memory=ags_memory,
            ags_score_tool_id=ags_score_tool_id,
        )
    else:
        hidden = run_hidden_tests(
            task,
            workspace,
            case_root,
            timeout_s=score_timeout_s,
            keep_image=keep_image,
        )
    pytest_result = hidden["pytest"]
    result.setdefault("reward_history", []).append(previous_reward)
    result["score_backend"] = score_backend
    result["hidden_tests"] = hidden
    result["quality_score"] = (
        pytest_result["quality_score"] if result.get("integrity_ok") else 0.0
    )
    result["success"] = bool(
        result.get("agent_ok")
        and result.get("integrity_ok")
        and result.get("protocol_ok")
        and pytest_result["all_passed"]
    )
    team = _team_metrics(workspace)
    try:
        hidden_log = (case_root / "hidden-tests.log").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        hidden_log = ""
    result["failure_class"] = classify_failure(
        agent_ok=bool(result.get("agent_ok")),
        integrity_ok=bool(result.get("integrity_ok")),
        protocol_ok=bool(result.get("protocol_ok")),
        team=team,
        hidden=hidden,
        hidden_log=hidden_log,
    )
    result["reward_skipped"] = False
    result["rescored_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(result_path, result)
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
    execution_backend: str = "local",
    score_backend: str = "docker",
    ags_image: str | None = None,
    ags_env_file: Path | None = None,
    ags_timeout: str = "3h",
    ags_cpu: str = "2",
    ags_memory: str = "4Gi",
    ags_score_tool_id: str | None = None,
) -> dict[str, Any]:
    """Run and score one case sequentially for API compatibility."""
    rollout = run_rollout(
        task,
        mode,
        output_root,
        provider=provider,
        model=model,
        max_turns=max_turns,
        teammate_max_turns=teammate_max_turns,
        max_output_tokens=max_output_tokens,
        agent_timeout_s=agent_timeout_s,
        stream=stream,
        execution_backend=execution_backend,
        ags_image=ags_image,
        ags_env_file=ags_env_file,
        ags_timeout=ags_timeout,
        ags_cpu=ags_cpu,
        ags_memory=ags_memory,
    )
    return score_rollout(
        rollout,
        provider=provider,
        model=model,
        score_timeout_s=score_timeout_s,
        keep_image=keep_image,
        execution_backend=execution_backend,
        score_backend=score_backend,
        ags_image=ags_image,
        ags_env_file=ags_env_file,
        ags_timeout=ags_timeout,
        ags_cpu=ags_cpu,
        ags_memory=ags_memory,
        ags_score_tool_id=ags_score_tool_id,
    )


def _failed_case_result(
    task: dict[str, Any],
    mode: str,
    phase: str,
    error: Exception,
    *,
    output_root: Path,
    provider: str,
    model: str,
    execution_backend: str,
    score_backend: str,
) -> dict[str, Any]:
    """Persist a scheduler-level failure without aborting the remaining cases."""
    case_root = output_root / task["id"] / mode
    case_root.mkdir(parents=True, exist_ok=True)
    message = f"{type(error).__name__}: {error}"
    (case_root / f"{phase}-error.log").write_text(message + "\n", encoding="utf-8")
    pytest_result = parse_pytest_output("", int(task["expected_tests"]), 1)
    result = {
        "task": task["id"],
        "difficulty": task["difficulty"],
        "mode": mode,
        "provider": provider,
        "model": model,
        "execution_backend": execution_backend,
        "score_backend": score_backend,
        "agent_elapsed_s": 0.0,
        "agent_timed_out": False,
        "agent_returncode": 1,
        "agent_ok": False,
        "agent_error": f"{phase} failed: {message}",
        "integrity_ok": False,
        "protocol_ok": False,
        "used_team": False,
        "quality_score": 0.0,
        "success": False,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "lead_turns": 0,
            "worker_turns": 0,
        },
        "calls": {"model": 0, "tools": 0},
        "team": {
            "present": False,
            "agents": [],
            "peer_messages": 0,
            "worker_usage": {},
        },
        "hidden_tests": {"error": message, "pytest": pytest_result},
        "workspace": str(case_root / "workspace"),
        "failure_phase": phase,
    }
    _write_json(case_root / "result.json", result)
    return result


def run_evaluation_pool(
    cases: list[tuple[dict[str, Any], str]],
    rollout_fn: Callable[[dict[str, Any], str], RolloutArtifact],
    reward_fn: Callable[[RolloutArtifact], dict[str, Any]],
    *,
    rollout_concurrency: int,
    reward_concurrency: int,
    failure_fn: Callable[
        [dict[str, Any], str, str, Exception], dict[str, Any]
    ] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Continuously refill rollout slots while scoring in a separate pool."""
    if rollout_concurrency < 1 or reward_concurrency < 1:
        raise ValueError("rollout and reward concurrency must be positive")
    started = time.monotonic()
    event_lock = threading.Lock()

    def emit(event: str, task: dict[str, Any], mode: str, **extra: Any) -> None:
        if on_event is None:
            return
        value = {
            "event": event,
            "task": task["id"],
            "mode": mode,
            "elapsed_s": round(time.monotonic() - started, 3),
            **extra,
        }
        with event_lock:
            on_event(value)

    def perform_rollout(
        index: int, task: dict[str, Any], mode: str
    ) -> tuple[int, dict[str, Any], str, RolloutArtifact | None, Exception | None]:
        emit("rollout.started", task, mode)
        try:
            artifact = rollout_fn(task, mode)
        except Exception as exc:
            emit("rollout.failed", task, mode, error_type=type(exc).__name__)
            return index, task, mode, None, exc
        emit("rollout.completed", task, mode)
        return index, task, mode, artifact, None

    def perform_reward(
        index: int, task: dict[str, Any], mode: str, artifact: RolloutArtifact
    ) -> tuple[int, dict[str, Any], str, dict[str, Any] | None, Exception | None]:
        emit("reward.started", task, mode)
        try:
            result = reward_fn(artifact)
        except Exception as exc:
            emit("reward.failed", task, mode, error_type=type(exc).__name__)
            return index, task, mode, None, exc
        emit(
            "reward.completed",
            task,
            mode,
            quality_score=result.get("quality_score"),
            success=bool(result.get("success")),
        )
        return index, task, mode, result, None

    indexed_results: list[tuple[int, dict[str, Any]]] = []
    reward_futures: dict[Any, tuple[int, dict[str, Any], str]] = {}
    with (
        ThreadPoolExecutor(
            max_workers=rollout_concurrency, thread_name_prefix="nl2repo-rollout"
        ) as rollout_pool,
        ThreadPoolExecutor(
            max_workers=reward_concurrency, thread_name_prefix="nl2repo-reward"
        ) as reward_pool,
    ):
        rollout_futures = [
            rollout_pool.submit(perform_rollout, index, task, mode)
            for index, (task, mode) in enumerate(cases)
        ]
        for future in as_completed(rollout_futures):
            index, task, mode, artifact, error = future.result()
            if error is not None:
                if failure_fn is None:
                    raise error
                indexed_results.append((index, failure_fn(task, mode, "rollout", error)))
                continue
            assert artifact is not None
            reward_future = reward_pool.submit(
                perform_reward, index, task, mode, artifact
            )
            reward_futures[reward_future] = (index, task, mode)

        for future in as_completed(reward_futures):
            index, task, mode, result, error = future.result()
            if error is not None:
                if failure_fn is None:
                    raise error
                result = failure_fn(task, mode, "reward", error)
            assert result is not None
            indexed_results.append((index, result))

    return [result for _, result in sorted(indexed_results, key=lambda item: item[0])]


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
    parser.add_argument("--teammate-min-timeout", type=float, default=0)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--progress-file", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("solo", "adaptive", "adaptive-team-v2", "forced-team"),
        default="adaptive",
    )
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--execution-backend", choices=("local", "ags"), default="local")
    parser.add_argument("--ags-image")
    parser.add_argument("--ags-env-file", type=Path)
    parser.add_argument("--ags-timeout", default="3h")
    parser.add_argument("--ags-cpu", default="2")
    parser.add_argument("--ags-memory", default="4Gi")
    return parser


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_run-one":
        args = _child_parser().parse_args()
        def terminate_child(signum: int, frame: Any) -> None:
            raise InterruptedError(f"agent child received signal {signum}")

        signal.signal(signal.SIGTERM, terminate_child)
        signal.signal(signal.SIGINT, terminate_child)
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
            teammate_min_timeout_s=args.teammate_min_timeout or None,
            mode=args.mode,
            execution_backend=args.execution_backend,
            ags_image=args.ags_image,
            ags_env_file=args.ags_env_file.resolve() if args.ags_env_file else None,
            ags_timeout=args.ags_timeout,
            ags_cpu=args.ags_cpu,
            ags_memory=args.ags_memory,
        )

    parser = argparse.ArgumentParser(description="Run Clawd against pinned NL2Repo-Bench tasks.")
    parser.add_argument("--list", action="store_true", help="List all upstream tasks")
    parser.add_argument("--validate", action="store_true", help="Validate task metadata and images")
    parser.add_argument("--plan", action="store_true", help="Print the resolved cases without running")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Re-run only reward evaluation for completed --task cases in --output",
    )
    parser.add_argument("--task", action="append", help="Task ID; repeat to override --task-set")
    parser.add_argument(
        "--task-set",
        choices=("pilot", "qwen32"),
        default="pilot",
        help="Built-in task selection; qwen32 is the fixed latency-probe subset",
    )
    parser.add_argument(
        "--mode",
        choices=("solo", "adaptive", "adaptive-team-v2", "forced-team", "both", "all"),
        help="Defaults to adaptive for qwen32 and both for the pilot set",
    )
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--teammate-max-turns", type=int, default=160)
    parser.add_argument(
        "--teammate-min-timeout",
        type=float,
        default=900.0,
        help="Minimum effective timeout for each TeamRun call",
    )
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--agent-timeout", type=float, default=7200.0)
    parser.add_argument("--score-timeout", type=float, default=1200.0)
    parser.add_argument(
        "--rollout-concurrency",
        type=int,
        help="Agent rollout slots; defaults to 8 for qwen32 and 1 otherwise",
    )
    parser.add_argument(
        "--reward-concurrency",
        type=int,
        default=4,
        help="Independent hidden-test workers; these never occupy rollout slots",
    )
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-image", action="store_true")
    parser.add_argument(
        "--execution-backend",
        choices=("local", "ags"),
        default="local",
        help="Where Bash and file tools execute",
    )
    parser.add_argument(
        "--score-backend",
        choices=("docker", "ags"),
        default="docker",
        help="Where the official hidden suite executes",
    )
    parser.add_argument("--ags-env-file", type=Path)
    parser.add_argument("--ags-timeout", default="3h", help="AGS instance TTL")
    parser.add_argument("--ags-cpu", default="2")
    parser.add_argument("--ags-memory", default="4Gi")
    parser.add_argument(
        "--ags-score-tool-id",
        help="Dedicated AGS SandboxTool configured with NetworkMode=SANDBOX",
    )
    parser.add_argument(
        "--ags-image-template",
        default=AGS_IMAGE_TEMPLATE,
        help="Task image template; {task} is replaced with the NL2Repo task ID",
    )
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

    selected_task_set = "custom" if args.task else args.task_set
    if args.task:
        task_names = args.task
    elif args.task_set == "qwen32":
        task_names = select_task_subset(list_tasks(upstream_root))
    else:
        task_names = list(PILOT_TASKS)
    if len(set(task_names)) != len(task_names):
        parser.error("task IDs must be unique")
    tasks = [load_task(upstream_root, name) for name in task_names]
    rollout_concurrency = args.rollout_concurrency
    if rollout_concurrency is None:
        rollout_concurrency = 8 if selected_task_set == "qwen32" else 1
    if rollout_concurrency < 1:
        parser.error("--rollout-concurrency must be positive")
    if args.reward_concurrency < 1:
        parser.error("--reward-concurrency must be positive")

    selected_mode = args.mode
    if selected_mode is None:
        selected_mode = "adaptive" if selected_task_set == "qwen32" else "both"
    if selected_mode == "both":
        modes = ("solo", "adaptive")
    elif selected_mode == "all":
        modes = ("solo", "adaptive", "forced-team")
    else:
        modes = (selected_mode,)
    cases = [(task, mode) for task in tasks for mode in modes]

    if args.rescore:
        if args.output is None:
            parser.error("--rescore requires --output pointing to an existing run")
        output_root = args.output.resolve()
        ags_env_file = args.ags_env_file.resolve() if args.ags_env_file else None
        rescored: list[dict[str, Any]] = []
        for task, mode in cases:
            print(f"[{task['id']}] rescoring {mode}...", flush=True)
            result = rescore_existing_case(
                task,
                mode,
                output_root,
                score_backend=args.score_backend,
                score_timeout_s=args.score_timeout,
                keep_image=args.keep_image,
                ags_image=format_ags_image(args.ags_image_template, task["id"]),
                ags_env_file=ags_env_file,
                ags_timeout=args.ags_timeout,
                ags_cpu=args.ags_cpu,
                ags_memory=args.ags_memory,
                ags_score_tool_id=args.ags_score_tool_id,
            )
            tests = result["hidden_tests"]["pytest"]
            print(
                f"  quality={result['quality_score']:.2f} "
                f"passed={tests['passed']}/{tests['expected']} "
                f"success={result['success']}",
                flush=True,
            )
            rescored.append(result)
        return 0 if all(not result["hidden_tests"].get("error") for result in rescored) else 2

    if args.plan:
        print(
            json.dumps(
                {
                    "task_set": selected_task_set,
                    "tasks": task_names,
                    "modes": list(modes),
                    "cases": len(cases),
                    "max_turns": args.max_turns,
                    "rollout_concurrency": rollout_concurrency,
                    "reward_concurrency": args.reward_concurrency,
                    "reward_uses_rollout_slots": False,
                },
                indent=2,
            )
        )
        return 0

    if args.validate:
        if args.execution_backend == "ags" or args.score_backend == "ags":
            from src.execution.ags import AGSSettings, ensure_swerex_importable

            image = format_ags_image(args.ags_image_template, tasks[0]["id"])
            settings = AGSSettings.from_env(
                image=image,
                env_file=args.ags_env_file,
                timeout=args.ags_timeout,
                cpu=args.ags_cpu,
                memory=args.ags_memory,
            )
            settings.validate()
            ensure_swerex_importable(settings)
            errors = []
            if args.score_backend == "ags":
                score_tool_id = args.ags_score_tool_id or os.environ.get(
                    "AGS_SCORE_TOOL_ID", ""
                ).strip()
                if not score_tool_id:
                    errors.append(
                        "AGS scoring requires AGS_SCORE_TOOL_ID for a dedicated SANDBOX "
                        "network-mode tool; otherwise use --score-backend docker"
                    )
        else:
            errors = validate_tasks(tasks)
        if errors:
            for error in errors:
                print(f"- {error}")
            return 1
        print(
            f"{len(tasks)} NL2Repo tasks are configured for "
            f"execution={args.execution_backend} scoring={args.score_backend}; "
            f"cases={len(cases)} rollout_pool={rollout_concurrency} "
            f"reward_pool={args.reward_concurrency}"
        )
        return 0

    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    output_root = (args.output or ROOT / "runs" / run_id).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "run-metadata.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "upstream_ref": UPSTREAM_REF,
            "task_set": selected_task_set,
            "tasks": task_names,
            "modes": list(modes),
            "cases": len(cases),
            "provider": args.provider,
            "model": args.model,
            "execution_backend": args.execution_backend,
            "score_backend": args.score_backend,
            "max_turns": args.max_turns,
            "teammate_max_turns": args.teammate_max_turns,
            "teammate_min_timeout_s": args.teammate_min_timeout,
            "rollout_concurrency": rollout_concurrency,
            "reward_concurrency": args.reward_concurrency,
            "reward_uses_rollout_slots": False,
        },
    )
    scheduler_path = output_root / "scheduler.jsonl"
    ags_env_file = args.ags_env_file.resolve() if args.ags_env_file else None

    def rollout_task(task: dict[str, Any], mode: str) -> RolloutArtifact:
        return run_rollout(
            task,
            mode,
            output_root,
            provider=args.provider,
            model=args.model,
            max_turns=args.max_turns,
            teammate_max_turns=args.teammate_max_turns,
            teammate_min_timeout_s=args.teammate_min_timeout,
            max_output_tokens=args.max_output_tokens,
            agent_timeout_s=args.agent_timeout,
            stream=args.stream,
            execution_backend=args.execution_backend,
            ags_image=format_ags_image(args.ags_image_template, task["id"]),
            ags_env_file=ags_env_file,
            ags_timeout=args.ags_timeout,
            ags_cpu=args.ags_cpu,
            ags_memory=args.ags_memory,
        )

    def reward_task(rollout: RolloutArtifact) -> dict[str, Any]:
        return score_rollout(
            rollout,
            provider=args.provider,
            model=args.model,
            score_timeout_s=args.score_timeout,
            keep_image=args.keep_image,
            execution_backend=args.execution_backend,
            score_backend=args.score_backend,
            ags_image=format_ags_image(args.ags_image_template, rollout.task["id"]),
            ags_env_file=ags_env_file,
            ags_timeout=args.ags_timeout,
            ags_cpu=args.ags_cpu,
            ags_memory=args.ags_memory,
            ags_score_tool_id=args.ags_score_tool_id,
        )

    def failed_task(
        task: dict[str, Any], mode: str, phase: str, error: Exception
    ) -> dict[str, Any]:
        return _failed_case_result(
            task,
            mode,
            phase,
            error,
            output_root=output_root,
            provider=args.provider,
            model=args.model,
            execution_backend=args.execution_backend,
            score_backend=args.score_backend,
        )

    def scheduler_event(event: dict[str, Any]) -> None:
        _append_jsonl(scheduler_path, event)
        event_name = event["event"]
        if event_name == "rollout.started":
            print(f"[{event['task']}] rollout started ({event['mode']})", flush=True)
        elif event_name == "rollout.completed":
            print(
                f"[{event['task']}] rollout complete; slot released, reward queued",
                flush=True,
            )
        elif event_name == "reward.completed":
            print(
                f"[{event['task']}] reward={event.get('quality_score', 0):.2f} "
                f"success={event.get('success', False)}",
                flush=True,
            )
        elif event_name.endswith(".failed"):
            print(
                f"[{event['task']}] {event_name}: {event.get('error_type')}",
                flush=True,
            )

    print(
        f"Starting {len(cases)} cases with rollout_pool={rollout_concurrency}, "
        f"reward_pool={args.reward_concurrency}, max_turns={args.max_turns}",
        flush=True,
    )
    results = run_evaluation_pool(
        cases,
        rollout_task,
        reward_task,
        rollout_concurrency=rollout_concurrency,
        reward_concurrency=args.reward_concurrency,
        failure_fn=failed_task,
        on_event=scheduler_event,
    )
    aggregate = {
        "run_id": run_id,
        "upstream_url": UPSTREAM_URL,
        "upstream_ref": UPSTREAM_REF,
        "provider": args.provider,
        "model": args.model,
        "execution_backend": args.execution_backend,
        "score_backend": args.score_backend,
        "task_set": selected_task_set,
        "run_config": {
            "tasks": len(tasks),
            "cases": len(cases),
            "max_turns": args.max_turns,
            "teammate_max_turns": args.teammate_max_turns,
            "teammate_min_timeout_s": args.teammate_min_timeout,
            "max_output_tokens": args.max_output_tokens,
            "agent_timeout_s": args.agent_timeout,
            "score_timeout_s": args.score_timeout,
            "rollout_concurrency": rollout_concurrency,
            "reward_concurrency": args.reward_concurrency,
            "reward_uses_rollout_slots": False,
            "stream": args.stream,
            "ags_timeout": args.ags_timeout if "ags" in {args.execution_backend, args.score_backend} else None,
            "ags_cpu": args.ags_cpu if "ags" in {args.execution_backend, args.score_backend} else None,
            "ags_memory": args.ags_memory if "ags" in {args.execution_backend, args.score_backend} else None,
            "ags_image_template": args.ags_image_template if "ags" in {args.execution_backend, args.score_backend} else None,
        },
        "results": results,
    }
    _write_json(output_root / "results.json", aggregate)
    report = render_report(results, run_id, UPSTREAM_REF)
    (output_root / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"\n{report}\nArtifacts: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
