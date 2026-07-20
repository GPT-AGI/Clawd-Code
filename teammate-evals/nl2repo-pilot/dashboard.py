#!/usr/bin/env python3
"""Serve a live dashboard for an NL2Repo benchmark run."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import time
import webbrowser
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import parse_qs, urlparse


HERE = Path(__file__).resolve().parent
DEFAULT_HTML = HERE / "dashboard.html"


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    values: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _tail(path: Path, lines: int = 80, max_chars: int = 16_000) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(content.splitlines()[-lines:])[-max_chars:]


def _compact(value: Any, limit: int = 360) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError:
            text = str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = max(1, int(limit * 0.62))
    tail = max(1, limit - head - 3)
    return f"{text[:head]} … {text[-tail:]}"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _team_directory(case_root: Path) -> Path | None:
    candidates: list[Path] = []
    for teams_root in (
        case_root / "workspace" / ".clawd" / "teams",
        case_root / ".clawd" / "teams",
    ):
        try:
            candidates.extend(path for path in teams_root.iterdir() if path.is_dir())
        except OSError:
            continue
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (path / "events.jsonl").stat().st_mtime
        if (path / "events.jsonl").exists()
        else 0,
    )


def _team_detail(case_root: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return a compact team snapshot and an actor-aware trace for the drawer."""
    team_root = _team_directory(case_root)
    if team_root is None:
        return None, []

    team = _read_json(team_root / "team.json", {})
    tasks = _read_json(team_root / "tasks.json", {})
    if not isinstance(team, dict):
        team = {}
    if not isinstance(tasks, dict):
        tasks = {}

    trace: list[dict[str, Any]] = []
    actor_turns: dict[str, int] = {}
    actor_names: dict[str, str] = {}
    actor_stats: dict[str, Counter[str]] = {}
    for event in _read_jsonl(team_root / "events.jsonl"):
        event_type = str(event.get("type") or "unknown")
        data = event.get("data")
        if not isinstance(data, dict):
            data = {}

        nested_agent = data.get("agent")
        if isinstance(nested_agent, dict):
            agent_id = str(nested_agent.get("agent_id") or "")
            agent_name = str(nested_agent.get("name") or "")
            if agent_id and agent_name:
                actor_names[agent_id] = agent_name
        agent_id = str(data.get("actor_id") or data.get("agent_id") or "")
        actor = str(
            data.get("actor_name")
            or data.get("name")
            or actor_names.get(agent_id)
            or ("team" if event_type.startswith(("team.", "task.", "agent.")) else "lead")
        )
        turn = data.get("turn")
        if isinstance(turn, int):
            actor_turns[actor] = turn
        else:
            turn = actor_turns.get(actor)

        tool = str(data.get("tool_name") or "")
        detail = ""
        if event_type == "tool.started":
            detail = _compact(data.get("tool_input"))
        elif event_type in {"tool.completed", "tool.failed"}:
            output = data.get("tool_output")
            if isinstance(output, dict):
                parts: list[str] = []
                if output.get("exit_code") is not None:
                    parts.append(f"exit {output['exit_code']}")
                parts.append(_compact(output.get("stderr") or output.get("stdout") or output))
                detail = " · ".join(part for part in parts if part)
            else:
                detail = _compact(output or data.get("error"))
        elif event_type == "model.response":
            detail = _compact(data.get("content") or data.get("error"), limit=720)
        elif event_type.startswith("task."):
            task = data.get("task")
            if isinstance(task, dict):
                detail = _compact(task.get("subject") or task.get("output") or task.get("last_error"))
            else:
                detail = _compact(data.get("subject") or data.get("output") or data.get("error"))
        elif event_type.startswith("agent."):
            detail = _compact(
                (nested_agent or {}).get("role")
                if isinstance(nested_agent, dict)
                else data.get("status")
            )

        stats = actor_stats.setdefault(actor, Counter())
        if event_type == "model.response":
            stats["turns"] += 1
        if event_type == "tool.started":
            stats["tools"] += 1
        if event_type.endswith("failed"):
            stats["errors"] += 1
        trace.append(
            {
                "kind": event_type.replace(".", "_"),
                "actor": actor,
                "turn": turn,
                "tool": tool,
                "tool_use_id": data.get("tool_use_id"),
                "created_at": event.get("created_at"),
                "is_error": event_type.endswith("failed"),
                "duration_ms": data.get("duration_ms"),
                "detail": detail,
            }
        )

    task_rows = []
    for task in tasks.values():
        if not isinstance(task, dict):
            continue
        owner_id = str(task.get("owner") or "")
        task_rows.append(
            {
                "subject": str(task.get("subject") or task.get("description") or "task"),
                "status": str(task.get("status") or "unknown"),
                "owner": actor_names.get(owner_id, owner_id or "unassigned"),
                "output": _compact(task.get("output") or task.get("last_error"), limit=520),
            }
        )
    actors = [
        {
            "name": name,
            "turns": stats["turns"],
            "tools": stats["tools"],
            "errors": stats["errors"],
        }
        for name, stats in actor_stats.items()
        if name != "team"
    ]
    snapshot = {
        "id": str(team.get("team_id") or team_root.name),
        "name": str(team.get("team_name") or "team"),
        "status": str(team.get("status") or "unknown"),
        "actors": actors,
        "tasks": task_rows,
    }
    return snapshot, trace[-1_400:]


@dataclass
class ProgressAccumulator:
    offset: int = 0
    inode: int | None = None
    counts: Counter[str] = field(default_factory=Counter)
    error_count: int = 0
    model_duration_ms: float = 0.0
    model_duration_count: int = 0
    last_kind: str = ""
    last_turn: int = 0
    last_tool: str = ""
    last_at: float | None = None
    terminal: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    current_turn: int | None = None
    recent: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=1_400))
    text_fragments: deque[str] = field(default_factory=lambda: deque(maxlen=100))

    def reset(self) -> None:
        fresh = ProgressAccumulator()
        self.__dict__.update(fresh.__dict__)

    def consume(self, path: Path) -> None:
        try:
            stat = path.stat()
        except OSError:
            return
        if self.inode not in (None, stat.st_ino) or stat.st_size < self.offset:
            self.reset()
        self.inode = stat.st_ino
        with path.open("rb") as handle:
            handle.seek(self.offset)
            while True:
                line_start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    handle.seek(line_start)
                    break
                self.offset = handle.tell()
                try:
                    event = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(event, dict):
                    self._consume_event(event)

    def _consume_event(self, event: dict[str, Any]) -> None:
        kind = str(event.get("kind") or "unknown")
        self.counts[kind] += 1
        self.last_kind = kind
        turn = event.get("turn")
        if isinstance(turn, int):
            self.last_turn = max(self.last_turn, turn)
            self.current_turn = turn
        created_at = _timestamp(event.get("created_at"))
        if created_at is not None:
            self.last_at = created_at
        usage = event.get("usage")
        if isinstance(usage, dict) and usage:
            self.usage = usage
        if kind == "model_response":
            duration = event.get("duration_ms")
            if isinstance(duration, (int, float)):
                self.model_duration_ms += float(duration)
                self.model_duration_count += 1
        if kind == "tool_use":
            self.last_tool = str(event.get("tool_name") or event.get("name") or "")
        if event.get("is_error") or kind in {"tool_error", "run_failed", "run_cancelled"}:
            self.error_count += 1
        if kind in {"run_completed", "run_failed", "run_cancelled"}:
            self.terminal = kind
        if kind == "text_chunk":
            content = event.get("content")
            if isinstance(content, str) and content:
                self.text_fragments.append(content)
            return
        self.recent.append(self._event_summary(event))

    def _event_summary(self, event: dict[str, Any]) -> dict[str, Any]:
        kind = str(event.get("kind") or "unknown")
        tool = str(event.get("tool_name") or event.get("name") or "")
        detail = ""
        if kind == "tool_use":
            detail = _compact(event.get("tool_input") or event.get("input"))
        elif kind in {"tool_result", "tool_error"}:
            output = event.get("tool_output")
            if isinstance(output, dict):
                parts = []
                if output.get("exit_code") is not None:
                    parts.append(f"exit {output['exit_code']}")
                parts.append(_compact(output.get("stderr") or output.get("stdout") or output))
                detail = " · ".join(part for part in parts if part)
            else:
                detail = _compact(output or event.get("error"))
        else:
            detail = _compact(event.get("error") or event.get("content"))
        return {
            "kind": kind,
            "tool": tool,
            "actor": "lead",
            "turn": event.get("turn") if isinstance(event.get("turn"), int) else self.current_turn,
            "tool_use_id": event.get("tool_use_id"),
            "created_at": event.get("created_at"),
            "is_error": bool(event.get("is_error") or kind in {"tool_error", "run_failed"}),
            "duration_ms": event.get("duration_ms"),
            "detail": detail,
        }

    def summary(self, path: Path, now: float) -> dict[str, Any]:
        self.consume(path)
        if self.last_at is None:
            try:
                self.last_at = path.stat().st_mtime
            except OSError:
                pass
        avg_model_s = (
            self.model_duration_ms / self.model_duration_count / 1000
            if self.model_duration_count
            else None
        )
        return {
            "model_calls": self.counts["model_response"],
            "tool_calls": self.counts["tool_use"],
            "errors": self.error_count,
            "last_kind": self.last_kind,
            "last_turn": self.last_turn,
            "last_tool": self.last_tool,
            "last_at": self.last_at,
            "activity_age_s": max(0.0, now - self.last_at) if self.last_at else None,
            "terminal": self.terminal,
            "usage": self.usage,
            "avg_model_s": avg_model_s,
        }


class DashboardStore:
    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root.expanduser().resolve()
        if not self.run_root.is_dir():
            raise FileNotFoundError(f"run directory not found: {self.run_root}")
        self._progress: dict[tuple[str, str], ProgressAccumulator] = {}
        self._comparison_cache: dict[str, Any] | None = None
        self._comparison_cache_key = ""
        self._comparison_cached_at = 0.0

    def _metadata(self) -> dict[str, Any]:
        metadata = _read_json(self.run_root / "run-metadata.json", {})
        if not isinstance(metadata, dict):
            metadata = {}
        aggregate = _read_json(self.run_root / "results.json", {})
        if isinstance(aggregate, dict):
            metadata.setdefault("provider", aggregate.get("provider"))
            metadata.setdefault("model", aggregate.get("model"))
            metadata.setdefault("execution_backend", aggregate.get("execution_backend"))
            metadata.setdefault("score_backend", aggregate.get("score_backend"))
            config = aggregate.get("run_config")
            if isinstance(config, dict):
                for key, value in config.items():
                    metadata.setdefault(key, value)
            if not metadata.get("tasks") and isinstance(aggregate.get("results"), list):
                metadata["tasks"] = sorted(
                    {str(item.get("task")) for item in aggregate["results"] if item.get("task")}
                )
        metadata.setdefault("run_id", self.run_root.name)
        return metadata

    def _scheduler(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.run_root / "scheduler.jsonl")

    def _queue_cases(self) -> list[dict[str, Any]]:
        path = self.run_root / "queue.sqlite3"
        if not path.is_file():
            return []
        try:
            connection = sqlite3.connect(
                f"file:{path}?mode=ro", uri=True, timeout=2
            )
            connection.row_factory = sqlite3.Row
            try:
                rows = connection.execute(
                    "SELECT * FROM cases ORDER BY priority DESC, id ASC"
                ).fetchall()
            finally:
                connection.close()
        except (sqlite3.Error, OSError):
            return []
        return [dict(row) for row in rows]

    def _queue_concurrency(self) -> dict[str, Any] | None:
        path = self.run_root / "queue.sqlite3"
        if not path.is_file():
            return None
        try:
            connection = sqlite3.connect(
                f"file:{path}?mode=ro", uri=True, timeout=2
            )
            connection.row_factory = sqlite3.Row
            try:
                row = connection.execute(
                    "SELECT * FROM worker_config WHERE id=1"
                ).fetchone()
            finally:
                connection.close()
        except (sqlite3.Error, OSError):
            return None
        return dict(row) if row is not None else None

    def set_concurrency(
        self,
        *,
        rollout: int | None = None,
        reward: int | None = None,
    ) -> dict[str, Any]:
        """Update desired worker slots for a continuous queue."""
        if rollout is None and reward is None:
            raise ValueError("provide rollout or reward concurrency")
        if rollout is not None and rollout < 0:
            raise ValueError("rollout concurrency must be non-negative")
        if reward is not None and reward < 0:
            raise ValueError("reward concurrency must be non-negative")
        path = self.run_root / "queue.sqlite3"
        if not path.is_file():
            raise RuntimeError("selected run does not have a live queue")
        connection = sqlite3.connect(path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM worker_config WHERE id=1"
            ).fetchone()
            if row is None:
                raise RuntimeError("this worker must be restarted once to enable live scaling")
            desired_rollout = int(row["rollout_concurrency"]) if rollout is None else rollout
            desired_reward = int(row["reward_concurrency"]) if reward is None else reward
            max_rollout = int(row["max_rollout_concurrency"])
            max_reward = int(row["max_reward_concurrency"])
            if desired_rollout > max_rollout:
                raise ValueError(f"rollout concurrency cannot exceed {max_rollout}")
            if desired_reward > max_reward:
                raise ValueError(f"reward concurrency cannot exceed {max_reward}")
            connection.execute(
                """
                UPDATE worker_config
                SET rollout_concurrency=?, reward_concurrency=?, updated_at=?
                WHERE id=1
                """,
                (
                    desired_rollout,
                    desired_reward,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        metadata = self._metadata()
        metadata.update(
            {
                "rollout_concurrency": desired_rollout,
                "reward_concurrency": desired_reward,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json_atomic(self.run_root / "run-metadata.json", metadata)
        configured = self._queue_concurrency()
        assert configured is not None
        return configured

    def _case_specs(
        self,
        metadata: dict[str, Any],
        scheduler: list[dict[str, Any]],
        queue_cases: list[dict[str, Any]],
    ) -> list[tuple[str, str]]:
        configured = metadata.get("tasks")
        if isinstance(configured, list):
            values = []
            for item in configured:
                name = item.get("id") if isinstance(item, dict) else item
                if name:
                    values.append(str(name))
            if values:
                modes = metadata.get("modes")
                selected_modes = (
                    [str(mode) for mode in modes]
                    if isinstance(modes, list) and modes
                    else ["adaptive"]
                )
                return [(task, mode) for task in values for mode in selected_modes]
        queued = [
            (str(case["task"]), str(case.get("mode") or "adaptive"))
            for case in queue_cases
            if case.get("task")
        ]
        if queued:
            return list(dict.fromkeys(queued))
        discovered = {
            (path.parent.parent.name, path.parent.name)
            for path in self.run_root.glob("*/*/progress.jsonl")
        }
        discovered.update(
            (str(event["task"]), str(event.get("mode") or "adaptive"))
            for event in scheduler
            if event.get("task")
        )
        return sorted(discovered)

    @staticmethod
    def _mode_for_task(root: Path, task: str, metadata: dict[str, Any]) -> str:
        modes = metadata.get("modes")
        if isinstance(modes, list) and modes:
            return str(modes[0])
        task_root = root / task
        if task_root.is_dir():
            candidates = sorted(path.name for path in task_root.iterdir() if path.is_dir())
            if candidates:
                return candidates[0]
        return "adaptive"

    @staticmethod
    def _start_epoch(
        metadata: dict[str, Any], scheduler: list[dict[str, Any]], scheduler_path: Path
    ) -> float | None:
        explicit = _timestamp(metadata.get("started_at"))
        if explicit is not None:
            return explicit
        if scheduler:
            elapsed = scheduler[-1].get("elapsed_s")
            if isinstance(elapsed, (int, float)):
                try:
                    return scheduler_path.stat().st_mtime - float(elapsed)
                except OSError:
                    pass
        return None

    @staticmethod
    def _comparison_case(
        result: dict[str, Any], source_run: str
    ) -> dict[str, Any]:
        calls = result.get("calls") if isinstance(result.get("calls"), dict) else {}
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        hidden = (
            result.get("hidden_tests")
            if isinstance(result.get("hidden_tests"), dict)
            else {}
        )
        tokens = _number(usage.get("total_tokens"))
        # Streaming responses without include_usage were historically persisted as
        # zero. Treat those as missing measurements instead of free rollouts.
        if tokens is not None and tokens <= 0:
            tokens = None
        return {
            "source_run": source_run,
            "model": result.get("model"),
            "quality_score": (
                None if hidden.get("error") else _number(result.get("quality_score"))
            ),
            "runtime_s": _number(result.get("agent_elapsed_s")),
            "model_calls": _number(calls.get("model")),
            "tool_calls": _number(calls.get("tools")),
            "total_tokens": tokens,
            "success": bool(result.get("success")),
            "infrastructure_error": str(hidden.get("error") or "") or None,
        }

    @staticmethod
    def _comparison_mode_summary(
        rows: list[dict[str, Any]], mode: str
    ) -> dict[str, Any]:
        cases = [row[mode] for row in rows]

        def values(key: str) -> list[float]:
            return [
                value
                for case in cases
                if (value := _number(case.get(key))) is not None
            ]

        quality = values("quality_score")
        runtime = values("runtime_s")
        model_calls = values("model_calls")
        tool_calls = values("tool_calls")
        tokens = values("total_tokens")
        return {
            "count": len(cases),
            "quality_count": len(quality),
            "average_quality": _average(quality),
            "median_quality": median(quality) if quality else None,
            "average_runtime_s": _average(runtime),
            "median_runtime_s": median(runtime) if runtime else None,
            "cumulative_runtime_s": sum(runtime),
            "average_model_calls": _average(model_calls),
            "average_tool_calls": _average(tool_calls),
            "total_tokens": sum(tokens),
            "token_coverage": len(tokens),
            "strict_successes": sum(bool(case.get("success")) for case in cases),
            "models": sorted(
                {str(case["model"]) for case in cases if case.get("model")}
            ),
            "source_runs": dict(
                Counter(str(case["source_run"]) for case in cases)
            ),
        }

    def _comparison(self, metadata: dict[str, Any]) -> dict[str, Any] | None:
        config = metadata.get("comparison")
        if not isinstance(config, dict):
            return None
        cache_key = json.dumps(config, sort_keys=True, ensure_ascii=False)
        if (
            self._comparison_cache is not None
            and self._comparison_cache_key == cache_key
            and time.monotonic() - self._comparison_cached_at < 10
        ):
            return self._comparison_cache
        configured_modes = config.get("modes")
        modes = (
            [str(mode) for mode in configured_modes]
            if isinstance(configured_modes, list)
            else ["adaptive", "forced-team"]
        )
        if len(modes) != 2 or modes[0] == modes[1]:
            return None
        left_mode, right_mode = modes

        indexed: dict[tuple[str, str], dict[str, Any]] = {}

        def load_run(root: Path, source_run: str, only_mode: str | None = None) -> None:
            pattern = f"*/{only_mode}/result.json" if only_mode else "*/*/result.json"
            for path in root.glob(pattern):
                task = path.parent.parent.name
                mode = path.parent.name
                key = (task, mode)
                if key in indexed:
                    continue
                result = _read_json(path, {})
                if isinstance(result, dict) and result:
                    indexed[key] = self._comparison_case(result, source_run)

        load_run(self.run_root, self.run_root.name)
        baselines = config.get("baseline_runs")
        baseline_runs: dict[str, str] = {}
        if isinstance(baselines, dict):
            for mode, raw_run_id in baselines.items():
                run_id = str(raw_run_id)
                if Path(run_id).name != run_id:
                    continue
                root = (self.run_root.parent / run_id).resolve()
                if root.parent != self.run_root.parent or not root.is_dir():
                    continue
                baseline_runs[str(mode)] = run_id
                load_run(root, run_id, str(mode))

        paired_tasks = sorted(
            {task for task, mode in indexed if mode == left_mode}
            & {task for task, mode in indexed if mode == right_mode}
        )
        rows: list[dict[str, Any]] = []
        for task in paired_tasks:
            left = indexed[(task, left_mode)]
            right = indexed[(task, right_mode)]
            left_quality = _number(left.get("quality_score"))
            right_quality = _number(right.get("quality_score"))
            left_runtime = _number(left.get("runtime_s"))
            right_runtime = _number(right.get("runtime_s"))
            rows.append(
                {
                    "task": task,
                    left_mode: left,
                    right_mode: right,
                    "quality_delta": (
                        right_quality - left_quality
                        if left_quality is not None and right_quality is not None
                        else None
                    ),
                    "runtime_delta_s": (
                        right_runtime - left_runtime
                        if left_runtime is not None and right_runtime is not None
                        else None
                    ),
                    "runtime_ratio": (
                        right_runtime / left_runtime
                        if left_runtime and right_runtime is not None
                        else None
                    ),
                    "cross_run": left["source_run"] != right["source_run"],
                    "deployment_mismatch": left.get("model") != right.get("model"),
                }
            )

        if not rows:
            return None
        quality_deltas = [
            value
            for row in rows
            if (value := _number(row.get("quality_delta"))) is not None
        ]
        runtime_deltas = [
            value
            for row in rows
            if (value := _number(row.get("runtime_delta_s"))) is not None
        ]
        comparison = {
            "modes": modes,
            "paired_count": len(rows),
            "cross_run_count": sum(bool(row["cross_run"]) for row in rows),
            "deployment_mismatch_count": sum(
                bool(row["deployment_mismatch"]) for row in rows
            ),
            "baseline_runs": baseline_runs,
            "mode_summaries": {
                left_mode: self._comparison_mode_summary(rows, left_mode),
                right_mode: self._comparison_mode_summary(rows, right_mode),
            },
            "paired": {
                "average_quality_delta": _average(quality_deltas),
                "average_runtime_delta_s": _average(runtime_deltas),
                "right_quality_wins": sum(value > 0 for value in quality_deltas),
                "ties": sum(value == 0 for value in quality_deltas),
                "left_quality_wins": sum(value < 0 for value in quality_deltas),
                "right_faster": sum(value < 0 for value in runtime_deltas),
                "left_faster": sum(value > 0 for value in runtime_deltas),
            },
            "rows": rows,
            "notes": [
                "Cross-run rows use agent_elapsed_s and can reflect different concurrency or service load.",
                "Token totals exclude zero-valued historical streaming usage; coverage is shown explicitly.",
            ],
        }
        self._comparison_cache = comparison
        self._comparison_cache_key = cache_key
        self._comparison_cached_at = time.monotonic()
        return comparison

    def state(self) -> dict[str, Any]:
        now = time.time()
        metadata = self._metadata()
        concurrency = self._queue_concurrency()
        if concurrency is not None:
            metadata.update(
                {
                    "rollout_concurrency": concurrency["rollout_concurrency"],
                    "reward_concurrency": concurrency["reward_concurrency"],
                    "max_rollout_concurrency": concurrency[
                        "max_rollout_concurrency"
                    ],
                    "max_reward_concurrency": concurrency[
                        "max_reward_concurrency"
                    ],
                }
            )
        scheduler = self._scheduler()
        queue_cases = self._queue_cases()
        scheduler_path = self.run_root / "scheduler.jsonl"
        case_specs = self._case_specs(metadata, scheduler, queue_cases)
        total = len(case_specs)
        start_epoch = self._start_epoch(metadata, scheduler, scheduler_path)
        now_elapsed = max(0.0, now - start_epoch) if start_epoch else 0.0

        events_by_case: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for event in scheduler:
            task = event.get("task")
            if task:
                key = (str(task), str(event.get("mode") or "adaptive"))
                events_by_case.setdefault(key, []).append(event)
        queue_by_task = {
            (str(case.get("task")), str(case.get("mode"))): case
            for case in queue_cases
        }

        tasks: list[dict[str, Any]] = []
        for index, (task, mode) in enumerate(case_specs):
            queued_case = queue_by_task.get((task, mode))
            queue_status = str(queued_case.get("status")) if queued_case else ""
            case_root = self.run_root / task / mode
            progress_path = case_root / "progress.jsonl"
            accumulator = self._progress.setdefault((task, mode), ProgressAccumulator())
            progress = accumulator.summary(progress_path, now)
            task_events = events_by_case.get((task, mode), [])
            event_map = {event.get("event"): event for event in task_events}
            rollout_started = event_map.get("rollout.started")
            rollout_completed = event_map.get("rollout.completed")
            reward_started = event_map.get("reward.started")
            reward_completed = event_map.get("reward.completed")
            result = _read_json(case_root / "result.json", {})
            if not isinstance(result, dict):
                result = {}
            hidden = result.get("hidden_tests") if isinstance(result.get("hidden_tests"), dict) else {}
            pytest = hidden.get("pytest") if isinstance(hidden.get("pytest"), dict) else {}
            queue_error = str(queued_case.get("error") or "") if queued_case else ""
            infrastructure_error = str(hidden.get("error") or queue_error or "")

            if queue_status == "queued":
                status = "queued"
            elif queue_status == "rollout":
                status = "running"
            elif queue_status == "reward_pending":
                status = "reward_waiting"
            elif queue_status == "rewarding":
                status = "rewarding"
            elif queue_status == "failed":
                status = "infra_error"
            elif queue_status == "done" and result:
                status = "infra_error" if infrastructure_error else (
                    "success" if result.get("success") else "scored"
                )
            elif queue_status == "done":
                status = "infra_error"
                infrastructure_error = "queue completed without result.json"
            elif result:
                status = "infra_error" if infrastructure_error else (
                    "success" if result.get("success") else "scored"
                )
            elif reward_started and not reward_completed:
                status = "rewarding"
            elif rollout_completed:
                status = "reward_waiting"
            elif rollout_started:
                status = "running"
            else:
                status = "queued"

            queued_started = bool(queue_status and queue_status != "queued")
            queued_rollout_done = queue_status in {
                "reward_pending", "rewarding", "done", "failed"
            }
            queued_reward_started = queue_status in {"rewarding", "done", "failed"}
            queued_reward_done = queue_status in {"done", "failed"}
            if queued_case:
                rollout_started_flag = queued_started
                rollout_completed_flag = queued_rollout_done
                reward_started_flag = queued_reward_started
                reward_completed_flag = queued_reward_done
                queued_started_at = _timestamp(queued_case.get("started_at"))
                queued_completed_at = _timestamp(queued_case.get("rollout_finished_at"))
                started_elapsed = (
                    max(0.0, queued_started_at - start_epoch)
                    if queued_started_at is not None and start_epoch is not None
                    else None
                )
                completed_elapsed = (
                    max(0.0, queued_completed_at - start_epoch)
                    if queued_completed_at is not None and start_epoch is not None
                    else None
                )
            else:
                rollout_started_flag = bool(rollout_started)
                rollout_completed_flag = bool(rollout_completed)
                reward_started_flag = bool(reward_started)
                reward_completed_flag = bool(result or reward_completed)
                started_elapsed = (
                    float(rollout_started.get("elapsed_s", 0))
                    if rollout_started
                    else None
                )
                completed_elapsed = (
                    float(rollout_completed.get("elapsed_s", 0))
                    if rollout_completed
                    else None
                )
            runtime_s = None
            if started_elapsed is not None:
                runtime_s = max(
                    0.0,
                    (completed_elapsed if completed_elapsed is not None else now_elapsed)
                    - started_elapsed,
                )
            quality = result.get("quality_score") if result else None
            turns = (
                (result.get("usage") or {}).get("lead_turns")
                if isinstance(result.get("usage"), dict)
                else None
            )
            if not isinstance(turns, int):
                turns = progress["model_calls"]
            tasks.append(
                {
                    "index": index + 1,
                    "task": task,
                    "mode": mode,
                    "case_key": f"{task}::{mode}",
                    "status": status,
                    "rollout_started": rollout_started_flag,
                    "rollout_completed": rollout_completed_flag,
                    "reward_started": reward_started_flag,
                    "reward_completed": reward_completed_flag,
                    "runtime_s": runtime_s,
                    "turns": turns,
                    "max_turns": metadata.get("max_turns", 300),
                    "model_calls": progress["model_calls"],
                    "tool_calls": progress["tool_calls"],
                    "tool_errors": progress["errors"],
                    "last_tool": progress["last_tool"],
                    "last_kind": progress["last_kind"],
                    "activity_age_s": progress["activity_age_s"],
                    "avg_model_s": progress["avg_model_s"],
                    "quality_score": quality,
                    "success": bool(result.get("success")) if result else None,
                    "passed": pytest.get("passed"),
                    "failed": pytest.get("failed"),
                    "errors": pytest.get("errors"),
                    "expected": pytest.get("expected"),
                    "infrastructure_error": infrastructure_error or None,
                    "failure_class": result.get("failure_class") if result else None,
                    "agent_ok": result.get("agent_ok") if result else None,
                    "rescored": bool(result.get("rescored_at")),
                    "queue_id": queued_case.get("id") if queued_case else None,
                    "queue_status": queue_status or None,
                    "priority": queued_case.get("priority") if queued_case else None,
                    "attempt": queued_case.get("attempt") if queued_case else None,
                    "enqueued_at": queued_case.get("enqueued_at") if queued_case else None,
                }
            )

        started_count = sum(task["rollout_started"] for task in tasks)
        completed_count = sum(task["rollout_completed"] for task in tasks)
        rewards_done = sum(task["reward_completed"] for task in tasks)
        active_count = sum(task["status"] == "running" for task in tasks)
        reward_active = sum(task["status"] == "rewarding" for task in tasks)
        infrastructure_errors = sum(task["status"] == "infra_error" for task in tasks)
        valid_scores = [
            float(task["quality_score"])
            for task in tasks
            if task["quality_score"] is not None and not task["infrastructure_error"]
        ]
        strict_successes = sum(task["status"] == "success" for task in tasks)
        passed_total = sum(int(task["passed"] or 0) for task in tasks if not task["infrastructure_error"])
        expected_total = sum(
            int(task["expected"] or 0) for task in tasks if not task["infrastructure_error"]
        )
        completed_elapsed_values = [
            float(event["elapsed_s"])
            for event in scheduler
            if event.get("event") == "rollout.completed"
            and isinstance(event.get("elapsed_s"), (int, float))
        ]
        rollout_rate_h = completed_count / now_elapsed * 3600 if now_elapsed else 0.0
        eta_s = (
            (total - completed_count) / rollout_rate_h * 3600
            if rollout_rate_h > 0 and total > completed_count
            else 0.0
        )
        latest_activity_age = min(
            (
                float(task["activity_age_s"])
                for task in tasks
                if task["activity_age_s"] is not None and task["status"] == "running"
            ),
            default=None,
        )
        aggregate_exists = (self.run_root / "results.json").is_file()
        continuous = metadata.get("queue_mode") == "continuous"
        if aggregate_exists and not continuous:
            run_status = "completed"
        elif active_count and latest_activity_age is not None and latest_activity_age < 180:
            run_status = "live"
        elif active_count:
            run_status = "stale"
        elif continuous:
            run_status = "waiting"
        else:
            run_status = "idle"

        timeline = [
            {
                "event": event.get("event"),
                "task": event.get("task"),
                "mode": event.get("mode"),
                "elapsed_s": event.get("elapsed_s"),
                "quality_score": event.get("quality_score"),
                "success": event.get("success"),
            }
            for event in scheduler[-160:]
        ]
        comparison = self._comparison(metadata)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run": {
                "id": metadata.get("run_id", self.run_root.name),
                "path": str(self.run_root),
                "status": run_status,
                "started_at": (
                    datetime.fromtimestamp(start_epoch, timezone.utc).isoformat()
                    if start_epoch
                    else None
                ),
                "elapsed_s": now_elapsed,
                "provider": metadata.get("provider"),
                "model": metadata.get("model"),
                "execution_backend": metadata.get("execution_backend"),
                "score_backend": metadata.get("score_backend"),
                "rollout_concurrency": metadata.get("rollout_concurrency", 0),
                "reward_concurrency": metadata.get("reward_concurrency", 0),
                "max_rollout_concurrency": metadata.get(
                    "max_rollout_concurrency", metadata.get("rollout_concurrency", 0)
                ),
                "max_reward_concurrency": metadata.get(
                    "max_reward_concurrency", metadata.get("reward_concurrency", 0)
                ),
                "max_turns": metadata.get("max_turns", 300),
                "queue_mode": metadata.get("queue_mode"),
            },
            "summary": {
                "total": total,
                "queued": max(0, total - started_count),
                "started": started_count,
                "active": active_count,
                "rollouts_completed": completed_count,
                "reward_active": reward_active,
                "rewards_completed": rewards_done,
                "strict_successes": strict_successes,
                "infrastructure_errors": infrastructure_errors,
                "average_quality": sum(valid_scores) / len(valid_scores) if valid_scores else None,
                "valid_rewards": len(valid_scores),
                "passed_total": passed_total,
                "expected_total": expected_total,
                "model_calls": sum(int(task["model_calls"]) for task in tasks),
                "tool_calls": sum(int(task["tool_calls"]) for task in tasks),
                "tool_errors": sum(int(task["tool_errors"]) for task in tasks),
                "rollout_rate_h": rollout_rate_h,
                "eta_s": eta_s,
                "rollout_progress": completed_count / total if total else 0,
                "reward_progress": rewards_done / total if total else 0,
                "overall_progress": (
                    (completed_count + rewards_done) / (2 * total) if total else 0
                ),
                "last_completion_s": max(completed_elapsed_values, default=None),
                "queue_depth": sum(task["status"] == "queued" for task in tasks),
                "queue_low": (
                    sum(task["status"] == "queued" for task in tasks)
                    < int(metadata.get("rollout_concurrency") or 1)
                    if continuous
                    else False
                ),
            },
            "tasks": tasks,
            "timeline": timeline,
            "comparison": comparison,
        }

    def task_detail(
        self, task_name: str, mode_name: str | None = None
    ) -> dict[str, Any]:
        state = self.state()
        task = next(
            (
                item
                for item in state["tasks"]
                if item["task"] == task_name
                and (mode_name is None or item["mode"] == mode_name)
            ),
            None,
        )
        if task is None:
            raise KeyError(task_name)
        mode = str(task["mode"])
        case_root = self.run_root / task_name / mode
        accumulator = self._progress.setdefault((task_name, mode), ProgressAccumulator())
        accumulator.consume(case_root / "progress.jsonl")
        team, team_trace = _team_detail(case_root)
        agent = _read_json(case_root / "agent-result.json", {})
        if not isinstance(agent, dict):
            agent = {}
        return {
            "task": task,
            "recent_events": list(accumulator.recent),
            "trace_events": team_trace or list(accumulator.recent),
            "team": team,
            "recent_text": "".join(accumulator.text_fragments)[-2_400:],
            "agent_response": str(agent.get("response_text") or agent.get("error") or "")[-4_000:],
            "hidden_log": _tail(case_root / "hidden-tests.log", lines=100),
            "docker_build_log": _tail(case_root / "docker-build.log", lines=50),
            "stdout_log": _tail(case_root / "stdout.log", lines=60),
            "stderr_log": _tail(case_root / "stderr.log", lines=60),
        }


class DashboardRegistry:
    """Discover sibling runs and keep one incremental store per selected run."""

    def __init__(self, default_run: Path) -> None:
        resolved = default_run.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"run directory not found: {resolved}")
        self.runs_root = resolved.parent
        self.default_run_id = resolved.name
        self._stores: dict[str, DashboardStore] = {
            self.default_run_id: DashboardStore(resolved)
        }

    @staticmethod
    def _is_run(path: Path) -> bool:
        return path.is_dir() and any(
            (path / marker).exists()
            for marker in (
                "run-metadata.json",
                "scheduler.jsonl",
                "queue.sqlite3",
                "results.json",
            )
        )

    def run_ids(self) -> list[str]:
        values = [path.name for path in self.runs_root.iterdir() if self._is_run(path)]
        return sorted(values, reverse=True)

    def get(self, run_id: str | None = None) -> DashboardStore:
        selected = run_id or self.default_run_id
        if not selected or Path(selected).name != selected:
            raise KeyError(selected)
        path = (self.runs_root / selected).resolve()
        if path.parent != self.runs_root or not self._is_run(path):
            raise KeyError(selected)
        store = self._stores.get(selected)
        if store is None:
            store = DashboardStore(path)
            self._stores[selected] = store
        return store

    def listing(self) -> dict[str, Any]:
        runs = []
        for run_id in self.run_ids():
            path = self.runs_root / run_id
            metadata = _read_json(path / "run-metadata.json", {})
            if not isinstance(metadata, dict):
                metadata = {}
            runs.append(
                {
                    "id": run_id,
                    "provider": metadata.get("provider"),
                    "model": metadata.get("model"),
                    "queue_mode": metadata.get("queue_mode"),
                    "invalidated": bool(metadata.get("invalidated_at")),
                }
            )
        return {"default": self.default_run_id, "runs": runs}

    @staticmethod
    def _campaign_id(run_id: str) -> str:
        """Collapse Adaptive/Forced repetitions into one dashboard campaign."""
        return re.sub(
            r"-(?:adaptive-team-v\d+|forced-team(?:-fixed)?)-pool\d+-r\d+$",
            "",
            run_id,
        )

    def _quick_queue_state(self, run_id: str) -> dict[str, Any] | None:
        path = self.runs_root / run_id
        database = path / "queue.sqlite3"
        if not database.is_file():
            return None
        metadata = _read_json(path / "run-metadata.json", {})
        if not isinstance(metadata, dict) or metadata.get("queue_mode") != "continuous":
            return None
        try:
            connection = sqlite3.connect(
                f"file:{database}?mode=ro", uri=True, timeout=2
            )
            connection.row_factory = sqlite3.Row
            try:
                count_rows = connection.execute(
                    "SELECT status, COUNT(*) AS count FROM cases GROUP BY status"
                ).fetchall()
                config = connection.execute(
                    "SELECT * FROM worker_config WHERE id=1"
                ).fetchone()
                score_row = connection.execute(
                    "SELECT AVG(quality_score) AS average_quality, "
                    "COUNT(quality_score) AS valid_rewards FROM cases "
                    "WHERE status='done'"
                ).fetchone()
            finally:
                connection.close()
        except (sqlite3.Error, OSError):
            return None
        counts = {
            status: 0
            for status in (
                "queued",
                "rollout",
                "reward_pending",
                "rewarding",
                "done",
                "failed",
            )
        }
        for row in count_rows:
            counts[str(row["status"])] = int(row["count"])
        total = sum(counts.values())
        rollout_done = (
            counts["reward_pending"]
            + counts["rewarding"]
            + counts["done"]
            + counts["failed"]
        )
        rewards_done = counts["done"] + counts["failed"]
        config_value = dict(config) if config is not None else {}
        average_quality = score_row["average_quality"] if score_row else None
        return {
            "id": run_id,
            "campaign": self._campaign_id(run_id),
            "provider": metadata.get("provider"),
            "model": metadata.get("model"),
            "rollout_slots": int(config_value.get("rollout_concurrency") or 0),
            "reward_slots": int(config_value.get("reward_concurrency") or 0),
            "counts": counts,
            "total": total,
            "started": total - counts["queued"],
            "rollouts_completed": rollout_done,
            "rewards_completed": rewards_done,
            "average_quality": average_quality,
            "valid_rewards": int(score_row["valid_rewards"] or 0) if score_row else 0,
            "overall_progress": (
                (rollout_done + rewards_done) / (2 * total) if total else 0
            ),
        }

    def global_state(self, selected_run_id: str | None = None) -> dict[str, Any]:
        """Return a low-cost global pool snapshot for sibling queue runs."""
        selected = selected_run_id or self.default_run_id
        campaign = self._campaign_id(selected)
        runs = [
            state
            for run_id in self.run_ids()
            if (state := self._quick_queue_state(run_id)) is not None
            and state["campaign"] == campaign
        ]
        runs.sort(key=lambda item: item["id"])

        persisted = _read_json(self.runs_root / "global-pool-state.json", {})
        if not isinstance(persisted, dict):
            persisted = {}
        persisted_ids = {
            str(item.get("run"))
            for item in persisted.get("runs", [])
            if isinstance(item, dict)
        }
        cohort_ids = {str(item["id"]) for item in runs}
        uses_persisted = bool(persisted_ids) and persisted_ids.issubset(cohort_ids)
        allocated_rollout = sum(int(item["rollout_slots"]) for item in runs)
        allocated_reward = sum(int(item["reward_slots"]) for item in runs)
        total = sum(int(item["total"]) for item in runs)
        rollout_done = sum(int(item["rollouts_completed"]) for item in runs)
        rewards_done = sum(int(item["rewards_completed"]) for item in runs)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "campaign": campaign,
            "pool": {
                "status": persisted.get("status") if uses_persisted else "observed",
                "updated_at": persisted.get("updated_at") if uses_persisted else None,
                "rollout_capacity": (
                    int(persisted.get("rollout_capacity") or 0)
                    if uses_persisted
                    else allocated_rollout
                ),
                "reward_capacity": (
                    int(persisted.get("reward_capacity") or 0)
                    if uses_persisted
                    else allocated_reward
                ),
                "allocated_rollout": allocated_rollout,
                "allocated_reward": allocated_reward,
            },
            "summary": {
                "runs": len(runs),
                "total": total,
                "queued": sum(int(item["counts"]["queued"]) for item in runs),
                "rollout_active": sum(
                    int(item["counts"]["rollout"]) for item in runs
                ),
                "reward_active": sum(
                    int(item["counts"]["rewarding"]) for item in runs
                ),
                "reward_pending": sum(
                    int(item["counts"]["reward_pending"]) for item in runs
                ),
                "rollouts_completed": rollout_done,
                "rewards_completed": rewards_done,
                "failed": sum(int(item["counts"]["failed"]) for item in runs),
                "overall_progress": (
                    (rollout_done + rewards_done) / (2 * total) if total else 0
                ),
            },
            "runs": runs,
        }


class DashboardHandler(BaseHTTPRequestHandler):
    registry: DashboardRegistry
    html_path: Path

    def log_message(self, format: str, *args: Any) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(format, *args)

    def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            try:
                payload = self.html_path.read_bytes()
            except OSError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/api/state":
            run_id = parse_qs(parsed.query).get("run", [""])[0] or None
            try:
                store = self.registry.get(run_id)
            except KeyError:
                self._send_json({"error": "unknown run"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(store.state())
            return
        if parsed.path == "/api/runs":
            self._send_json(self.registry.listing())
            return
        if parsed.path == "/api/global":
            run_id = parse_qs(parsed.query).get("run", [""])[0] or None
            self._send_json(self.registry.global_state(run_id))
            return
        if parsed.path == "/api/task":
            query = parse_qs(parsed.query)
            task = query.get("task", [""])[0]
            mode = query.get("mode", [""])[0] or None
            run_id = query.get("run", [""])[0] or None
            try:
                detail = self.registry.get(run_id).task_detail(task, mode)
            except KeyError as exc:
                self._send_json({"error": f"unknown run or task: {exc}"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(detail)
            return
        if parsed.path == "/api/health":
            self._send_json(
                {"ok": True, "run": self.registry.default_run_id}
            )
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/concurrency":
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(
                {"error": "Content-Type must be application/json"},
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 16_384:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            run_id = str(payload.get("run") or "") or None
            rollout = payload.get("rollout_concurrency")
            reward = payload.get("reward_concurrency")
            if rollout is not None and (
                isinstance(rollout, bool) or not isinstance(rollout, int)
            ):
                raise ValueError("rollout_concurrency must be an integer")
            if reward is not None and (
                isinstance(reward, bool) or not isinstance(reward, int)
            ):
                raise ValueError("reward_concurrency must be an integer")
            configured = self.registry.get(run_id).set_concurrency(
                rollout=rollout,
                reward=reward,
            )
        except KeyError:
            self._send_json({"error": "unknown run"}, HTTPStatus.NOT_FOUND)
            return
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except (RuntimeError, sqlite3.Error, OSError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
            return
        self._send_json({"ok": True, "concurrency": configured})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Benchmark run directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--open", action="store_true", help="Open the dashboard in a browser")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = DashboardRegistry(args.run)
    handler = type(
        "BoundDashboardHandler",
        (DashboardHandler,),
        {"registry": registry, "html_path": args.html.resolve()},
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.verbose = args.verbose  # type: ignore[attr-defined]
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    print(f"NL2Repo dashboard: {url}", flush=True)
    print(f"Runs root: {registry.runs_root}", flush=True)
    print(f"Default run: {registry.default_run_id}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
