#!/usr/bin/env python3
"""Run NL2Repo evaluation as a persistent, dynamically refillable queue."""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import shutil
import sqlite3
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
BENCHMARK_PATH = HERE / "benchmark.py"
SPEC = importlib.util.spec_from_file_location("nl2repo_queue_benchmark", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)

QUEUE_DB = "queue.sqlite3"
QUEUE_STATUSES = (
    "queued",
    "rollout",
    "reward_pending",
    "rewarding",
    "done",
    "failed",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class QueueStore:
    """Small SQLite state machine shared by add/status/serve processes."""

    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root.expanduser().resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.path = self.run_root / QUEUE_DB
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'queued',
                    attempt INTEGER NOT NULL DEFAULT 0,
                    enqueued_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    rollout_finished_at TEXT,
                    reward_started_at TEXT,
                    finished_at TEXT,
                    quality_score REAL,
                    success INTEGER,
                    error TEXT,
                    UNIQUE(task, mode),
                    CHECK(
                        status IN (
                            'queued','rollout','reward_pending','rewarding','done','failed'
                        )
                    )
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS cases_status_priority "
                "ON cases(status, priority DESC, id ASC)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS worker_config (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    rollout_concurrency INTEGER NOT NULL,
                    reward_concurrency INTEGER NOT NULL,
                    max_rollout_concurrency INTEGER NOT NULL,
                    max_reward_concurrency INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK(rollout_concurrency >= 0),
                    CHECK(reward_concurrency >= 0),
                    CHECK(max_rollout_concurrency >= 1),
                    CHECK(max_reward_concurrency >= 1),
                    CHECK(rollout_concurrency <= max_rollout_concurrency),
                    CHECK(reward_concurrency <= max_reward_concurrency)
                )
                """
            )

    def initialize_concurrency(
        self,
        rollout: int,
        reward: int,
        *,
        max_rollout: int,
        max_reward: int,
    ) -> dict[str, Any]:
        """Initialize dynamic limits while preserving an existing desired size."""
        if rollout < 0 or reward < 0:
            raise ValueError("initial concurrency must be non-negative")
        if max_rollout < 1 or max_reward < 1:
            raise ValueError("maximum concurrency must be positive")
        if rollout > max_rollout or reward > max_reward:
            raise ValueError("initial concurrency cannot exceed worker capacity")
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO worker_config
                    (id, rollout_concurrency, reward_concurrency,
                     max_rollout_concurrency, max_reward_concurrency, updated_at)
                VALUES (1, ?, ?, ?, ?, ?)
                """,
                (rollout, reward, max_rollout, max_reward, now),
            )
            connection.execute(
                """
                UPDATE worker_config
                SET rollout_concurrency=MIN(rollout_concurrency, ?),
                    reward_concurrency=MIN(reward_concurrency, ?),
                    max_rollout_concurrency=?, max_reward_concurrency=?, updated_at=?
                WHERE id=1
                """,
                (max_rollout, max_reward, max_rollout, max_reward, now),
            )
        configured = self.concurrency()
        assert configured is not None
        return configured

    def concurrency(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM worker_config WHERE id=1"
            ).fetchone()
        return dict(row) if row is not None else None

    def set_concurrency(
        self,
        *,
        rollout: int | None = None,
        reward: int | None = None,
    ) -> dict[str, Any]:
        """Persist desired pool sizes; zero pauses new work for that pool."""
        if rollout is None and reward is None:
            raise ValueError("provide rollout or reward concurrency")
        if rollout is not None and rollout < 0:
            raise ValueError("rollout concurrency must be non-negative")
        if reward is not None and reward < 0:
            raise ValueError("reward concurrency must be non-negative")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM worker_config WHERE id=1"
            ).fetchone()
            if row is None:
                raise RuntimeError("dynamic worker configuration is not initialized")
            desired_rollout = int(row["rollout_concurrency"]) if rollout is None else rollout
            desired_reward = int(row["reward_concurrency"]) if reward is None else reward
            if desired_rollout > int(row["max_rollout_concurrency"]):
                raise ValueError(
                    f"rollout concurrency exceeds worker capacity "
                    f"{row['max_rollout_concurrency']}"
                )
            if desired_reward > int(row["max_reward_concurrency"]):
                raise ValueError(
                    f"reward concurrency exceeds worker capacity "
                    f"{row['max_reward_concurrency']}"
                )
            connection.execute(
                """
                UPDATE worker_config
                SET rollout_concurrency=?, reward_concurrency=?, updated_at=?
                WHERE id=1
                """,
                (desired_rollout, desired_reward, utc_now()),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        configured = self.concurrency()
        assert configured is not None
        return configured

    def enqueue(
        self, task_names: list[str], mode: str, *, priority: int = 0
    ) -> tuple[list[str], list[str]]:
        added: list[str] = []
        skipped: list[str] = []
        now = utc_now()
        with self._connect() as connection:
            for task in task_names:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO cases
                        (task, mode, priority, status, enqueued_at, updated_at)
                    VALUES (?, ?, ?, 'queued', ?, ?)
                    """,
                    (task, mode, priority, now, now),
                )
                (added if cursor.rowcount else skipped).append(task)
        return added, skipped

    def retry(self, task_names: list[str], mode: str) -> tuple[list[str], list[str]]:
        retried: list[str] = []
        missing: list[str] = []
        attempts: dict[str, int] = {}
        now = utc_now()
        with self._connect() as connection:
            for task in task_names:
                row = connection.execute(
                    "SELECT attempt FROM cases WHERE task=? AND mode=? "
                    "AND status IN ('done','failed')",
                    (task, mode),
                ).fetchone()
                cursor = connection.execute(
                    """
                    UPDATE cases
                    SET status='queued', updated_at=?, started_at=NULL,
                        rollout_finished_at=NULL, reward_started_at=NULL,
                        finished_at=NULL, quality_score=NULL, success=NULL, error=NULL
                    WHERE task=? AND mode=? AND status IN ('done','failed')
                    """,
                    (now, task, mode),
                )
                if cursor.rowcount:
                    retried.append(task)
                    attempts[task] = int(row["attempt"]) if row else 0
                else:
                    missing.append(task)
        archive_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for task in retried:
            case_root = self.run_root / task / mode
            if not case_root.exists():
                continue
            archive = (
                self.run_root
                / "_attempts"
                / task
                / mode
                / f"attempt-{attempts[task]}-{archive_stamp}"
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(case_root), str(archive))
        return retried, missing

    def recover_interrupted(self) -> dict[str, int]:
        """Return interrupted rollouts to queue and interrupted rewards to reward queue."""
        now = utc_now()
        with self._connect() as connection:
            interrupted_rollouts = connection.execute(
                "SELECT task, mode, attempt FROM cases WHERE status='rollout'"
            ).fetchall()
            rollout = connection.execute(
                """
                UPDATE cases
                SET status='queued', updated_at=?, error='runner restarted during rollout'
                WHERE status='rollout'
                """,
                (now,),
            ).rowcount
            rewarding = connection.execute(
                """
                UPDATE cases
                SET status='reward_pending', updated_at=?,
                    error='runner restarted during reward'
                WHERE status='rewarding'
                """,
                (now,),
            ).rowcount
        archive_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for row in interrupted_rollouts:
            case_root = self.run_root / str(row["task"]) / str(row["mode"])
            if not case_root.exists():
                continue
            archive = (
                self.run_root
                / "_attempts"
                / str(row["task"])
                / str(row["mode"])
                / f"attempt-{int(row['attempt'])}-interrupted-{archive_stamp}"
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(case_root), str(archive))
        return {"rollout": rollout, "reward": rewarding}

    def salvage_completed_rollouts(
        self, *, exclude_case_ids: set[int] | None = None
    ) -> list[dict[str, Any]]:
        """Promote completed rollouts left in-flight by a draining predecessor."""
        excluded = exclude_case_ids or set()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cases WHERE status='rollout' ORDER BY id"
            ).fetchall()
        salvaged: list[dict[str, Any]] = []
        for row in rows:
            case = dict(row)
            case_id = int(case["id"])
            if case_id in excluded:
                continue
            case_root = self.run_root / str(case["task"]) / str(case["mode"])
            workspace = case_root / "workspace"
            agent_path = case_root / "agent-result.json"
            start_path = workspace / "start.md"
            if not agent_path.is_file() or not start_path.is_file():
                continue
            try:
                agent = benchmark._read_json(agent_path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(agent, dict):
                continue
            started_at = datetime.fromisoformat(str(case["started_at"]))
            elapsed = max(
                0.0, (datetime.now(timezone.utc) - started_at).total_seconds()
            )
            benchmark._write_json(
                artifact_path(self.run_root, str(case["task"]), str(case["mode"])),
                {
                    "start_hash": benchmark._hash_file(start_path),
                    "agent_elapsed_s": elapsed,
                    "agent_timed_out": False,
                    "agent_returncode": 0 if agent.get("ok") else 1,
                    "salvaged": True,
                },
            )
            now = utc_now()
            with self._connect() as connection:
                updated = connection.execute(
                    """
                    UPDATE cases
                    SET status='reward_pending', updated_at=?, rollout_finished_at=?,
                        error='rollout salvaged during live worker handoff'
                    WHERE id=? AND status='rollout'
                    """,
                    (now, now, case_id),
                ).rowcount
            if updated:
                salvaged.append(case)
        return salvaged

    def claim(self, from_status: str, to_status: str, limit: int) -> list[dict[str, Any]]:
        if from_status not in QUEUE_STATUSES or to_status not in QUEUE_STATUSES:
            raise ValueError("invalid queue status")
        if limit < 1:
            return []
        now = utc_now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM cases WHERE status=? "
                "ORDER BY priority DESC, id ASC LIMIT ?",
                (from_status, limit),
            ).fetchall()
            ids = [int(row["id"]) for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                assignments = "status=?, updated_at=?"
                values: list[Any] = [to_status, now]
                if to_status == "rollout":
                    assignments += ", started_at=?, attempt=attempt+1, error=NULL"
                    values.append(now)
                elif to_status == "rewarding":
                    assignments += ", reward_started_at=?, error=NULL"
                    values.append(now)
                connection.execute(
                    f"UPDATE cases SET {assignments} WHERE id IN ({placeholders})",
                    (*values, *ids),
                )
            connection.commit()
            return [dict(row) for row in rows]
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def mark_rollout_complete(self, case_id: int) -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE cases SET status='reward_pending', updated_at=?,
                    rollout_finished_at=?, error=NULL WHERE id=?
                """,
                (now, now, case_id),
            )

    def mark_done(self, case_id: int, result: dict[str, Any]) -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE cases SET status='done', updated_at=?, finished_at=?,
                    quality_score=?, success=?, error=NULL WHERE id=?
                """,
                (
                    now,
                    now,
                    result.get("quality_score"),
                    int(bool(result.get("success"))),
                    case_id,
                ),
            )

    def mark_failed(self, case_id: int, error: BaseException | str) -> None:
        now = utc_now()
        message = str(error)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE cases SET status='failed', updated_at=?, finished_at=?,
                    success=0, error=? WHERE id=?
                """,
                (now, now, message[-4000:], case_id),
            )

    def cases(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cases ORDER BY priority DESC, id ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def counts(self) -> dict[str, int]:
        values = {status: 0 for status in QUEUE_STATUSES}
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM cases GROUP BY status"
            ).fetchall()
        for row in rows:
            values[str(row["status"])] = int(row["count"])
        values["total"] = sum(values.values())
        return values


def artifact_path(run_root: Path, task_name: str, mode: str) -> Path:
    return run_root / task_name / mode / "rollout-artifact.json"


def persist_artifact(artifact: Any) -> None:
    agent_path = artifact.case_root / "agent-result.json"
    if not agent_path.is_file():
        raise FileNotFoundError(
            f"refusing to persist incomplete rollout without {agent_path.name}: "
            f"{artifact.case_root}"
        )
    benchmark._write_json(
        artifact.case_root / "rollout-artifact.json",
        {
            "start_hash": artifact.start_hash,
            "agent_elapsed_s": artifact.agent_elapsed_s,
            "agent_timed_out": artifact.agent_timed_out,
            "agent_returncode": artifact.agent_returncode,
        },
    )


def restore_artifact(run_root: Path, task: dict[str, Any], mode: str) -> Any:
    case_root = run_root / task["id"] / mode
    stored = benchmark._read_json(case_root / "rollout-artifact.json")
    agent_path = case_root / "agent-result.json"
    if not isinstance(stored, dict) or not agent_path.is_file():
        raise FileNotFoundError(f"persisted rollout artifact is incomplete: {case_root}")
    return benchmark.RolloutArtifact(
        task=task,
        mode=mode,
        case_root=case_root,
        workspace=case_root / "workspace",
        start_hash=str(stored["start_hash"]),
        agent=benchmark._read_json(agent_path),
        agent_elapsed_s=float(stored.get("agent_elapsed_s", 0)),
        agent_timed_out=bool(stored.get("agent_timed_out")),
        agent_returncode=int(stored.get("agent_returncode", 1)),
    )


def score_with_infrastructure_retries(
    score_fn: Callable[[], dict[str, Any]],
    *,
    attempts: int,
    delay_s: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Retry scorer exceptions and explicit hidden-test infrastructure errors."""
    if attempts < 1:
        raise ValueError("reward attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            result = score_fn()
            hidden = result.get("hidden_tests")
            error = str(hidden.get("error") or "") if isinstance(hidden, dict) else ""
            if not error or attempt == attempts:
                return result
        except Exception as exc:
            if attempt == attempts:
                raise
            error = f"{type(exc).__name__}: {exc}"
        if on_retry is not None:
            on_retry(attempt, error)
        sleep_fn(delay_s)
    raise AssertionError("unreachable")


def allocate_global_slots(
    snapshots: list[dict[str, int]],
    capacity: int,
    *,
    active_key: str,
    pending_key: str,
) -> list[int]:
    """Allocate one global pool across ordered, independently persisted runs."""
    if capacity < 0:
        raise ValueError("global capacity must be non-negative")
    remaining = capacity
    allocations: list[int] = []
    for counts in snapshots:
        active = max(0, int(counts.get(active_key, 0)))
        pending = max(0, int(counts.get(pending_key, 0)))
        allocated = min(remaining, active + pending)
        allocations.append(allocated)
        remaining -= allocated
    return allocations


def run_queue_loop(
    store: QueueStore,
    task_loader: Callable[[str], dict[str, Any]],
    rollout_fn: Callable[[dict[str, Any], str], Any],
    reward_fn: Callable[[Any], dict[str, Any]],
    *,
    rollout_concurrency: int,
    reward_concurrency: int,
    max_rollout_concurrency: int | None = None,
    max_reward_concurrency: int | None = None,
    concurrency_loader: Callable[[], dict[str, Any] | None] | None = None,
    recover_interrupted: bool = True,
    adopt_external_inflight: bool = False,
    stop_event: threading.Event,
    poll_interval_s: float = 1.0,
    stop_when_empty: bool = False,
    failure_fn: Callable[[dict[str, Any], str, str, Exception], dict[str, Any]] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    on_resize: Callable[[dict[str, int]], None] | None = None,
) -> None:
    """Keep both pools full while cases can be appended from another process."""
    if rollout_concurrency < 0 or reward_concurrency < 0:
        raise ValueError("rollout and reward concurrency must be non-negative")
    max_rollout = max_rollout_concurrency or rollout_concurrency
    max_reward = max_reward_concurrency or reward_concurrency
    if max_rollout < 1 or max_reward < 1:
        raise ValueError("maximum concurrency must be positive")
    if max_rollout < rollout_concurrency or max_reward < reward_concurrency:
        raise ValueError("maximum concurrency cannot be below initial concurrency")
    desired_rollout = rollout_concurrency
    desired_reward = reward_concurrency
    started = time.monotonic()
    event_lock = threading.Lock()

    def emit(name: str, task: dict[str, Any], mode: str, **extra: Any) -> None:
        if on_event is None:
            return
        value = {
            "event": name,
            "task": task["id"],
            "mode": mode,
            "elapsed_s": round(time.monotonic() - started, 3),
            "recorded_at": utc_now(),
            **extra,
        }
        with event_lock:
            on_event(value)

    rollout_futures: dict[Future[Any], tuple[dict[str, Any], dict[str, Any]]] = {}
    reward_futures: dict[Future[Any], tuple[dict[str, Any], dict[str, Any]]] = {}
    if recover_interrupted:
        store.recover_interrupted()

    def do_rollout(case: dict[str, Any], task: dict[str, Any]) -> Any:
        emit("rollout.started", task, str(case["mode"]), queue_id=case["id"])
        return rollout_fn(task, str(case["mode"]))

    def do_reward(case: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
        emit("reward.started", task, str(case["mode"]), queue_id=case["id"])
        artifact = restore_artifact(store.run_root, task, str(case["mode"]))
        return reward_fn(artifact)

    with (
        ThreadPoolExecutor(
            max_workers=max_rollout, thread_name_prefix="queue-rollout"
        ) as rollout_pool,
        ThreadPoolExecutor(
            max_workers=max_reward, thread_name_prefix="queue-reward"
        ) as reward_pool,
    ):
        while not stop_event.is_set():
            if concurrency_loader is not None:
                configured = concurrency_loader()
                if configured is not None:
                    next_rollout = int(configured["rollout_concurrency"])
                    next_reward = int(configured["reward_concurrency"])
                    if not 0 <= next_rollout <= max_rollout:
                        raise ValueError("dynamic rollout concurrency is outside worker capacity")
                    if not 0 <= next_reward <= max_reward:
                        raise ValueError("dynamic reward concurrency is outside worker capacity")
                    if (next_rollout, next_reward) != (desired_rollout, desired_reward):
                        desired_rollout, desired_reward = next_rollout, next_reward
                        if on_resize is not None:
                            on_resize(
                                {
                                    "rollout_concurrency": desired_rollout,
                                    "reward_concurrency": desired_reward,
                                    "rollout_active": len(rollout_futures),
                                    "reward_active": len(reward_futures),
                                }
                            )
            if adopt_external_inflight:
                owned_ids = {
                    int(case["id"]) for case, _task in rollout_futures.values()
                }
                for case in store.salvage_completed_rollouts(
                    exclude_case_ids=owned_ids
                ):
                    task = task_loader(str(case["task"]))
                    emit(
                        "rollout.salvaged",
                        task,
                        str(case["mode"]),
                        queue_id=case["id"],
                    )
            for future in [item for item in rollout_futures if item.done()]:
                case, task = rollout_futures.pop(future)
                try:
                    artifact = future.result()
                    persist_artifact(artifact)
                    store.mark_rollout_complete(int(case["id"]))
                    emit("rollout.completed", task, str(case["mode"]), queue_id=case["id"])
                except Exception as exc:
                    if failure_fn is not None:
                        failure_fn(task, str(case["mode"]), "rollout", exc)
                    store.mark_failed(int(case["id"]), exc)
                    emit(
                        "rollout.failed", task, str(case["mode"]),
                        queue_id=case["id"], error_type=type(exc).__name__,
                    )

            for future in [item for item in reward_futures if item.done()]:
                case, task = reward_futures.pop(future)
                try:
                    result = future.result()
                    store.mark_done(int(case["id"]), result)
                    emit(
                        "reward.completed", task, str(case["mode"]),
                        queue_id=case["id"], quality_score=result.get("quality_score"),
                        success=bool(result.get("success")),
                    )
                except Exception as exc:
                    if failure_fn is not None:
                        failure_fn(task, str(case["mode"]), "reward", exc)
                    store.mark_failed(int(case["id"]), exc)
                    emit(
                        "reward.failed", task, str(case["mode"]),
                        queue_id=case["id"], error_type=type(exc).__name__,
                    )

            active_counts = store.counts() if adopt_external_inflight else None
            reward_slots = desired_reward - (
                active_counts["rewarding"]
                if active_counts is not None
                else len(reward_futures)
            )
            for case in store.claim("reward_pending", "rewarding", reward_slots):
                try:
                    task = task_loader(str(case["task"]))
                except Exception as exc:
                    store.mark_failed(int(case["id"]), exc)
                    continue
                future = reward_pool.submit(do_reward, case, task)
                reward_futures[future] = (case, task)

            active_counts = store.counts() if adopt_external_inflight else None
            rollout_slots = desired_rollout - (
                active_counts["rollout"]
                if active_counts is not None
                else len(rollout_futures)
            )
            for case in store.claim("queued", "rollout", rollout_slots):
                try:
                    task = task_loader(str(case["task"]))
                except Exception as exc:
                    store.mark_failed(int(case["id"]), exc)
                    continue
                future = rollout_pool.submit(do_rollout, case, task)
                rollout_futures[future] = (case, task)

            counts = store.counts()
            if (
                stop_when_empty
                and not rollout_futures
                and not reward_futures
                and not counts["queued"]
                and not counts["reward_pending"]
                and not counts["rollout"]
                and not counts["rewarding"]
            ):
                break
            stop_event.wait(poll_interval_s)


def ensure_metadata(run_root: Path, values: dict[str, Any]) -> dict[str, Any]:
    path = run_root / "run-metadata.json"
    current: dict[str, Any] = {}
    if path.is_file():
        loaded = benchmark._read_json(path)
        if isinstance(loaded, dict):
            current = loaded
    if not current.get("started_at"):
        current["started_at"] = utc_now()
    current.update(values)
    current.update(
        {
            "schema_version": 2,
            "queue_mode": "continuous",
            "run_id": current.get("run_id") or run_root.name,
            "queue_db": QUEUE_DB,
            "updated_at": utc_now(),
        }
    )
    benchmark._write_json(path, current)
    return current


def resolve_task_names(args: argparse.Namespace, upstream_root: Path) -> list[str]:
    if args.task:
        names = list(dict.fromkeys(args.task))
    elif args.task_set == "qwen32":
        names = benchmark.select_task_subset(benchmark.list_tasks(upstream_root))
    elif args.task_set == "remaining-qwen32":
        all_tasks = benchmark.list_tasks(upstream_root)
        selected = set(benchmark.select_task_subset(all_tasks))
        names = [task["id"] for task in all_tasks if task["id"] not in selected]
    elif args.task_set == "all":
        names = [task["id"] for task in benchmark.list_tasks(upstream_root)]
    else:
        raise ValueError("provide --task or --task-set")
    for name in names:
        benchmark.load_task(upstream_root, name)
    return names


def add_common_task_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", action="append", help="Task ID; repeat to enqueue several")
    parser.add_argument(
        "--task-set", choices=("qwen32", "remaining-qwen32", "all")
    )
    parser.add_argument(
        "--mode",
        choices=("solo", "adaptive", "adaptive-team-v2", "forced-team"),
        default="adaptive",
    )
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--cache-root", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Persistent queue run directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Append validated cases to the live queue")
    add_common_task_args(add_parser)
    add_parser.add_argument("--priority", type=int, default=0)

    retry_parser = subparsers.add_parser("retry", help="Requeue completed or failed cases")
    add_common_task_args(retry_parser)

    subparsers.add_parser("status", help="Print queue counts and cases")

    scale_parser = subparsers.add_parser(
        "scale", help="Change live rollout/reward slots without restarting workers"
    )
    scale_parser.add_argument("--rollout-concurrency", type=int)
    scale_parser.add_argument("--reward-concurrency", type=int)

    serve = subparsers.add_parser("serve", help="Run persistent rollout and reward workers")
    serve.add_argument("--provider", default="qwen")
    serve.add_argument("--model", default="ms-mnhdj86z")
    serve.add_argument("--max-turns", type=int, default=300)
    serve.add_argument("--teammate-max-turns", type=int, default=160)
    serve.add_argument("--teammate-min-timeout", type=float, default=900.0)
    serve.add_argument("--max-output-tokens", type=int, default=16384)
    serve.add_argument("--agent-timeout", type=float, default=7200)
    serve.add_argument("--score-timeout", type=float, default=1200)
    serve.add_argument("--rollout-concurrency", type=int, default=8)
    serve.add_argument("--reward-concurrency", type=int, default=4)
    serve.add_argument("--max-rollout-concurrency", type=int, default=64)
    serve.add_argument("--max-reward-concurrency", type=int, default=16)
    serve.add_argument("--execution-backend", choices=("local", "ags"), default="ags")
    serve.add_argument("--score-backend", choices=("docker", "ags"), default="docker")
    serve.add_argument("--upstream-root", type=Path)
    serve.add_argument("--cache-root", type=Path)
    serve.add_argument("--ags-env-file", type=Path)
    serve.add_argument("--ags-timeout", default="3h")
    serve.add_argument("--ags-cpu", default="2")
    serve.add_argument("--ags-memory", default="4Gi")
    serve.add_argument("--ags-score-tool-id")
    serve.add_argument("--ags-image-template", default=benchmark.AGS_IMAGE_TEMPLATE)
    serve.add_argument("--keep-image", action="store_true")
    serve.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    serve.add_argument("--poll-interval", type=float, default=1.0)
    serve.add_argument("--reward-attempts", type=int, default=3)
    serve.add_argument("--reward-retry-delay", type=float, default=5.0)
    serve.add_argument(
        "--start-after",
        type=Path,
        help="Wait for another run directory's results.json before claiming work",
    )
    serve.add_argument("--stop-when-empty", action="store_true", help=argparse.SUPPRESS)
    serve.add_argument("--adopt-inflight", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = args.run.resolve()
    store = QueueStore(run_root)

    if args.command == "status":
        print(
            json.dumps(
                {
                    "counts": store.counts(),
                    "concurrency": store.concurrency(),
                    "cases": store.cases(),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "scale":
        try:
            configured = store.set_concurrency(
                rollout=args.rollout_concurrency,
                reward=args.reward_concurrency,
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        ensure_metadata(
            run_root,
            {
                "rollout_concurrency": configured["rollout_concurrency"],
                "reward_concurrency": configured["reward_concurrency"],
            },
        )
        print(json.dumps(configured, indent=2))
        return 0

    if args.command in {"add", "retry"}:
        upstream_root = benchmark.resolve_upstream(args.upstream_root, cache_root=args.cache_root)
        try:
            task_names = resolve_task_names(args, upstream_root)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        ensure_metadata(run_root, {})
        if args.command == "add":
            changed, unchanged = store.enqueue(task_names, args.mode, priority=args.priority)
            print(f"queued {len(changed)} case(s): {', '.join(changed) or '-'}")
            if unchanged:
                print(f"already present {len(unchanged)}: {', '.join(unchanged)}")
        else:
            changed, unchanged = store.retry(task_names, args.mode)
            print(f"requeued {len(changed)} case(s): {', '.join(changed) or '-'}")
            if unchanged:
                print(f"not retryable {len(unchanged)}: {', '.join(unchanged)}")
        return 0

    if args.rollout_concurrency < 0 or args.reward_concurrency < 0:
        raise SystemExit("concurrency must be non-negative")
    if (
        args.max_rollout_concurrency < args.rollout_concurrency
        or args.max_reward_concurrency < args.reward_concurrency
    ):
        raise SystemExit("maximum concurrency must cover initial concurrency")
    if args.reward_attempts < 1 or args.reward_retry_delay < 0:
        raise SystemExit("reward retry settings must be non-negative with positive attempts")
    upstream_root = benchmark.resolve_upstream(args.upstream_root, cache_root=args.cache_root)
    ags_env_file = args.ags_env_file.resolve() if args.ags_env_file else None
    configured = store.initialize_concurrency(
        args.rollout_concurrency,
        args.reward_concurrency,
        max_rollout=args.max_rollout_concurrency,
        max_reward=args.max_reward_concurrency,
    )
    ensure_metadata(
        run_root,
        {
            "provider": args.provider,
            "model": args.model,
            "execution_backend": args.execution_backend,
            "score_backend": args.score_backend,
            "max_turns": args.max_turns,
            "teammate_max_turns": args.teammate_max_turns,
            "teammate_min_timeout_s": args.teammate_min_timeout,
            "rollout_concurrency": configured["rollout_concurrency"],
            "reward_concurrency": configured["reward_concurrency"],
            "max_rollout_concurrency": configured["max_rollout_concurrency"],
            "max_reward_concurrency": configured["max_reward_concurrency"],
            "reward_uses_rollout_slots": False,
        },
    )
    scheduler_path = run_root / "scheduler.jsonl"

    def load_task(name: str) -> dict[str, Any]:
        return benchmark.load_task(upstream_root, name)

    def rollout(task: dict[str, Any], mode: str) -> Any:
        return benchmark.run_rollout(
            task, mode, run_root,
            provider=args.provider, model=args.model, max_turns=args.max_turns,
            teammate_max_turns=args.teammate_max_turns,
            teammate_min_timeout_s=args.teammate_min_timeout,
            max_output_tokens=args.max_output_tokens, agent_timeout_s=args.agent_timeout,
            stream=args.stream, execution_backend=args.execution_backend,
            ags_image=benchmark.format_ags_image(args.ags_image_template, task["id"]),
            ags_env_file=ags_env_file, ags_timeout=args.ags_timeout,
            ags_cpu=args.ags_cpu, ags_memory=args.ags_memory,
        )

    def reward(artifact: Any) -> dict[str, Any]:
        return score_with_infrastructure_retries(
            lambda: benchmark.score_rollout(
                artifact,
                provider=args.provider,
                model=args.model,
                score_timeout_s=args.score_timeout,
                keep_image=args.keep_image,
                execution_backend=args.execution_backend,
                score_backend=args.score_backend,
                ags_image=benchmark.format_ags_image(
                    args.ags_image_template, artifact.task["id"]
                ),
                ags_env_file=ags_env_file,
                ags_timeout=args.ags_timeout,
                ags_cpu=args.ags_cpu,
                ags_memory=args.ags_memory,
                ags_score_tool_id=args.ags_score_tool_id,
            ),
            attempts=args.reward_attempts,
            delay_s=args.reward_retry_delay,
            on_retry=lambda attempt, error: print(
                f"[{artifact.task['id']}] reward infrastructure retry "
                f"{attempt}/{args.reward_attempts}: {error}",
                flush=True,
            ),
        )

    def failed(task: dict[str, Any], mode: str, phase: str, error: Exception) -> dict[str, Any]:
        return benchmark._failed_case_result(
            task, mode, phase, error, output_root=run_root,
            provider=args.provider, model=args.model,
            execution_backend=args.execution_backend, score_backend=args.score_backend,
        )

    def event(value: dict[str, Any]) -> None:
        benchmark._append_jsonl(scheduler_path, value)
        print(
            f"[{value['task']}] {value['event']} · queue={store.counts()}",
            flush=True,
        )

    def resized(value: dict[str, int]) -> None:
        ensure_metadata(
            run_root,
            {
                "rollout_concurrency": value["rollout_concurrency"],
                "reward_concurrency": value["reward_concurrency"],
            },
        )
        benchmark._append_jsonl(
            scheduler_path,
            {
                "event": "pool.resized",
                "elapsed_s": None,
                "recorded_at": utc_now(),
                **value,
            },
        )
        print(
            "pool resized: "
            f"rollout={value['rollout_concurrency']} "
            f"reward={value['reward_concurrency']} "
            f"active={value['rollout_active']}+{value['reward_active']}",
            flush=True,
        )

    stop_event = threading.Event()

    def stop(signum: int, frame: Any) -> None:
        print(f"received signal {signum}; waiting for active workers", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if args.start_after:
        marker = args.start_after.expanduser().resolve()
        if marker.suffix != ".json":
            marker = marker / "results.json"
        print(f"waiting for predecessor: {marker}", flush=True)
        while not marker.is_file() and not stop_event.wait(max(0.2, args.poll_interval)):
            pass
        if stop_event.is_set():
            return 0
        print("predecessor completed; queue workers released", flush=True)
    print(
        f"Continuous queue ready: {run_root}\n"
        f"rollout_pool={configured['rollout_concurrency']} "
        f"reward_pool={configured['reward_concurrency']} "
        f"capacity={configured['max_rollout_concurrency']}+"
        f"{configured['max_reward_concurrency']} "
        f"queued={store.counts()['queued']}",
        flush=True,
    )
    run_queue_loop(
        store, load_task, rollout, reward,
        rollout_concurrency=int(configured["rollout_concurrency"]),
        reward_concurrency=int(configured["reward_concurrency"]),
        max_rollout_concurrency=int(configured["max_rollout_concurrency"]),
        max_reward_concurrency=int(configured["max_reward_concurrency"]),
        concurrency_loader=store.concurrency,
        recover_interrupted=not args.adopt_inflight,
        adopt_external_inflight=args.adopt_inflight,
        stop_event=stop_event,
        poll_interval_s=args.poll_interval,
        stop_when_empty=args.stop_when_empty,
        failure_fn=failed,
        on_event=event,
        on_resize=resized,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
