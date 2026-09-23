#!/usr/bin/env python3
"""Run NL2Repo evaluation as a persistent, dynamically refillable queue."""

from __future__ import annotations

import argparse
import errno
import fcntl
import importlib.util
import json
import os
import signal
import shutil
import sqlite3
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


HERE = Path(__file__).resolve().parent
BENCHMARK_PATH = HERE / "benchmark.py"
SPEC = importlib.util.spec_from_file_location("nl2repo_queue_benchmark", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)

QUEUE_DB = "queue.sqlite3"
GLOBAL_POOL_WORKER_ENV = "CLAWD_NL2REPO_GLOBAL_POOL_WORKER"
GLOBAL_POOL_WORKER_MARKER = "global_pool_supervisor.v1"
GLOBAL_POOL_LOCK_PATH = HERE / "runs" / "global-pool.lock"
QUEUE_STATUSES = (
    "queued",
    "rollout",
    "reward_pending",
    "rewarding",
    "done",
    "failed",
)


def enforce_serve_launch_policy(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
    lock_path: Path | None = None,
    parent_pid: int | None = None,
) -> None:
    """Prevent direct worker/slot controls from bypassing the global pool.

    A marker environment variable alone is intentionally insufficient: a
    managed worker must be a direct child of the process recorded in the live
    pilot-wide flock, and its resolved run directory must be registered there.
    The check happens before ``QueueStore`` is opened so an unauthorized second
    worker cannot recover/move another worker's in-flight cases.
    """
    if args.command == "scale":
        raise SystemExit(
            "direct 'evaluation_queue.py ... scale' is disabled: the global pool "
            "supervisor exclusively owns rollout/reward slot allocation. Restart "
            "global_pool_supervisor.py with the desired capacities."
        )
    if args.command != "serve":
        return
    environment = os.environ if environ is None else environ
    if _is_authorized_global_pool_worker(
        args.run,
        environ=environment,
        lock_path=GLOBAL_POOL_LOCK_PATH if lock_path is None else lock_path,
        parent_pid=os.getppid() if parent_pid is None else parent_pid,
    ):
        return
    raise SystemExit(
        "direct 'evaluation_queue.py ... serve' is disabled: start queue workers "
        "with global_pool_supervisor.py so rollout/reward capacity is shared. A "
        "marker environment variable without the matching live supervisor lock, "
        "parent PID, and registered run is not accepted."
    )


def _global_pool_lock_is_held(path: Path) -> bool:
    """Return whether another process currently owns the pilot-wide flock."""
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        return False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return True
            return False
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return False
    finally:
        handle.close()


def _is_authorized_global_pool_worker(
    run_root: Path,
    *,
    environ: Mapping[str, str],
    lock_path: Path,
    parent_pid: int,
) -> bool:
    """Validate the supervisor-to-worker launch relationship."""
    if environ.get(GLOBAL_POOL_WORKER_ENV) != GLOBAL_POOL_WORKER_MARKER:
        return False
    try:
        metadata = json.loads(lock_path.read_text(encoding="utf-8"))
        owner_pid = int(metadata["pid"])
        registered_values = {str(value) for value in metadata["runs"]}
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False
    if owner_pid != parent_pid:
        return False
    resolved_run = run_root.expanduser().resolve()
    # Name-only metadata is accepted solely for workers whose parent owns the
    # live lock, allowing a supervisor started before the full-path metadata
    # migration to restart its children without abandoning in-flight work.
    if (
        str(resolved_run) not in registered_values
        and resolved_run.name not in registered_values
    ):
        return False
    return _global_pool_lock_is_held(lock_path)


def start_supervisor_lease_watchdog(
    run_root: Path,
    stop_event: threading.Event,
    *,
    interval_s: float = 1.0,
    exit_fn: Callable[[int], Any] | None = None,
) -> threading.Thread:
    """Hard-stop an orphan queue worker after its supervisor loses the flock.

    Executor threads cannot be safely cancelled while they are inside model or
    sandbox clients.  Exiting the child process is therefore intentional: a new
    supervisor can recover the SQLite in-flight states without an orphan worker
    continuing to consume global capacity or later committing duplicate output.
    """
    if interval_s <= 0:
        raise ValueError("supervisor watchdog interval must be positive")
    terminate = _terminate_orphan_worker_tree if exit_fn is None else exit_fn

    def monitor() -> None:
        consecutive_failures = 0
        while not stop_event.wait(interval_s):
            valid = _is_authorized_global_pool_worker(
                run_root,
                environ=os.environ,
                lock_path=GLOBAL_POOL_LOCK_PATH,
                parent_pid=os.getppid(),
            )
            if valid:
                consecutive_failures = 0
                continue
            # Lock metadata is updated in place to preserve the flock inode. A
            # reader may catch its tiny truncate/write window, so require three
            # consecutive misses before treating the lease as lost.
            consecutive_failures += 1
            if consecutive_failures < 3:
                continue
            print(
                "global pool supervisor lease lost; terminating orphan queue worker",
                file=sys.stderr,
                flush=True,
            )
            terminate(75)
            return

    thread = threading.Thread(
        target=monitor,
        name="global-pool-lease-watchdog",
        daemon=True,
    )
    thread.start()
    return thread


def _terminate_orphan_worker_tree(exit_code: int) -> None:
    """Gracefully stop a managed worker group, then enforce a hard deadline."""
    process_group = os.getpgrp()
    if process_group != os.getpid():
        # New supervisors always launch each worker as a session leader. Avoid
        # signaling an unrelated shell/process group if an old manual process
        # somehow reaches this path.
        os._exit(exit_code)

    def force_kill() -> None:
        time.sleep(30)
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass

    threading.Thread(
        target=force_kill,
        name="orphan-worker-force-kill",
        daemon=True,
    ).start()
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        os._exit(exit_code)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def has_reusable_rollout(run_root: Path, task: str, mode: str) -> bool:
    """Whether a failed case can be rescored without regenerating its workspace."""
    case_root = run_root / task / mode
    artifact_path = case_root / "rollout-artifact.json"
    agent_path = case_root / "agent-result.json"
    if not artifact_path.is_file() or not agent_path.is_file():
        return False
    if not (case_root / "workspace" / "start.md").is_file():
        return False
    try:
        stored = benchmark._read_json(artifact_path)
        agent = benchmark._read_json(agent_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(stored, dict) or not isinstance(agent, dict):
        return False
    outcome = str(agent.get("rollout_outcome") or "")
    return not (
        outcome in {"infra_error", "harness_error"}
        or agent.get("rollout_infrastructure")
        or agent.get("workspace_download_error")
    )


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
        """Retry terminal cases without exposing an unarchived rollout as queued.

        SQLite's write transaction is deliberately held across the filesystem
        rename.  A live worker attempting to claim the case therefore cannot
        observe ``queued`` until the previous attempt is safely archived.
        """
        retried: list[str] = []
        missing: list[str] = []
        for task in task_names:
            connection = self._connect()
            case_root = self.run_root / task / mode
            archive: Path | None = None
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT status, attempt FROM cases WHERE task=? AND mode=? "
                    "AND status IN ('done','failed')",
                    (task, mode),
                ).fetchone()
                if row is None:
                    connection.rollback()
                    missing.append(task)
                    continue
                resume_reward = bool(
                    row["status"] == "failed"
                    and has_reusable_rollout(self.run_root, task, mode)
                )
                if resume_reward:
                    cursor = connection.execute(
                        """
                        UPDATE cases
                        SET status='reward_pending', updated_at=?, reward_started_at=NULL,
                            finished_at=NULL, quality_score=NULL, success=NULL, error=NULL
                        WHERE task=? AND mode=? AND status='failed'
                        """,
                        (utc_now(), task, mode),
                    )
                else:
                    if case_root.exists():
                        archive = rollout_attempt_archive_path(
                            self.run_root,
                            {"task": task, "mode": mode, "attempt": row["attempt"]},
                            label="manual-retry",
                        )
                        archive.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(case_root), str(archive))
                    cursor = connection.execute(
                        """
                        UPDATE cases
                        SET status='queued', updated_at=?, started_at=NULL,
                            rollout_finished_at=NULL, reward_started_at=NULL,
                            finished_at=NULL, quality_score=NULL, success=NULL, error=NULL
                        WHERE task=? AND mode=? AND status IN ('done','failed')
                        """,
                        (utc_now(), task, mode),
                    )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        f"retry state changed unexpectedly for {task}/{mode}"
                    )
                connection.commit()
                retried.append(task)
            except Exception:
                connection.rollback()
                # A same-filesystem rename is atomic.  If anything after it
                # fails, restore the terminal attempt before surfacing the
                # error so a later retry remains safe.
                if (
                    archive is not None
                    and archive.exists()
                    and not case_root.exists()
                ):
                    case_root.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(archive), str(case_root))
                raise
            finally:
                connection.close()
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
                refreshed = connection.execute(
                    f"SELECT * FROM cases WHERE id IN ({placeholders})",
                    ids,
                ).fetchall()
                refreshed_by_id = {int(row["id"]): row for row in refreshed}
                rows = [refreshed_by_id[case_id] for case_id in ids]
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

    def mark_rollout_retry(self, case_id: int, error: BaseException | str) -> bool:
        """Return a failed infrastructure rollout to the rollout queue.

        The attempt counter is intentionally retained.  It is incremented only
        when the case is claimed again, which makes it a durable retry budget
        across worker restarts.
        """
        now = utc_now()
        message = str(error)
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE cases SET status='queued', updated_at=?, started_at=NULL,
                    rollout_finished_at=NULL, reward_started_at=NULL,
                    finished_at=NULL, quality_score=NULL, success=NULL, error=?
                WHERE id=? AND status IN ('rollout','rewarding')
                """,
                (now, message[-4000:], case_id),
            ).rowcount
        return bool(updated)

    def mark_done(self, case_id: int, result: dict[str, Any]) -> None:
        if reward_result_error(result) is not None:
            raise ValueError("cannot mark a case done without a valid reward score")
        score = reward_numeric_score(result)
        assert score is not None
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
                    score,
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
                    quality_score=NULL, success=0, error=? WHERE id=?
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


def rollout_result_error(artifact: Any) -> tuple[str, bool] | None:
    """Return ``(reason, retryable)`` for non-scorable rollout outcomes."""
    agent = getattr(artifact, "agent", None)
    if not isinstance(agent, dict):
        return None
    outcome = str(agent.get("rollout_outcome") or "")
    infrastructure = bool(agent.get("rollout_infrastructure"))
    if outcome not in {"infra_error", "harness_error"} and not infrastructure:
        return None
    reason = str(
        agent.get("workspace_download_error")
        or agent.get("error")
        or agent.get("failure_reason")
        or outcome
        or "rollout infrastructure failure"
    )
    # Harness failures are terminal even if an older result accidentally set
    # rollout_infrastructure=true.  Only an explicitly retryable infrastructure
    # outcome is allowed to consume another rollout attempt.
    retryable = bool(
        outcome == "infra_error" and agent.get("rollout_retryable")
    )
    return reason, retryable


def retryable_rollout_exception(error: BaseException) -> bool:
    """Conservatively identify transient infrastructure exceptions.

    Explicit structured metadata wins.  Otherwise only network/timeout error
    types, transient network errno values, and well-known infrastructure
    messages are accepted.  Broad ``OSError`` and programming exceptions are
    intentionally not retried.
    """
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    transient_errnos = {
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.EPIPE,
        errno.ETIMEDOUT,
    }
    markers = (
        "connection reset",
        "connection refused",
        "connection error",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "rate limit",
        "too many requests",
        "sandbox unavailable",
        "deployment unavailable",
        "ags backend",
        "image is still preparing",
        "pending request was cancelled",
    )

    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (ConnectionError, TimeoutError)):
            return True
        if isinstance(current, OSError) and current.errno in transient_errnos:
            return True

        metadata: list[Any] = [current]
        metadata.extend(
            getattr(current, name, None)
            for name in ("payload", "details", "metadata", "result")
        )
        for value in metadata:
            if isinstance(value, dict):
                retryable = value.get("retryable") is True
                infrastructure = bool(
                    value.get("is_infrastructure")
                    or value.get("infrastructure")
                    or value.get("failure_domain") == "infrastructure"
                    or value.get("rollout_outcome") == "infra_error"
                )
            else:
                retryable = getattr(value, "retryable", None) is True
                infrastructure = bool(
                    getattr(value, "is_infrastructure", False)
                    or getattr(value, "infrastructure", False)
                    or getattr(value, "failure_domain", None) == "infrastructure"
                    or getattr(value, "rollout_outcome", None) == "infra_error"
                )
                class_name = type(value).__name__.casefold()
                infrastructure = infrastructure or (
                    retryable and "infra" in class_name
                )
            if retryable and infrastructure:
                return True

        if any(marker in str(current).casefold() for marker in markers):
            return True
        for nested in (
            current.__cause__,
            current.__context__,
            getattr(current, "cause", None),
            getattr(current, "original_error", None),
        ):
            if isinstance(nested, BaseException):
                pending.append(nested)
        pending.extend(arg for arg in current.args if isinstance(arg, BaseException))
    return False


def reward_numeric_score(result: dict[str, Any]) -> int | float | None:
    for key in ("code_quality_score", "quality_score"):
        value = result.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def reward_result_error(result: Any) -> str | None:
    """Explain why a reward result cannot be a successful queue terminal state."""
    if not isinstance(result, dict):
        return "reward returned a non-object result"
    score = reward_numeric_score(result)
    numeric_score = score is not None
    if "reward_outcome" not in result and "reward_score_valid" not in result:
        return None if numeric_score else "reward did not return a numeric quality score"
    outcome = str(result.get("reward_outcome") or "pending")
    score_valid = result.get("reward_score_valid")
    if score_valid is None:
        score_valid = outcome == "scored" and numeric_score
    if outcome == "scored" and bool(score_valid) and numeric_score:
        return None
    hidden = result.get("hidden_tests")
    detail = ""
    if isinstance(hidden, dict):
        detail = str(hidden.get("error") or hidden.get("skip_reason") or "")
    reason = f"reward_outcome={outcome}; reward_score_valid={bool(score_valid)}"
    return f"{reason}: {detail}" if detail else reason


def rollout_attempt_archive_path(
    run_root: Path,
    case: dict[str, Any],
    *,
    label: str = "infra-retry",
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (
        run_root
        / "_attempts"
        / str(case["task"])
        / str(case["mode"])
        / f"attempt-{int(case['attempt'])}-{label}-{stamp}"
    )


def archive_rollout_attempt(
    run_root: Path,
    case: dict[str, Any],
    *,
    label: str = "infra-retry",
) -> Path | None:
    """Archive one rollout attempt before its workspace is generated again."""
    case_root = run_root / str(case["task"]) / str(case["mode"])
    if not case_root.exists():
        return None
    archive = rollout_attempt_archive_path(run_root, case, label=label)
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(case_root), str(archive))
    return archive


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
    """Fairly allocate one pool while preserving already-active work.

    Existing active work consumes capacity first because shrinking a desired
    concurrency does not preempt an in-flight rollout/reward.  Remaining slots
    water-fill the least-allocated run that still has pending demand, so one
    large first queue cannot starve every later run.
    """
    if capacity < 0:
        raise ValueError("global capacity must be non-negative")
    active = [max(0, int(counts.get(active_key, 0))) for counts in snapshots]
    demand = [
        active[index] + max(0, int(counts.get(pending_key, 0)))
        for index, counts in enumerate(snapshots)
    ]

    if sum(active) <= capacity:
        allocations = active.copy()
        remaining = capacity - sum(allocations)
        limits = demand
    else:
        # The supervisor may inherit more active work than its new capacity.
        # It cannot preempt that work, but lower fair desired values prevent any
        # run from claiming replacements until the aggregate drains.
        allocations = [0] * len(snapshots)
        remaining = capacity
        limits = active

    while remaining:
        eligible = [
            index
            for index, limit in enumerate(limits)
            if allocations[index] < limit
        ]
        if not eligible:
            break
        target = min(eligible, key=lambda index: (allocations[index], index))
        allocations[target] += 1
        remaining -= 1
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
    rollout_attempts: int = 3,
    failure_fn: Callable[[dict[str, Any], str, str, Exception], dict[str, Any]] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    on_resize: Callable[[dict[str, int]], None] | None = None,
) -> None:
    """Keep both pools full while cases can be appended from another process."""
    if rollout_concurrency < 0 or reward_concurrency < 0:
        raise ValueError("rollout and reward concurrency must be non-negative")
    if rollout_attempts < 1:
        raise ValueError("rollout attempts must be positive")
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
                    rollout_error = rollout_result_error(artifact)
                    if rollout_error is None:
                        store.mark_rollout_complete(int(case["id"]))
                        emit(
                            "rollout.completed", task, str(case["mode"]),
                            queue_id=case["id"],
                        )
                    else:
                        reason, retryable = rollout_error
                        attempt = int(case["attempt"])
                        if retryable and attempt < rollout_attempts:
                            archive_rollout_attempt(store.run_root, case)
                            store.mark_rollout_retry(int(case["id"]), reason)
                            emit(
                                "rollout.requeued", task, str(case["mode"]),
                                queue_id=case["id"], attempt=attempt,
                                max_attempts=rollout_attempts, error=reason,
                            )
                        else:
                            store.mark_failed(int(case["id"]), reason)
                            emit(
                                "rollout.failed", task, str(case["mode"]),
                                queue_id=case["id"],
                                error_type=(
                                    "InfrastructureError" if retryable
                                    else "HarnessError"
                                ),
                                attempt=attempt, max_attempts=rollout_attempts,
                            )
                except Exception as exc:
                    if failure_fn is not None:
                        failure_fn(task, str(case["mode"]), "rollout", exc)
                    attempt = int(case["attempt"])
                    if (
                        retryable_rollout_exception(exc)
                        and attempt < rollout_attempts
                    ):
                        archive_rollout_attempt(
                            store.run_root, case, label="exception-retry"
                        )
                        store.mark_rollout_retry(int(case["id"]), exc)
                        emit(
                            "rollout.requeued", task, str(case["mode"]),
                            queue_id=case["id"], attempt=attempt,
                            max_attempts=rollout_attempts, error=str(exc),
                            error_type=type(exc).__name__,
                        )
                    else:
                        store.mark_failed(int(case["id"]), exc)
                        emit(
                            "rollout.failed", task, str(case["mode"]),
                            queue_id=case["id"], error_type=type(exc).__name__,
                            attempt=attempt, max_attempts=rollout_attempts,
                        )

            for future in [item for item in reward_futures if item.done()]:
                case, task = reward_futures.pop(future)
                try:
                    result = future.result()
                    invalid_reward = reward_result_error(result)
                    if invalid_reward is None:
                        store.mark_done(int(case["id"]), result)
                        emit(
                            "reward.completed", task, str(case["mode"]),
                            queue_id=case["id"], quality_score=result.get("quality_score"),
                            success=bool(result.get("success")),
                        )
                    else:
                        store.mark_failed(int(case["id"]), invalid_reward)
                        emit(
                            "reward.failed", task, str(case["mode"]),
                            queue_id=case["id"], error_type="InvalidRewardResult",
                            reward_outcome=result.get("reward_outcome"),
                            retryable=bool(result.get("retryable")),
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
    serve.add_argument("--rollout-attempts", type=int, default=3)
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
    enforce_serve_launch_policy(args)
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
    if args.rollout_attempts < 1:
        raise SystemExit("rollout attempts must be positive")
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
            "rollout_attempts": args.rollout_attempts,
            "reward_attempts": args.reward_attempts,
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
    start_supervisor_lease_watchdog(run_root, stop_event)
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
        rollout_attempts=args.rollout_attempts,
        failure_fn=failed,
        on_event=event,
        on_resize=resized,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
