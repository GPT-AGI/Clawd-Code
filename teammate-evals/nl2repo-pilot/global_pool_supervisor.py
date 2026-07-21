#!/usr/bin/env python3
"""Run several NL2Repo queues through shared global rollout and reward pools."""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluation_queue as queue


GLOBAL_POOL_ROOT = Path(__file__).resolve().parent / "runs"


class GlobalPoolLock:
    """Process-wide lease preventing multiple supervisors from allocating slots.

    The lock file is intentionally persistent: deleting a flock file while a
    process owns it can split future contenders across different inodes.  Its
    contents are only diagnostic; the kernel lock is authoritative.
    """

    def __init__(self, path: Path, run_roots: list[Path]) -> None:
        self.path = path
        self.run_roots = run_roots
        self._handle: Any = None
        self._metadata: dict[str, Any] = {}

    def _write_metadata(self) -> None:
        handle = self._handle
        if handle is None:
            raise RuntimeError(f"global pool lock is not acquired: {self.path}")
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(self._metadata, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def acquire(self) -> None:
        if self._handle is not None:
            raise RuntimeError(f"global pool lock is already acquired: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                handle.close()
                raise
            try:
                handle.seek(0)
                owner = handle.read(4096).strip() or "owner metadata unavailable"
            finally:
                handle.close()
            raise RuntimeError(
                f"another global pool supervisor already owns {self.path}: {owner}"
            ) from exc
        metadata = {
            "schema_version": 2,
            "pid": os.getpid(),
            "acquired_at": queue.utc_now(),
            # Full resolved paths are part of the worker-launch authorization.
            # Names alone are ambiguous when independent directories contain
            # runs with the same basename.
            "runs": [str(root.expanduser().resolve()) for root in self.run_roots],
            "worker_pids": [],
        }
        self._handle = handle
        self._metadata = metadata
        try:
            self._write_metadata()
        except BaseException:
            self.release()
            raise

    def update_worker_pids(self, workers: list["ManagedWorker"]) -> None:
        """Publish the only queue-worker parents allowed to spawn agent children."""
        self._metadata["worker_pids"] = [
            int(worker.process.pid)
            for worker in workers
            if worker.process is not None and worker.process.poll() is None
        ]
        self._metadata["updated_at"] = queue.utc_now()
        self._write_metadata()

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        self._metadata = {}
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "GlobalPoolLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


def build_worker_environment() -> dict[str, str]:
    """Mark a queue worker as managed by this global supervisor."""
    environment = os.environ.copy()
    environment[queue.GLOBAL_POOL_WORKER_ENV] = queue.GLOBAL_POOL_WORKER_MARKER
    return environment


def validate_pool_capacities(
    rollout_capacity: int,
    reward_capacity: int,
    worker_capacity: int,
) -> None:
    """Allow a disabled stage, but never a supervisor with no usable pool."""
    if rollout_capacity < 0 or reward_capacity < 0:
        raise ValueError("global capacities must be non-negative")
    if rollout_capacity == 0 and reward_capacity == 0:
        raise ValueError("at least one global capacity must be positive")
    if worker_capacity < max(rollout_capacity, reward_capacity):
        raise ValueError("worker capacity must cover each global pool")


def enabled_pool_has_work(
    snapshots: list[dict[str, int]],
    *,
    rollout_capacity: int,
    reward_capacity: int,
) -> bool:
    """Whether any status serviced by an enabled pool still has work."""
    statuses: list[str] = []
    if rollout_capacity > 0:
        statuses.extend(("queued", "rollout"))
    if reward_capacity > 0:
        statuses.extend(("reward_pending", "rewarding"))
    return any(counts.get(status, 0) for counts in snapshots for status in statuses)


def write_pool_state(path: Path, value: dict[str, Any]) -> None:
    """Publish a dashboard-readable snapshot without exposing a partial write."""
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


@dataclass
class ManagedWorker:
    run_root: Path
    command: list[str]
    process: subprocess.Popen[str] | None = None
    log_handle: Any = None

    def start(self) -> None:
        self.log_handle = (self.run_root / "worker.log").open(
            "a", encoding="utf-8"
        )
        self.process = subprocess.Popen(
            self.command,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=build_worker_environment(),
            start_new_session=True,
        )

    def needs_restart(self) -> bool:
        return self.process is None or self.process.poll() is not None

    def restart(self) -> None:
        if self.log_handle is not None:
            self.log_handle.close()
        self.start()

    def stop(self, timeout_s: float = 30.0) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


def restart_worker_safely(worker: ManagedWorker, store: queue.QueueStore) -> bool:
    """Restart a dead worker without exposing its previously assigned slots.

    ``worker_config`` survives a child-process crash.  Freeze it before spawning
    the replacement so the new queue loop cannot claim work under a stale global
    allocation during startup recovery.
    """
    if not worker.needs_restart():
        return False
    store.set_concurrency(rollout=0, reward=0)
    worker.restart()
    return True


def reconcile_worker_concurrency(
    stores: list[queue.QueueStore],
    allocation: tuple[tuple[int, int], ...],
) -> list[int]:
    """Make persisted per-run slots match the global allocation.

    Reconcile against the database every supervisor tick, rather than only
    against the previous calculated tuple.  This restores slots after a worker
    restart and promptly corrects an out-of-band ``evaluation_queue scale``.
    """
    corrected: list[int] = []
    for index, (store, (rollout_slots, reward_slots)) in enumerate(
        zip(stores, allocation, strict=True)
    ):
        configured = store.concurrency()
        current = (
            int(configured["rollout_concurrency"]),
            int(configured["reward_concurrency"]),
        ) if configured is not None else None
        desired = (rollout_slots, reward_slots)
        if current == desired:
            continue
        store.set_concurrency(rollout=rollout_slots, reward=reward_slots)
        corrected.append(index)
    return corrected


def build_worker_command(args: argparse.Namespace, run_root: Path) -> list[str]:
    script = Path(__file__).with_name("evaluation_queue.py")
    command = [
        sys.executable,
        str(script),
        "--run",
        str(run_root),
        "serve",
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--max-turns",
        str(args.max_turns),
        "--teammate-max-turns",
        str(args.teammate_max_turns),
        "--teammate-min-timeout",
        str(args.teammate_min_timeout),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--agent-timeout",
        str(args.agent_timeout),
        "--score-timeout",
        str(args.score_timeout),
        "--rollout-concurrency",
        "0",
        "--reward-concurrency",
        "0",
        "--max-rollout-concurrency",
        str(args.worker_capacity),
        "--max-reward-concurrency",
        str(args.worker_capacity),
        "--execution-backend",
        "ags",
        "--score-backend",
        "ags",
        "--ags-env-file",
        str(args.ags_env_file),
        "--ags-timeout",
        args.ags_timeout,
        "--ags-cpu",
        args.ags_cpu,
        "--ags-memory",
        args.ags_memory,
        "--reward-attempts",
        str(args.reward_attempts),
        "--rollout-attempts",
        str(args.rollout_attempts),
        "--reward-retry-delay",
        str(args.reward_retry_delay),
    ]
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--rollout-capacity", type=int, default=32)
    parser.add_argument("--reward-capacity", type=int, default=32)
    parser.add_argument("--worker-capacity", type=int, default=64)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--provider", default="qwen")
    parser.add_argument("--model", default="ms-rns547kc")
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--teammate-max-turns", type=int, default=160)
    parser.add_argument("--teammate-min-timeout", type=float, default=900.0)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--agent-timeout", type=float, default=7200.0)
    parser.add_argument("--score-timeout", type=float, default=1200.0)
    parser.add_argument("--ags-env-file", type=Path, required=True)
    parser.add_argument("--ags-timeout", default="3h")
    parser.add_argument("--ags-cpu", default="2")
    parser.add_argument("--ags-memory", default="4Gi")
    parser.add_argument("--reward-attempts", type=int, default=3)
    parser.add_argument("--rollout-attempts", type=int, default=3)
    parser.add_argument("--reward-retry-delay", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_pool_capacities(
            args.rollout_capacity,
            args.reward_capacity,
            args.worker_capacity,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.poll_interval <= 0:
        raise SystemExit("poll interval must be positive")

    run_roots = [path.expanduser().resolve() for path in args.run]
    if len(set(run_roots)) != len(run_roots):
        raise SystemExit("each --run must identify a distinct queue")
    pool_root = GLOBAL_POOL_ROOT.expanduser().resolve()
    state_path = pool_root / "global-pool-state.json"
    pool_lock = GlobalPoolLock(pool_root / "global-pool.lock", run_roots)
    try:
        pool_lock.acquire()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    stores: list[queue.QueueStore] = []
    workers: list[ManagedWorker] = []
    try:
        stores = [queue.QueueStore(path) for path in run_roots]
        for store in stores:
            store.initialize_concurrency(
                0,
                0,
                max_rollout=args.worker_capacity,
                max_reward=args.worker_capacity,
            )
            store.set_concurrency(rollout=0, reward=0)

        workers = [
            ManagedWorker(path, build_worker_command(args, path))
            for path in run_roots
        ]
        stopping = False

        def request_stop(signum: int, frame: Any) -> None:
            nonlocal stopping
            stopping = True
            print(f"received signal {signum}; stopping global workers", flush=True)

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        os.environ.setdefault("QWEN_ENABLE_THINKING", "1")
        os.environ.setdefault("AGS_SCORE_SETUP_CONCURRENCY", "8")

        for worker in workers:
            worker.start()
        pool_lock.update_worker_pids(workers)
        # Give every worker time to recover stale per-run states before allocating.
        time.sleep(min(2.0, args.poll_interval))
    except BaseException:
        try:
            for store in stores:
                try:
                    store.set_concurrency(rollout=0, reward=0)
                except Exception:
                    pass
            for worker in workers:
                worker.stop()
        finally:
            pool_lock.release()
        raise

    previous: tuple[tuple[int, int], ...] | None = None
    try:
        while not stopping:
            workers_restarted = False
            for index, worker in enumerate(workers):
                if restart_worker_safely(worker, stores[index]):
                    workers_restarted = True
                    print(f"restarted worker: {worker.run_root.name}", flush=True)
            if workers_restarted:
                pool_lock.update_worker_pids(workers)

            snapshots = [store.counts() for store in stores]
            has_work = enabled_pool_has_work(
                snapshots,
                rollout_capacity=args.rollout_capacity,
                reward_capacity=args.reward_capacity,
            )
            rollout = queue.allocate_global_slots(
                snapshots,
                args.rollout_capacity,
                active_key="rollout",
                pending_key="queued",
            )
            reward = queue.allocate_global_slots(
                snapshots,
                args.reward_capacity,
                active_key="rewarding",
                pending_key="reward_pending",
            )
            allocation = tuple(zip(rollout, reward, strict=True))
            pool_state = {
                "status": "running" if has_work else "idle",
                "updated_at": queue.utc_now(),
                "pid": os.getpid(),
                "worker_pids": [
                    int(worker.process.pid)
                    for worker in workers
                    if worker.process is not None and worker.process.poll() is None
                ],
                "rollout_capacity": args.rollout_capacity,
                "reward_capacity": args.reward_capacity,
                "runs": [
                    {
                        "run": root.name,
                        "rollout_slots": rollout_slots,
                        "reward_slots": reward_slots,
                        "counts": counts,
                    }
                    for root, rollout_slots, reward_slots, counts in zip(
                        run_roots, rollout, reward, snapshots, strict=True
                    )
                ],
            }
            write_pool_state(state_path, pool_state)
            corrected = reconcile_worker_concurrency(stores, allocation)
            if allocation != previous:
                print(
                    json.dumps(
                        {
                            "event": "global_pool.allocated",
                            "rollout_capacity": args.rollout_capacity,
                            "reward_capacity": args.reward_capacity,
                            "runs": [
                                {
                                    "run": root.name,
                                    "rollout_slots": rollout_slots,
                                    "reward_slots": reward_slots,
                                    "counts": counts,
                                }
                                for root, rollout_slots, reward_slots, counts in zip(
                                    run_roots,
                                    rollout,
                                    reward,
                                    snapshots,
                                    strict=True,
                                )
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            elif corrected:
                print(
                    json.dumps(
                        {
                            "event": "global_pool.reconciled",
                            "runs": [run_roots[index].name for index in corrected],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            previous = allocation

            time.sleep(args.poll_interval)
    finally:
        try:
            for store in stores:
                try:
                    store.set_concurrency(rollout=0, reward=0)
                except Exception:
                    pass
            for worker in workers:
                worker.stop()
        finally:
            pool_lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
