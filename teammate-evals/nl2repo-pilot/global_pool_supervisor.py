#!/usr/bin/env python3
"""Run several NL2Repo queues through shared global rollout and reward pools."""

from __future__ import annotations

import argparse
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
            env=os.environ.copy(),
        )

    def restart_if_needed(self) -> bool:
        if self.process is not None and self.process.poll() is None:
            return False
        if self.log_handle is not None:
            self.log_handle.close()
        self.start()
        return True

    def stop(self, timeout_s: float = 30.0) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


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
    parser.add_argument("--reward-retry-delay", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rollout_capacity < 1 or args.reward_capacity < 1:
        raise SystemExit("global capacities must be positive")
    if args.worker_capacity < max(args.rollout_capacity, args.reward_capacity):
        raise SystemExit("worker capacity must cover each global pool")
    if args.poll_interval <= 0:
        raise SystemExit("poll interval must be positive")

    run_roots = [path.expanduser().resolve() for path in args.run]
    common_root = Path(os.path.commonpath([str(path.parent) for path in run_roots]))
    state_path = common_root / "global-pool-state.json"
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
        ManagedWorker(path, build_worker_command(args, path)) for path in run_roots
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
    # Give every worker time to recover stale per-run states before allocating.
    time.sleep(min(2.0, args.poll_interval))

    previous: tuple[tuple[int, int], ...] | None = None
    try:
        while not stopping:
            for index, worker in enumerate(workers):
                if worker.restart_if_needed():
                    stores[index].set_concurrency(rollout=0, reward=0)
                    print(f"restarted worker: {worker.run_root.name}", flush=True)

            snapshots = [store.counts() for store in stores]
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
                "status": "running",
                "updated_at": queue.utc_now(),
                "pid": os.getpid(),
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
            if allocation != previous:
                for store, rollout_slots, reward_slots in zip(
                    stores, rollout, reward, strict=True
                ):
                    store.set_concurrency(
                        rollout=rollout_slots,
                        reward=reward_slots,
                    )
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
                previous = allocation

            if all(
                not any(
                    counts[status]
                    for status in ("queued", "rollout", "reward_pending", "rewarding")
                )
                for counts in snapshots
            ):
                print("all global queues completed", flush=True)
                pool_state["status"] = "completed"
                pool_state["updated_at"] = queue.utc_now()
                write_pool_state(state_path, pool_state)
                break
            time.sleep(args.poll_interval)
    finally:
        for store in stores:
            try:
                store.set_concurrency(rollout=0, reward=0)
            except Exception:
                pass
        for worker in workers:
            worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
