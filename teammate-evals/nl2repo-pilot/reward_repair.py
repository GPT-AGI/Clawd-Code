#!/usr/bin/env python3
"""Repair NL2Repo reward infrastructure failures without rerunning agents."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BENCHMARK_PATH = HERE / "benchmark.py"
SPEC = importlib.util.spec_from_file_location("nl2repo_reward_repair_benchmark", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def discover_infrastructure_failures(
    run_root: Path,
) -> list[tuple[str, str, str]]:
    failures: list[tuple[str, str, str]] = []
    for path in sorted(run_root.glob("*/*/result.json")):
        result = read_json(path)
        if result is None:
            continue
        hidden = result.get("hidden_tests")
        error = str(hidden.get("error") or "") if isinstance(hidden, dict) else ""
        if error:
            failures.append((path.parent.parent.name, path.parent.name, error))
    return failures


def refresh_aggregate(run_root: Path) -> bool:
    """Replace stale in-memory scheduler results with repaired per-case results."""
    aggregate_path = run_root / "results.json"
    aggregate = read_json(aggregate_path)
    if aggregate is None or not isinstance(aggregate.get("results"), list):
        return False
    refreshed: list[dict[str, Any]] = []
    for old_result in aggregate["results"]:
        if not isinstance(old_result, dict):
            continue
        task = str(old_result.get("task") or "")
        mode = str(old_result.get("mode") or "adaptive")
        current = read_json(run_root / task / mode / "result.json")
        refreshed.append(current or old_result)
    aggregate["results"] = refreshed
    benchmark._write_json(aggregate_path, aggregate)
    report = benchmark.render_report(
        refreshed,
        str(aggregate.get("run_id") or run_root.name),
        str(aggregate.get("upstream_ref") or benchmark.UPSTREAM_REF),
    )
    (run_root / "REPORT.md").write_text(report, encoding="utf-8")
    return True


def repair_case(
    run_root: Path,
    upstream_root: Path,
    task_name: str,
    mode: str,
    *,
    score_backend: str,
    score_timeout_s: float,
    keep_image: bool,
) -> dict[str, Any]:
    task = benchmark.load_task(upstream_root, task_name)
    return benchmark.rescore_existing_case(
        task,
        mode,
        run_root,
        score_backend=score_backend,
        score_timeout_s=score_timeout_s,
        keep_image=keep_image,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--score-backend", choices=("docker",), default="docker")
    parser.add_argument("--score-timeout", type=float, default=1200)
    parser.add_argument("--keep-image", action="store_true")
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise SystemExit(
        "direct reward_repair.py scoring pools are disabled. Requeue reusable "
        "rollouts and run global_pool_supervisor.py with --rollout-capacity 0 "
        "and the desired --reward-capacity."
    )
    # Kept below for import-level migration tooling; the production CLI cannot
    # reach this legacy private ThreadPoolExecutor path.
    if args.concurrency < 1 or args.max_attempts < 1 or args.poll_interval < 0:
        raise SystemExit("repair concurrency/attempts must be positive")
    run_root = args.run.expanduser().resolve()
    if not run_root.is_dir():
        raise SystemExit(f"run directory not found: {run_root}")
    upstream_root = benchmark.resolve_upstream(
        args.upstream_root, cache_root=args.cache_root
    )
    attempts: dict[tuple[str, str], int] = {}

    while True:
        failures = discover_infrastructure_failures(run_root)
        eligible = [
            failure
            for failure in failures
            if attempts.get((failure[0], failure[1]), 0) < args.max_attempts
        ]
        if eligible:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {}
                for task, mode, error in eligible:
                    key = (task, mode)
                    attempts[key] = attempts.get(key, 0) + 1
                    print(
                        f"[{task}] repairing {mode} reward "
                        f"({attempts[key]}/{args.max_attempts}): {error}",
                        flush=True,
                    )
                    future = pool.submit(
                        repair_case,
                        run_root,
                        upstream_root,
                        task,
                        mode,
                        score_backend=args.score_backend,
                        score_timeout_s=args.score_timeout,
                        keep_image=args.keep_image,
                    )
                    futures[future] = key
                for future in as_completed(futures):
                    task, mode = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        print(f"[{task}] repair raised: {type(exc).__name__}: {exc}", flush=True)
                        continue
                    hidden = result.get("hidden_tests")
                    error = hidden.get("error") if isinstance(hidden, dict) else None
                    tests = hidden.get("pytest", {}) if isinstance(hidden, dict) else {}
                    print(
                        f"[{task}] reward repaired: quality={result.get('quality_score', 0):.2f} "
                        f"passed={tests.get('passed', 0)}/{tests.get('expected', 0)} "
                        f"infra_error={error or '-'}",
                        flush=True,
                    )

        remaining = discover_infrastructure_failures(run_root)
        completed = (run_root / "results.json").is_file()
        if not args.watch or (completed and not remaining):
            if completed and not remaining:
                refresh_aggregate(run_root)
            return 0 if not remaining else 2
        exhausted = all(
            attempts.get((task, mode), 0) >= args.max_attempts
            for task, mode, _ in remaining
        )
        if completed and remaining and exhausted:
            print(f"unresolved reward infrastructure failures: {remaining}", flush=True)
            return 2
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
