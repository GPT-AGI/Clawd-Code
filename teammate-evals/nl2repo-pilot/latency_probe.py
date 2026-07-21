#!/usr/bin/env python3
"""Measure Qwen latency on a deterministic NL2Repo prompt subset.

This probe deliberately does not run agents or hidden tests.  It sends the same
real ``start.md`` documents once serially and once at the requested concurrency
so model-service queueing can be separated from sandbox and tool overhead.
Prompt contents and credentials are never written to the result artifacts.
"""

from __future__ import annotations

import argparse
import csv
import errno
import fcntl
import json
import math
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import get_provider_config  # noqa: E402
from src.providers.qwen_provider import QwenProvider  # noqa: E402


DEFAULT_SEED = 20260715
DEFAULT_MAX_PROMPT_BYTES = 64 * 1024
PINNED_REVISION = "781a1da1ee41fb8edb0bed22f586d69111610edf"
GLOBAL_POOL_LOCK_PATH = Path(__file__).resolve().parent / "runs" / "global-pool.lock"


@dataclass(frozen=True)
class TaskPrompt:
    name: str
    path: Path
    prompt_bytes: int
    difficulty: str


@dataclass
class RequestResult:
    task: str
    prompt_bytes: int
    difficulty: str
    success: bool
    ttft_seconds: float | None
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    output_chars: int
    finish_reason: str | None
    error_type: str | None = None
    error_status: int | None = None
    error_message: str | None = None


def global_pool_is_active(lock_path: Path | None = None) -> bool:
    """Probe the persistent global-pool lock without modifying it."""
    path = (lock_path or GLOBAL_POOL_LOCK_PATH).expanduser().resolve()
    if not path.is_file():
        return False
    try:
        handle = path.open("r+", encoding="utf-8")
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return True
            raise
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


def reject_if_global_pool_active(lock_path: Path | None = None) -> None:
    if global_pool_is_active(lock_path):
        raise SystemExit(
            "latency_probe.py is disabled while the NL2Repo global pool is active; "
            "wait for the evaluator to stop so the probe cannot overcommit model capacity."
        )


def find_upstream_root(explicit: str | None = None) -> Path:
    """Find an NL2Repo-Bench checkout containing ``test_files``."""
    if explicit:
        candidates = [Path(explicit).expanduser()]
    else:
        cache = Path.home() / ".cache" / "clawd-code" / "nl2repo-bench"
        candidates = [
            cache / PINNED_REVISION[:12],
            cache / PINNED_REVISION,
            cache,
        ]
        if cache.is_dir():
            candidates.extend(sorted(cache.iterdir(), reverse=True))

    for candidate in candidates:
        if (candidate / "test_files" / "task_difficulty.csv").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "NL2Repo-Bench checkout not found; pass --upstream-root or run benchmark.py --list"
    )


def load_tasks(upstream_root: Path) -> list[TaskPrompt]:
    test_files = upstream_root / "test_files"
    difficulties: dict[str, str] = {}
    with (test_files / "task_difficulty.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("task-name") or row.get("task") or "").strip()
            if name:
                difficulties[name] = (row.get("Level") or row.get("difficulty") or "").strip()

    tasks: list[TaskPrompt] = []
    for start_path in sorted(test_files.glob("*/start.md")):
        tasks.append(
            TaskPrompt(
                name=start_path.parent.name,
                path=start_path,
                prompt_bytes=start_path.stat().st_size,
                difficulty=difficulties.get(start_path.parent.name, ""),
            )
        )
    if not tasks:
        raise ValueError(f"no NL2Repo start.md files found under {test_files}")
    return tasks


def select_tasks(
    tasks: Iterable[TaskPrompt],
    subset_size: int,
    seed: int,
    max_prompt_bytes: int,
) -> tuple[list[TaskPrompt], int]:
    """Select a stable random subset after applying a prompt-size ceiling."""
    eligible = sorted(
        (task for task in tasks if task.prompt_bytes <= max_prompt_bytes),
        key=lambda task: task.name,
    )
    if subset_size < 1:
        raise ValueError("subset size must be positive")
    if len(eligible) < subset_size:
        raise ValueError(
            f"only {len(eligible)} tasks are <= {max_prompt_bytes} bytes; "
            f"cannot select {subset_size}"
        )
    selected = random.Random(seed).sample(eligible, subset_size)
    return sorted(selected, key=lambda task: task.name), len(eligible)


def percentile(values: Iterable[float], percent: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("mean", "p50", "p95", "p99", "max")}
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def summarize(results: list[RequestResult], duration_seconds: float) -> dict[str, Any]:
    successes = [result for result in results if result.success]
    errors: dict[str, int] = {}
    for result in results:
        if not result.success:
            label = result.error_type or "UnknownError"
            if result.error_status is not None:
                label += f":{result.error_status}"
            errors[label] = errors.get(label, 0) + 1
    return {
        "requests": len(results),
        "successes": len(successes),
        "errors": len(results) - len(successes),
        "error_breakdown": errors,
        "duration_seconds": duration_seconds,
        "requests_per_second": len(successes) / duration_seconds if duration_seconds else 0,
        "input_tokens": sum(result.input_tokens for result in successes),
        "input_tokens_per_second": (
            sum(result.input_tokens for result in successes) / duration_seconds
            if duration_seconds
            else 0
        ),
        "output_tokens": sum(result.output_tokens for result in successes),
        "output_tokens_per_second": (
            sum(result.output_tokens for result in successes) / duration_seconds
            if duration_seconds
            else 0
        ),
        "ttft_seconds": _distribution(
            [result.ttft_seconds for result in successes if result.ttft_seconds is not None]
        ),
        "latency_seconds": _distribution(
            [result.latency_seconds for result in successes]
        ),
    }


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def compare_runs(
    baseline: dict[str, Any],
    concurrent: dict[str, Any],
    baseline_concurrency: int,
    concurrency: int,
    baseline_results: list[RequestResult] | None = None,
    concurrent_results: list[RequestResult] | None = None,
) -> dict[str, Any]:
    scale = _safe_ratio(
        concurrent["requests_per_second"], baseline["requests_per_second"]
    )
    ratios: list[float] = []
    if baseline_results is not None and concurrent_results is not None:
        by_task = {
            result.task: result
            for result in baseline_results
            if result.success and result.latency_seconds > 0
        }
        for result in concurrent_results:
            base = by_task.get(result.task)
            if result.success and base is not None:
                ratios.append(result.latency_seconds / base.latency_seconds)

    return {
        "throughput_scale": scale,
        "ideal_concurrency_scale": concurrency / baseline_concurrency,
        "scaling_efficiency": (
            scale / (concurrency / baseline_concurrency) if scale is not None else None
        ),
        "ttft_p50_ratio": _safe_ratio(
            concurrent["ttft_seconds"]["p50"], baseline["ttft_seconds"]["p50"]
        ),
        "ttft_p95_ratio": _safe_ratio(
            concurrent["ttft_seconds"]["p95"], baseline["ttft_seconds"]["p95"]
        ),
        "latency_p50_ratio": _safe_ratio(
            concurrent["latency_seconds"]["p50"], baseline["latency_seconds"]["p50"]
        ),
        "latency_p95_ratio": _safe_ratio(
            concurrent["latency_seconds"]["p95"], baseline["latency_seconds"]["p95"]
        ),
        "paired_latency_ratio": _distribution(ratios),
    }


def _error_details(exc: Exception, secret: str) -> tuple[str, int | None, str]:
    status = getattr(exc, "status_code", None)
    message = str(exc).replace(secret, "[REDACTED]") if secret else str(exc)
    return type(exc).__name__, status if isinstance(status, int) else None, message[:500]


def request_once(
    client: Any,
    task: TaskPrompt,
    model: str,
    max_tokens: int,
    gate: threading.Event,
    secret: str,
) -> RequestResult:
    gate.wait()
    started = time.perf_counter()
    first_text_at: float | None = None
    usage: Any = None
    finish_reason: str | None = None
    output_parts: list[str] = []
    try:
        prompt = task.path.read_text(encoding="utf-8")
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "This is a latency benchmark. Read the repository specification and "
                        "reply with exactly one compact line: OK:<package-name>. Do not solve it."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        for chunk in stream:
            usage_candidate = getattr(chunk, "usage", None)
            if usage_candidate is not None:
                usage = usage_candidate
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = str(choice.finish_reason)
            delta = getattr(choice, "delta", None)
            text = getattr(delta, "content", None) if delta is not None else None
            if text:
                if first_text_at is None:
                    first_text_at = time.perf_counter()
                output_parts.append(str(text))
        finished = time.perf_counter()
        return RequestResult(
            task=task.name,
            prompt_bytes=task.prompt_bytes,
            difficulty=task.difficulty,
            success=True,
            ttft_seconds=(first_text_at - started) if first_text_at is not None else None,
            latency_seconds=finished - started,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            output_chars=sum(len(part) for part in output_parts),
            finish_reason=finish_reason,
        )
    except Exception as exc:
        finished = time.perf_counter()
        error_type, error_status, error_message = _error_details(exc, secret)
        return RequestResult(
            task=task.name,
            prompt_bytes=task.prompt_bytes,
            difficulty=task.difficulty,
            success=False,
            ttft_seconds=(first_text_at - started) if first_text_at is not None else None,
            latency_seconds=finished - started,
            input_tokens=0,
            output_tokens=0,
            output_chars=sum(len(part) for part in output_parts),
            finish_reason=finish_reason,
            error_type=error_type,
            error_status=error_status,
            error_message=error_message,
        )


def run_batch(
    client: Any,
    tasks: list[TaskPrompt],
    concurrency: int,
    model: str,
    max_tokens: int,
    secret: str,
) -> tuple[list[RequestResult], float]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    gate = threading.Event()
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(request_once, client, task, model, max_tokens, gate, secret)
            for task in tasks
        ]
        started = time.perf_counter()
        gate.set()
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            state = "ok" if result.success else f"error:{result.error_type}"
            print(
                f"[{completed:02d}/{len(tasks):02d}] {result.task:<24} "
                f"{state:<24} {result.latency_seconds:7.2f}s",
                flush=True,
            )
    duration = time.perf_counter() - started
    return sorted(results, key=lambda result: result.task), duration


def _fmt(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.3f}{suffix}"


def parse_concurrency_sweep(value: str) -> list[int]:
    """Parse an ordered, comma-separated set of positive concurrency levels."""
    try:
        levels = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sweep values must be integers") from exc
    if not levels or any(level < 1 for level in levels):
        raise argparse.ArgumentTypeError("sweep values must be positive")
    if len(set(levels)) != len(levels):
        raise argparse.ArgumentTypeError("sweep values must be unique")
    return levels


def write_report(path: Path, payload: dict[str, Any]) -> None:
    baseline = payload["runs"]["baseline"]["summary"]
    concurrent = payload["runs"]["concurrent"]["summary"]
    comparison = payload["comparison"]
    lines = [
        "# NL2Repo Qwen latency probe",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "This is a model-service latency probe, not a full agent or hidden-test evaluation. "
        "It uses the same fixed NL2Repo prompts in both runs, disables model thinking, "
        "streams a short answer, and performs no retry.",
        "",
        "## Configuration",
        "",
        f"- Model: `{payload['model']}`",
        (
            f"- Subset: {payload['subset']['selected']} of "
            f"{payload['subset']['eligible']} eligible tasks"
        ),
        f"- Seed: `{payload['subset']['seed']}`",
        f"- Prompt ceiling: {payload['subset']['max_prompt_bytes']} bytes",
        f"- Output limit: {payload['max_tokens']} tokens",
        "",
        "## Results",
        "",
        (
            "| Run | Concurrency | Success | Errors | Wall time | Req/s | "
            "TTFT p50 | TTFT p95 | Latency p50 | Latency p95 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary, concurrency in (
        ("baseline", baseline, payload["runs"]["baseline"]["concurrency"]),
        ("concurrent", concurrent, payload["runs"]["concurrent"]["concurrency"]),
    ):
        lines.append(
            f"| {label} | {concurrency} | {summary['successes']} | {summary['errors']} | "
            f"{_fmt(summary['duration_seconds'], 's')} | {_fmt(summary['requests_per_second'])} | "
            f"{_fmt(summary['ttft_seconds']['p50'], 's')} | "
            f"{_fmt(summary['ttft_seconds']['p95'], 's')} | "
            f"{_fmt(summary['latency_seconds']['p50'], 's')} | "
            f"{_fmt(summary['latency_seconds']['p95'], 's')} |"
        )
    efficiency = comparison["scaling_efficiency"]
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- Throughput scale: {_fmt(comparison['throughput_scale'], 'x')}",
            (
                "- Scaling efficiency: "
                f"{_fmt(efficiency * 100 if efficiency is not None else None, '%')}"
            ),
            (
                "- Aggregate input throughput: "
                f"{_fmt(baseline['input_tokens_per_second'], ' tok/s')} baseline; "
                f"{_fmt(concurrent['input_tokens_per_second'], ' tok/s')} concurrent"
            ),
            (
                "- TTFT p50 / p95 ratio: "
                f"{_fmt(comparison['ttft_p50_ratio'], 'x')} / "
                f"{_fmt(comparison['ttft_p95_ratio'], 'x')}"
            ),
            (
                "- Total latency p50 / p95 ratio: "
                f"{_fmt(comparison['latency_p50_ratio'], 'x')} / "
                f"{_fmt(comparison['latency_p95_ratio'], 'x')}"
            ),
            (
                "- Paired per-task latency ratio p50 / p95: "
                f"{_fmt(comparison['paired_latency_ratio']['p50'], 'x')} / "
                f"{_fmt(comparison['paired_latency_ratio']['p95'], 'x')}"
            ),
            "",
            "## Selected tasks",
            "",
            "| Task | Difficulty | Prompt bytes |",
            "|---|---|---:|",
        ]
    )
    for task in payload["tasks"]:
        lines.append(f"| {task['name']} | {task['difficulty'] or 'n/a'} | {task['prompt_bytes']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sweep_report(path: Path, payload: dict[str, Any]) -> None:
    baseline_level = str(payload["concurrency_levels"][0])
    lines = [
        "# NL2Repo Qwen concurrency sweep",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "This is a model-service latency probe, not a full agent or hidden-test "
        "evaluation. Every level uses the same fixed NL2Repo prompts, disables "
        "model thinking, streams a short answer, and performs no retry.",
        "",
        "## Configuration",
        "",
        f"- Model: `{payload['model']}`",
        (
            f"- Subset: {payload['subset']['selected']} of "
            f"{payload['subset']['eligible']} eligible tasks"
        ),
        f"- Seed: `{payload['subset']['seed']}`",
        f"- Prompt ceiling: {payload['subset']['max_prompt_bytes']} bytes",
        f"- Output limit: {payload['max_tokens']} tokens",
        "",
        "## Results",
        "",
        (
            "| Concurrency | Success | Errors | Wall time | Req/s | Input tok/s | "
            "TTFT p50 | TTFT p95 | Latency p50 | Latency p95 |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in payload["concurrency_levels"]:
        summary = payload["runs"][str(level)]["summary"]
        lines.append(
            f"| {level} | {summary['successes']} | {summary['errors']} | "
            f"{_fmt(summary['duration_seconds'], 's')} | "
            f"{_fmt(summary['requests_per_second'])} | "
            f"{_fmt(summary['input_tokens_per_second'])} | "
            f"{_fmt(summary['ttft_seconds']['p50'], 's')} | "
            f"{_fmt(summary['ttft_seconds']['p95'], 's')} | "
            f"{_fmt(summary['latency_seconds']['p50'], 's')} | "
            f"{_fmt(summary['latency_seconds']['p95'], 's')} |"
        )
    lines.extend(
        [
            "",
            f"## Scaling relative to concurrency {baseline_level}",
            "",
            "| Concurrency | Throughput scale | Efficiency | TTFT p95 ratio | Latency p95 ratio |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for level in payload["concurrency_levels"]:
        comparison = payload["comparisons"][str(level)]
        efficiency = comparison["scaling_efficiency"]
        lines.append(
            f"| {level} | {_fmt(comparison['throughput_scale'], 'x')} | "
            f"{_fmt(efficiency * 100 if efficiency is not None else None, '%')} | "
            f"{_fmt(comparison['ttft_p95_ratio'], 'x')} | "
            f"{_fmt(comparison['latency_p95_ratio'], 'x')} |"
        )
    lines.extend(
        [
            "",
            "## Selected tasks",
            "",
            "| Task | Difficulty | Prompt bytes |",
            "|---|---|---:|",
        ]
    )
    for task in payload["tasks"]:
        lines.append(
            f"| {task['name']} | {task['difficulty'] or 'n/a'} | "
            f"{task['prompt_bytes']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root")
    parser.add_argument("--subset-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-prompt-bytes", type=int, default=DEFAULT_MAX_PROMPT_BYTES)
    parser.add_argument("--baseline-concurrency", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument(
        "--sweep",
        type=parse_concurrency_sweep,
        help="comma-separated levels, for example 1,2,4,8; overrides the two-run mode",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run:
        raise SystemExit(
            "live latency_probe.py request pools are disabled under the "
            "global-pool-only policy; --dry-run remains available for planning"
        )
    upstream = find_upstream_root(args.upstream_root)
    all_tasks = load_tasks(upstream)
    tasks, eligible_count = select_tasks(
        all_tasks, args.subset_size, args.seed, args.max_prompt_bytes
    )
    print(
        f"Selected {len(tasks)} of {eligible_count} eligible tasks "
        f"(all={len(all_tasks)}, seed={args.seed}, ceiling={args.max_prompt_bytes} bytes)."
    )
    for task in tasks:
        print(f"  {task.name:<24} {task.prompt_bytes:>7} bytes {task.difficulty}")
    if args.dry_run:
        return 0

    provider_config = get_provider_config("qwen")
    api_key = str(provider_config.get("api_key") or "")
    if not api_key:
        raise RuntimeError("Qwen API key is not configured; run `clawd config --use qwen3.5`")
    model = str(provider_config.get("default_model") or QwenProvider.DEFAULT_MODEL)
    provider = QwenProvider(
        api_key=api_key,
        base_url=provider_config.get("base_url"),
        model=model,
    )
    client = provider.client.with_options(timeout=args.timeout, max_retries=0)

    warmup_task = min(tasks, key=lambda task: task.prompt_bytes)
    print(f"\nWarmup: {warmup_task.name}", flush=True)
    warmup, _ = run_batch(client, [warmup_task], 1, model, args.max_tokens, api_key)
    if not warmup[0].success:
        raise RuntimeError(
            f"warmup failed: {warmup[0].error_type}: {warmup[0].error_message}"
        )

    if args.sweep:
        runs: dict[str, dict[str, Any]] = {}
        for level in args.sweep:
            print(f"\nSweep run (concurrency={level})", flush=True)
            results, duration = run_batch(
                client, tasks, level, model, args.max_tokens, api_key
            )
            runs[str(level)] = {
                "concurrency": level,
                "summary": summarize(results, duration),
                "results": [asdict(result) for result in results],
            }

        baseline_level = args.sweep[0]
        baseline_run = runs[str(baseline_level)]
        comparisons = {
            str(level): compare_runs(
                baseline_run["summary"],
                runs[str(level)]["summary"],
                baseline_level,
                level,
                [RequestResult(**result) for result in baseline_run["results"]],
                [RequestResult(**result) for result in runs[str(level)]["results"]],
            )
            for level in args.sweep
        }
        output_dir = args.output or (
            Path(__file__).resolve().parent
            / "latency-runs"
            / datetime.now().strftime("%Y%m%d-%H%M%S-sweep")
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        payload: dict[str, Any] = {
            "schema_version": 2,
            "mode": "concurrency_sweep",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "upstream_root": str(upstream),
            "model": model,
            "max_tokens": args.max_tokens,
            "timeout_seconds": args.timeout,
            "concurrency_levels": args.sweep,
            "subset": {
                "selected": len(tasks),
                "eligible": eligible_count,
                "all_tasks": len(all_tasks),
                "seed": args.seed,
                "max_prompt_bytes": args.max_prompt_bytes,
            },
            "tasks": [
                {
                    "name": task.name,
                    "prompt_bytes": task.prompt_bytes,
                    "difficulty": task.difficulty,
                }
                for task in tasks
            ],
            "warmup": asdict(warmup[0]),
            "runs": runs,
            "comparisons": comparisons,
        }
        (output_dir / "results.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_sweep_report(output_dir / "REPORT.md", payload)
        print(f"\nResults: {output_dir}")
        return 0 if all(run["summary"]["errors"] == 0 for run in runs.values()) else 2

    print(f"\nBaseline run (concurrency={args.baseline_concurrency})", flush=True)
    baseline_results, baseline_duration = run_batch(
        client,
        tasks,
        args.baseline_concurrency,
        model,
        args.max_tokens,
        api_key,
    )
    print(f"\nConcurrent run (concurrency={args.concurrency})", flush=True)
    concurrent_results, concurrent_duration = run_batch(
        client, tasks, args.concurrency, model, args.max_tokens, api_key
    )

    baseline_summary = summarize(baseline_results, baseline_duration)
    concurrent_summary = summarize(concurrent_results, concurrent_duration)
    comparison = compare_runs(
        baseline_summary,
        concurrent_summary,
        args.baseline_concurrency,
        args.concurrency,
        baseline_results,
        concurrent_results,
    )
    output_dir = args.output or (
        Path(__file__).resolve().parent
        / "latency-runs"
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "upstream_root": str(upstream),
        "model": model,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout,
        "subset": {
            "selected": len(tasks),
            "eligible": eligible_count,
            "all_tasks": len(all_tasks),
            "seed": args.seed,
            "max_prompt_bytes": args.max_prompt_bytes,
        },
        "tasks": [
            {
                "name": task.name,
                "prompt_bytes": task.prompt_bytes,
                "difficulty": task.difficulty,
            }
            for task in tasks
        ],
        "warmup": asdict(warmup[0]),
        "runs": {
            "baseline": {
                "concurrency": args.baseline_concurrency,
                "summary": baseline_summary,
                "results": [asdict(result) for result in baseline_results],
            },
            "concurrent": {
                "concurrency": args.concurrency,
                "summary": concurrent_summary,
                "results": [asdict(result) for result in concurrent_results],
            },
        },
        "comparison": comparison,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(output_dir / "REPORT.md", payload)
    print(f"\nResults: {output_dir}")
    print(json.dumps(comparison, indent=2))
    return 0 if concurrent_summary["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
