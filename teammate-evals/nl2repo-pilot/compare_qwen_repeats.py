#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_results(run: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(run.glob("*/*/result.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            results.append(value)
    return results


def has_complete_usage(result: dict[str, Any]) -> bool:
    usage = result.get("usage") or {}
    if not isinstance(usage, dict):
        return False
    if "complete" in usage:
        return bool(usage["complete"])
    return int(usage.get("total_tokens") or 0) > 0


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in results if not (r.get("hidden_tests") or {}).get("error")]
    quality = [float(r.get("quality_score") or 0) for r in valid]
    elapsed = [float(r.get("agent_elapsed_s") or 0) for r in results]
    tokens = [int((r.get("usage") or {}).get("total_tokens") or 0) for r in results]
    token_coverage = sum(has_complete_usage(result) for result in results)
    return {
        "cases": len(results),
        "valid": len(valid),
        "infra": len(results) - len(valid),
        "quality": statistics.fmean(quality) if quality else 0.0,
        "success": sum(bool(r.get("success")) for r in valid),
        "success_rate": sum(bool(r.get("success")) for r in valid) / len(valid) if valid else 0,
        "elapsed_mean": statistics.fmean(elapsed) if elapsed else 0.0,
        "elapsed_median": statistics.median(elapsed) if elapsed else 0.0,
        "tokens_total": sum(tokens),
        "token_coverage": token_coverage,
        "team_rate": sum(bool(r.get("used_team")) for r in results) / len(results) if results else 0,
        "backends": ",".join(sorted({str(r.get("score_backend") or "missing") for r in results})),
    }


def row(run_name: str, scope: str, summary: dict[str, Any]) -> str:
    return (
        f"| {run_name} | {scope} | {summary['cases']} | {summary['valid']} | "
        f"{summary['infra']} | {summary['quality']:.2f} | "
        f"{summary['success']}/{summary['valid']} ({summary['success_rate']:.1%}) | "
        f"{summary['elapsed_mean']:.1f} | {summary['elapsed_median']:.1f} | "
        f"{summary['tokens_total']:,} ({summary['token_coverage']}/{summary['cases']}) | "
        f"{summary['team_rate']:.1%} | {summary['backends']} |"
    )


def main() -> int:
    runs = [Path(value).resolve() for value in sys.argv[1:]]
    if not runs:
        raise SystemExit("provide one or more run directories")
    print("# Qwen NL2Repo three-repeat comparison (AGS Reward)\n")
    print(
        "| Run | Scope | Cases | Valid rewards | Infra errors | Mean quality | "
        "Success | Mean rollout s | Median rollout s | Total tokens (coverage) | Team usage | Reward backend |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for run in runs:
        results = load_results(run)
        print(row(run.name.rsplit("-", 1)[-1], "all", summarize(results)))
        for mode in ("adaptive", "forced-team"):
            print(row(run.name.rsplit("-", 1)[-1], mode, summarize([r for r in results if r.get("mode") == mode])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
