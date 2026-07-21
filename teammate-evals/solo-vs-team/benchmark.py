from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
SCENARIOS_ROOT = ROOT / "scenarios"
DEFAULT_TEST_COMMAND = [
    sys.executable,
    "-m",
    "unittest",
    "discover",
    "-s",
    "tests",
    "-v",
]
PROTECTED_PATTERNS = ("TASK.md", "requirements.md", "tests/**/*.py")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected an object in {path}")
    return data


def load_scenarios(selected: list[str] | None = None) -> list[dict[str, Any]]:
    wanted = set(selected or [])
    scenarios: list[dict[str, Any]] = []
    for manifest_path in sorted(SCENARIOS_ROOT.glob("*/scenario.json")):
        manifest = _read_json(manifest_path)
        scenario_id = str(manifest.get("id") or manifest_path.parent.name)
        if wanted and scenario_id not in wanted:
            continue
        workspace = manifest_path.parent / "workspace"
        if not workspace.is_dir():
            raise ValueError(f"missing workspace fixture for {scenario_id}")
        manifest["id"] = scenario_id
        manifest["fixture"] = str(workspace)
        scenarios.append(manifest)
    missing = wanted - {scenario["id"] for scenario in scenarios}
    if missing:
        raise ValueError(f"unknown scenarios: {', '.join(sorted(missing))}")
    return scenarios


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_snapshot(workspace: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for pattern in PROTECTED_PATTERNS:
        for path in sorted(workspace.glob(pattern)):
            if path.is_file():
                snapshot[path.relative_to(workspace).as_posix()] = _hash_file(path)
    return snapshot


def build_prompt(workspace: Path, mode: str) -> str:
    task = (workspace / "TASK.md").read_text(encoding="utf-8").strip()
    common = f"""{task}

Read requirements.md before editing. Do not modify TASK.md, requirements.md, or
anything under tests/. Run this command before finishing:

python -m unittest discover -s tests -v
"""
    if mode == "solo":
        return common + """

Execution protocol: work directly as one agent. Do not create a teammate team
or call any Team*/Teammate* tool. Inspect, implement, test, and review your own
change, then summarize the result.
"""
    if mode == "adaptive":
        return common + """

Execution protocol: act as the lead and decide whether this task benefits from a
team. It is valid to solve it directly without creating any teammate. If you do
delegate, choose the number of teammates, their task-specific roles, models,
tool allowlists, workspace modes, task graph, and concurrency yourself. Do not
default to a fixed role pipeline. Teammates may communicate directly through
SendMessage and ReadMessages when useful. Observe their state, intervene or
stop work when needed, and personally perform final integration and validation.
The communication topology should emerge from the work rather than be imposed.
"""
    if mode in {"team", "forced-team"}:
        return common + """

Diagnostic execution protocol: use at least one teammate, but choose the team
size, roles, models, tool allowlists, workspace modes, task graph, concurrency,
and communication topology from the task itself. Do not use a predefined role
pipeline. Teammates may communicate directly through SendMessage and
ReadMessages. Observe the team, adapt it when needed, and personally perform
final integration and validation.
"""
    raise ValueError(f"unknown mode: {mode}")


def _parse_unittest_output(output: str, returncode: int) -> dict[str, Any]:
    match = re.search(r"Ran\s+(\d+)\s+tests?", output)
    total = int(match.group(1)) if match else 0
    counts = {"failures": 0, "errors": 0, "skipped": 0}
    summary = re.search(r"FAILED\s*\(([^)]*)\)", output)
    if summary:
        for key, value in re.findall(r"(failures|errors|skipped)=(\d+)", summary.group(1)):
            counts[key] = int(value)
    ok_skipped = re.search(r"OK\s*\(([^)]*)\)", output)
    if ok_skipped:
        skipped = re.search(r"skipped=(\d+)", ok_skipped.group(1))
        if skipped:
            counts["skipped"] = int(skipped.group(1))
    if returncode != 0 and total and not counts["failures"] and not counts["errors"]:
        counts["errors"] = total
    passed = max(0, total - counts["failures"] - counts["errors"] - counts["skipped"])
    return {
        "total": total,
        "passed": passed,
        **counts,
        "returncode": returncode,
    }


def run_acceptance(workspace: Path, timeout_s: float = 60.0) -> dict[str, Any]:
    completed = subprocess.run(
        DEFAULT_TEST_COMMAND,
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    result = _parse_unittest_output(output, completed.returncode)
    result["output"] = output.strip()
    return result


def _load_team_metrics(workspace: Path) -> dict[str, Any]:
    active_path = workspace / ".clawd" / "team.json"
    active = active_path.exists()
    if active:
        team_path = active_path
    else:
        historical = list((workspace / ".clawd" / "teams").glob("*/team.json"))
        team_path = max(historical, key=lambda path: path.stat().st_mtime) if historical else None
    if team_path is None:
        return {
            "present": False,
            "active": False,
            "status": None,
            "agents": 0,
            "tasks": 0,
            "completed_tasks": 0,
            "messages": 0,
            "worker_usage": {},
            "trace_model_calls": 0,
            "trace_tool_calls": 0,
        }
    team = _read_json(team_path)
    team_id = str(team.get("team_id") or team_path.parent.name)
    team_dir = workspace / ".clawd" / "teams" / team_id
    agents = list((team_dir / "agents").glob("*.json"))
    messages = list((team_dir / "messages").glob("*.json"))
    tasks = _read_json(team_dir / "tasks.json") if (team_dir / "tasks.json").exists() else {}
    event_types: list[str] = []
    events_path = team_dir / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                event_types.append(str(json.loads(line).get("type") or ""))
            except (json.JSONDecodeError, AttributeError):
                continue
    return {
        "present": True,
        "active": active,
        "team_id": team_id,
        "status": team.get("status"),
        "agents": len(agents),
        "tasks": len(tasks),
        "completed_tasks": sum(
            1 for task in tasks.values() if isinstance(task, dict) and task.get("status") == "completed"
        ),
        "messages": len(messages),
        "worker_usage": team.get("usage") or {},
        "trace_model_calls": event_types.count("model.response"),
        "trace_tool_calls": event_types.count("tool.started"),
        "completed_events": event_types.count("team.completed"),
        "failed_events": event_types.count("team.failed"),
        "cancelled_events": event_types.count("team.cancelled"),
    }


def _protocol_ok(mode: str, team: dict[str, Any]) -> bool:
    if mode == "solo":
        return not team["present"]
    if mode == "adaptive" and not team["present"]:
        return True
    return bool(
        team["present"]
        and team["status"] == "completed"
        and team["agents"] >= 1
        and team["tasks"] >= 1
        and team["completed_tasks"] == team["tasks"]
    )


def _run_child(
    workspace: Path,
    prompt_path: Path,
    result_path: Path,
    provider: str,
    model: str,
    max_turns: int,
) -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.runner import run_prompt

    lead_events: list[dict[str, Any]] = []

    def capture(event: Any) -> None:
        lead_events.append(
            {
                "kind": event.kind,
                "tool_name": event.tool_name,
                "usage": event.usage,
                "duration_ms": event.duration_ms,
            }
        )

    payload: dict[str, Any]
    try:
        result = run_prompt(
            prompt_path.read_text(encoding="utf-8"),
            workspace=workspace,
            provider_name=provider,
            model=model,
            max_turns=max_turns,
            on_event=capture,
        )
        payload = {
            "ok": result.response_text != "[Max tool turns reached]",
            "response_text": result.response_text,
            "lead_usage": result.usage or {},
            "lead_turns": result.num_turns,
            "lead_model_calls": sum(event["kind"] == "model_response" for event in lead_events),
            "lead_tool_calls": sum(event["kind"] == "tool_use" for event in lead_events),
        }
    except Exception as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "lead_usage": {},
            "lead_turns": 0,
            "lead_model_calls": sum(event["kind"] == "model_response" for event in lead_events),
            "lead_tool_calls": sum(event["kind"] == "tool_use" for event in lead_events),
        }
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if payload["ok"] else 1


def run_case(
    scenario: dict[str, Any],
    mode: str,
    output_root: Path,
    *,
    provider: str,
    model: str,
    max_turns: int,
    timeout_s: float,
) -> dict[str, Any]:
    case_root = output_root / scenario["id"] / mode
    workspace = case_root / "workspace"
    case_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(scenario["fixture"]), workspace)
    before = protected_snapshot(workspace)
    prompt_path = case_root / "PROMPT.md"
    prompt_path.write_text(build_prompt(workspace, mode), encoding="utf-8")
    result_path = case_root / "agent-result.json"
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
    ]
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=os.environ.copy(),
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
    elapsed = time.monotonic() - started
    (case_root / "stdout.log").write_text(str(stdout), encoding="utf-8")
    (case_root / "stderr.log").write_text(str(stderr), encoding="utf-8")
    agent = _read_json(result_path) if result_path.exists() else {
        "ok": False,
        "error": "run timed out" if timed_out else "missing agent result",
        "lead_usage": {},
        "lead_turns": 0,
        "lead_model_calls": 0,
        "lead_tool_calls": 0,
    }
    result = _score_case(
        scenario,
        mode,
        workspace,
        agent,
        provider=provider,
        model=model,
        elapsed=elapsed,
        timed_out=timed_out,
        returncode=returncode,
        protected_before=before,
    )
    (case_root / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def _score_case(
    scenario: dict[str, Any],
    mode: str,
    workspace: Path,
    agent: dict[str, Any],
    *,
    provider: str,
    model: str,
    elapsed: float,
    timed_out: bool,
    returncode: int,
    protected_before: dict[str, str],
) -> dict[str, Any]:
    acceptance = run_acceptance(workspace)
    protected_after = protected_snapshot(workspace)
    integrity_ok = protected_before == protected_after
    changed_protected = sorted(
        path
        for path in set(protected_before) | set(protected_after)
        if protected_before.get(path) != protected_after.get(path)
    )
    team = _load_team_metrics(workspace)
    protocol_ok = _protocol_ok(mode, team)
    lead_usage = agent.get("lead_usage") or {}
    worker_usage = team.get("worker_usage") or {}
    input_tokens = int(lead_usage.get("input_tokens", 0) or 0) + int(
        worker_usage.get("input_tokens", 0) or 0
    )
    output_tokens = int(lead_usage.get("output_tokens", 0) or 0) + int(
        worker_usage.get("output_tokens", 0) or 0
    )
    total = int(acceptance["total"])
    quality = 100.0 * int(acceptance["passed"]) / total if total and integrity_ok else 0.0
    return {
        "scenario": scenario["id"],
        "title": scenario.get("title", scenario["id"]),
        "mode": mode,
        "provider": provider,
        "model": model,
        "elapsed_s": round(elapsed, 3),
        "timed_out": timed_out,
        "agent_returncode": returncode,
        "agent_ok": bool(agent.get("ok")),
        "agent_error": agent.get("error"),
        "acceptance": acceptance,
        "integrity_ok": integrity_ok,
        "changed_protected": changed_protected,
        "protocol_ok": protocol_ok,
        "used_team": bool(team["present"]),
        "quality_score": round(quality, 2),
        "success": bool(
            agent.get("ok")
            and acceptance["returncode"] == 0
            and integrity_ok
            and protocol_ok
        ),
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
        "workspace": str(workspace),
    }


def rescore_output(output_root: Path) -> tuple[list[dict[str, Any]], str]:
    aggregate_path = output_root / "results.json"
    aggregate = _read_json(aggregate_path)
    scenarios = {scenario["id"]: scenario for scenario in load_scenarios()}
    rescored: list[dict[str, Any]] = []
    for previous in aggregate.get("results", []):
        scenario_id = str(previous.get("scenario") or "")
        mode = str(previous.get("mode") or "")
        scenario = scenarios.get(scenario_id)
        if scenario is None or mode not in {"solo", "adaptive", "team", "forced-team"}:
            raise ValueError(f"cannot rescore unknown case: {scenario_id}/{mode}")
        case_root = output_root / scenario_id / mode
        workspace = case_root / "workspace"
        agent_path = case_root / "agent-result.json"
        agent = _read_json(agent_path) if agent_path.exists() else {
            "ok": False,
            "error": "missing agent result",
            "lead_usage": {},
            "lead_turns": 0,
            "lead_model_calls": 0,
            "lead_tool_calls": 0,
        }
        result = _score_case(
            scenario,
            mode,
            workspace,
            agent,
            provider=str(previous.get("provider") or aggregate.get("provider") or ""),
            model=str(previous.get("model") or aggregate.get("model") or ""),
            elapsed=float(previous.get("elapsed_s", 0) or 0),
            timed_out=bool(previous.get("timed_out")),
            returncode=int(previous.get("agent_returncode", 1) or 0),
            protected_before=protected_snapshot(Path(scenario["fixture"])),
        )
        (case_root / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        rescored.append(result)

    run_id = str(aggregate.get("run_id") or output_root.name)
    aggregate["results"] = rescored
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render_report(rescored, run_id)
    (output_root / "REPORT.md").write_text(report, encoding="utf-8")
    return rescored, report


def _format_cell(result: dict[str, Any] | None, key: Callable[[dict[str, Any]], Any]) -> str:
    if result is None:
        return "-"
    return str(key(result))


def render_report(results: list[dict[str, Any]], run_id: str) -> str:
    by_scenario: dict[str, dict[str, dict[str, Any]]] = {}
    for result in results:
        by_scenario.setdefault(result["scenario"], {})[result["mode"]] = result
    available_modes = {result["mode"] for result in results}
    if "adaptive" in available_modes:
        comparison_mode = "adaptive"
        comparison_label = "Adaptive"
    elif "forced-team" in available_modes:
        comparison_mode = "forced-team"
        comparison_label = "Forced team"
    else:
        comparison_mode = "team"
        comparison_label = "Team"
    lines = [
        "# Solo vs Team Benchmark",
        "",
        f"Run: `{run_id}`",
        "",
        f"| Scenario | Solo quality | {comparison_label} quality | Delta | Solo sec | {comparison_label} sec | Solo tokens | {comparison_label} tokens | Used team | Protocol |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    deltas: list[float] = []
    for scenario, modes in sorted(by_scenario.items()):
        solo = modes.get("solo")
        team = modes.get(comparison_mode)
        delta = None
        if solo and team:
            delta = float(team["quality_score"]) - float(solo["quality_score"])
            deltas.append(delta)
        lines.append(
            "| "
            + " | ".join(
                [
                    scenario,
                    _format_cell(solo, lambda item: item["quality_score"]),
                    _format_cell(team, lambda item: item["quality_score"]),
                    "-" if delta is None else f"{delta:+.2f}",
                    _format_cell(solo, lambda item: item["elapsed_s"]),
                    _format_cell(team, lambda item: item["elapsed_s"]),
                    _format_cell(solo, lambda item: item["usage"]["total_tokens"]),
                    _format_cell(team, lambda item: item["usage"]["total_tokens"]),
                    _format_cell(team, lambda item: "yes" if item.get("used_team") else "no"),
                    _format_cell(team, lambda item: "yes" if item["protocol_ok"] else "no"),
                ]
            )
            + " |"
        )
    success = sum(bool(result["success"]) for result in results)
    lines.extend(
        [
            "",
            f"Successful runs: **{success}/{len(results)}**",
            (
                f"Mean {comparison_label.lower()} quality delta: **{sum(deltas) / len(deltas):+.2f} points**"
                if deltas
                else f"Mean {comparison_label.lower()} quality delta: unavailable"
            ),
            "",
            "Quality is the percentage of deterministic acceptance tests passed. A run is only",
            "successful when protected files are unchanged and its execution protocol is respected.",
        ]
    )
    return "\n".join(lines) + "\n"


def validate_fixtures() -> list[str]:
    errors: list[str] = []
    scenarios = load_scenarios()
    if len(scenarios) != 5:
        errors.append(f"expected 5 scenarios, found {len(scenarios)}")
    for scenario in scenarios:
        workspace = Path(scenario["fixture"])
        for required in ("TASK.md", "requirements.md", "tests"):
            if not (workspace / required).exists():
                errors.append(f"{scenario['id']}: missing {required}")
        result = run_acceptance(workspace)
        if result["total"] < 5:
            errors.append(f"{scenario['id']}: expected at least 5 acceptance tests")
        if result["returncode"] == 0:
            errors.append(f"{scenario['id']}: fixture unexpectedly passes before repair")
        if not protected_snapshot(workspace):
            errors.append(f"{scenario['id']}: no protected files found")
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
    return parser


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_run-one":
        args = _child_parser().parse_args()
        return _run_child(
            args.workspace.resolve(),
            args.prompt_file.resolve(),
            args.result_file.resolve(),
            args.provider,
            args.model,
            args.max_turns,
        )

    parser = argparse.ArgumentParser(description="Compare solo and adaptive teammate execution.")
    parser.add_argument("--list", action="store_true", help="List benchmark scenarios")
    parser.add_argument("--validate-fixtures", action="store_true")
    parser.add_argument("--scenario", action="append", help="Scenario ID; repeat to select several")
    parser.add_argument(
        "--mode",
        choices=("solo", "adaptive", "forced-team", "team", "both", "all"),
        default="both",
        help="both runs solo+adaptive; team is a legacy alias for forced-team",
    )
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=1200.0, help="Seconds per agent run")
    parser.add_argument("--output", type=Path, help="Run output directory")
    parser.add_argument(
        "--rescore-output",
        type=Path,
        help="Recompute metrics for an existing output directory without calling a model",
    )
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenario)
    if args.list:
        for scenario in scenarios:
            print(f"{scenario['id']}: {scenario.get('title', '')}")
        return 0
    if args.validate_fixtures:
        errors = validate_fixtures()
        if errors:
            for error in errors:
                print(f"- {error}")
            return 1
        print(f"{len(scenarios)} scenarios valid and intentionally failing at baseline")
        return 0
    if args.rescore_output:
        output_root = args.rescore_output.resolve()
        _, report = rescore_output(output_root)
        print(report)
        print(f"Artifacts: {output_root}")
        return 0

    if args.max_turns < 1 or args.timeout <= 0:
        parser.error("--max-turns and --timeout must be positive")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (args.output or ROOT / "runs" / run_id).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "both":
        modes = ("solo", "adaptive")
    elif args.mode == "all":
        modes = ("solo", "adaptive", "forced-team")
    elif args.mode == "team":
        modes = ("forced-team",)
    else:
        modes = (args.mode,)
    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        for mode in modes:
            print(f"[{scenario['id']}] running {mode}...", flush=True)
            result = run_case(
                scenario,
                mode,
                output_root,
                provider=args.provider,
                model=args.model,
                max_turns=args.max_turns,
                timeout_s=args.timeout,
            )
            results.append(result)
            print(
                f"  quality={result['quality_score']:.2f} success={result['success']} "
                f"time={result['elapsed_s']:.1f}s tokens={result['usage']['total_tokens']}",
                flush=True,
            )
    payload = {
        "run_id": run_id,
        "provider": args.provider,
        "model": args.model,
        "results": results,
    }
    (output_root / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = render_report(results, run_id)
    (output_root / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"\n{report}")
    print(f"Artifacts: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
