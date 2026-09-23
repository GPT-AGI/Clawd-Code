from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..agent.conversation import Conversation
from ..tool_system.agent_loop import ToolEvent, run_agent_loop
from ..tool_system.context import ToolContext
from ..tool_system.ownership import task_test_scratch_prefix_for_id
from ..tool_system.permissions import ToolPermissionContext
from ..tool_system.registry import ToolRegistry
from .control import mark_teammate_stopped
from .models import AgentRecord, Message, Team, TeamTask, utc_now
from .store import TeamStore, execution_budget_manifest_errors
from .worktree import TeammateWorktreeManager


_MANDATORY_TEAMMATE_TOOLS = (
    "SendMessage",
    "ReadMessages",
    "TaskGet",
    "TaskList",
    "TaskUpdate",
    "StructuredOutput",
)
_FORBIDDEN_TEAMMATE_TOOLS = {
    "Agent",
    "TeamCreate",
    "TeamConfigure",
    "TeamPlan",
    "TeammateCreate",
    "TeamRun",
    "TeamVerify",
    "TeamResume",
    "TeamCancel",
    "TeamAbort",
    "TeamReplan",
    "TeamIntegrate",
    "TeamDelete",
    "TeammateStop",
    "TeammateResume",
    "TaskRetry",
}
_FROZEN_RUN_OPTION_KEYS = (
    "max_workers",
    "timeout_s",
    "token_budget",
    "turn_budget",
    "max_retries",
    "lease_timeout_s",
)


@dataclass(frozen=True)
class TeamRunOptions:
    max_workers: int = 1
    max_batches: int | None = None
    timeout_s: float | None = None
    token_budget: int | None = None
    turn_budget: int | None = None
    max_retries: int = 0
    lease_timeout_s: int = 900

    @classmethod
    def build(cls, persisted: dict[str, Any], overrides: dict[str, Any]) -> "TeamRunOptions":
        source = persisted
        manifest = persisted.get("execution_manifest")
        if isinstance(manifest, dict) and isinstance(manifest.get("execution"), dict):
            source = manifest["execution"]
        values: dict[str, Any] = {}
        for name in cls.__dataclass_fields__:
            if overrides.get(name) is not None:
                values[name] = overrides[name]
            elif name != "max_batches" and source.get(name) is not None:
                values[name] = source[name]
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_workers": self.max_workers,
            "max_batches": self.max_batches,
            "timeout_s": self.timeout_s,
            "token_budget": self.token_budget,
            "turn_budget": self.turn_budget,
            "max_retries": self.max_retries,
            "lease_timeout_s": self.lease_timeout_s,
        }


@dataclass(frozen=True)
class TaskOutcome:
    status: str
    task_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0
    error: str | None = None
    repair_required: bool = False
    infrastructure: bool = False


class TeammateRuntime:
    """Persistent teammate scheduler with recovery, budgets, and optional parallelism."""

    def __init__(
        self,
        provider: Any,
        registry: ToolRegistry,
        *,
        max_turns: int = 30,
        max_output_tokens: int = 4096,
        allowed_models: set[str] | None = None,
        minimum_timeout_s: float | None = None,
    ):
        self.provider = provider
        self.registry = registry
        self.max_turns = max_turns
        self.max_output_tokens = max_output_tokens
        self.allowed_models = {
            model.strip() for model in (allowed_models or set()) if model.strip()
        }
        self.minimum_timeout_s = minimum_timeout_s

    def validate_model(self, model: str | None) -> str | None:
        """Reject teammate model overrides unsupported by this endpoint."""
        if model is None:
            return None
        normalized = model.strip()
        if self.allowed_models and normalized not in self.allowed_models:
            allowed = ", ".join(sorted(self.allowed_models))
            raise ValueError(
                f"unsupported teammate model {normalized!r}; omit model to inherit the "
                f"lead model or choose one of: {allowed}"
            )
        return normalized

    def validate_tools(self, names: list[str]) -> list[str]:
        canonical: list[str] = []
        seen: set[str] = set()
        for name in names:
            tool = self.registry.get(name)
            if tool is None:
                raise ValueError(f"unknown tool in teammate allowlist: {name}")
            spec_name = tool.spec().name
            if spec_name in _FORBIDDEN_TEAMMATE_TOOLS:
                raise ValueError(f"teammates cannot use team-management tool: {spec_name}")
            key = spec_name.lower()
            if key not in seen:
                canonical.append(spec_name)
                seen.add(key)
        return canonical

    def run_team(
        self,
        lead_context: ToolContext,
        *,
        resume: bool = False,
        retry_failed: bool = False,
        retry_cancelled: bool = False,
        max_workers: int | None = None,
        max_batches: int | None = None,
        timeout_s: float | None = None,
        token_budget: int | None = None,
        turn_budget: int | None = None,
        max_retries: int | None = None,
        lease_timeout_s: int | None = None,
    ) -> dict[str, Any]:
        team = lead_context.team_store.load_active_team()
        if team is None:
            return {"status": "failed", "error": "no active team"}
        if self._is_v2(team) and team.lifecycle_state == "aborted":
            return {
                "status": "aborted",
                "team_id": team.team_id,
                "lifecycle_state": "aborted",
                "error": (
                    "protocol v2 abort is terminal; create a new top-level rollout "
                    "instead of resuming or replacing this team"
                ),
                "executed_task_ids": [],
                "usage": team.usage,
            }
        if self._is_v2(team) and team.lifecycle_state == "budget_exhausted":
            return self._budget_exhausted_result(
                lead_context,
                team,
                "the frozen rollout execution budget was already exhausted",
                [],
            )
        if self._is_v2(team):
            plan = team.settings.get("team_plan")
            plan = plan if isinstance(plan, dict) else {}
            manifest = team.settings.get("execution_manifest")
            manifest = manifest if isinstance(manifest, dict) else {}
            if int(plan.get("revision") or 0) > 0:
                budget_manifest_errors = execution_budget_manifest_errors(
                    plan, manifest, usage=team.usage
                )
                if budget_manifest_errors:
                    lead_context.team_store.append_event(
                        team.team_id,
                        "team.execution_budget_manifest_invalid",
                        {
                            "plan_hash": self._active_plan_hash(team),
                            "errors": budget_manifest_errors,
                        },
                    )
                    blocked = self._blocked_result(
                        lead_context,
                        team,
                        "execution budget manifest failed runtime integrity checks: "
                        + "; ".join(budget_manifest_errors),
                        [],
                    )
                    blocked.update(
                        {
                            "failure_domain": "harness",
                            "budget_manifest_errors": budget_manifest_errors,
                            "workspace_preserved": True,
                            "next_required_action": (
                                "Do not execute this plan or edit the manifest in place; "
                                "report the corrupted harness state and preserve the workspace."
                            ),
                        }
                    )
                    return blocked
        if team.status == "completed":
            lead_context.reload_team_state()
            all_tasks_completed = all(
                task.get("status") == "completed"
                for task in lead_context.tasks.values()
            )
            all_v2_tasks_accepted = bool(lead_context.tasks) and all(
                task.get("status") == "completed"
                and task.get("lifecycle_state") == "accepted"
                for task in lead_context.tasks.values()
            )
            validation_passed = (
                (self._quality_policy(team).get("validation") or {}).get("status")
                == "passed"
            )
            if (
                all_tasks_completed
                and (not self._is_strict(team) or validation_passed)
                and (not self._is_v2(team) or all_v2_tasks_accepted)
            ):
                return self._result(lead_context, team, [])
            if self._is_v2(team):
                return self._blocked_result(
                    lead_context,
                    team,
                    "protocol v2 completed teams cannot be reopened implicitly; "
                    "preserve this terminal team for scoring and start a new top-level "
                    "rollout instead of submitting another TeamPlan",
                    [],
                )
            team.transition_to("running")
            team.completed_at = None
            lead_context.team_store.save_team(team)
            lead_context.team_store.append_event(
                team.team_id,
                "team.reopened",
                {
                    "reason": (
                        "strict validation is pending"
                        if all_tasks_completed
                        else "unfinished tasks were added after completion"
                    ),
                    "unfinished_task_ids": [
                        task_id
                        for task_id, task in lead_context.tasks.items()
                        if task.get("status") != "completed"
                    ],
                },
            )
            lead_context.reload_team_state()
        if team.status == "cancelled" and not resume:
            return {"status": "cancelled", "error": "team is cancelled", "team_id": team.team_id}

        lead_context.reload_team_state()
        requested_execution = {
            "max_workers": max_workers,
            "timeout_s": timeout_s,
            "token_budget": token_budget,
            "turn_budget": turn_budget,
            "max_retries": max_retries,
            "lease_timeout_s": lease_timeout_s,
        }
        manifest_mismatches = self._execution_override_mismatches(
            team, requested_execution
        )
        if manifest_mismatches:
            lead_context.team_store.append_event(
                team.team_id,
                "team.execution_manifest_mismatch",
                {
                    "plan_hash": self._active_plan_hash(team),
                    "source": "TeamResume" if resume else "TeamRun",
                    "mismatches": manifest_mismatches,
                },
            )
            rendered = "; ".join(
                f"{item['field']}: planned {item['planned']!r}, "
                f"requested {item['requested']!r}"
                for item in manifest_mismatches
            )
            blocked = self._blocked_result(
                lead_context,
                team,
                "frozen TeamPlan execution manifest mismatch: " + rendered,
                [],
            )
            blocked["execution_manifest_mismatches"] = manifest_mismatches
            blocked["next_required_action"] = (
                "Call TeamReplan first to checkpoint the workspace, then submit one "
                "complete replacement TeamPlan with the new execution settings. Do not "
                "override the frozen manifest in TeamRun or TeamResume."
            )
            return blocked
        strict_errors = self._strict_plan_errors(
            lead_context,
            team,
            require_parallel_start=not self._quality_policy(team).get("plan_accepted", False),
        )
        if strict_errors:
            if self._is_v2(team) and team.lifecycle_state != "repair_required":
                team = self._set_lifecycle(
                    lead_context, team, "draft", event=False
                )
            lead_context.team_store.append_event(
                team.team_id,
                "team.plan_rejected",
                {"errors": strict_errors},
            )
            return self._blocked_result(
                lead_context,
                team,
                "strict team plan rejected: " + "; ".join(strict_errors),
                [],
            )
        if self._is_strict(team):
            team = lead_context.team_store.load_team(team.team_id) or team
            quality = self._quality_policy(team)
            if not quality.get("plan_accepted"):
                quality["plan_accepted"] = True
                quality["plan_accepted_at"] = utc_now()
                team.settings["quality_gates"] = quality
                manifest = team.settings.get("execution_manifest")
                if self._is_v2(team) and isinstance(manifest, dict):
                    manifest = dict(manifest)
                    manifest["status"] = "accepted"
                    manifest["accepted_at"] = utc_now()
                    team.settings["execution_manifest"] = manifest
                lead_context.team_store.save_team(team)
                lead_context.team_store.append_event(
                    team.team_id,
                    "team.plan_accepted",
                    {"task_count": len(lead_context.tasks)},
                )
                if self._is_v2(team):
                    team = self._set_lifecycle(lead_context, team, "ready")
            if any(
                task.get("status") != "completed"
                for task in lead_context.tasks.values()
            ):
                validation = dict(quality.get("validation") or {})
                if validation.get("status") == "passed":
                    quality["validation"] = {
                        "status": "pending",
                        "reason": "team tasks changed after validation",
                    }
                    team.settings["quality_gates"] = quality
                    lead_context.team_store.save_team(team)

        option_source = (
            self._frozen_execution(team) if self._is_v2(team) else team.settings
        )
        options = TeamRunOptions.build(
            option_source,
            {
                "max_workers": max_workers,
                "max_batches": max_batches,
                "timeout_s": timeout_s,
                "token_budget": token_budget,
                "turn_budget": turn_budget,
                "max_retries": max_retries,
                "lease_timeout_s": lease_timeout_s,
            },
        )
        requested_timeout_s = options.timeout_s
        if self.minimum_timeout_s is not None and (
            options.timeout_s is None or options.timeout_s < self.minimum_timeout_s
        ):
            options = replace(options, timeout_s=self.minimum_timeout_s)
            lead_context.team_store.append_event(
                team.team_id,
                "team.options_adjusted",
                {
                    "timeout_s": {
                        "requested": requested_timeout_s,
                        "effective": self.minimum_timeout_s,
                        "reason": "runtime minimum",
                    }
                },
            )
        persisted_options = options.to_dict()
        persisted_options.pop("max_batches", None)
        reconciliation_reasons = self._execution_setting_mismatches(
            team, persisted_options
        )
        if self._is_v2(team):
            for key in (
                *_FROZEN_RUN_OPTION_KEYS,
                "max_batches",
                "verify_timeout_s",
                "auto_verify",
            ):
                team.settings.pop(key, None)
            manifest = team.settings.get("execution_manifest")
            if isinstance(manifest, dict):
                manifest = dict(manifest)
                effective_execution = self._frozen_execution(team)
                effective_execution.update(persisted_options)
                manifest["effective_execution"] = effective_execution
                if requested_timeout_s != options.timeout_s:
                    adjustments = dict(manifest.get("runtime_adjustments") or {})
                    adjustments["timeout_s"] = {
                        "requested": requested_timeout_s,
                        "effective": options.timeout_s,
                        "reason": "runtime minimum",
                    }
                    manifest["runtime_adjustments"] = adjustments
                team.settings["execution_manifest"] = manifest
        else:
            team.settings.update(persisted_options)
        team.usage = self._normalized_usage(team.usage)
        team.started_at = team.started_at or utc_now()
        if resume:
            team.cancel_requested_at = None
        if reconciliation_reasons:
            lead_context.team_store.append_event(
                team.team_id,
                "team.execution_manifest_reconciled",
                {
                    "plan_hash": self._active_plan_hash(team),
                    "mismatches": reconciliation_reasons,
                    "effective_execution": persisted_options,
                },
            )

        try:
            if team.status in {"created", "failed", "cancelled"}:
                team.transition_to("running")
                if self._is_v2(team):
                    team.set_lifecycle_state("running")
                lead_context.team_store.save_team(team)
                lead_context.team_store.append_event(
                    team.team_id,
                    "team.resumed" if resume else "team.running",
                    {"settings": options.to_dict()},
                )
            else:
                if self._is_v2(team) and team.lifecycle_state != "running":
                    team.set_lifecycle_state("running")
                lead_context.team_store.save_team(team)

            self._recover_tasks(
                lead_context,
                team,
                options,
                retry_failed=retry_failed,
                retry_cancelled=retry_cancelled,
            )
            executed: list[str] = []
            completed_batches = 0
            run_started = time.monotonic()

            while True:
                current = lead_context.team_store.load_team(team.team_id) or team
                if current.status == "cancelled" or current.cancel_requested_at:
                    lead_context.reload_team_state()
                    return self._cancelled_result(lead_context, current, executed)

                lead_context.reload_team_state()
                tasks = lead_context.tasks
                if not tasks:
                    return self._fail_team(lead_context, current, "team has no tasks", executed)
                if all(task.get("status") == "completed" for task in tasks.values()):
                    budget_error = self._budget_error(current, options, run_started)
                    if (
                        self._is_v2(current)
                        and budget_error
                        and "budget exhausted" in budget_error
                    ):
                        return self._budget_exhausted_result(
                            lead_context, current, budget_error, executed
                        )
                    if self._is_strict(current):
                        if self._is_v2(current):
                            acceptance_failure = self._accept_produced_tasks(
                                lead_context,
                                current,
                                timeout_s=self._validation_timeout(current),
                                executed=executed,
                            )
                            if acceptance_failure is not None:
                                return acceptance_failure
                            current = self._set_lifecycle(
                                lead_context, current, "awaiting_verification"
                            )
                        coordination_errors = self._coordination_errors(
                            lead_context, current
                        )
                        if coordination_errors:
                            lead_context.team_store.append_event(
                                current.team_id,
                                "team.coordination_rejected",
                                {"errors": coordination_errors},
                            )
                            return self._blocked_result(
                                lead_context,
                                current,
                                "strict coordination gate failed: "
                                + "; ".join(coordination_errors),
                                executed,
                            )
                        quality = self._quality_policy(current)
                        if (quality.get("validation") or {}).get("status") != "passed":
                            if self._is_v2(current):
                                verified = self.verify_team(
                                    lead_context,
                                    timeout_s=self._validation_timeout(current),
                                )
                                verified["executed_task_ids"] = executed
                                return verified
                            return self._verification_required(
                                lead_context, current, executed
                            )
                    completed = self._complete_team(lead_context, current)
                    result = self._result(lead_context, completed, executed)
                    if budget_error:
                        lead_context.team_store.append_event(
                            current.team_id,
                            "team.budget_exceeded_after_completion",
                            {"warning": budget_error, "usage": completed.usage},
                        )
                        result["budget_warning"] = budget_error
                    return result

                budget_error = self._budget_error(current, options, run_started)
                if budget_error:
                    if self._is_v2(current) and "budget exhausted" in budget_error:
                        return self._budget_exhausted_result(
                            lead_context, current, budget_error, executed
                        )
                    return self._fail_team(
                        lead_context,
                        current,
                        budget_error,
                        executed,
                        status="paused" if self._is_v2(current) else "failed",
                        lifecycle_state="paused" if self._is_v2(current) else None,
                    )

                failed = [task for task in tasks.values() if task.get("status") == "failed"]
                if failed:
                    names = ", ".join(str(task.get("key") or task.get("id")) for task in failed)
                    return self._fail_team(lead_context, current, f"failed tasks: {names}", executed)

                agents = {
                    agent.agent_id: agent
                    for agent in lead_context.team_store.list_agents(team.team_id)
                }
                ready = [
                    task
                    for task in tasks.values()
                    if task.get("status") == "pending"
                    and self._dependencies_completed(task, tasks)
                    and task.get("owner") in agents
                    and agents[str(task.get("owner"))].status
                    not in {"stopping", "cancelled"}
                ]
                if not ready:
                    active = [
                        task for task in tasks.values() if task.get("status") == "in_progress"
                    ]
                    pending = [
                        str(task.get("key") or task.get("id"))
                        for task in tasks.values()
                        if task.get("status") in {"pending", "in_progress"}
                    ]
                    reason = "no runnable tasks; active leases or dependencies remain"
                    if pending:
                        reason += f" ({', '.join(pending)})"
                    if active:
                        return self._blocked_result(lead_context, current, reason, executed)
                    return self._blocked_result(lead_context, current, reason, executed)

                ready.sort(
                    key=lambda task: (
                        str(task.get("created_at") or ""),
                        str(task.get("id") or ""),
                    )
                )
                batch, turn_limits = self._build_batch(ready, current, options)
                if not batch:
                    if self._is_v2(current):
                        return self._budget_exhausted_result(
                            lead_context,
                            current,
                            "team plan-revision turn budget exhausted",
                            executed,
                        )
                    return self._fail_team(
                        lead_context, current, "turn budget exhausted", executed
                    )
                outcomes = self._run_batch(
                    lead_context, current, batch, options, turn_limits
                )
                self._record_usage(lead_context.team_store, current.team_id, outcomes)

                terminal_failures: list[TaskOutcome] = []
                infrastructure_failures: list[TaskOutcome] = []
                for outcome in outcomes:
                    executed.append(outcome.task_id)
                    if outcome.infrastructure or outcome.status == "infrastructure":
                        infrastructure_failures.append(outcome)
                    elif outcome.status == "failed":
                        task_data = lead_context.team_store.load_tasks(current.team_id).get(
                            outcome.task_id
                        )
                        if task_data and not outcome.repair_required and self._schedule_retry(
                            lead_context.team_store,
                            current,
                            TeamTask.from_dict(task_data),
                            options,
                        ):
                            continue
                        terminal_failures.append(outcome)
                    elif outcome.status == "cancelled":
                        latest = lead_context.team_store.load_team(current.team_id) or current
                        return self._cancelled_result(lead_context, latest, executed)
                    elif outcome.status == "stopped":
                        continue

                if infrastructure_failures:
                    first = infrastructure_failures[0]
                    paused = self._fail_team(
                        lead_context,
                        current,
                        first.error or "teammate infrastructure failure",
                        executed,
                        status="paused",
                        lifecycle_state="paused",
                    )
                    paused.update(
                        {
                            "failure_domain": "infrastructure",
                            "retryable": True,
                            "infrastructure_task_ids": [
                                outcome.task_id for outcome in infrastructure_failures
                            ],
                        }
                    )
                    return paused

                if terminal_failures:
                    first = terminal_failures[0]
                    needs_repair = self._is_v2(current) and first.repair_required
                    return self._fail_team(
                        lead_context,
                        current,
                        first.error or f"task failed: {first.task_id}",
                        executed,
                        status="repair_required" if needs_repair else "failed",
                        lifecycle_state="repair_required" if needs_repair else None,
                    )

                completed_batches += 1
                if (
                    options.max_batches is not None
                    and completed_batches >= options.max_batches
                ):
                    latest = lead_context.team_store.load_team(current.team_id) or current
                    lead_context.reload_team_state()
                    lead_context.team_store.append_event(
                        current.team_id,
                        "team.batch_paused",
                        {
                            "completed_batches": completed_batches,
                            "executed_task_ids": executed,
                        },
                    )
                    if self._is_v2(latest):
                        latest = self._set_lifecycle(
                            lead_context, latest, "paused"
                        )
                    return self._result(lead_context, latest, executed)
        except Exception as exc:
            current = lead_context.team_store.load_team(team.team_id) or team
            if self._is_v2(current) and self._is_infrastructure_exception(exc):
                return self._fail_team(
                    lead_context,
                    current,
                    str(exc),
                    locals().get("executed", []),
                    status="paused",
                    lifecycle_state="paused",
                )
            return self._fail_team(
                lead_context, current, str(exc), locals().get("executed", [])
            )

    @staticmethod
    def _quality_policy(team: Team) -> dict[str, Any]:
        value = team.settings.get("quality_gates")
        return dict(value) if isinstance(value, dict) else {}

    @classmethod
    def _is_strict(cls, team: Team) -> bool:
        return bool(cls._quality_policy(team).get("strict"))

    @classmethod
    def _protocol_version(cls, team: Team) -> int:
        quality = cls._quality_policy(team)
        versions = [1]
        for raw in (
            getattr(team, "protocol_version", None),
            team.settings.get("protocol_version"),
            quality.get("protocol_version"),
        ):
            try:
                versions.append(int(raw))
            except (TypeError, ValueError):
                continue
        return max(versions)

    @classmethod
    def _is_v2(cls, team: Team) -> bool:
        return cls._protocol_version(team) >= 2

    @staticmethod
    def _active_plan_hash(team: Team) -> str:
        plan = team.settings.get("team_plan")
        return str(plan.get("hash") or "") if isinstance(plan, dict) else ""

    @staticmethod
    def _execution_values_equal(left: Any, right: Any) -> bool:
        if isinstance(left, bool) or isinstance(right, bool):
            return type(left) is type(right) and left == right
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return float(left) == float(right)
        return left == right

    @classmethod
    def _frozen_execution(cls, team: Team) -> dict[str, Any]:
        """Return normalized immutable execution values for the active v2 plan."""

        defaults = TeamRunOptions().to_dict()
        defaults.pop("max_batches", None)
        defaults.update({"verify_timeout_s": 900, "auto_verify": True})
        plan = team.settings.get("team_plan")
        plan = plan if isinstance(plan, dict) else {}
        manifest = team.settings.get("execution_manifest")
        manifest = manifest if isinstance(manifest, dict) else {}
        source: Any = None
        if (
            manifest.get("plan_hash") == plan.get("hash")
            and isinstance(manifest.get("execution"), dict)
        ):
            source = manifest["execution"]
        elif isinstance(plan.get("execution"), dict):
            source = plan["execution"]
        if isinstance(source, dict):
            for key in (*_FROZEN_RUN_OPTION_KEYS, "verify_timeout_s", "auto_verify"):
                if key in source:
                    defaults[key] = source[key]
        return defaults

    @classmethod
    def _execution_override_mismatches(
        cls, team: Team, requested: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if not cls._is_v2(team) or not cls._active_plan_hash(team):
            return []
        frozen = cls._frozen_execution(team)
        mismatches: list[dict[str, Any]] = []
        for field in _FROZEN_RUN_OPTION_KEYS:
            value = requested.get(field)
            if value is None:
                continue
            planned = frozen.get(field)
            if cls._execution_values_equal(value, planned):
                continue
            mismatches.append(
                {
                    "field": field,
                    "planned": planned,
                    "requested": value,
                    "reason": (
                        "runtime override differs from the frozen TeamPlan execution "
                        "manifest"
                    ),
                }
            )
        return mismatches

    @classmethod
    def _execution_setting_mismatches(
        cls, team: Team, effective: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Describe stale top-level settings that the frozen plan will reconcile."""

        if not cls._is_v2(team) or not cls._active_plan_hash(team):
            return []
        reasons: list[dict[str, Any]] = []
        for field in (
            *_FROZEN_RUN_OPTION_KEYS,
            "max_batches",
            "verify_timeout_s",
            "auto_verify",
        ):
            if field not in team.settings:
                continue
            reasons.append(
                {
                    "field": field,
                    "planned_effective": effective.get(field),
                    "persisted": team.settings.get(field),
                    "reason": (
                        "legacy top-level execution setting is not authoritative in "
                        "protocol v2 and was removed"
                    ),
                }
            )
        return reasons

    @staticmethod
    def _is_infrastructure_exception(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
            return True
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "ags backend is not started",
                "sandbox unavailable",
                "deployment unavailable",
                "connection reset",
                "connection refused",
                "service unavailable",
                "pending request was cancelled",
            )
        )

    @staticmethod
    def _set_lifecycle(
        context: ToolContext, team: Team, state: str, *, event: bool = True
    ) -> Team:
        current = context.team_store.load_team(team.team_id) or team
        if current.lifecycle_state == state:
            return current
        previous = current.lifecycle_state
        current.set_lifecycle_state(state)
        context.team_store.save_team(current)
        if event:
            context.team_store.append_event(
                current.team_id,
                "team.lifecycle_changed",
                {"from": previous, "to": state},
            )
        context.reload_team_state()
        return current

    @staticmethod
    def _normalize_owned_path(value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized.rstrip("/")

    @classmethod
    def _owned_paths_overlap(cls, left: str, right: str) -> bool:
        first = cls._normalize_owned_path(left)
        second = cls._normalize_owned_path(right)
        return bool(
            first == second
            or first.startswith(second + "/")
            or second.startswith(first + "/")
        )

    def _strict_plan_errors(
        self,
        lead_context: ToolContext,
        team: Team,
        *,
        require_parallel_start: bool,
    ) -> list[str]:
        if not self._is_strict(team):
            return []
        quality = self._quality_policy(team)
        errors: list[str] = []
        if not quality.get("configured"):
            errors.append("call TeamConfigure before TeamRun")
        if not require_parallel_start and not quality.get("plan_accepted"):
            errors.append("strict plan has not been accepted by TeamRun")
        for field in (
            "architecture_contract",
            "install_command",
            "import_command",
            "integration_command",
        ):
            if not str(quality.get(field) or "").strip():
                errors.append(f"quality gate is missing {field}")

        lead_context.reload_team_state()
        agents = {
            agent.agent_id: agent
            for agent in lead_context.team_store.list_agents(team.team_id)
        }
        tasks = list(lead_context.tasks.values())
        teammate_tasks = [task for task in tasks if task.get("owner") in agents]
        if self._is_v2(team):
            plan = team.settings.get("team_plan")
            plan = plan if isinstance(plan, dict) else {}
            plan_hash = str(plan.get("hash") or "")
            try:
                revision = int(plan.get("revision") or 0)
            except (TypeError, ValueError):
                revision = 0
            if not plan_hash or revision < 1:
                errors.append(
                    "protocol v2 requires one committed atomic TeamPlan revision"
                )
            manifest = team.settings.get("execution_manifest")
            manifest = manifest if isinstance(manifest, dict) else {}
            if not manifest:
                errors.append(
                    "protocol v2 active plan is missing its frozen execution manifest"
                )
            else:
                if str(manifest.get("plan_hash") or "") != plan_hash:
                    errors.append(
                        "frozen execution manifest plan_hash does not match the active TeamPlan"
                    )
                try:
                    manifest_revision = int(manifest.get("plan_revision") or 0)
                except (TypeError, ValueError):
                    manifest_revision = 0
                if manifest_revision != revision:
                    errors.append(
                        "frozen execution manifest revision does not match the active TeamPlan"
                    )
                manifest_execution = manifest.get("execution")
                plan_execution = plan.get("execution")
                if not isinstance(manifest_execution, dict) or (
                    isinstance(plan_execution, dict)
                    and manifest_execution != plan_execution
                ):
                    errors.append(
                        "frozen execution manifest values do not match the active TeamPlan"
                    )
            validation = quality.get("validation")
            validation = validation if isinstance(validation, dict) else {}
            if (
                validation.get("requires_plan_revision")
                and validation.get("failed_plan_hash") == plan_hash
            ):
                errors.append(
                    "repair_required state requires a new TeamPlan revision"
                )
            for task in teammate_tasks:
                metadata = (
                    task.get("metadata")
                    if isinstance(task.get("metadata"), dict)
                    else {}
                )
                if not plan_hash or metadata.get("plan_hash") != plan_hash:
                    key = str(task.get("key") or task.get("id"))
                    errors.append(
                        f"task {key} is not bound to the active atomic TeamPlan hash"
                    )
            implementation_tasks = [
                task
                for task in teammate_tasks
                if (task.get("metadata") or {}).get("task_type")
                != "validation"
            ]
            implementation_owners = {
                str(task.get("owner"))
                for task in implementation_tasks
                if task.get("owned_files") and task.get("acceptance_checks")
            }
            if len(implementation_owners) < 2:
                errors.append(
                    "protocol v2 requires two distinct owners of real implementation tasks"
                )
        assigned_owners = {str(task.get("owner")) for task in teammate_tasks}
        if len(assigned_owners) < 2:
            errors.append("strict teams require at least two assigned teammates")
        if len(teammate_tasks) < 2:
            errors.append("strict teams require at least two teammate-owned tasks")

        paths: list[tuple[str, str, str]] = []
        providers: dict[str, list[dict[str, Any]]] = {}
        contract = quality.get("contract") if isinstance(quality.get("contract"), dict) else {}
        interface_modes = {
            str(item.get("name")): str(item.get("mode") or "handoff")
            for item in (contract.get("interfaces") or [])
            if isinstance(item, dict) and item.get("name")
        }
        for task in teammate_tasks:
            key = str(task.get("key") or task.get("id"))
            owned_files = list(task.get("owned_files") or [])
            acceptance_checks = list(task.get("acceptance_checks") or [])
            metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
            is_validation_task = metadata.get("task_type") == "validation"
            if not owned_files and not is_validation_task:
                errors.append(f"task {key} must declare ownedFiles")
            if not acceptance_checks:
                errors.append(f"task {key} must declare acceptanceChecks")
            for path in owned_files:
                normalized = self._normalize_owned_path(str(path))
                if (
                    not normalized
                    or normalized.startswith("/")
                    or ".." in normalized.split("/")
                    or any(mark in normalized for mark in "*?[]")
                ):
                    errors.append(
                        f"task {key} ownedFiles must use concrete workspace paths: {path!r}"
                    )
                if require_parallel_start or task.get("status") != "completed":
                    paths.append((key, str(task.get("owner")), normalized))
            for interface in task.get("provides_interfaces") or []:
                providers.setdefault(str(interface), []).append(task)

        for index, (left_key, left_owner, left_path) in enumerate(paths):
            for right_key, right_owner, right_path in paths[index + 1 :]:
                # A single worker may intentionally own a directory plus one of its
                # children.  The write-conflict gate is only meaningful across owners.
                if left_key == right_key or left_owner == right_owner:
                    continue
                if self._owned_paths_overlap(left_path, right_path):
                    errors.append(
                        "ownedFiles overlap between "
                        f"{left_key} ({left_owner}) and {right_key} ({right_owner}): "
                        f"{left_path!r} vs {right_path!r}"
                    )

        for task in teammate_tasks:
            key = str(task.get("key") or task.get("id"))
            dependencies = set(task.get("blockedBy") or [])
            for interface in task.get("depends_on_interfaces") or []:
                matches = providers.get(str(interface), [])
                if len(matches) != 1:
                    errors.append(
                        f"task {key} interface {interface!r} must have exactly one provider"
                    )
                    continue
                provider_task = matches[0]
                provider_id = str(provider_task.get("id"))
                if provider_id == str(task.get("id")):
                    errors.append(f"task {key} cannot depend on its own interface {interface!r}")
                elif (
                    interface_modes.get(str(interface), "handoff") != "frozen"
                    and provider_id not in dependencies
                ):
                    provider_key = str(provider_task.get("key") or provider_id)
                    errors.append(
                        f"task {key} must include provider task {provider_key} in blockedBy "
                        f"for interface {interface!r}"
                    )

        if require_parallel_start:
            ready_owners = {
                str(task.get("owner"))
                for task in teammate_tasks
                if not task.get("blockedBy")
                and (
                    task.get("status") == "pending"
                    or (
                        task.get("status") == "completed"
                        and task.get("lifecycle_state") == "produced"
                        and isinstance(task.get("metadata"), dict)
                        and isinstance(task["metadata"].get("carry_forward"), dict)
                        and task["metadata"]["carry_forward"].get(
                            "requires_acceptance"
                        )
                        is True
                    )
                )
            }
            if len(ready_owners) < 2:
                errors.append(
                    "strict teams require at least two initially ready tasks with distinct owners"
                )
        return list(dict.fromkeys(errors))

    def _coordination_errors(
        self, lead_context: ToolContext, team: Team
    ) -> list[str]:
        if not self._is_strict(team):
            return []
        # Protocol v2 freezes shared contracts in TeamPlan and encodes real handoff
        # dependencies in the DAG.  A ceremonial peer message is not evidence that
        # the contract is correct, so it is retained only for v1 compatibility.
        if self._is_v2(team):
            return []
        tasks = list(lead_context.team_store.load_tasks(team.team_id).values())
        providers: dict[str, dict[str, Any]] = {}
        for task in tasks:
            for interface in task.get("provides_interfaces") or []:
                providers[str(interface)] = task
        message_edges = {
            frozenset((message.sender_id, message.recipient_id))
            for message in lead_context.team_store.list_messages(team.team_id)
        }
        errors: list[str] = []
        for task in tasks:
            for interface in task.get("depends_on_interfaces") or []:
                provider = providers.get(str(interface))
                if provider is None:
                    continue
                owner = str(task.get("owner") or "")
                provider_owner = str(provider.get("owner") or "")
                if owner and provider_owner and owner != provider_owner:
                    if frozenset((owner, provider_owner)) not in message_edges:
                        errors.append(
                            f"owners of interface {interface!r} must exchange at least one "
                            "peer message before completion"
                        )
        return list(dict.fromkeys(errors))

    @classmethod
    def _validation_timeout(cls, team: Team) -> int:
        quality = cls._quality_policy(team)
        frozen = cls._frozen_execution(team) if cls._is_v2(team) else {}
        raw = (
            quality.get("verify_timeout_s")
            or frozen.get("verify_timeout_s")
            or team.settings.get("verify_timeout_s")
        )
        try:
            return max(1, int(raw or 900))
        except (TypeError, ValueError):
            return 900

    def _accept_produced_tasks(
        self,
        lead_context: ToolContext,
        team: Team,
        *,
        timeout_s: int,
        executed: list[str],
    ) -> dict[str, Any] | None:
        """Run harness-owned task acceptance between production and integration.

        v1 exposes only the coarse ``completed`` status.  In v2 a worker reaching
        that status means it produced a candidate; the harness must execute every
        declared check before the task is accepted.  Results are persisted on the
        task, making recovery and dashboard inspection deterministic.
        """
        if not self._is_v2(team):
            return None
        failures: list[dict[str, Any]] = []
        task_data = lead_context.team_store.load_tasks(team.team_id)
        for raw in task_data.values():
            task = TeamTask.from_dict(raw)
            if task.status != "completed" or task.lifecycle_state == "accepted":
                continue
            task.set_lifecycle_state("produced")
            stages: list[dict[str, Any]] = []
            for command in task.acceptance_checks:
                result = self._run_validation_command(
                    lead_context, str(command), timeout_s=timeout_s
                )
                stages.append({"command": str(command), **result})
                if result["exit_code"] != 0:
                    break
            passed = bool(task.acceptance_checks) and all(
                stage["exit_code"] == 0 for stage in stages
            )
            acceptance = {
                "status": "passed" if passed else "failed",
                "checked_at": utc_now(),
                "stages": stages,
            }
            task.metadata = dict(task.metadata)
            task.metadata["acceptance"] = acceptance
            if passed:
                task.set_lifecycle_state("accepted")
                lead_context.team_store.append_event(
                    team.team_id,
                    "task.accepted",
                    {"task_id": task.id, "task_key": task.key, "stages": stages},
                )
            else:
                failure = {
                    "task_id": task.id,
                    "task_key": task.key,
                    "error": (
                        "task has no acceptance checks"
                        if not task.acceptance_checks
                        else "acceptance check failed"
                    ),
                    "acceptance": acceptance,
                }
                failures.append(failure)
                task.last_error = failure["error"]
                lead_context.team_store.append_event(
                    team.team_id, "task.acceptance_failed", failure
                )
            self._save_task(lead_context.team_store, team.team_id, task)

        lead_context.reload_team_state()
        if not failures:
            return None

        current = lead_context.team_store.load_team(team.team_id) or team
        quality = self._quality_policy(current)
        quality["validation"] = {
            "status": "pending",
            "reason": "task acceptance failed",
            "requires_plan_revision": True,
            "failed_plan_hash": str(
                ((current.settings.get("team_plan") or {}).get("hash") or "")
            ),
        }
        current.settings["quality_gates"] = quality
        current.set_lifecycle_state("repair_required")
        lead_context.team_store.save_team(current)
        lead_context.team_store.append_event(
            current.team_id, "team.repair_required", {"task_failures": failures}
        )
        lead_context.reload_team_state()
        return {
            "status": "repair_required",
            "team_id": current.team_id,
            "lifecycle_state": current.lifecycle_state,
            "error": "one or more task acceptance checks failed",
            "task_failures": failures,
            "executed_task_ids": executed,
            "tasks": list(lead_context.tasks.values()),
            "quality_gates": quality,
            "usage": current.usage,
            "next_required_action": (
                "Call TeamReplan first to checkpoint the workspace, then submit one "
                "complete materially changed TeamPlan revision and call TeamRun again."
            ),
        }

    def verify_team(
        self, lead_context: ToolContext, *, timeout_s: int = 300
    ) -> dict[str, Any]:
        team = lead_context.team_store.load_active_team()
        if team is None:
            return {"status": "failed", "error": "no active team"}
        if not self._is_strict(team):
            return {
                "status": "failed",
                "team_id": team.team_id,
                "error": "TeamVerify requires strict quality gates",
            }
        lead_context.reload_team_state()
        quality = self._quality_policy(team)
        validation = dict(quality.get("validation") or {})
        v2_tasks_accepted = bool(lead_context.tasks) and all(
            task.get("status") == "completed"
            and task.get("lifecycle_state") == "accepted"
            for task in lead_context.tasks.values()
        )
        if (
            team.status == "completed"
            and team.lifecycle_state == "completed"
            and validation.get("status") == "passed"
            and (not self._is_v2(team) or v2_tasks_accepted)
        ):
            # Verification is deliberately idempotent: repeated tool calls must not
            # recreate environments, duplicate events, or reopen a settled team.
            result = self._result(lead_context, team, [])
            result["validation"] = validation
            result["verification_reused"] = True
            return result
        unfinished = [
            str(task.get("key") or task_id)
            for task_id, task in lead_context.tasks.items()
            if task.get("status") != "completed"
        ]
        if unfinished:
            return {
                "status": "failed",
                "team_id": team.team_id,
                "error": "unfinished teammate tasks: " + ", ".join(unfinished),
            }
        plan_errors = self._strict_plan_errors(
            lead_context, team, require_parallel_start=False
        )
        coordination_errors = self._coordination_errors(lead_context, team)
        errors = plan_errors + coordination_errors
        if errors:
            return {
                "status": (
                    "repair_required"
                    if self._is_v2(team)
                    and team.lifecycle_state == "repair_required"
                    else "failed"
                ),
                "team_id": team.team_id,
                "lifecycle_state": team.lifecycle_state,
                "error": "; ".join(errors),
            }
        if self._is_v2(team):
            acceptance_failure = self._accept_produced_tasks(
                lead_context, team, timeout_s=timeout_s, executed=[]
            )
            if acceptance_failure is not None:
                return acceptance_failure
            team = self._set_lifecycle(
                lead_context, team, "awaiting_verification"
            )

        quality = self._quality_policy(team)
        if self._is_v2(team):
            team = self._set_lifecycle(lead_context, team, "verifying")
        commands = [
            ("install", str(quality["install_command"])),
            ("import", str(quality["import_command"])),
            ("integration", str(quality["integration_command"])),
        ]
        validation_root = (
            f"/tmp/clawd-team-verify-{team.team_id}-{uuid.uuid4().hex[:12]}"
            if lead_context.workspace_backend is not None
            else tempfile.mkdtemp(prefix=f"clawd-team-verify-{team.team_id}-")
        )
        venv_bin = f"{validation_root}/bin"
        validation_workspace = (
            lead_context.execution_workspace_root or "/workspace"
            if lead_context.workspace_backend is not None
            else str(lead_context.workspace_root)
        )
        bootstrap_python = "python3" if lead_context.workspace_backend is not None else shlex.quote(sys.executable)
        bootstrap = (
            f"{bootstrap_python} -m venv --system-site-packages "
            f"{shlex.quote(validation_root)}"
        )
        stages: list[dict[str, Any]] = []
        verification_error: Exception | None = None
        cleanup_error: Exception | None = None
        try:
            bootstrap_result = self._run_validation_command(
                lead_context, bootstrap, timeout_s=timeout_s
            )
            stages.append(
                {"stage": "bootstrap", "command": bootstrap, **bootstrap_result}
            )
            if bootstrap_result["exit_code"] == 0:
                prefix = (
                    f"export VIRTUAL_ENV={shlex.quote(validation_root)}; "
                    f"export PATH={shlex.quote(venv_bin)}:$PATH; "
                    f"export PYTHONPATH={shlex.quote(validation_workspace)}; "
                )
                for stage, command in commands:
                    result = self._run_validation_command(
                        lead_context, prefix + command, timeout_s=timeout_s
                    )
                    stages.append({"stage": stage, "command": command, **result})
                    if result["exit_code"] != 0:
                        break
        except Exception as exc:
            verification_error = exc
        finally:
            try:
                self._cleanup_validation_root(lead_context, validation_root)
            except Exception as exc:
                cleanup_error = exc

        infrastructure_error = verification_error or cleanup_error
        if infrastructure_error is not None:
            if not self._is_v2(team):
                raise infrastructure_error
            current = lead_context.team_store.load_team(team.team_id) or team
            validation = {
                "status": "paused",
                "verified_at": utc_now(),
                "fresh_virtualenv": True,
                "stages": stages,
                "failure_domain": "infrastructure",
                "retryable": True,
                "error": f"{type(infrastructure_error).__name__}: {infrastructure_error}",
                "cleanup_error": (
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                    if cleanup_error is not None
                    else None
                ),
            }
            quality = self._quality_policy(current)
            quality["validation"] = validation
            current.settings["quality_gates"] = quality
            lead_context.team_store.save_team(current)
            current = self._set_lifecycle(lead_context, current, "paused")
            lead_context.team_store.append_event(
                current.team_id,
                "team.validation_paused",
                {
                    "error": validation["error"],
                    "cleanup_error": validation["cleanup_error"],
                    "retryable": True,
                },
            )
            return {
                "status": "paused",
                "team_id": current.team_id,
                "lifecycle_state": current.lifecycle_state,
                "failure_domain": "infrastructure",
                "retryable": True,
                "error": validation["error"],
                "validation": validation,
                "next_required_action": "Retry TeamRun when infrastructure is healthy.",
            }

        passed = len(stages) == 4 and all(stage["exit_code"] == 0 for stage in stages)
        validation = {
            "status": "passed" if passed else "failed",
            "verified_at": utc_now(),
            "fresh_virtualenv": True,
            "stages": stages,
        }
        team = lead_context.team_store.load_team(team.team_id) or team
        if not passed and self._is_v2(team):
            plan = team.settings.get("team_plan")
            plan = plan if isinstance(plan, dict) else {}
            validation.update(
                {
                    "requires_plan_revision": True,
                    "failed_plan_hash": str(plan.get("hash") or ""),
                }
            )
        quality = self._quality_policy(team)
        quality["validation"] = validation
        team.settings["quality_gates"] = quality
        lead_context.team_store.save_team(team)
        lead_context.team_store.append_event(
            team.team_id,
            "team.validation_passed" if passed else "team.validation_failed",
            {"stages": stages},
        )
        if not passed:
            failed_stage = next(
                (stage for stage in stages if stage["exit_code"] != 0), stages[-1]
            )
            if self._is_v2(team):
                team = self._set_lifecycle(
                    lead_context, team, "repair_required"
                )
                lead_context.team_store.append_event(
                    team.team_id,
                    "team.repair_required",
                    {"failed_stage": failed_stage["stage"]},
                )
            return {
                "status": "repair_required" if self._is_v2(team) else "failed",
                "team_id": team.team_id,
                "lifecycle_state": team.lifecycle_state,
                "error": f"{failed_stage['stage']} validation failed",
                "validation": validation,
                "next_required_action": (
                    "Call TeamReplan first to checkpoint the workspace, then submit one "
                    "complete materially changed TeamPlan revision and call TeamRun again."
                    if self._is_v2(team)
                    else "Create a repair task, run TeamRun, then retry TeamVerify."
                ),
            }
        completed = self._complete_team(lead_context, team)
        result = self._result(lead_context, completed, [])
        result["validation"] = validation
        return result

    @staticmethod
    def _run_validation_command(
        context: ToolContext, command: str, *, timeout_s: int
    ) -> dict[str, Any]:
        if context.workspace_backend is not None:
            outcome = context.workspace_backend.exec(
                command,
                cwd=context.execution_workspace_root or "/workspace",
                timeout_s=timeout_s,
            )
            return {
                "exit_code": int(outcome.exit_code),
                "stdout": str(outcome.stdout or "")[-20_000:],
                "stderr": str(outcome.stderr or "")[-20_000:],
            }
        try:
            environment = os.environ.copy()
            environment["PATH"] = (
                str(Path(sys.executable).parent)
                + os.pathsep
                + environment.get("PATH", "")
            )
            completed = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(context.workspace_root),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=environment,
            )
            return {
                "exit_code": completed.returncode,
                "stdout": (completed.stdout or "")[-20_000:],
                "stderr": (completed.stderr or "")[-20_000:],
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "exit_code": 124,
                "stdout": str(exc.stdout or "")[-20_000:],
                "stderr": str(exc.stderr or "")[-20_000:],
            }

    @staticmethod
    def _cleanup_validation_root(context: ToolContext, path: str) -> None:
        if context.workspace_backend is not None:
            outcome = context.workspace_backend.exec(
                f"rm -rf {shlex.quote(path)}",
                cwd=context.execution_workspace_root or "/workspace",
                timeout_s=60,
            )
            if int(outcome.exit_code) != 0:
                raise OSError(
                    "failed to remove validation environment "
                    f"{path}: {str(outcome.stderr or outcome.stdout or '').strip()}"
                )
        else:
            shutil.rmtree(path, ignore_errors=True)

    def _verification_required(
        self, lead_context: ToolContext, team: Team, executed: list[str]
    ) -> dict[str, Any]:
        lead_context.reload_team_state()
        quality = self._quality_policy(team)
        lead_context.team_store.append_event(
            team.team_id,
            "team.verification_required",
            {"validation_status": (quality.get("validation") or {}).get("status")},
        )
        return {
            "status": "verification_required",
            "team_id": team.team_id,
            "executed_task_ids": executed,
            "tasks": list(lead_context.tasks.values()),
            "quality_gates": quality,
            "next_required_action": (
                "Call TeamVerify. The team remains running and protocol completion is false "
                "until clean install, import, and integration checks pass."
            ),
            "usage": team.usage,
        }

    @staticmethod
    def _dependencies_completed(
        task: dict[str, Any], tasks: dict[str, dict[str, Any]]
    ) -> bool:
        dependencies = list(task.get("blockedBy") or [])
        return all(
            dependency in tasks and tasks[dependency].get("status") == "completed"
            for dependency in dependencies
        )

    @staticmethod
    def _build_batch(
        ready: list[dict[str, Any]], team: Team, options: TeamRunOptions
    ) -> tuple[list[dict[str, Any]], list[int]]:
        workers = min(options.max_workers, len(ready))
        usage = TeammateRuntime._normalized_usage(team.usage)
        remaining_turns = None
        if options.turn_budget is not None:
            budget = TeammateRuntime._budget_window(team, options)
            ceiling = budget["hard_ceiling"]["turns"]
            remaining_turns = max(0, int(ceiling) - usage["turns"])
            workers = min(workers, remaining_turns)
        if workers <= 0:
            return [], []
        batch: list[dict[str, Any]] = []
        owners: set[str] = set()
        for task in ready:
            owner_key = str(task.get("owner") or f"task:{task.get('id')}")
            if owner_key in owners:
                continue
            owners.add(owner_key)
            batch.append(task)
            if len(batch) >= workers:
                break
        workers = len(batch)
        if workers == 0:
            return [], []
        if remaining_turns is None:
            return batch, [0] * len(batch)
        base = max(1, remaining_turns // len(batch))
        limits = [base] * len(batch)
        for index in range(remaining_turns - base * len(batch)):
            limits[index] += 1
        return batch, limits

    def _run_batch(
        self,
        lead_context: ToolContext,
        team: Team,
        batch: list[dict[str, Any]],
        options: TeamRunOptions,
        turn_limits: list[int],
    ) -> list[TaskOutcome]:
        tasks = [TeamTask.from_dict(task) for task in batch]
        effective_limits = [
            self.max_turns if limit <= 0 else min(self.max_turns, limit)
            for limit in turn_limits
        ]
        if len(tasks) == 1:
            return [
                self._run_task(
                    lead_context, team, tasks[0], options, effective_limits[0]
                )
            ]
        with ThreadPoolExecutor(
            max_workers=len(tasks), thread_name_prefix=f"clawd-{team.team_id}"
        ) as pool:
            futures = [
                pool.submit(
                    self._run_task,
                    lead_context,
                    team,
                    task,
                    options,
                    limit,
                )
                for task, limit in zip(tasks, effective_limits)
            ]
            return [future.result() for future in futures]

    def _run_task(
        self,
        lead_context: ToolContext,
        team: Team,
        task: TeamTask,
        options: TeamRunOptions,
        task_max_turns: int,
    ) -> TaskOutcome:
        store = lead_context.team_store
        input_tokens = 0
        output_tokens = 0
        turns = 0
        if not task.owner:
            self._set_task_failed(store, team, task, "task has no owner")
            return TaskOutcome("failed", task.id, error="task has no owner")
        agent = store.load_agent(team.team_id, task.owner)
        if agent is None:
            error = f"unknown task owner: {task.owner}"
            self._set_task_failed(store, team, task, error)
            return TaskOutcome("failed", task.id, error=error)
        if agent.status in {"stopping", "cancelled"} or agent.stop_requested_at:
            return TaskOutcome("stopped", task.id)

        lease_id = uuid.uuid4().hex
        try:
            claimed = store.claim_task(
                team.team_id,
                task.id,
                lease_id=lease_id,
                lease_expires_at=self._lease_expiry(options.lease_timeout_s),
                max_retries=options.max_retries,
                expected_plan_hash=self._active_plan_hash(team),
            )
            if claimed is None:
                return TaskOutcome("leased", task.id)
            task = claimed
            agent = self._transition_agent(store, agent, "running")
            if agent.status in {"stopping", "cancelled"} or agent.stop_requested_at:
                return self._finalize_worker_stop(store, agent, task, {})
            store.append_event(
                team.team_id,
                "task.started",
                {
                    "task_id": task.id,
                    "task_key": task.key,
                    "agent_id": agent.agent_id,
                    "attempt": task.attempt,
                    "lease_id": lease_id,
                },
            )

            conversation = self._load_conversation(store, agent)
            incoming = self._consume_messages(store, team, agent)
            conversation.add_user_message(self._task_prompt(team, agent, task, incoming))
            self._save_session(store, agent, conversation)

            child_context = self._child_context(lead_context, team, agent, task)

            def heartbeat(event: ToolEvent) -> None:
                if event.kind in {
                    "model_started",
                    "model_response",
                    "tool_use",
                    "tool_result",
                    "tool_error",
                }:
                    self._refresh_lease(
                        store,
                        team.team_id,
                        task.id,
                        lease_id,
                        options.lease_timeout_s,
                    )

            def should_stop() -> bool:
                current_team = store.load_team(team.team_id) or team
                current_agent = store.load_agent(team.team_id, agent.agent_id) or agent
                return bool(
                    current_team.status == "cancelled"
                    or current_team.cancel_requested_at
                    or current_agent.status in {"stopping", "cancelled"}
                    or current_agent.stop_requested_at
                )

            result = run_agent_loop(
                conversation=conversation,
                provider=self.provider,
                tool_registry=self._child_registry(agent),
                tool_context=child_context,
                max_turns=task_max_turns,
                max_output_tokens=self.max_output_tokens,
                stream=False,
                verbose=False,
                on_event=heartbeat,
                should_stop=should_stop,
            )
            self._save_session(store, agent, conversation)
            usage = result.usage or {}
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            turns = result.num_turns
            outcome_usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "turns": turns,
            }

            persisted_data = store.load_tasks(team.team_id).get(task.id)
            persisted = TeamTask.from_dict(persisted_data or task.to_dict())
            latest_team = store.load_team(team.team_id) or team
            if latest_team.status == "cancelled" or latest_team.cancel_requested_at:
                if persisted.status == "in_progress":
                    persisted.transition_to("cancelled")
                self._clear_lease(persisted)
                persisted.last_error = "team cancelled"
                self._save_task(store, team.team_id, persisted)
                self._transition_agent(store, agent, "cancelled")
                store.append_event(
                    team.team_id,
                    "task.cancelled",
                    {"task_id": task.id, "task_key": task.key, "agent_id": agent.agent_id},
                )
                return TaskOutcome("cancelled", task.id, **outcome_usage)

            latest_agent = store.load_agent(team.team_id, agent.agent_id) or agent
            if latest_agent.status in {"stopping", "cancelled"} or latest_agent.stop_requested_at:
                return self._finalize_worker_stop(
                    store,
                    latest_agent,
                    persisted,
                    outcome_usage,
                )

            if child_context.ownership_violations:
                error = self._set_ownership_failed(
                    store,
                    team,
                    persisted,
                    child_context.ownership_violations,
                )
                transitioned = self._transition_agent(store, agent, "failed")
                if transitioned.status in {"stopping", "cancelled"} or transitioned.stop_requested_at:
                    return self._finalize_worker_stop(
                        store, transitioned, persisted, outcome_usage
                    )
                return TaskOutcome(
                    "failed",
                    task.id,
                    error=error,
                    repair_required=True,
                    **outcome_usage,
                )

            if result.response_text == "[Max tool turns reached]":
                self._set_task_failed(store, team, persisted, result.response_text)
                transitioned = self._transition_agent(store, agent, "failed")
                if transitioned.status in {"stopping", "cancelled"} or transitioned.stop_requested_at:
                    return self._finalize_worker_stop(
                        store, transitioned, persisted, outcome_usage
                    )
                return TaskOutcome(
                    "failed", task.id, error=result.response_text, **outcome_usage
                )
            if persisted.status == "failed":
                if not persisted.output:
                    persisted.output = result.response_text
                    persisted.updated_at = utc_now()
                    self._save_task(store, team.team_id, persisted)
                transitioned = self._transition_agent(store, agent, "failed")
                if transitioned.status in {"stopping", "cancelled"} or transitioned.stop_requested_at:
                    return self._finalize_worker_stop(
                        store, transitioned, persisted, outcome_usage
                    )
                return TaskOutcome(
                    "failed", task.id, error=persisted.output, **outcome_usage
                )
            if persisted.status == "cancelled":
                self._transition_agent(store, agent, "cancelled")
                return TaskOutcome("cancelled", task.id, **outcome_usage)
            if persisted.status == "in_progress":
                persisted.output = result.response_text
                persisted.transition_to("completed")
            elif persisted.status == "completed" and not persisted.output:
                persisted.output = result.response_text
                persisted.updated_at = utc_now()
            if self._is_v2(team):
                persisted.set_lifecycle_state("produced")
            if agent.workspace_mode == "worktree" and agent.auto_integrate:
                integration = TeammateWorktreeManager(
                    lead_context.workspace_root
                ).integrate(agent, persisted)
                store.append_event(
                    team.team_id,
                    "worktree.integrated",
                    {"agent_id": agent.agent_id, "task_id": task.id, **integration},
                )
            self._clear_lease(persisted)
            persisted.completed_at = utc_now()
            self._save_task(store, team.team_id, persisted)
            transitioned = self._transition_agent(store, agent, "idle")
            if transitioned.status in {"stopping", "cancelled"} or transitioned.stop_requested_at:
                return self._finalize_worker_stop(
                    store, transitioned, persisted, outcome_usage
                )
            store.append_event(
                team.team_id,
                "task.produced" if self._is_v2(team) else "task.completed",
                {
                    "task_id": task.id,
                    "task_key": task.key,
                    "agent_id": agent.agent_id,
                    "attempt": persisted.attempt,
                    "lifecycle_state": persisted.lifecycle_state,
                },
            )
            return TaskOutcome("completed", task.id, **outcome_usage)
        except Exception as exc:
            current_data = store.load_tasks(team.team_id).get(task.id)
            current = TeamTask.from_dict(current_data or task.to_dict())
            violations = list(
                getattr(locals().get("child_context"), "ownership_violations", [])
                or []
            )
            if violations:
                error = self._set_ownership_failed(store, team, current, violations)
                latest_agent = store.load_agent(team.team_id, agent.agent_id) or agent
                if latest_agent.status in {"created", "running", "idle"}:
                    self._transition_agent(store, latest_agent, "failed")
                return TaskOutcome(
                    "failed",
                    task.id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    turns=turns,
                    error=error,
                    repair_required=True,
                )
            if self._is_v2(team) and self._is_infrastructure_exception(exc):
                error = self._set_task_infrastructure_paused(
                    store, team, current, str(exc)
                )
                latest_agent = store.load_agent(team.team_id, agent.agent_id) or agent
                if latest_agent.status == "running":
                    self._transition_agent(store, latest_agent, "idle")
                return TaskOutcome(
                    "infrastructure",
                    task.id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    turns=turns,
                    error=error,
                    infrastructure=True,
                )
            if current.status in {"pending", "in_progress"}:
                self._set_task_failed(store, team, current, str(exc))
            latest_agent = store.load_agent(team.team_id, agent.agent_id) or agent
            if latest_agent.status in {"created", "running", "idle"}:
                latest_agent = self._transition_agent(store, latest_agent, "failed")
            if latest_agent.status in {"stopping", "cancelled"} and latest_agent.stop_requested_at:
                return self._finalize_worker_stop(
                    store,
                    latest_agent,
                    current,
                    {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "turns": turns,
                    },
                )
            return TaskOutcome(
                "failed",
                task.id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                turns=turns,
                error=str(exc),
            )

    @staticmethod
    def _finalize_worker_stop(
        store: TeamStore,
        agent: AgentRecord,
        task: TeamTask,
        usage: dict[str, int],
    ) -> TaskOutcome:
        reason = agent.stop_reason or "worker stopped by lead"
        policy = agent.stop_task_policy or "requeue"
        state: dict[str, Any] = {"outcome": "stopped", "event": None}

        def mutate(tasks: dict[str, TeamTask]) -> None:
            current = tasks.get(task.id)
            if current is None:
                current = task
                tasks[task.id] = current
            if current.status == "completed":
                state["outcome"] = "completed"
                TeammateRuntime._clear_lease(current)
                current.completed_at = current.completed_at or utc_now()
                state["event"] = (
                    "task.completed",
                    {
                        "task_id": current.id,
                        "task_key": current.key,
                        "agent_id": agent.agent_id,
                        "attempt": current.attempt,
                    },
                )
            elif policy == "requeue":
                already_requeued = (
                    current.status == "pending"
                    and current.owner is None
                    and current.lease_id is None
                    and current.lease_expires_at is None
                )
                if already_requeued:
                    return
                if current.status != "pending":
                    current.transition_to("pending")
                current.owner = None
                current.output = ""
                current.completed_at = None
                current.last_error = reason
                TeammateRuntime._clear_lease(current)
                state["event"] = (
                    "task.requeued",
                    {
                        "task_id": current.id,
                        "agent_id": agent.agent_id,
                        "reason": reason,
                    },
                )
            else:
                already_cancelled = (
                    current.status == "cancelled"
                    and current.lease_id is None
                    and current.lease_expires_at is None
                    and current.last_error == reason
                )
                if already_cancelled:
                    return
                if current.status != "cancelled":
                    current.transition_to("cancelled")
                current.completed_at = utc_now()
                current.last_error = reason
                TeammateRuntime._clear_lease(current)
                state["event"] = (
                    "task.cancelled",
                    {
                        "task_id": current.id,
                        "agent_id": agent.agent_id,
                        "reason": reason,
                    },
                )

        store.mutate_tasks(agent.team_id, mutate)
        if state["event"] is not None:
            event_type, payload = state["event"]
            store.append_event(agent.team_id, event_type, payload)

        mark_teammate_stopped(store, agent.team_id, agent.agent_id)
        return TaskOutcome(str(state["outcome"]), task.id, **usage)

    def _recover_tasks(
        self,
        lead_context: ToolContext,
        team: Team,
        options: TeamRunOptions,
        *,
        retry_failed: bool,
        retry_cancelled: bool,
    ) -> None:
        store = lead_context.team_store
        for task_data in store.load_tasks(team.team_id).values():
            task = TeamTask.from_dict(task_data)
            reason: str | None = None
            if task.status == "in_progress" and self._lease_expired(task.lease_expires_at):
                reason = "expired or missing task lease"
            elif task.status == "failed" and (
                retry_failed or task.attempt <= max(task.max_retries, options.max_retries)
            ):
                reason = "retrying failed task"
            elif task.status == "cancelled" and retry_cancelled:
                reason = "resuming cancelled task"
            if reason is None:
                continue
            previous = task.status
            task.transition_to("pending")
            self._clear_lease(task)
            task.completed_at = None
            task.max_retries = max(task.max_retries, options.max_retries)
            task.last_error = reason
            self._save_task(store, team.team_id, task)
            self._reset_agent_for_retry(store, team.team_id, task.owner)
            store.append_event(
                team.team_id,
                "task.recovered",
                {
                    "task_id": task.id,
                    "from": previous,
                    "reason": reason,
                    "attempt": task.attempt,
                },
            )
        lead_context.reload_team_state()

    def _schedule_retry(
        self,
        store: TeamStore,
        team: Team,
        task: TeamTask,
        options: TeamRunOptions,
    ) -> bool:
        retry_limit = max(task.max_retries, options.max_retries)
        if task.status != "failed" or task.attempt > retry_limit:
            return False
        task.transition_to("pending")
        self._clear_lease(task)
        task.completed_at = None
        self._save_task(store, team.team_id, task)
        self._reset_agent_for_retry(store, team.team_id, task.owner)
        store.append_event(
            team.team_id,
            "task.retry_scheduled",
            {"task_id": task.id, "attempt": task.attempt, "max_retries": retry_limit},
        )
        return True

    def _child_registry(self, agent: AgentRecord) -> ToolRegistry:
        names = self.validate_tools([*agent.tools, *_MANDATORY_TEAMMATE_TOOLS])
        return ToolRegistry(self.registry.get(name) for name in names)

    @staticmethod
    def _child_context(
        lead_context: ToolContext, team: Team, agent: AgentRecord, task: TeamTask
    ) -> ToolContext:
        workspace_root = lead_context.workspace_root
        if agent.workspace_mode == "worktree" and agent.workspace_path:
            workspace_root = Path(agent.workspace_path)
        permissions = ToolPermissionContext.from_iterables(
            workspace_root=workspace_root,
            allow_docs=lead_context.permission_context.allow_docs,
        )
        context = ToolContext(
            workspace_root=workspace_root,
            permission_context=permissions,
            cwd=workspace_root,
            workspace_backend=lead_context.workspace_backend,
            execution_workspace_root=lead_context.execution_workspace_root,
            execution_cwd=lead_context.execution_workspace_root,
            actor_id=agent.agent_id,
            current_task_id=task.id,
            mutation_lock=lead_context.mutation_lock,
            model_override=agent.model,
            system_prompt_extra=(
                "## Teammate Identity\n"
                f"You are teammate `{agent.name}` with role `{agent.role}` in team `{team.team_name}`.\n"
                f"Role instructions: {agent.instructions}\n"
                f"Your current task is `{task.key or task.id}`. Work only on this task. "
                "Use SendMessage to coordinate directly with any teammate or the lead when it helps; "
                "use ReadMessages to receive peer replies during parallel work. Communicate useful "
                "decisions, interfaces, blockers, and handoffs rather than sending ceremonial updates. "
                "Do not claim another teammate's work. "
                "Protocol-v2 file ownership is enforced: Write/Edit paths must be inside this "
                "task's owned_files, and Bash workspace changes are audited after each command. "
                "Prefer disposable self-tests under `"
                f"{task_test_scratch_prefix_for_id(task.id)}/`. A new `tests/test_*.py` "
                "path is also task-local only if it did not exist at plan start and no "
                "other task declared it. Existing or reserved tests are never writable; "
                "persistent deliverable tests must be listed in this task's owned_files. "
                "Any out-of-scope write fails the task and requires a repair plan. "
                "If the task cannot be completed, set your current task to failed with TaskUpdate and explain why."
            ),
        )
        if workspace_root != lead_context.workspace_root:
            context.team_store = lead_context.team_store
            context.team = team.to_dict()
            context.tasks = lead_context.team_store.load_tasks(team.team_id)
        context.permission_handler = lead_context.permission_handler
        return context

    @staticmethod
    def _task_prompt(
        team: Team, agent: AgentRecord, task: TeamTask, incoming: list[Message]
    ) -> str:
        quality = TeammateRuntime._quality_policy(team)
        lines = [
            f"Team: {team.team_name}",
            f"Task ID: {task.id}",
            f"Task key: {task.key or task.id}",
            f"Attempt: {task.attempt}",
            f"Subject: {task.subject}",
            f"Description: {task.description}",
        ]
        if quality.get("strict"):
            lines.extend(
                [
                    f"Architecture contract: {quality.get('architecture_contract') or 'not configured'}",
                    "Owned files/directories: " + ", ".join(task.owned_files),
                    "Interfaces provided: "
                    + (", ".join(task.provides_interfaces) or "none"),
                    "Interfaces consumed: "
                    + (", ".join(task.depends_on_interfaces) or "none"),
                    "Acceptance checks: " + "; ".join(task.acceptance_checks),
                    (
                        "Strict ownership is active: do not modify files owned by another task. "
                        "Prefer disposable self-tests under `"
                        + task_test_scratch_prefix_for_id(task.id)
                        + "/`. A new `tests/test_*.py` is task-local only when it did not "
                        "exist at plan start and no other task declared it. Existing or "
                        "reserved tests are forbidden; persistent deliverable tests must "
                        "be declared in owned_files. "
                        "If you consume another task's interface, exchange a concrete interface "
                        "message with that owner before finishing."
                    ),
                ]
            )
        if incoming:
            lines.append("Incoming teammate messages:")
            for message in incoming:
                lines.append(
                    f"- from {message.sender_id}: {message.summary or ''}\n  {message.content}"
                )
        else:
            lines.append("Incoming teammate messages: none")
        lines.append(
            "Complete the task using only your available tools, run every declared acceptance "
            "check, and report concrete evidence. Team-level validation will run independently."
        )
        return "\n".join(lines)

    @staticmethod
    def _load_conversation(store: TeamStore, agent: AgentRecord) -> Conversation:
        data = store.load_session(agent.team_id, agent.session_id)
        if data is None:
            return Conversation()
        conversation = data.get("conversation")
        return Conversation.from_dict(conversation) if isinstance(conversation, dict) else Conversation()

    def _save_session(
        self, store: TeamStore, agent: AgentRecord, conversation: Conversation
    ) -> None:
        store.save_session(
            agent.team_id,
            agent.session_id,
            {
                "session_id": agent.session_id,
                "team_id": agent.team_id,
                "agent_id": agent.agent_id,
                "model": agent.model or getattr(self.provider, "model", None),
                "conversation": conversation.to_dict(),
                "updated_at": utc_now(),
            },
        )

    @staticmethod
    def _consume_messages(
        store: TeamStore, team: Team, agent: AgentRecord
    ) -> list[Message]:
        return store.consume_messages(team.team_id, agent.agent_id)

    @staticmethod
    def _save_task(store: TeamStore, team_id: str, task: TeamTask) -> None:
        store.update_task(
            team_id,
            task,
            expected_plan_hash=str((task.metadata or {}).get("plan_hash") or "")
            or None,
        )

    @staticmethod
    def _set_task_failed(
        store: TeamStore, team: Team, task: TeamTask, error: str
    ) -> None:
        if task.status == "pending":
            task.transition_to("in_progress")
        if task.status == "in_progress":
            task.transition_to("failed")
        TeammateRuntime._clear_lease(task)
        task.output = error
        task.last_error = error
        task.completed_at = utc_now()
        task.updated_at = utc_now()
        TeammateRuntime._save_task(store, team.team_id, task)
        store.append_event(
            team.team_id,
            "task.failed",
            {
                "task_id": task.id,
                "task_key": task.key,
                "attempt": task.attempt,
                "error": error,
            },
        )

    @staticmethod
    def _set_ownership_failed(
        store: TeamStore,
        team: Team,
        task: TeamTask,
        violations: list[dict[str, Any]],
    ) -> str:
        """Persist a sticky ownership failure at the scheduler boundary."""

        paths = sorted(
            {
                str(path)
                for violation in violations
                for path in (violation.get("paths") or [])
            }
        )
        rendered = ", ".join(paths[:8])
        if len(paths) > 8:
            rendered += f", ... (+{len(paths) - 8} more)"
        error = f"protocol v2 task ownership violation: {rendered or 'unknown path'}"
        # A teammate may catch a tool error and update its own task. Override that
        # self-reported state so an ownership breach can never become completed.
        task.status = "failed"
        task.set_lifecycle_state("failed")
        TeammateRuntime._clear_lease(task)
        task.output = error
        task.last_error = error
        task.completed_at = utc_now()
        task.metadata = dict(task.metadata)
        task.metadata["ownership_audit"] = {
            "status": "failed",
            "violations": violations,
        }
        TeammateRuntime._save_task(store, team.team_id, task)
        store.append_event(
            team.team_id,
            "task.ownership_failed",
            {
                "task_id": task.id,
                "task_key": task.key,
                "attempt": task.attempt,
                "paths": paths,
                "error": error,
            },
        )
        return error

    @staticmethod
    def _set_task_infrastructure_paused(
        store: TeamStore, team: Team, task: TeamTask, error: str
    ) -> str:
        """Release a v2 task lease without turning transport failure into a candidate."""

        task.status = "pending"
        task.set_lifecycle_state("pending")
        TeammateRuntime._clear_lease(task)
        task.completed_at = None
        task.last_error = error
        task.metadata = dict(task.metadata)
        task.metadata["infrastructure_failure"] = {
            "retryable": True,
            "error": error,
            "recorded_at": utc_now(),
        }
        TeammateRuntime._save_task(store, team.team_id, task)
        store.append_event(
            team.team_id,
            "task.infrastructure_paused",
            {
                "task_id": task.id,
                "task_key": task.key,
                "attempt": task.attempt,
                "retryable": True,
                "error": error,
            },
        )
        return error

    @staticmethod
    def _transition_agent(
        store: TeamStore, agent: AgentRecord, status: str
    ) -> AgentRecord:
        changed = False

        def mutate(current: AgentRecord) -> None:
            nonlocal changed
            if current.stop_requested_at and status != "cancelled":
                return
            if current.status != status:
                current.transition_to(status)
                changed = True

        updated = store.mutate_agent(agent.team_id, agent.agent_id, mutate)
        current = updated or agent
        if changed:
            store.append_event(
                current.team_id,
                f"agent.{status}",
                {"agent_id": current.agent_id, "name": current.name},
            )
        return current

    @staticmethod
    def _reset_agent_for_retry(
        store: TeamStore, team_id: str, agent_id: str | None
    ) -> None:
        if not agent_id:
            return
        agent = store.load_agent(team_id, agent_id)
        if agent is None:
            return
        if agent.stop_requested_at or agent.status == "stopping":
            return
        if agent.status in {"failed", "cancelled"}:
            agent = TeammateRuntime._transition_agent(store, agent, "running")
        if agent.status == "running":
            TeammateRuntime._transition_agent(store, agent, "idle")

    @staticmethod
    def _refresh_lease(
        store: TeamStore,
        team_id: str,
        task_id: str,
        lease_id: str,
        lease_timeout_s: int,
    ) -> None:
        task_data = store.load_tasks(team_id).get(task_id)
        if task_data is None:
            return
        task = TeamTask.from_dict(task_data)
        if task.status != "in_progress" or task.lease_id != lease_id:
            return
        task.lease_expires_at = TeammateRuntime._lease_expiry(lease_timeout_s)
        TeammateRuntime._save_task(store, team_id, task)

    @staticmethod
    def _lease_expiry(seconds: int) -> str:
        return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()

    @staticmethod
    def _lease_expired(value: str | None) -> bool:
        if not value:
            return True
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return True
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed <= datetime.now(timezone.utc)

    @staticmethod
    def _clear_lease(task: TeamTask) -> None:
        task.lease_id = None
        task.lease_expires_at = None

    @staticmethod
    def _normalized_usage(usage: dict[str, Any]) -> dict[str, int]:
        return {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "turns": int(usage.get("turns", 0) or 0),
        }

    @staticmethod
    def _record_usage(
        store: TeamStore, team_id: str, outcomes: list[TaskOutcome]
    ) -> None:
        team = store.load_team(team_id)
        if team is None:
            return
        usage = TeammateRuntime._normalized_usage(team.usage)
        for outcome in outcomes:
            usage["input_tokens"] += outcome.input_tokens
            usage["output_tokens"] += outcome.output_tokens
            usage["turns"] += outcome.turns
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        team.usage = usage
        store.save_team(team)

    @staticmethod
    def _budget_error(
        team: Team, options: TeamRunOptions, run_started: float
    ) -> str | None:
        usage = TeammateRuntime._normalized_usage(team.usage)
        budget = TeammateRuntime._budget_window(team, options)
        if options.timeout_s is not None and time.monotonic() - run_started >= options.timeout_s:
            return f"team timeout exceeded ({options.timeout_s}s)"
        token_ceiling = budget["hard_ceiling"]["total_tokens"]
        if token_ceiling is not None and usage["total_tokens"] >= int(token_ceiling):
            return (
                "team plan-revision token budget exhausted "
                f"(incremental={options.token_budget}, "
                f"baseline={budget['baseline']['total_tokens']}, "
                f"hard_ceiling={token_ceiling})"
            )
        turn_ceiling = budget["hard_ceiling"]["turns"]
        if turn_ceiling is not None and usage["turns"] >= int(turn_ceiling):
            return (
                "team plan-revision turn budget exhausted "
                f"(incremental={options.turn_budget}, "
                f"baseline={budget['baseline']['turns']}, "
                f"hard_ceiling={turn_ceiling})"
            )
        return None

    @staticmethod
    def _budget_window(
        team: Team, options: TeamRunOptions
    ) -> dict[str, dict[str, int | None] | str]:
        """Resolve the active plan revision's incremental budget window.

        The scheduler still compares against an absolute, team-wide hard ceiling;
        only the baseline moves when a newly authorized repair plan is committed.
        """

        baseline = {"total_tokens": 0, "turns": 0}
        stored_ceiling: dict[str, int | None] | None = None
        global_cap = {"total_tokens": None, "turns": None}
        manifest = team.settings.get("execution_manifest")
        if TeammateRuntime._is_v2(team):
            plan = team.settings.get("team_plan")
            plan = plan if isinstance(plan, dict) else {}
            if not isinstance(manifest, dict):
                raise ValueError("execution budget manifest is missing")
            errors = execution_budget_manifest_errors(
                plan, manifest, usage=team.usage
            )
            if errors:
                raise ValueError(
                    "invalid execution budget manifest: " + "; ".join(errors)
                )
        if isinstance(manifest, dict):
            plan = team.settings.get("team_plan")
            plan_hash = str(plan.get("hash") or "") if isinstance(plan, dict) else ""
            window = manifest.get("budget_window")
            if manifest.get("plan_hash") == plan_hash and isinstance(window, dict):
                raw_baseline = window.get("baseline")
                if isinstance(raw_baseline, dict):
                    baseline = {
                        "total_tokens": int(raw_baseline.get("total_tokens", 0) or 0),
                        "turns": int(raw_baseline.get("turns", 0) or 0),
                    }
                raw_ceiling = window.get("hard_ceiling")
                if isinstance(raw_ceiling, dict):
                    stored_ceiling = {
                        "total_tokens": (
                            int(raw_ceiling["total_tokens"])
                            if raw_ceiling.get("total_tokens") is not None
                            else None
                        ),
                        "turns": (
                            int(raw_ceiling["turns"])
                            if raw_ceiling.get("turns") is not None
                            else None
                        ),
                    }
                raw_global = manifest.get("global_cap")
                if isinstance(raw_global, dict):
                    global_cap = {
                        "total_tokens": (
                            int(raw_global["total_tokens"])
                            if raw_global.get("total_tokens") is not None
                            else None
                        ),
                        "turns": (
                            int(raw_global["turns"])
                            if raw_global.get("turns") is not None
                            else None
                        ),
                    }
        computed_ceiling = {
            "total_tokens": (
                baseline["total_tokens"] + options.token_budget
                if options.token_budget is not None
                else None
            ),
            "turns": (
                baseline["turns"] + options.turn_budget
                if options.turn_budget is not None
                else None
            ),
        }
        return {
            "scope": "plan_revision",
            "baseline": baseline,
            "global_cap": global_cap,
            "hard_ceiling": stored_ceiling or computed_ceiling,
        }

    @staticmethod
    def _budget_exhausted_result(
        lead_context: ToolContext,
        team: Team,
        error: str,
        executed: list[str],
    ) -> dict[str, Any]:
        current = lead_context.team_store.load_team(team.team_id) or team
        already_terminal = current.lifecycle_state == "budget_exhausted"
        if not already_terminal:
            if current.status == "created":
                current.transition_to("running")
            if current.status == "running":
                current.transition_to("failed")
            current.set_lifecycle_state("budget_exhausted")
            current.completed_at = utc_now()
            lead_context.team_store.save_team(current)
            lead_context.team_store.append_event(
                current.team_id,
                "team.budget_exhausted",
                {"error": error, "usage": current.usage},
            )
        lead_context.reload_team_state()
        return {
            "status": "budget_exhausted",
            "team_id": current.team_id,
            "lifecycle_state": "budget_exhausted",
            "error": error,
            "terminal": True,
            "replan_allowed": False,
            "resume_allowed": False,
            "workspace_preserved": True,
            "executed_task_ids": executed,
            "usage": current.usage,
        }

    def _complete_team(self, lead_context: ToolContext, team: Team) -> Team:
        store = lead_context.team_store
        for agent in store.list_agents(team.team_id):
            if agent.status == "created":
                agent = self._transition_agent(store, agent, "running")
                self._transition_agent(store, agent, "completed")
            elif agent.status in {"running", "idle"}:
                self._transition_agent(store, agent, "completed")
            elif agent.status == "failed":
                agent = self._transition_agent(store, agent, "running")
                self._transition_agent(store, agent, "completed")
        current = store.load_team(team.team_id) or team
        if current.status != "completed":
            current.transition_to("completed")
            current.completed_at = utc_now()
            store.save_team(current)
            store.append_event(current.team_id, "team.completed", {"usage": current.usage})
        elif self._is_v2(current) and current.lifecycle_state != "completed":
            current.set_lifecycle_state("completed")
            current.completed_at = current.completed_at or utc_now()
            store.save_team(current)
        lead_context.reload_team_state()
        return current

    @staticmethod
    def _fail_team(
        lead_context: ToolContext,
        team: Team,
        error: str,
        executed: list[str],
        *,
        status: str = "failed",
        lifecycle_state: str | None = None,
    ) -> dict[str, Any]:
        current = lead_context.team_store.load_team(team.team_id) or team
        if lifecycle_state is not None:
            current.set_lifecycle_state(lifecycle_state)
            lead_context.team_store.save_team(current)
            lead_context.team_store.append_event(
                current.team_id,
                f"team.{lifecycle_state}",
                {"error": error, "usage": current.usage},
            )
        elif current.status == "running":
            current.transition_to("failed")
            lead_context.team_store.save_team(current)
            lead_context.team_store.append_event(
                current.team_id, "team.failed", {"error": error, "usage": current.usage}
            )
        lead_context.reload_team_state()
        return {
            "status": status,
            "team_id": current.team_id,
            "lifecycle_state": current.lifecycle_state,
            "error": error,
            "executed_task_ids": executed,
            "usage": current.usage,
        }

    @staticmethod
    def _cancelled_result(
        lead_context: ToolContext, team: Team, executed: list[str]
    ) -> dict[str, Any]:
        return {
            "status": "cancelled",
            "team_id": team.team_id,
            "lifecycle_state": team.lifecycle_state,
            "error": "team cancellation requested",
            "executed_task_ids": executed,
            "usage": team.usage,
        }

    @staticmethod
    def _blocked_result(
        lead_context: ToolContext, team: Team, error: str, executed: list[str]
    ) -> dict[str, Any]:
        lead_context.reload_team_state()
        return {
            "status": "blocked",
            "team_id": team.team_id,
            "lifecycle_state": team.lifecycle_state,
            "error": error,
            "executed_task_ids": executed,
            "usage": team.usage,
        }

    @staticmethod
    def _result(
        lead_context: ToolContext, team: Team, executed: list[str]
    ) -> dict[str, Any]:
        agents = lead_context.team_store.list_agents(team.team_id)
        names = {agent.agent_id: agent.name for agent in agents}
        names[team.lead_agent_id] = "lead"
        messages = [
            {
                "message_id": message.message_id,
                "from": names.get(message.sender_id, message.sender_id),
                "to": names.get(message.recipient_id, message.recipient_id),
                "summary": message.summary,
                "status": message.status,
            }
            for message in lead_context.team_store.list_messages(team.team_id)
        ]
        return {
            "status": team.status,
            "team_id": team.team_id,
            "protocol_version": TeammateRuntime._protocol_version(team),
            "lifecycle_state": team.lifecycle_state,
            "executed_task_ids": executed,
            "tasks": list(lead_context.tasks.values()),
            "messages": messages,
            "quality_gates": TeammateRuntime._quality_policy(team),
            "usage": team.usage,
        }
