from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

from .models import AgentRecord, Message, Team, TeamTask, utc_now


_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
_TEAM_TRANSACTION_LOCAL = threading.local()
_TEAM_PLAN_EXECUTION_KEYS = {
    "max_workers",
    "max_batches",
    "timeout_s",
    "token_budget",
    "turn_budget",
    "max_retries",
    "lease_timeout_s",
    "verify_timeout_s",
    "auto_verify",
}
_BUDGET_FIELDS = {
    "total_tokens": "token_budget",
    "turns": "turn_budget",
}


def _budget_integrity_hash(
    *,
    plan_hash: str,
    plan_revision: int,
    execution: dict[str, Any],
    global_cap: dict[str, Any],
    budget_window: dict[str, Any],
) -> str:
    payload = {
        "plan_hash": plan_hash,
        "plan_revision": plan_revision,
        "execution": execution,
        "global_cap": global_cap,
        "budget_window": budget_window,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def execution_budget_manifest_errors(
    plan: dict[str, Any],
    manifest: dict[str, Any],
    *,
    usage: dict[str, Any] | None = None,
) -> list[str]:
    """Validate every runtime-authoritative execution-budget field.

    Protocol-v2 schema 2 treats the derived window as frozen state.  The plan hash
    alone covers only the requested incremental limits, so the manifest also binds
    the usage baseline, rollout-wide cap, and absolute ceilings to the plan revision.
    """

    errors: list[str] = []
    try:
        schema_version = int(manifest.get("schema_version") or 0)
    except (TypeError, ValueError):
        schema_version = 0
    if schema_version != 2:
        errors.append("execution manifest budget schema_version must be 2")

    plan_hash = str(plan.get("hash") or "")
    try:
        plan_revision = int(plan.get("revision") or 0)
    except (TypeError, ValueError):
        plan_revision = 0
    if str(manifest.get("plan_hash") or "") != plan_hash:
        errors.append("budget manifest plan_hash does not match TeamPlan")
    try:
        manifest_revision = int(manifest.get("plan_revision") or 0)
    except (TypeError, ValueError):
        manifest_revision = 0
    if manifest_revision != plan_revision:
        errors.append("budget manifest plan_revision does not match TeamPlan")
    execution = plan.get("execution")
    execution = execution if isinstance(execution, dict) else {}
    manifest_execution = manifest.get("execution")
    manifest_execution = (
        manifest_execution if isinstance(manifest_execution, dict) else {}
    )
    if manifest_execution != execution:
        errors.append("budget manifest execution does not match TeamPlan")

    global_cap = manifest.get("global_cap")
    window = manifest.get("budget_window")
    if not isinstance(global_cap, dict):
        errors.append("budget manifest global_cap must be an object")
        global_cap = {}
    if not isinstance(window, dict):
        errors.append("budget manifest budget_window must be an object")
        window = {}
    if window.get("scope") != "plan_revision":
        errors.append("budget_window.scope must be plan_revision")

    baseline = window.get("baseline")
    incremental = window.get("incremental_limit")
    hard_ceiling = window.get("hard_ceiling")
    if not isinstance(baseline, dict):
        errors.append("budget_window.baseline must be an object")
        baseline = {}
    if not isinstance(incremental, dict):
        errors.append("budget_window.incremental_limit must be an object")
        incremental = {}
    if not isinstance(hard_ceiling, dict):
        errors.append("budget_window.hard_ceiling must be an object")
        hard_ceiling = {}

    normalized_usage: dict[str, int] = {}
    if usage is not None:
        raw_total = int(usage.get("total_tokens", 0) or 0)
        component_total = int(usage.get("input_tokens", 0) or 0) + int(
            usage.get("output_tokens", 0) or 0
        )
        normalized_usage = {
            "total_tokens": max(raw_total, component_total),
            "turns": int(usage.get("turns", 0) or 0),
        }

    normalized_global: dict[str, int | None] = {}
    normalized_baseline: dict[str, int] = {}
    normalized_incremental: dict[str, int | None] = {}
    normalized_ceiling: dict[str, int | None] = {}

    def optional_non_negative(value: Any, field: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"{field} must be a non-negative integer or null")
            return None
        return value

    for metric, budget_key in _BUDGET_FIELDS.items():
        raw_baseline = baseline.get(metric)
        if (
            isinstance(raw_baseline, bool)
            or not isinstance(raw_baseline, int)
            or raw_baseline < 0
        ):
            errors.append(
                f"budget_window.baseline.{metric} must be a non-negative integer"
            )
            baseline_value = 0
        else:
            baseline_value = raw_baseline
        normalized_baseline[metric] = baseline_value
        if normalized_usage and baseline_value > normalized_usage[metric]:
            errors.append(
                f"budget_window.baseline.{metric} exceeds recorded team usage"
            )

        planned_limit = optional_non_negative(
            execution.get(budget_key), f"execution.{budget_key}"
        )
        recorded_limit = optional_non_negative(
            incremental.get(metric),
            f"budget_window.incremental_limit.{metric}",
        )
        normalized_incremental[metric] = recorded_limit
        if recorded_limit != planned_limit:
            errors.append(
                f"budget_window.incremental_limit.{metric} does not match TeamPlan"
            )

        global_value = optional_non_negative(
            global_cap.get(metric), f"global_cap.{metric}"
        )
        normalized_global[metric] = global_value
        ceiling_value = optional_non_negative(
            hard_ceiling.get(metric), f"budget_window.hard_ceiling.{metric}"
        )
        normalized_ceiling[metric] = ceiling_value

        allocated = (
            baseline_value + planned_limit if planned_limit is not None else None
        )
        expected_ceiling = (
            global_value
            if allocated is None
            else allocated
            if global_value is None
            else min(allocated, global_value)
        )
        if ceiling_value != expected_ceiling:
            errors.append(
                f"budget_window.hard_ceiling.{metric} is inconsistent with its "
                "baseline, incremental limit, and global cap"
            )
        if plan_revision == 1 and global_value != allocated:
            errors.append(
                f"global_cap.{metric} does not match the first plan allocation"
            )

    normalized_window = {
        "scope": window.get("scope"),
        "baseline": normalized_baseline,
        "incremental_limit": normalized_incremental,
        "hard_ceiling": normalized_ceiling,
    }
    expected_integrity = _budget_integrity_hash(
        plan_hash=plan_hash,
        plan_revision=plan_revision,
        execution=execution,
        global_cap=normalized_global,
        budget_window=normalized_window,
    )
    if str(manifest.get("budget_integrity_hash") or "") != expected_integrity:
        errors.append("budget_integrity_hash mismatch")
    return list(dict.fromkeys(errors))


def _accepted_task_evidence(task: TeamTask) -> dict[str, Any] | None:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    evidence = metadata.get("acceptance")
    if not isinstance(evidence, dict) or evidence.get("status") != "passed":
        return None
    stages = evidence.get("stages")
    if not isinstance(stages, list) or len(stages) != len(task.acceptance_checks):
        return None
    for command, stage in zip(task.acceptance_checks, stages):
        if (
            not isinstance(stage, dict)
            or str(stage.get("command") or "") != str(command)
            or stage.get("exit_code") != 0
        ):
            return None
    return evidence


def _logical_task_dependencies(
    tasks: dict[str, dict[str, Any]],
) -> dict[str, set[str]]:
    id_to_key = {
        task_id: str(raw.get("key") or task_id).lower()
        for task_id, raw in tasks.items()
        if isinstance(raw, dict)
    }
    providers: dict[str, set[str]] = {}
    for task_id, raw in tasks.items():
        if not isinstance(raw, dict):
            continue
        key = id_to_key[task_id]
        for interface in raw.get("provides_interfaces") or []:
            providers.setdefault(str(interface), set()).add(key)
    dependencies: dict[str, set[str]] = {}
    for task_id, raw in tasks.items():
        if not isinstance(raw, dict):
            continue
        key = id_to_key[task_id]
        task_dependencies = {
            id_to_key[dependency_id]
            for dependency_id in raw.get("blockedBy") or []
            if dependency_id in id_to_key
        }
        for interface in raw.get("depends_on_interfaces") or []:
            task_dependencies.update(providers.get(str(interface), set()))
        task_dependencies.discard(key)
        dependencies[key] = task_dependencies
    return dependencies


def _carry_forward_accepted_tasks(
    *,
    current_tasks: dict[str, dict[str, Any]],
    candidate_tasks: dict[str, dict[str, Any]],
    current_plan: dict[str, Any],
    next_plan: dict[str, Any],
    checkpoint: dict[str, Any],
) -> list[dict[str, Any]]:
    """Carry exact accepted artifacts into a revision without re-running workers.

    Candidate records retain their new IDs, owners, DAG edges, and active plan hash.
    Only the old artifact/output and acceptance evidence are inherited.  Lifecycle is
    deliberately downgraded to ``produced`` so the new revision's harness must run
    task acceptance again before final verification.
    """

    current_hash = str(current_plan.get("hash") or "")
    next_hash = str(next_plan.get("hash") or "")
    current_contract_hash = str(current_plan.get("contract_hash") or "")
    next_contract_hash = str(next_plan.get("contract_hash") or "")
    if not current_hash or not next_hash or current_hash == next_hash:
        return []
    if not current_contract_hash or current_contract_hash != next_contract_hash:
        return []
    artifact_keys = {
        str(key).lower() for key in checkpoint.get("artifact_tasks") or []
    }
    current_by_key = {
        str(raw.get("key") or task_id).lower(): (task_id, raw)
        for task_id, raw in current_tasks.items()
        if isinstance(raw, dict)
    }
    candidate_by_key = {
        str(raw.get("key") or task_id).lower(): (task_id, raw)
        for task_id, raw in candidate_tasks.items()
        if isinstance(raw, dict)
    }
    evidence_by_key: dict[str, dict[str, Any]] = {}
    reusable: set[str] = set()
    for key, (_, candidate_raw) in candidate_by_key.items():
        previous = current_by_key.get(key)
        if previous is None or key not in artifact_keys:
            continue
        _, current_raw = previous
        current_task = TeamTask.from_dict(current_raw)
        current_metadata = current_task.metadata or {}
        candidate_metadata = (
            candidate_raw.get("metadata")
            if isinstance(candidate_raw.get("metadata"), dict)
            else {}
        )
        current_fingerprint = str(
            current_metadata.get("task_contract_fingerprint") or ""
        )
        candidate_fingerprint = str(
            candidate_metadata.get("task_contract_fingerprint") or ""
        )
        evidence = _accepted_task_evidence(current_task)
        if (
            current_task.status == "completed"
            and current_task.lifecycle_state == "accepted"
            and evidence is not None
            and current_fingerprint
            and current_fingerprint == candidate_fingerprint
            and str(current_metadata.get("plan_hash") or "") == current_hash
            and str(candidate_metadata.get("plan_hash") or "") == next_hash
            and str(current_metadata.get("contract_hash") or "")
            == current_contract_hash
            and str(candidate_metadata.get("contract_hash") or "")
            == next_contract_hash
        ):
            reusable.add(key)
            evidence_by_key[key] = evidence

    dependencies = _logical_task_dependencies(candidate_tasks)
    while True:
        invalidated = {
            key
            for key in reusable
            if any(
                dependency not in reusable
                for dependency in dependencies.get(key, set())
            )
        }
        if not invalidated:
            break
        reusable.difference_update(invalidated)

    carried: list[dict[str, Any]] = []
    for key in sorted(reusable):
        old_task_id, current_raw = current_by_key[key]
        candidate_task_id, candidate_raw = candidate_by_key[key]
        old_task = TeamTask.from_dict(current_raw)
        task = TeamTask.from_dict(candidate_raw)
        task.transition_to("completed")
        task.set_lifecycle_state("produced")
        task.output = old_task.output
        task.started_at = old_task.started_at
        task.completed_at = old_task.completed_at
        task.last_error = None
        task.metadata = dict(task.metadata)
        task.metadata.pop("acceptance", None)
        task.metadata["carry_forward"] = {
            "from_task_id": old_task_id,
            "from_plan_hash": current_hash,
            "from_plan_revision": int(current_plan.get("revision") or 0),
            "accepted_evidence": json.loads(
                json.dumps(evidence_by_key[key], ensure_ascii=False)
            ),
            "source_attempt": old_task.attempt,
            "requires_acceptance": True,
            "carried_at": utc_now(),
        }
        task.updated_at = utc_now()
        candidate_tasks[candidate_task_id] = task.to_dict()
        carried.append(
            {
                "key": str(task.key or candidate_task_id),
                "from_task_id": old_task_id,
                "task_id": candidate_task_id,
                "from_plan_hash": current_hash,
                "contract_fingerprint": str(
                    task.metadata.get("task_contract_fingerprint") or ""
                ),
                "lifecycle_state": "produced",
                "requires_acceptance": True,
            }
        )
    return carried


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _locked_path(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock(lock_path):
        with lock_path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _locked_team_transaction(path: Path) -> Iterator[None]:
    """Acquire one re-entrant write transaction lock for a whole team.

    Every team/tasks/agents/sessions writer takes this lock before its narrower
    file lock. Re-entry in the same thread deliberately skips a second ``flock``
    so callbacks cannot deadlock on a lock already owned by their transaction.
    """

    key = str(path.resolve())
    held = getattr(_TEAM_TRANSACTION_LOCAL, "held", None)
    if held is None:
        held = set()
        _TEAM_TRANSACTION_LOCAL.held = held
    if key in held:
        yield
        return
    with _locked_path(path):
        held.add(key)
        try:
            yield
        finally:
            held.remove(key)


class TeamStore:
    """Filesystem-backed storage for the active team and its shared state."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root).resolve()
        self.clawd_dir = self.workspace_root / ".clawd"
        self.teams_dir = self.clawd_dir / "teams"
        self.active_team_path = self.clawd_dir / "team.json"

    def team_dir(self, team_id: str) -> Path:
        return self.teams_dir / team_id

    def _transaction_path(self, team_id: str) -> Path:
        return self.team_dir(team_id) / "team-transaction"

    def create_team(
        self,
        team_name: str,
        description: str | None = None,
        agent_type: str | None = None,
    ) -> Team:
        if self.load_active_team() is not None:
            raise ValueError("an active team already exists")

        team = Team(
            team_id=uuid.uuid4().hex[:12],
            team_name=team_name,
            description=description,
            agent_type=agent_type,
            lead_agent_id=uuid.uuid4().hex[:12],
        )
        directory = self.team_dir(team.team_id)
        for name in ("agents", "sessions", "messages"):
            (directory / name).mkdir(parents=True, exist_ok=True)

        self._write_json(directory / "team.json", team.to_dict())
        self._write_json(directory / "tasks.json", {})
        (directory / "events.jsonl").touch(exist_ok=True)
        self._write_json(self.active_team_path, team.to_dict())
        self.append_event(team.team_id, "team.created", {"team": team.to_dict()})
        return team

    def load_active_team(self) -> Team | None:
        if not self.active_team_path.exists():
            return None
        data = self._read_json(self.active_team_path)
        if "team_id" not in data:
            data = self._migrate_legacy_team(data)
        return Team.from_dict(data)

    def load_team(self, team_id: str) -> Team | None:
        path = self.team_dir(team_id) / "team.json"
        if not path.exists():
            return None
        return Team.from_dict(self._read_json(path))

    def save_team(self, team: Team) -> Path:
        path = self.team_dir(team.team_id) / "team.json"
        with _locked_team_transaction(self._transaction_path(team.team_id)):
            self._write_json(path, team.to_dict())
            active = self.load_active_team()
            if active is not None and active.team_id == team.team_id:
                self._write_json(self.active_team_path, team.to_dict())
        return path

    def load_tasks(self, team_id: str) -> dict[str, dict[str, Any]]:
        path = self.team_dir(team_id) / "tasks.json"
        if not path.exists():
            return {}
        data = self._read_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"invalid task store at {path}")
        return {task_id: TeamTask.from_dict(task).to_dict() for task_id, task in data.items()}

    def save_tasks(self, team_id: str, tasks: dict[str, dict[str, Any]]) -> None:
        serialized = {task_id: TeamTask.from_dict(task).to_dict() for task_id, task in tasks.items()}
        with _locked_team_transaction(self._transaction_path(team_id)):
            self._write_json(self.team_dir(team_id) / "tasks.json", serialized)

    def mutate_tasks(
        self,
        team_id: str,
        mutator: Callable[[dict[str, TeamTask]], Any],
    ) -> Any:
        """Apply one atomic mutation across the team's task collection."""
        path = self.team_dir(team_id) / "tasks.json"
        with _locked_team_transaction(self._transaction_path(team_id)):
            with _locked_path(path):
                data = self._read_json(path) if path.exists() else {}
                tasks = {
                    task_id: TeamTask.from_dict(task)
                    for task_id, task in data.items()
                }
                result = mutator(tasks)
                self._write_json_unlocked(
                    path,
                    {task_id: task.to_dict() for task_id, task in tasks.items()},
                )
        return result

    def update_task(
        self,
        team_id: str,
        task: TeamTask,
        *,
        expected_plan_hash: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        path = self.team_dir(team_id) / "tasks.json"
        with _locked_team_transaction(self._transaction_path(team_id)):
            with _locked_path(path):
                data = self._read_json(path) if path.exists() else {}
                current = data.get(task.id)
                # A worker may finish after a TeamReplan/TeamPlan replacement.  Its
                # stale in-memory task must never recreate the deleted revision.
                if not isinstance(current, dict):
                    return {
                        task_id: TeamTask.from_dict(raw).to_dict()
                        for task_id, raw in data.items()
                    }
                team_path = self.team_dir(team_id) / "team.json"
                team = (
                    Team.from_dict(self._read_json(team_path))
                    if team_path.exists()
                    else None
                )
                if team is not None and team.protocol_version >= 2:
                    plan = team.settings.get("team_plan")
                    plan = plan if isinstance(plan, dict) else {}
                    active_hash = str(plan.get("hash") or "")
                    task_hash = str(
                        expected_plan_hash
                        or (task.metadata or {}).get("plan_hash")
                        or ""
                    )
                    current_hash = str(
                        ((current.get("metadata") or {}).get("plan_hash") or "")
                        if isinstance(current.get("metadata"), dict)
                        else ""
                    )
                    if (
                        not active_hash
                        or task_hash != active_hash
                        or current_hash != active_hash
                    ):
                        return {
                            task_id: TeamTask.from_dict(raw).to_dict()
                            for task_id, raw in data.items()
                        }
                data[task.id] = task.to_dict()
                self._write_json_unlocked(path, data)
        return self.load_tasks(team_id)

    def claim_task(
        self,
        team_id: str,
        task_id: str,
        *,
        lease_id: str,
        lease_expires_at: str,
        max_retries: int,
        expected_plan_hash: str | None = None,
    ) -> TeamTask | None:
        path = self.team_dir(team_id) / "tasks.json"
        with _locked_team_transaction(self._transaction_path(team_id)):
            with _locked_path(path):
                data = self._read_json(path) if path.exists() else {}
                raw = data.get(task_id)
                if not isinstance(raw, dict):
                    return None
                task = TeamTask.from_dict(raw)
                team_path = self.team_dir(team_id) / "team.json"
                team = (
                    Team.from_dict(self._read_json(team_path))
                    if team_path.exists()
                    else None
                )
                if team is not None and team.protocol_version >= 2:
                    plan = team.settings.get("team_plan")
                    plan = plan if isinstance(plan, dict) else {}
                    active_hash = str(plan.get("hash") or "")
                    task_hash = str(
                        expected_plan_hash
                        or (task.metadata or {}).get("plan_hash")
                        or ""
                    )
                    if (
                        team.lifecycle_state != "running"
                        or not active_hash
                        or task_hash != active_hash
                        or str((task.metadata or {}).get("plan_hash") or "")
                        != active_hash
                    ):
                        return None
                if task.status != "pending":
                    return None
                task.transition_to("in_progress")
                task.attempt += 1
                task.max_retries = max(task.max_retries, max_retries)
                task.lease_id = lease_id
                task.lease_expires_at = lease_expires_at
                task.started_at = utc_now()
                task.completed_at = None
                task.last_error = None
                data[task_id] = task.to_dict()
                self._write_json_unlocked(path, data)
                return task

    def delete_task(self, team_id: str, task_id: str) -> dict[str, dict[str, Any]]:
        path = self.team_dir(team_id) / "tasks.json"
        with _locked_team_transaction(self._transaction_path(team_id)):
            with _locked_path(path):
                data = self._read_json(path) if path.exists() else {}
                data.pop(task_id, None)
                self._write_json_unlocked(path, data)
        return self.load_tasks(team_id)

    def save_agent(self, agent: AgentRecord) -> Path:
        path = self.team_dir(agent.team_id) / "agents" / f"{agent.agent_id}.json"
        with _locked_team_transaction(self._transaction_path(agent.team_id)):
            self._write_json(path, agent.to_dict())
        return path

    def mutate_agent(
        self,
        team_id: str,
        agent_id: str,
        mutator: Callable[[AgentRecord], Any],
    ) -> AgentRecord | None:
        """Atomically load, mutate, and persist one agent record."""
        path = self.team_dir(team_id) / "agents" / f"{agent_id}.json"
        with _locked_team_transaction(self._transaction_path(team_id)):
            with _locked_path(path):
                if not path.exists():
                    return None
                agent = AgentRecord.from_dict(self._read_json(path))
                mutator(agent)
                self._write_json_unlocked(path, agent.to_dict())
        return agent

    def load_agent(self, team_id: str, agent_id: str) -> AgentRecord | None:
        path = self.team_dir(team_id) / "agents" / f"{agent_id}.json"
        if not path.exists():
            return None
        return AgentRecord.from_dict(self._read_json(path))

    def list_agents(self, team_id: str) -> list[AgentRecord]:
        directory = self.team_dir(team_id) / "agents"
        if not directory.exists():
            return []
        return [AgentRecord.from_dict(self._read_json(path)) for path in sorted(directory.glob("*.json"))]

    def find_agent(self, team_id: str, identity: str) -> AgentRecord | None:
        normalized = identity.strip().lower()
        for agent in self.list_agents(team_id):
            if agent.agent_id == identity or agent.name.lower() == normalized:
                return agent
        return None

    def save_message(self, message: Message) -> Path:
        path = self.team_dir(message.team_id) / "messages" / f"{message.message_id}.json"
        # Message writes participate in the team transaction so a protocol-v2
        # teammate Bash audit can freeze legitimate harness control-state changes
        # while attributing its before/after filesystem diff.
        with _locked_team_transaction(self._transaction_path(message.team_id)):
            self._write_json(path, message.to_dict())
        return path

    def load_message(self, team_id: str, message_id: str) -> Message | None:
        path = self.team_dir(team_id) / "messages" / f"{message_id}.json"
        if not path.exists():
            return None
        return Message.from_dict(self._read_json(path))

    def list_messages(self, team_id: str) -> list[Message]:
        directory = self.team_dir(team_id) / "messages"
        if not directory.exists():
            return []
        messages = [Message.from_dict(self._read_json(path)) for path in directory.glob("*.json")]
        messages.sort(key=lambda message: (message.created_at, message.message_id))
        return messages

    def consume_messages(self, team_id: str, recipient_id: str) -> list[Message]:
        incoming = [
            message
            for message in self.list_messages(team_id)
            if message.recipient_id == recipient_id and message.status == "delivered"
        ]
        for message in incoming:
            message.transition_to("consumed")
            self.save_message(message)
            self.append_event(
                team_id,
                "message.consumed",
                {"message_id": message.message_id, "agent_id": recipient_id},
            )
        return incoming

    def save_session(self, team_id: str, session_id: str, data: dict[str, Any]) -> Path:
        path = self.team_dir(team_id) / "sessions" / f"{session_id}.json"
        with _locked_team_transaction(self._transaction_path(team_id)):
            self._write_json(path, data)
        return path

    def request_team_replan(
        self,
        team_id: str,
        *,
        reason: str,
        replace_completed_work: bool,
    ) -> tuple[Team, dict[str, Any]]:
        """Checkpoint a recoverable plan transition in one team transaction.

        The busy check, lifecycle change, checkpoint, and event share the same
        transaction as task claims. Therefore either a claim wins and replan is
        rejected as busy, or replan wins and the claim observes repair_required.
        """

        team_path = self.team_dir(team_id) / "team.json"
        tasks_path = self.team_dir(team_id) / "tasks.json"
        with _locked_team_transaction(self._transaction_path(team_id)):
            if not team_path.exists():
                raise ValueError("active team state is unavailable")
            team = Team.from_dict(self._read_json(team_path))
            quality = team.settings.get("quality_gates")
            quality = quality if isinstance(quality, dict) else {}
            if team.protocol_version < 2 or not quality.get("strict"):
                raise ValueError(
                    "TeamReplan is only available to strict protocol v2 teams"
                )
            lifecycle = str(team.lifecycle_state or team.status)
            if lifecycle in {"completed", "aborted", "budget_exhausted"} or (
                team.status == "completed"
            ):
                raise ValueError(
                    f"a {lifecycle} team is terminal and cannot be replanned; "
                    "preserve its workspace for scoring and start a new top-level rollout"
                )

            raw_tasks = self._read_json(tasks_path) if tasks_path.exists() else {}
            active_tasks = [
                str(raw.get("key") or task_id)
                for task_id, raw in raw_tasks.items()
                if isinstance(raw, dict) and raw.get("status") == "in_progress"
            ]
            running_agents = [
                agent.name
                for agent in self.list_agents(team_id)
                if agent.status in {"running", "stopping"}
            ]
            if active_tasks or running_agents:
                details: list[str] = []
                if active_tasks:
                    details.append("active tasks: " + ", ".join(active_tasks))
                if running_agents:
                    details.append("running teammates: " + ", ".join(running_agents))
                raise ValueError(
                    "cannot replan while workers are active; call TeamCancel, wait for "
                    "cooperative shutdown, then call TeamReplan. Do not use TeamAbort "
                    "for restart ("
                    + "; ".join(details)
                    + ")"
                )

            artifact_tasks = [
                str(raw.get("key") or task_id)
                for task_id, raw in raw_tasks.items()
                if isinstance(raw, dict)
                and (
                    raw.get("status") == "completed"
                    or raw.get("lifecycle_state") in {"produced", "accepted"}
                    or bool(str(raw.get("output") or "").strip())
                )
            ]
            if artifact_tasks and not replace_completed_work:
                raise ValueError(
                    "the current plan has completed or produced task artifacts: "
                    + ", ".join(artifact_tasks)
                    + ". They remain preserved. If replacement is intentional, call "
                    "TeamReplan again with replace_completed_work=true"
                )

            current_plan = team.settings.get("team_plan")
            current_plan = current_plan if isinstance(current_plan, dict) else {}
            checkpoint = {
                "checkpoint_id": uuid.uuid4().hex[:12],
                "created_at": utc_now(),
                "plan_revision": int(current_plan.get("revision") or 0),
                "plan_hash": current_plan.get("hash"),
                "prior_status": team.status,
                "prior_lifecycle_state": lifecycle,
                "artifact_tasks": artifact_tasks,
                "workspace_preserved": True,
            }
            quality = dict(quality)
            quality["plan_accepted"] = False
            quality["validation"] = {
                "status": "pending",
                "reason": "recoverable replan requested",
            }
            team.settings["quality_gates"] = quality
            team.settings["last_replan_checkpoint"] = checkpoint
            if team.status in {"failed", "cancelled"}:
                team.transition_to("running")
            team.set_lifecycle_state("repair_required")
            team.cancel_requested_at = None
            team.completed_at = None
            self.save_team(team)
            self.append_event(
                team.team_id,
                "team.replan_requested",
                {
                    "reason": reason,
                    "replace_completed_work": replace_completed_work,
                    "checkpoint": checkpoint,
                },
            )
            return team, checkpoint

    def replace_team_plan(
        self,
        team_id: str,
        *,
        tasks: dict[str, dict[str, Any]],
        agents: list[AgentRecord],
        sessions: dict[str, dict[str, Any]],
        settings_updates: dict[str, Any],
        plan_record: dict[str, Any],
        expected_revision: int | None,
        idempotency_key: str | None,
    ) -> tuple[Team, bool]:
        """Replace a complete materialized plan with optimistic concurrency.

        TeamPlan validates its candidate entirely before calling this method.  This
        final storage boundary rechecks revision/idempotency under one plan lock and
        rolls every materialized file back if any write fails.  Readers still use
        the established team/tasks/agents/session layout, so older runtimes remain
        compatible with v2 plans.

        Returns ``(team, changed)``.  ``changed`` is false for an idempotent retry.
        """

        directory = self.team_dir(team_id)
        team_path = directory / "team.json"
        tasks_path = directory / "tasks.json"
        agent_dir = directory / "agents"
        session_dir = directory / "sessions"

        with _locked_team_transaction(self._transaction_path(team_id)):
            if not team_path.exists():
                raise ValueError("active team state is unavailable")
            team = Team.from_dict(self._read_json(team_path))
            if team.lifecycle_state in {"aborted", "budget_exhausted"}:
                raise ValueError(
                    f"a {team.lifecycle_state} protocol v2 team is terminal and "
                    "cannot accept a new plan"
                )
            if team.status == "completed" or team.lifecycle_state == "completed":
                raise ValueError(
                    "a completed protocol v2 team is terminal and cannot accept a new plan"
                )
            current_plan = team.settings.get("team_plan")
            current_plan = current_plan if isinstance(current_plan, dict) else {}
            current_revision = int(current_plan.get("revision") or 0)
            current_key = current_plan.get("idempotency_key")
            candidate_hash = str(plan_record.get("hash") or "")

            if (
                candidate_hash
                and str(current_plan.get("hash") or "") == candidate_hash
                and team.lifecycle_state == "repair_required"
            ):
                raise ValueError(
                    "repair_required cannot consume TeamReplan with an unchanged plan; "
                    "submit a materially revised TeamPlan that addresses the reported "
                    "failure"
                )
            if candidate_hash and str(current_plan.get("hash") or "") == candidate_hash:
                return team, False
            if idempotency_key and current_key == idempotency_key:
                if str(current_plan.get("hash") or "") != candidate_hash:
                    raise ValueError(
                        "idempotency_key was already used for a different plan"
                    )
                return team, False
            if expected_revision is not None and expected_revision != current_revision:
                raise ValueError(
                    f"expected plan revision {expected_revision}, current revision is "
                    f"{current_revision}"
                )

            quality = team.settings.get("quality_gates")
            quality = quality if isinstance(quality, dict) else {}
            current_manifest = team.settings.get("execution_manifest")
            current_manifest = (
                current_manifest if isinstance(current_manifest, dict) else {}
            )
            if current_revision > 0:
                budget_errors = execution_budget_manifest_errors(
                    current_plan, current_manifest, usage=team.usage
                )
                if budget_errors:
                    raise ValueError(
                        "active execution budget manifest is invalid: "
                        + "; ".join(budget_errors)
                    )

            current_task_records = (
                self._read_json(tasks_path) if tasks_path.exists() else {}
            )
            active_tasks = [
                str(raw.get("key") or task_id)
                for task_id, raw in current_task_records.items()
                if isinstance(raw, dict) and raw.get("status") == "in_progress"
            ]
            running_agents = []
            for path in agent_dir.glob("*.json"):
                agent = AgentRecord.from_dict(self._read_json(path))
                if agent.status in {"running", "stopping"}:
                    running_agents.append(agent.name)
            if active_tasks or running_agents:
                details = []
                if active_tasks:
                    details.append("active tasks: " + ", ".join(active_tasks))
                if running_agents:
                    details.append("running teammates: " + ", ".join(running_agents))
                raise ValueError(
                    "a running plan cannot be replaced (" + "; ".join(details) + ")"
                )
            accepted_plan = bool(
                current_plan
                and (
                    quality.get("plan_accepted")
                    or current_manifest.get("status") == "accepted"
                )
            )
            checkpoint = team.settings.get("last_replan_checkpoint")
            checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
            checkpoint_valid = bool(
                team.lifecycle_state == "repair_required"
                and checkpoint.get("workspace_preserved") is True
                and int(checkpoint.get("plan_revision") or 0) == current_revision
                and str(checkpoint.get("plan_hash") or "")
                == str(current_plan.get("hash") or "")
                and checkpoint.get("consumed_by_revision") is None
            )
            if accepted_plan and not checkpoint_valid:
                raise ValueError(
                    "an accepted protocol v2 plan is frozen; call TeamReplan before "
                    "submitting a replacement TeamPlan"
                )

            next_plan = dict(plan_record)
            next_plan["revision"] = current_revision + 1
            next_plan["idempotency_key"] = idempotency_key
            next_plan["updated_at"] = utc_now()
            carried_forward = (
                _carry_forward_accepted_tasks(
                    current_tasks=current_task_records,
                    candidate_tasks=tasks,
                    current_plan=current_plan,
                    next_plan=next_plan,
                    checkpoint=checkpoint,
                )
                if checkpoint_valid
                else []
            )
            execution = next_plan.get("execution")
            execution = dict(execution) if isinstance(execution, dict) else {}
            recorded_total = max(
                int(team.usage.get("total_tokens", 0) or 0),
                int(team.usage.get("input_tokens", 0) or 0)
                + int(team.usage.get("output_tokens", 0) or 0),
            )
            usage = {
                "total_tokens": recorded_total,
                "turns": int(team.usage.get("turns", 0) or 0),
            }

            def allocated_ceiling(metric: str, budget_key: str) -> int | None:
                raw = execution.get(budget_key)
                if raw is None:
                    return None
                return usage[metric] + int(raw)

            previous_global = current_manifest.get("global_cap")
            previous_global = (
                previous_global if isinstance(previous_global, dict) else {}
            )

            def inherited_global_cap(metric: str, budget_key: str) -> int | None:
                if current_revision == 0:
                    return allocated_ceiling(metric, budget_key)
                if metric in previous_global:
                    raw = previous_global.get(metric)
                    return int(raw) if raw is not None else None
                # Migration for manifests created before global_cap existed.
                previous_window = current_manifest.get("budget_window")
                if isinstance(previous_window, dict):
                    hard = previous_window.get("hard_ceiling")
                    if isinstance(hard, dict) and metric in hard:
                        raw = hard.get(metric)
                        return int(raw) if raw is not None else None
                return None

            global_cap = {
                "total_tokens": inherited_global_cap(
                    "total_tokens", "token_budget"
                ),
                "turns": inherited_global_cap("turns", "turn_budget"),
            }

            def effective_ceiling(metric: str, budget_key: str) -> int | None:
                allocated = allocated_ceiling(metric, budget_key)
                global_limit = global_cap[metric]
                if allocated is None:
                    return global_limit
                if global_limit is None:
                    return allocated
                return min(allocated, global_limit)

            frozen_at = utc_now()
            execution_manifest = {
                "schema_version": 2,
                "status": "frozen",
                "plan_revision": next_plan["revision"],
                "plan_hash": candidate_hash,
                "execution": execution,
                "frozen_at": frozen_at,
                # The first plan freezes a rollout-wide absolute cap.  Repair
                # revisions receive an incremental window from their current usage
                # baseline, but that window is always clamped to the inherited cap.
                "global_cap": global_cap,
                "budget_window": {
                    "scope": "plan_revision",
                    "baseline": usage,
                    "incremental_limit": {
                        "total_tokens": execution.get("token_budget"),
                        "turns": execution.get("turn_budget"),
                    },
                    "hard_ceiling": {
                        "total_tokens": effective_ceiling(
                            "total_tokens", "token_budget"
                        ),
                        "turns": effective_ceiling("turns", "turn_budget"),
                    },
                },
            }
            execution_manifest["budget_integrity_hash"] = _budget_integrity_hash(
                plan_hash=candidate_hash,
                plan_revision=next_plan["revision"],
                execution=execution,
                global_cap=global_cap,
                budget_window=execution_manifest["budget_window"],
            )
            for key in _TEAM_PLAN_EXECUTION_KEYS:
                team.settings.pop(key, None)
            team.settings.update(settings_updates)
            team.settings["team_plan"] = next_plan
            team.settings["execution_manifest"] = execution_manifest
            team.settings["last_plan_carry_forward"] = {
                "from_revision": current_revision,
                "plan_revision": next_plan["revision"],
                "tasks": carried_forward,
                "requires_acceptance": bool(carried_forward),
                "recorded_at": utc_now(),
            }
            if checkpoint_valid:
                consumed_checkpoint = dict(checkpoint)
                consumed_checkpoint["consumed_by_revision"] = next_plan["revision"]
                consumed_checkpoint["consumed_at"] = utc_now()
                consumed_checkpoint["carried_forward_task_keys"] = [
                    item["key"] for item in carried_forward
                ]
                team.settings["last_replan_checkpoint"] = consumed_checkpoint
            team.settings["protocol_version"] = 2
            # ``protocol_version`` is a first-class Team field in schema v2.  The
            # setattr keeps this method source-compatible while older serialized
            # teams are migrated by Team.from_dict.
            team.protocol_version = 2  # type: ignore[attr-defined]
            # Replacing a completed/failed plan is the explicit v2 reopen action;
            # TeamRun itself never reopens a completed team implicitly.
            if team.status in {"failed", "cancelled"}:
                team.transition_to("running")
            team.set_lifecycle_state("ready")
            team.completed_at = None
            team.cancel_requested_at = None
            team.updated_at = utc_now()

            new_agent_paths = {
                agent_dir / f"{agent.agent_id}.json" for agent in agents
            }
            new_session_paths = {
                session_dir / f"{session_id}.json" for session_id in sessions
            }
            affected_paths = {
                team_path,
                tasks_path,
                self.active_team_path,
                *agent_dir.glob("*.json"),
                *session_dir.glob("*.json"),
                *new_agent_paths,
                *new_session_paths,
            }
            snapshots = {
                path: path.read_bytes() if path.exists() else None
                for path in affected_paths
            }

            try:
                agent_dir.mkdir(parents=True, exist_ok=True)
                session_dir.mkdir(parents=True, exist_ok=True)
                self._write_json_unlocked(team_path, team.to_dict())
                self._write_json_unlocked(
                    tasks_path,
                    {
                        task_id: TeamTask.from_dict(task).to_dict()
                        for task_id, task in tasks.items()
                    },
                )
                for path in agent_dir.glob("*.json"):
                    if path not in new_agent_paths:
                        path.unlink(missing_ok=True)
                for path in session_dir.glob("*.json"):
                    if path not in new_session_paths:
                        path.unlink(missing_ok=True)
                for agent in agents:
                    self._write_json_unlocked(
                        agent_dir / f"{agent.agent_id}.json", agent.to_dict()
                    )
                for session_id, data in sessions.items():
                    self._write_json_unlocked(
                        session_dir / f"{session_id}.json", data
                    )
                active = (
                    Team.from_dict(self._read_json(self.active_team_path))
                    if self.active_team_path.exists()
                    else None
                )
                if active is not None and active.team_id == team_id:
                    self._write_json_unlocked(self.active_team_path, team.to_dict())
            except Exception:
                for path, content in snapshots.items():
                    if content is None:
                        path.unlink(missing_ok=True)
                    else:
                        self._write_bytes_unlocked(path, content)
                raise
            return team, True

    def load_session(self, team_id: str, session_id: str) -> dict[str, Any] | None:
        path = self.team_dir(team_id) / "sessions" / f"{session_id}.json"
        if not path.exists():
            return None
        return self._read_json(path)

    def list_events(self, team_id: str) -> list[dict[str, Any]]:
        path = self.team_dir(team_id) / "events.jsonl"
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
        return events

    def append_event(
        self,
        team_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
        *,
        created_at: str | None = None,
        event_id: str | None = None,
    ) -> None:
        path = self.team_dir(team_id) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "event_id": event_id or uuid.uuid4().hex,
            "team_id": team_id,
            "type": event_type,
            "created_at": created_at or utc_now(),
            "data": data or {},
        }
        # See save_message: event appends must not race a teammate Bash control
        # snapshot, otherwise a valid harness event could be reported as tampering.
        with _locked_team_transaction(self._transaction_path(team_id)):
            with _locked_path(path):
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())

    def disband_active_team(self) -> Team | None:
        team = self.load_active_team()
        if team is None:
            return None
        cancelled = False
        with _locked_team_transaction(self._transaction_path(team.team_id)):
            team = self.load_active_team()
            if team is None:
                return None
            if team.status in {"created", "running", "failed"}:
                team.transition_to("cancelled")
                self._write_json_unlocked(
                    self.team_dir(team.team_id) / "team.json", team.to_dict()
                )
                self._write_json_unlocked(self.active_team_path, team.to_dict())
                cancelled = True
            self.active_team_path.unlink(missing_ok=True)
        if cancelled:
            self.append_event(team.team_id, "team.cancelled")
        return team

    def _migrate_legacy_team(self, data: dict[str, Any]) -> dict[str, Any]:
        team_name = str(data.get("team_name") or "legacy-team")
        seed = f"{self.workspace_root}:{team_name}"
        team_id = uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:12]
        lead_agent_id = str(data.get("lead_agent_id") or uuid.uuid5(uuid.NAMESPACE_OID, seed).hex[:12])
        team = Team(
            team_id=team_id,
            team_name=team_name,
            lead_agent_id=lead_agent_id,
            description=data.get("description"),
            agent_type=data.get("agent_type"),
        )
        directory = self.team_dir(team_id)
        for name in ("agents", "sessions", "messages"):
            (directory / name).mkdir(parents=True, exist_ok=True)
        self._write_json(directory / "team.json", team.to_dict())
        if not (directory / "tasks.json").exists():
            self._write_json(directory / "tasks.json", {})
        (directory / "events.jsonl").touch(exist_ok=True)
        self._write_json(self.active_team_path, team.to_dict())
        return team.to_dict()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"expected a JSON object in {path}")
        return data

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        with _locked_path(path):
            TeamStore._write_json_unlocked(path, data)

    @staticmethod
    def _write_json_unlocked(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
        TeamStore._write_bytes_unlocked(path, payload)

    @staticmethod
    def _write_bytes_unlocked(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp_path.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
