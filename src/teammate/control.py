"""Human and lead control operations for persistent teammate teams."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import AgentRecord, Team, TeamTask, utc_now
from .store import TeamStore


def _active_team(store: TeamStore) -> Team:
    team = store.load_active_team()
    if team is None:
        raise ValueError("no active team")
    return team


def _resolve_task(tasks: dict[str, TeamTask], identity: str) -> TeamTask:
    if identity in tasks:
        return tasks[identity]
    normalized = identity.strip().lower()
    matches = [task for task in tasks.values() if (task.key or "").lower() == normalized]
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous task: {identity}")
    return matches[0]


def mark_teammate_stopped(
    store: TeamStore,
    team_id: str,
    agent_id: str,
) -> AgentRecord:
    """Persist the terminal worker state and emit its event exactly once."""
    changed = False

    def mutate(agent: AgentRecord) -> None:
        nonlocal changed
        if agent.status != "cancelled":
            agent.transition_to("cancelled")
        if agent.stopped_at is None:
            agent.stopped_at = utc_now()
            changed = True

    agent = store.mutate_agent(team_id, agent_id, mutate)
    if agent is None:
        raise ValueError(f"unknown teammate: {agent_id}")
    if changed:
        store.append_event(
            team_id,
            "agent.stopped",
            {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "reason": agent.stop_reason or "worker stopped by lead",
                "task_policy": agent.stop_task_policy or "requeue",
            },
        )
    return agent


def stop_teammate(
    store: TeamStore,
    identity: str,
    *,
    task_policy: str = "requeue",
    reason: str | None = None,
) -> dict[str, Any]:
    """Request one worker to stop without cancelling its team."""
    if task_policy not in {"requeue", "cancel"}:
        raise ValueError("task_policy must be requeue or cancel")
    team = _active_team(store)
    agent = store.find_agent(team.team_id, identity)
    if agent is None:
        raise ValueError(f"unknown teammate: {identity}")
    already_requested = False

    def request_stop(current: AgentRecord) -> None:
        nonlocal already_requested
        if current.status == "completed":
            raise ValueError("completed teammates cannot be stopped")
        if current.stop_requested_at:
            already_requested = True
            return
        current.stop_requested_at = utc_now()
        current.stop_reason = (reason or "stopped by lead").strip()
        current.stop_task_policy = task_policy
        if current.status not in {"stopping", "cancelled"}:
            current.transition_to("stopping")

    updated = store.mutate_agent(team.team_id, agent.agent_id, request_stop)
    if updated is None:
        raise ValueError(f"unknown teammate: {identity}")
    agent = updated
    if already_requested:
        return {
            "team_id": team.team_id,
            "agent_id": agent.agent_id,
            "name": agent.name,
            "status": agent.status,
            "task_policy": agent.stop_task_policy,
            "already_requested": True,
        }

    requested_at = agent.stop_requested_at or utc_now()
    stop_reason = agent.stop_reason or "stopped by lead"
    store.append_event(
        team.team_id,
        "agent.stop_requested",
        {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "reason": stop_reason,
            "task_policy": task_policy,
        },
    )

    changes: dict[str, list[str]] = {
        "active": [],
        "requeued": [],
        "cancelled": [],
    }

    def mutate(tasks: dict[str, TeamTask]) -> None:
        for task in tasks.values():
            if task.owner != agent.agent_id or task.status == "completed":
                continue
            if task.status == "in_progress":
                changes["active"].append(task.id)
                continue
            if task_policy == "requeue":
                if task.status in {"failed", "cancelled"}:
                    task.transition_to("pending")
                task.owner = None
                task.lease_id = None
                task.lease_expires_at = None
                task.completed_at = None
                task.last_error = stop_reason
                task.updated_at = requested_at
                changes["requeued"].append(task.id)
            else:
                already_cancelled = (
                    task.status == "cancelled"
                    and task.lease_id is None
                    and task.lease_expires_at is None
                    and task.last_error == stop_reason
                )
                if already_cancelled:
                    continue
                if task.status != "cancelled":
                    task.transition_to("cancelled")
                task.lease_id = None
                task.lease_expires_at = None
                task.completed_at = requested_at
                task.last_error = stop_reason
                changes["cancelled"].append(task.id)

    store.mutate_tasks(team.team_id, mutate)
    for task_id in changes["requeued"]:
        store.append_event(
            team.team_id,
            "task.requeued",
            {"task_id": task_id, "agent_id": agent.agent_id, "reason": stop_reason},
        )
    for task_id in changes["cancelled"]:
        store.append_event(
            team.team_id,
            "task.cancelled",
            {"task_id": task_id, "agent_id": agent.agent_id, "reason": stop_reason},
        )

    if not changes["active"]:
        agent = mark_teammate_stopped(store, team.team_id, agent.agent_id)

    return {
        "team_id": team.team_id,
        "agent_id": agent.agent_id,
        "name": agent.name,
        "status": agent.status,
        "task_policy": task_policy,
        "active_task_ids": changes["active"],
        "requeued_task_ids": changes["requeued"],
        "cancelled_task_ids": changes["cancelled"],
        "already_requested": False,
    }


def resume_teammate(store: TeamStore, identity: str) -> dict[str, Any]:
    """Make a fully stopped worker available for newly assigned work."""
    team = _active_team(store)
    if team.status == "completed":
        raise ValueError("workers in a completed team cannot be resumed")
    agent = store.find_agent(team.team_id, identity)
    if agent is None:
        raise ValueError(f"unknown teammate: {identity}")

    def resume(current: AgentRecord) -> None:
        if current.status == "stopping":
            raise ValueError("teammate is still stopping")
        if current.status != "cancelled":
            raise ValueError("only cancelled teammates can be resumed")
        current.transition_to("running")
        current.transition_to("idle")
        current.stop_requested_at = None
        current.stop_reason = None
        current.stop_task_policy = None
        current.stopped_at = None

    updated = store.mutate_agent(team.team_id, agent.agent_id, resume)
    if updated is None:
        raise ValueError(f"unknown teammate: {identity}")
    agent = updated
    store.append_event(
        team.team_id,
        "agent.resumed",
        {"agent_id": agent.agent_id, "name": agent.name},
    )
    return {
        "team_id": team.team_id,
        "agent_id": agent.agent_id,
        "name": agent.name,
        "status": agent.status,
    }


def reassign_task(
    store: TeamStore,
    task_identity: str,
    teammate_identity: str,
) -> dict[str, Any]:
    """Assign a non-running task to an available teammate."""
    team = _active_team(store)
    if team.status == "completed":
        raise ValueError("tasks in a completed team cannot be reassigned")
    agent = store.find_agent(team.team_id, teammate_identity)
    if agent is None:
        raise ValueError(f"unknown teammate: {teammate_identity}")
    if agent.status not in {"created", "running", "idle"}:
        raise ValueError(f"teammate is not available: {agent.status}")

    result: dict[str, Any] = {}

    def mutate(tasks: dict[str, TeamTask]) -> None:
        task = _resolve_task(tasks, task_identity)
        if task.status == "completed":
            raise ValueError("completed tasks cannot be reassigned")
        if task.status == "in_progress":
            raise ValueError("stop the current worker before reassigning an active task")
        previous_owner = task.owner
        previous_status = task.status
        if task.status in {"failed", "cancelled"}:
            task.transition_to("pending")
        task.owner = agent.agent_id
        task.lease_id = None
        task.lease_expires_at = None
        task.completed_at = None
        task.last_error = None
        task.updated_at = utc_now()
        result.update(
            {
                "task_id": task.id,
                "task_key": task.key,
                "previous_owner": previous_owner,
                "previous_status": previous_status,
            }
        )

    store.mutate_tasks(team.team_id, mutate)
    store.append_event(
        team.team_id,
        "task.reassigned",
        {
            **result,
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
        },
    )
    return {
        "team_id": team.team_id,
        **result,
        "owner": agent.agent_id,
        "owner_name": agent.name,
        "status": "pending",
    }


def cancel_team(store: TeamStore, reason: str | None = None) -> dict[str, Any]:
    team = _active_team(store)
    if team.status == "completed":
        raise ValueError("completed teams cannot be cancelled")
    team.cancel_requested_at = utc_now()
    if team.status != "cancelled":
        team.transition_to("cancelled")
    store.save_team(team)
    store.append_event(
        team.team_id,
        "team.cancel_requested",
        {"reason": reason or "cancelled by user"},
    )
    return {"team_id": team.team_id, "status": team.status}


def list_teams(workspace_root: str | Path) -> list[dict[str, Any]]:
    store = TeamStore(Path(workspace_root))
    if not store.teams_dir.exists():
        return []
    teams: list[dict[str, Any]] = []
    active = store.load_active_team()
    active_id = active.team_id if active else None
    for directory in sorted(store.teams_dir.iterdir()):
        if not directory.is_dir():
            continue
        team = store.load_team(directory.name)
        if team is None:
            continue
        teams.append(
            {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "status": team.status,
                "active": team.team_id == active_id,
                "updated_at": team.updated_at,
            }
        )
    return teams


def team_status(workspace_root: str | Path, team_id: str | None = None) -> dict[str, Any]:
    store = TeamStore(Path(workspace_root))
    team = store.load_team(team_id) if team_id else store.load_active_team()
    if team is None:
        raise ValueError("team not found")
    return {
        "team": team.to_dict(),
        "agents": [agent.to_dict() for agent in store.list_agents(team.team_id)],
        "tasks": list(store.load_tasks(team.team_id).values()),
        "message_count": len(store.list_messages(team.team_id)),
        "event_count": len(store.list_events(team.team_id)),
    }
