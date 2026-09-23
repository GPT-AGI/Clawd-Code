from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..config import get_default_provider, get_provider_config
from ..providers import get_provider_class, normalize_provider_name
from ..runner import _provider_settings
from ..tool_system.defaults import build_default_registry
from ..tool_system.registry import ToolRegistry
from .backend import AgentLoopPeerBackend, PeerSessionBackend, PeerSessionSpec
from .models import PeerRunConfig
from .runtime import PeerRuntime


def build_peer_provider_factory(
    provider_name: str | None, model: str | None
) -> tuple[str, str | None, Callable[[PeerSessionSpec], Any]]:
    selected = normalize_provider_name(provider_name or get_default_provider())
    provider_config = get_provider_config(selected)
    api_key, base_url, selected_model = _provider_settings(
        selected, provider_config, model
    )
    if not api_key:
        raise ValueError(
            f"API key is not configured for {selected}; configure it before a real peer run"
        )
    provider_class = get_provider_class(selected)

    def factory(spec: PeerSessionSpec) -> Any:
        return provider_class(
            api_key=api_key,
            base_url=base_url,
            model=spec.model or selected_model,
        )

    return selected, selected_model, factory


def run_peer_collaboration(
    mission: str,
    *,
    repo: str | Path,
    peers: int,
    communication: str,
    workspace_mode: str,
    provider_name: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 300.0,
    max_turns: int = 30,
    max_output_tokens: int = 4096,
    token_budget: int | None = None,
    turn_budget: int | None = None,
    output_dir: str | Path | None = None,
    coordinator_peer: str | None = None,
    acceptance_command: list[str] | None = None,
    cleanup_worktrees: bool = True,
    backend: PeerSessionBackend | None = None,
    base_registry: ToolRegistry | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(mission, str) or not mission.strip():
        raise ValueError("mission must be a non-empty string")
    selected_provider = provider_name or "scripted"
    selected_model = model
    selected_backend = backend
    if selected_backend is None:
        selected_provider, selected_model, factory = build_peer_provider_factory(
            provider_name, model
        )
        selected_backend = AgentLoopPeerBackend(factory)
    registry = base_registry or build_default_registry(include_user_tools=False)
    config = PeerRunConfig(
        repo_path=str(Path(repo).expanduser().resolve()),
        mission=mission.strip(),
        peers=peers,
        communication=communication,
        workspace_mode=workspace_mode,
        provider=selected_provider,
        model=selected_model,
        timeout_seconds=timeout_seconds,
        max_turns=max_turns,
        max_output_tokens=max_output_tokens,
        token_budget=token_budget,
        turn_budget=turn_budget,
        output_dir=(str(Path(output_dir).expanduser().resolve()) if output_dir else None),
        coordinator_peer=coordinator_peer,
        acceptance_command=acceptance_command,
        cleanup_worktrees=cleanup_worktrees,
    )
    return PeerRuntime(selected_backend, registry).run(config, run_id=run_id)
