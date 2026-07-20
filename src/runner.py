"""Non-interactive agent execution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .agent.conversation import Conversation
from .config import get_default_provider, get_provider_config
from .providers import get_provider_class, normalize_provider_name
from .teammate.runtime import TeammateRuntime
from .tool_system.agent_loop import (
    AgentLoopResult,
    TextChunkHandler,
    ToolEventHandler,
    run_agent_loop,
)
from .tool_system.context import ToolContext
from .tool_system.defaults import build_default_registry


_PROVIDER_ENV: dict[str, dict[str, tuple[str, ...]]] = {
    "anthropic": {
        "api_key": ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"),
        "base_url": ("ANTHROPIC_BASE_URL",),
        "model": ("ANTHROPIC_MODEL",),
    },
    "openai": {
        "api_key": ("OPENAI_API_KEY",),
        "base_url": ("OPENAI_BASE_URL",),
        "model": ("OPENAI_MODEL",),
    },
    "glm": {
        "api_key": ("ZAI_API_KEY", "ZHIPUAI_API_KEY"),
        "base_url": ("ZAI_BASE_URL", "ZHIPUAI_BASE_URL"),
        "model": ("ZAI_MODEL", "ZHIPUAI_MODEL"),
    },
    "qwen": {
        "api_key": ("QWEN_API_KEY", "TENCENT_TIONE_AUTH_TOKEN"),
        "base_url": ("QWEN_BASE_URL",),
        "model": ("QWEN_MODEL",),
    },
    "minimax": {
        "api_key": ("MINIMAX_API_KEY",),
        "base_url": ("MINIMAX_BASE_URL",),
        "model": ("MINIMAX_MODEL",),
    },
}


def _first_env(
    names: tuple[str, ...], env_overrides: Mapping[str, str] | None = None
) -> str | None:
    for name in names:
        value = (env_overrides or {}).get(name) or os.environ.get(name)
        if value:
            return value
    return None


def _provider_settings(
    provider_name: str,
    config: dict[str, Any],
    model_override: str | None,
    env_overrides: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None, str | None]:
    env = _PROVIDER_ENV.get(provider_name, {})
    api_key = _first_env(env.get("api_key", ()), env_overrides) or config.get(
        "api_key"
    )
    base_url = _first_env(env.get("base_url", ()), env_overrides) or config.get(
        "base_url"
    )
    selected_model = (
        model_override
        or _first_env(env.get("model", ()), env_overrides)
        or config.get("default_model")
    )
    return api_key, base_url, selected_model


def _build_runtime_context(
    workspace: str | Path,
    provider_name: str | None,
    model: str | None,
    *,
    teammate_max_turns: int = 30,
    teammate_min_timeout_s: float | None = None,
    max_output_tokens: int = 4096,
    workspace_backend: Any | None = None,
    provider_env: Mapping[str, str] | None = None,
    include_team_tools: bool = True,
) -> tuple[Any, Any, ToolContext, TeammateRuntime]:
    workspace_root = Path(workspace).expanduser().resolve()
    if not workspace_root.is_dir():
        raise ValueError(f"workspace is not a directory: {workspace_root}")

    selected_provider = normalize_provider_name(provider_name or get_default_provider())
    config = get_provider_config(selected_provider)
    api_key, base_url, selected_model = _provider_settings(
        selected_provider, config, model, provider_env
    )
    if not api_key:
        raise ValueError(
            f"API key is not configured for {selected_provider}; use its environment "
            "variable or run `clawd login`"
        )

    provider_class = get_provider_class(selected_provider)
    provider_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "model": selected_model,
    }
    if selected_provider == "qwen" and provider_env is not None:
        if routing_key := provider_env.get("QWEN_ROUTING_KEY"):
            provider_kwargs["routing_key"] = routing_key
        if "QWEN_ENABLE_THINKING" in provider_env:
            provider_kwargs["enable_thinking"] = provider_env[
                "QWEN_ENABLE_THINKING"
            ].strip().casefold() in {"1", "true", "yes", "on"}
    provider: Any = provider_class(
        **provider_kwargs,
    )
    registry = build_default_registry(
        include_user_tools=workspace_backend is None,
        workspace_backend=workspace_backend,
        include_team_tools=include_team_tools,
    )
    context = ToolContext(
        workspace_root=workspace_root,
        workspace_backend=workspace_backend,
        execution_workspace_root=(
            str(getattr(workspace_backend, "workspace_root", "/workspace"))
            if workspace_backend is not None
            else None
        ),
    )
    runtime = TeammateRuntime(
        provider,
        registry,
        max_turns=teammate_max_turns,
        max_output_tokens=max_output_tokens,
        allowed_models={selected_model} if selected_model else None,
        minimum_timeout_s=teammate_min_timeout_s,
    )
    context.teammate_runtime = runtime
    if model:
        context.model_override = model
    return provider, registry, context, runtime


def run_prompt(
    prompt: str,
    *,
    workspace: str | Path = ".",
    provider_name: str | None = None,
    model: str | None = None,
    max_turns: int = 100,
    teammate_max_turns: int = 30,
    teammate_min_timeout_s: float | None = None,
    max_output_tokens: int = 4096,
    stream: bool = False,
    on_event: ToolEventHandler | None = None,
    on_text_chunk: TextChunkHandler | None = None,
    workspace_backend: Any | None = None,
    provider_env: Mapping[str, str] | None = None,
    include_team_tools: bool = True,
) -> AgentLoopResult:
    """Run one prompt to completion without starting the interactive REPL."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be non-empty")
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if teammate_max_turns < 1:
        raise ValueError("teammate_max_turns must be at least 1")
    if teammate_min_timeout_s is not None and teammate_min_timeout_s < 1:
        raise ValueError("teammate_min_timeout_s must be at least 1 when provided")
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be at least 1")

    provider, registry, context, _ = _build_runtime_context(
        workspace,
        provider_name,
        model,
        teammate_max_turns=teammate_max_turns,
        teammate_min_timeout_s=teammate_min_timeout_s,
        max_output_tokens=max_output_tokens,
        workspace_backend=workspace_backend,
        provider_env=provider_env,
        include_team_tools=include_team_tools,
    )

    conversation = Conversation()
    conversation.add_user_message(prompt.strip())
    return run_agent_loop(
        conversation=conversation,
        provider=provider,
        tool_registry=registry,
        tool_context=context,
        max_turns=max_turns,
        max_output_tokens=max_output_tokens,
        stream=stream,
        verbose=False,
        on_event=on_event,
        on_text_chunk=on_text_chunk,
    )


def resume_team(
    *,
    workspace: str | Path = ".",
    provider_name: str | None = None,
    model: str | None = None,
    max_turns: int = 30,
    max_workers: int | None = None,
    timeout_s: float | None = None,
    token_budget: int | None = None,
    turn_budget: int | None = None,
    max_retries: int | None = None,
    lease_timeout_s: int | None = None,
    retry_failed: bool = True,
    retry_cancelled: bool = True,
) -> dict[str, Any]:
    """Resume the active persisted team without a lead model round trip."""
    _, _, context, runtime = _build_runtime_context(
        workspace,
        provider_name,
        model,
        teammate_max_turns=max_turns,
    )
    if context.team is None:
        raise ValueError("no active team")
    return runtime.run_team(
        context,
        resume=True,
        retry_failed=retry_failed,
        retry_cancelled=retry_cancelled,
        max_workers=max_workers,
        timeout_s=timeout_s,
        token_budget=token_budget,
        turn_budget=turn_budget,
        max_retries=max_retries,
        lease_timeout_s=lease_timeout_s,
    )
