"""Agent loop for multi-turn tool calling."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from .registry import ToolRegistry
from .context import ToolContext
from ..agent.conversation import Conversation, TextContentBlock, ToolUseContentBlock
from ..context_system import build_context_prompt
from ..outputStyles import resolve_output_style
from ..providers.base import BaseProvider, ChatResponse
from ..providers.anthropic_provider import AnthropicProvider
from ..providers.minimax_provider import MinimaxProvider
from ..teammate.trace import TeamTraceRecorder


_LOCAL_TOOL_GUIDANCE = """## Local Engineering Tools
- Local workspace files are accessed with `Read`, `Glob`, and `Grep`.
- Use `Write` to create files and `Edit` for exact replacements in files you have read.
- Use `Bash` to run local commands and tests.
- Use web tools only for HTTP or HTTPS resources. Never send local paths or `file://` URLs to `webReader`, `WebFetch`, or other web tools.
- These local tools are registered even if a provider also offers built-in web tools. If unsure, call `ToolSearch` with focused keywords or query `*`; do not conclude that a tool is unavailable after one empty search."""

_LEADER_TEAM_GUIDANCE = """## Adaptive Team Orchestration
- You are the lead and remain responsible for the final result. First decide whether delegation is likely to improve quality, latency, or context coverage enough to justify its cost. It is valid to complete the task without creating a team.
- When a team is useful, choose the number of teammates, their roles, models, tool allowlists, workspace modes, tasks, dependencies, and concurrency from the task itself. Do not default to a fixed planner/coder/reviewer pipeline.
- Teammates may communicate directly with one another through `SendMessage` and receive new peer messages through `ReadMessages`; useful peer coordination does not need to be routed through the lead.
- Use stable task keys and explicit ownership. Prefer dependencies only when work truly must be sequential, and allow independent tasks to run in parallel.
- Observe persisted task, message, and agent state. You may run a bounded number of scheduling batches, then add or reassign tasks, adjust dependencies, stop or resume workers, recover failures, and continue.
- Treat teammate output as evidence, not authority. Integrate the work, run final verification yourself, and stop unnecessary work when the expected value of more collaboration is low.
- The resulting communication topology is an execution outcome, not a prescribed shape."""


def _is_anthropic_provider(provider: BaseProvider) -> bool:
    return isinstance(provider, (AnthropicProvider, MinimaxProvider))


def _build_tool_result_content(result_output: Any) -> str:
    """Serialize structured tool output into provider-safe text."""
    if isinstance(result_output, str):
        return result_output
    return json.dumps(result_output, ensure_ascii=False)

def summarize_tool_result(name: str, output: Any) -> str:
    """Create a concise, single-line summary for tool result output."""
    if not isinstance(output, dict):
        return str(output)
    if name.lower() == "write":
        path = output.get("filePath") or output.get("file_path")
        op = output.get("type")
        return f"{name} · {path} · {op}"
    if name.lower() == "edit":
        path = output.get("filePath") or output.get("file_path")
        replace_all = output.get("replaceAll")
        return f"{name} · {path} · replaceAll={replace_all}"
    if name.lower() == "read":
        if output.get("type") == "text" and isinstance(output.get("file"), dict):
            f = output["file"]
            path = f.get("filePath")
            num = f.get("numLines")
            total = f.get("totalLines")
            start = f.get("startLine")
            return f"{name} · {path} · lines={start}-{(start or 1) + (num or 0) - 1}/{total}"
        if output.get("type") == "file_unchanged" and isinstance(output.get("file"), dict):
            return f"{name} · {output['file'].get('filePath')} · unchanged"
        if output.get("type") in {"image", "pdf", "notebook"} and isinstance(output.get("file"), dict):
            return f"{name} · {output['file'].get('filePath')} · {output.get('type')}"
        return f"{name}"
    if name.lower() == "glob":
        n = output.get("numFiles")
        return f"{name} · matches={n}"
    if name.lower() == "grep":
        n = output.get("numFiles")
        mode = output.get("mode")
        return f"{name} · mode={mode} · files={n}"
    if name.lower() == "bash":
        code = output.get("exit_code")
        return f"{name} · exit={code}"
    if name.lower() == "webfetch":
        url = output.get("url")
        ct = output.get("content_type")
        return f"{name} · {url} · {ct}"
    if name.lower() == "websearch":
        q = output.get("query")
        results = output.get("results")
        n = len(results) if isinstance(results, list) else None
        return f"{name} · \"{q}\" · results={n}"
    if name.lower() == "config":
        op = output.get("operation")
        setting = output.get("setting")
        return f"{name} · {op} · {setting}"
    if name.lower() == "taskstop":
        tid = output.get("task_id")
        stopped = output.get("stopped")
        return f"{name} · {tid} · stopped={stopped}"
    if name.lower() == "sendusermessage":
        n = 0
        atts = output.get("attachments")
        if isinstance(atts, list):
            n = len(atts)
        return f"{name} · attachments={n}"
    # default: truncate dict keys for brevity
    keys = ", ".join(list(output.keys())[:3])
    return f"{name} · {keys}"


@dataclass(frozen=True)
class ToolEvent:
    kind: str
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: Any | None = None
    tool_use_id: str | None = None
    is_error: bool = False
    error: str | None = None
    content: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    turn: int | None = None
    duration_ms: int | None = None
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class AgentLoopResult:
    """Result of running the agent loop."""
    response_text: str
    usage: dict[str, Any] | None = None  # {"input_tokens": int, "output_tokens": int}
    num_turns: int = 0
    cancelled: bool = False


ToolEventHandler = Callable[[ToolEvent], None]
TextChunkHandler = Callable[[str], None]
StopPredicate = Callable[[], bool]


def _safe_call_handler(handler: ToolEventHandler | None, event: ToolEvent) -> None:
    if handler is None:
        return
    try:
        handler(event)
    except Exception:
        return


def _safe_should_stop(predicate: StopPredicate | None) -> bool:
    if predicate is None:
        return False
    try:
        return bool(predicate())
    except Exception:
        return False


def _emit_text_chunks(handler: TextChunkHandler | None, text: str, *, chunk_size: int = 12) -> None:
    """Emit text in small chunks for user-visible streaming without changing loop semantics."""
    if handler is None or not text:
        return
    if chunk_size <= 0:
        chunk_size = len(text)
    for idx in range(0, len(text), chunk_size):
        try:
            handler(text[idx: idx + chunk_size])
        except Exception:
            return


def _call_provider_for_turn(
    *,
    provider: BaseProvider,
    api_messages: list[dict[str, Any]],
    call_kwargs: dict[str, Any],
    stream: bool,
    on_text_chunk: TextChunkHandler | None,
) -> tuple[Any, bool]:
    """Call the provider, preferring structured streaming when available.

    Returns (response, streamed_live_text).
    """
    if stream:
        try:
            response = provider.chat_stream_response(
                api_messages,
                on_text_chunk=on_text_chunk,
                **call_kwargs,
            )
            if not isinstance(response, ChatResponse):
                raise TypeError("Structured streaming must return ChatResponse")
            return response, True
        except NotImplementedError:
            pass
        except Exception:
            # Preserve existing stable behavior if streaming is unsupported or fails.
            pass

    response = provider.chat(api_messages, **call_kwargs)
    return response, False


def _build_effective_system_prompt(style_prompt: str, tool_context: ToolContext) -> str:
    try:
        context_prompt = build_context_prompt(
            tool_context.workspace_root,
            cwd=tool_context.cwd,
        )
    except Exception:
        context_prompt = ""
    sections = [style_prompt, _LOCAL_TOOL_GUIDANCE]
    if tool_context.actor_id is None:
        sections.append(_LEADER_TEAM_GUIDANCE)
    if tool_context.system_prompt_extra and tool_context.system_prompt_extra.strip():
        sections.append(tool_context.system_prompt_extra.strip())
    if context_prompt.strip():
        sections.append(context_prompt)
    return "\n\n".join(sections)


def summarize_tool_use(name: str, tool_input: dict[str, Any]) -> str:
    lowered = name.lower()
    if lowered == "bash":
        cmd = tool_input.get("command")
        if isinstance(cmd, str):
            s = cmd.strip().replace("\n", " ")
            return s if len(s) <= 80 else s[:77] + "..."
        return ""
    if lowered in {"read", "write", "edit"}:
        p = tool_input.get("file_path") or tool_input.get("filePath") or tool_input.get("path")
        if isinstance(p, str):
            extra = ""
            if lowered == "read":
                off = tool_input.get("offset")
                lim = tool_input.get("limit")
                if isinstance(off, int) or isinstance(lim, int):
                    start = off if isinstance(off, int) else 1
                    if isinstance(lim, int):
                        extra = f" · lines {start}-{start + lim - 1}"
            return f"{p}{extra}"
        return ""
    if lowered == "glob":
        pat = tool_input.get("pattern")
        base = tool_input.get("path")
        if isinstance(pat, str) and isinstance(base, str):
            return f"{pat} · {base}"
        if isinstance(pat, str):
            return pat
        return ""
    if lowered == "grep":
        pat = tool_input.get("pattern")
        base = tool_input.get("path")
        if isinstance(pat, str) and isinstance(base, str):
            return f"{pat} · {base}"
        if isinstance(pat, str):
            return pat
        return ""
    if lowered == "webfetch":
        url = tool_input.get("url")
        return url if isinstance(url, str) else ""
    if lowered == "websearch":
        q = tool_input.get("query")
        return q if isinstance(q, str) else ""
    if lowered == "toolsearch":
        q = tool_input.get("query")
        return q if isinstance(q, str) else ""
    if lowered == "askuserquestion":
        qs = tool_input.get("questions")
        if isinstance(qs, list):
            return f"{len(qs)} question(s)"
        return ""
    if lowered == "sendusermessage":
        status = tool_input.get("status")
        return status if isinstance(status, str) else ""
    return ""



def run_agent_loop(
    conversation: Conversation,
    provider: BaseProvider,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    max_turns: int = 20,
    stream: bool = False,
    verbose: bool = False,
    on_event: ToolEventHandler | None = None,
    on_text_chunk: TextChunkHandler | None = None,
    should_stop: StopPredicate | None = None,
) -> AgentLoopResult:
    """Run agent loop: LLM -> tools -> LLM until no more tools or max turns.

    Args:
        conversation: Conversation with initial user message
        provider: LLM provider
        tool_registry: Tool registry to use
        tool_context: Tool context
        max_turns: Maximum tool turns before stopping
        stream: Whether to stream responses
        verbose: Whether to print tool calls/results
        on_event: Optional callback for tool events
        on_text_chunk: Optional callback for incremental user-visible text chunks
        should_stop: Optional cooperative cancellation check at model/tool boundaries

    Returns:
        AgentLoopResult with final text response, usage info, and turn count
    """
    # Convert tools to schemas (Anthropic format)
    tool_schemas = []
    for spec in tool_registry.list_specs():
        tool_schemas.append({
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        })

    # For OpenAI/GLM, keep separate message list in OpenAI format
    openai_messages: list[dict[str, Any]] = []
    last_user_visible_message: str | None = None
    style_name = getattr(tool_context, "output_style_name", None)
    style_dir = getattr(tool_context, "output_style_dir", None)
    style_prompt = resolve_output_style(style_name, style_dir).prompt
    effective_system_prompt = _build_effective_system_prompt(style_prompt, tool_context)

    # Seed OpenAI messages from initial conversation messages
    for msg in conversation.messages:
        if isinstance(msg.content, str):
            openai_messages.append({"role": msg.role, "content": msg.content})
        else:
            # If there are already block messages, we are probably Anthropic; leave as is
            pass

    # Track usage across all turns
    total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    turn_count = 0
    trace_recorder = TeamTraceRecorder(tool_context)

    def emit(event: ToolEvent) -> None:
        _safe_call_handler(on_event, event)
        trace_recorder.record(event)

    def finish(
        response_text: str,
        *,
        failed: bool = False,
        cancelled: bool = False,
    ) -> AgentLoopResult:
        usage = total_usage if total_usage["input_tokens"] > 0 or total_usage["output_tokens"] > 0 else None
        emit(ToolEvent(
            kind="run_cancelled" if cancelled else ("run_failed" if failed else "run_completed"),
            content=response_text,
            model=tool_context.model_override or getattr(provider, "model", None),
            usage=usage,
            turn=turn_count,
            is_error=failed,
            error=response_text if failed else None,
        ))
        return AgentLoopResult(
            response_text=response_text,
            usage=usage,
            num_turns=turn_count,
            cancelled=cancelled,
        )

    emit(ToolEvent(
        kind="run_started",
        model=tool_context.model_override or getattr(provider, "model", None),
        turn=0,
    ))

    for turn in range(max_turns):
        if _safe_should_stop(should_stop):
            return finish("[Run stopped]", cancelled=True)
        if _is_anthropic_provider(provider):
            api_messages = conversation.get_messages()
        else:
            # Use OpenAI formatted messages for non-Anthropic
            api_messages = openai_messages

        call_kwargs: dict[str, Any] = {"tools": tool_schemas}
        if tool_context.model_override:
            call_kwargs["model"] = tool_context.model_override
        if _is_anthropic_provider(provider):
            call_kwargs["system"] = effective_system_prompt
        else:
            if turn == 0:
                api_messages = [{"role": "system", "content": effective_system_prompt}, *api_messages]
        model_name = tool_context.model_override or getattr(provider, "model", None)
        emit(ToolEvent(kind="model_started", model=model_name, turn=turn + 1))
        model_started_at = time.monotonic()
        try:
            response, streamed_live_text = _call_provider_for_turn(
                provider=provider,
                api_messages=api_messages,
                call_kwargs=call_kwargs,
                stream=stream,
                on_text_chunk=on_text_chunk,
            )
        except Exception as exc:
            duration_ms = round((time.monotonic() - model_started_at) * 1000)
            error = f"{type(exc).__name__}: {exc}"
            emit(ToolEvent(
                kind="model_error",
                model=model_name,
                turn=turn + 1,
                duration_ms=duration_ms,
                is_error=True,
                error=error,
            ))
            finish(error, failed=True)
            raise
        turn_count += 1

        # Collect usage info
        if response.usage:
            total_usage["input_tokens"] += response.usage.get("input_tokens", 0)
            total_usage["output_tokens"] += response.usage.get("output_tokens", 0)

        # Build assistant content for Anthropic or just text for OpenAI
        final_assistant_content = response.content or ""
        emit(ToolEvent(
            kind="model_response",
            content=final_assistant_content,
            model=response.model or model_name,
            usage=response.usage,
            finish_reason=response.finish_reason,
            turn=turn_count,
            duration_ms=round((time.monotonic() - model_started_at) * 1000),
        ))
        if _safe_should_stop(should_stop):
            return finish("[Run stopped]", cancelled=True)

        if _is_anthropic_provider(provider):
            assistant_blocks: list = []
            if response.content:
                assistant_blocks.append(TextContentBlock(type="text", text=response.content))

            tool_uses = response.tool_uses or []
            for tool_use in tool_uses:
                assistant_blocks.append(ToolUseContentBlock(
                    type="tool_use",
                    id=tool_use["id"],
                    name=tool_use["name"],
                    input=tool_use["input"],
                ))

            conversation.add_assistant_message(assistant_blocks if assistant_blocks else "")
        else:
            # Persist assistant text for session history features like /render-last
            # and for subsequent non-Anthropic turns seeded from conversation.
            conversation.add_assistant_message(final_assistant_content)
            # Add assistant message to OpenAI messages (text only)
            openai_assistant_msg: dict[str, Any] = {"role": "assistant", "content": final_assistant_content}
            # If there are tool_uses, add them in OpenAI format
            if response.tool_uses:
                # Build OpenAI tool_calls
                tool_calls = []
                for tu in response.tool_uses:
                    tool_calls.append({
                        "id": tu["id"],
                        "type": "function",
                        "function": {
                            "name": tu["name"],
                            "arguments": json.dumps(tu["input"], ensure_ascii=False)
                        }
                    })
                openai_assistant_msg["tool_calls"] = tool_calls
            openai_messages.append(openai_assistant_msg)

        tool_uses = response.tool_uses or []

        if not tool_uses:
            # No more tools, done
            if stream and final_assistant_content and not streamed_live_text:
                _emit_text_chunks(on_text_chunk, final_assistant_content)
            if (final_assistant_content or "").strip() == "" and last_user_visible_message is not None:
                return finish(last_user_visible_message)
            return finish(final_assistant_content)

        # Call each tool
        for tool_use in tool_uses:
            if _safe_should_stop(should_stop):
                return finish("[Run stopped]", cancelled=True)
            tool_id = tool_use["id"]
            tool_name = tool_use["name"]
            tool_input = tool_use["input"]
            tool_started_at = time.monotonic()

            try:
                emit(
                    ToolEvent(
                        kind="tool_use",
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_use_id=tool_id,
                    ),
                )
                # Use dispatch to get proper validation and result wrapping
                from ..tool_system.protocol import ToolCall
                call = ToolCall(name=tool_name, input=tool_input, tool_use_id=tool_id)
                result = tool_registry.dispatch(call, tool_context)
                result_output = result.output
                if tool_name.lower() == "sendusermessage" and isinstance(result_output, dict):
                    msg = result_output.get("message")
                    if isinstance(msg, str):
                        last_user_visible_message = msg
                if tool_name.lower() == "structuredoutput" and isinstance(result_output, dict):
                    payload = result_output.get("structured_output")
                    try:
                        last_user_visible_message = json.dumps(payload, ensure_ascii=False, indent=2)
                    except Exception:
                        last_user_visible_message = str(payload)

                if verbose:
                    use_summary = summarize_tool_use(tool_name, tool_input)
                    if use_summary:
                        print(f"{tool_name} · {use_summary}")
                    summary = summarize_tool_result(tool_name, result_output)
                    print(f"{summary}")

                emit(
                    ToolEvent(
                        kind="tool_result",
                        tool_name=tool_name,
                        tool_output=result_output,
                        tool_use_id=tool_id,
                        is_error=result.is_error,
                        duration_ms=round((time.monotonic() - tool_started_at) * 1000),
                    ),
                )
                if _is_anthropic_provider(provider):
                    conversation.add_tool_result_message(tool_id, _build_tool_result_content(result_output))
                else:
                    # Add tool result in OpenAI format
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": _build_tool_result_content(result_output)
                    })
            except Exception as e:
                error_str = f"Error: {e}"
                if verbose:
                    print(f"[Tool Error] {error_str}")
                emit(
                    ToolEvent(
                        kind="tool_error",
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_use_id=tool_id,
                        is_error=True,
                        error=error_str,
                        duration_ms=round((time.monotonic() - tool_started_at) * 1000),
                    ),
                )
                if _is_anthropic_provider(provider):
                    conversation.add_tool_result_message(tool_id, error_str, is_error=True)
                else:
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": error_str
                    })

            if _safe_should_stop(should_stop):
                return finish("[Run stopped]", cancelled=True)

    # Reached max turns
    return finish("[Max tool turns reached]", failed=True)
