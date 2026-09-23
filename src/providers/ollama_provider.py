"""Ollama provider — local models via Ollama's native /api/chat endpoint.

Ollama exposes an OpenAI-compatible facade, but it does not forward the
`think` parameter that Qwen 3.5 / DeepSeek-R1 reasoning models need to
disable thinking mode. Without `think=false` those models burn the
entire token budget inside `<think>...</think>` blocks before emitting
any content, and the OpenAI facade returns an empty string.

This provider talks to `/api/chat` directly, passes `think` through, and
streams messages the Ollama-native way. It also exposes the model's
reasoning trace via `ChatResponse.reasoning_content` when the model
emits one (think=true).

Env:
    OLLAMA_BASE_URL  default http://localhost:11434
    OLLAMA_THINK     "true" / "false" — default false (fast, content-first)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Generator, Optional

from .base import BaseProvider, ChatResponse, MessageInput, TextChunkCallback


DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    """Drive a locally-running Ollama daemon.

    Tested models:
      - huihui_ai/qwen3.5-abliterated:4B  (refusal-removed Qwen 3.5)
      - llama3.2:3b
      - deepseek-r1, qwen3, and any other chat-capable Ollama model
    """

    def __init__(
        self,
        api_key: str = "",  # Ollama has no auth
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        *,
        think: Optional[bool] = None,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL,
            model=model or "llama3.2:3b",
        )
        if think is None:
            env_think = os.environ.get("OLLAMA_THINK", "false").lower()
            think = env_think in ("1", "true", "yes", "on")
        self.think = think

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, body: dict[str, Any], stream: bool):
        url = f"{self.base_url.rstrip('/')}{path}"
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        return urllib.request.urlopen(req)

    def _chat_body(
        self,
        messages: list[MessageInput],
        stream: bool,
        **kwargs,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._get_model(**kwargs),
            "messages": self._prepare_messages(messages),
            "stream": stream,
            "think": bool(kwargs.get("think", self.think)),
        }
        options: dict[str, Any] = {}
        if (n := kwargs.get("max_tokens")) is not None:
            options["num_predict"] = int(n)
        if (n := kwargs.get("num_predict")) is not None:
            options["num_predict"] = int(n)
        if (t := kwargs.get("temperature")) is not None:
            options["temperature"] = float(t)
        if (t := kwargs.get("top_p")) is not None:
            options["top_p"] = float(t)
        if (t := kwargs.get("top_k")) is not None:
            options["top_k"] = int(t)
        if options:
            body["options"] = options
        return body

    # ------------------------------------------------------------------
    # BaseProvider surface
    # ------------------------------------------------------------------

    def get_available_models(self) -> list[str]:
        """Query /api/tags for installed models."""
        try:
            with self._post("/api/tags", {}, stream=False) as r:
                payload = json.load(r)
        except urllib.error.URLError:
            # Fallback: try GET on /api/tags (that's what Ollama actually uses)
            req = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/api/tags", method="GET"
            )
            try:
                with urllib.request.urlopen(req) as r:
                    payload = json.load(r)
            except Exception:
                return []
        return [m["name"] for m in payload.get("models", [])]

    def chat(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Non-streaming chat. Collects the whole response in one shot."""
        body = self._chat_body(messages, stream=False, **kwargs)
        if tools:
            body["tools"] = tools
        with self._post("/api/chat", body, stream=False) as r:
            payload = json.load(r)
        msg = payload.get("message", {})
        return ChatResponse(
            content=msg.get("content", "") or "",
            model=payload.get("model", body["model"]),
            usage={
                "input_tokens": payload.get("prompt_eval_count", 0),
                "output_tokens": payload.get("eval_count", 0),
                "total_tokens": (payload.get("prompt_eval_count", 0)
                                 + payload.get("eval_count", 0)),
            },
            finish_reason=payload.get("done_reason", "stop"),
            reasoning_content=msg.get("thinking") or None,
            tool_uses=msg.get("tool_calls"),
        )

    def chat_stream(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream assistant content (not reasoning) chunk by chunk."""
        body = self._chat_body(messages, stream=True, **kwargs)
        if tools:
            body["tools"] = tools
        with self._post("/api/chat", body, stream=True) as r:
            for raw in r:
                line = raw.strip()
                if not line:
                    continue
                msg = json.loads(line)
                chunk = msg.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if msg.get("done"):
                    break

    def chat_stream_response(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        on_text_chunk: TextChunkCallback | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Stream and return the rebuilt ChatResponse — preserves reasoning."""
        body = self._chat_body(messages, stream=True, **kwargs)
        if tools:
            body["tools"] = tools
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        last_msg: dict[str, Any] = {}
        final_payload: dict[str, Any] = {}
        with self._post("/api/chat", body, stream=True) as r:
            for raw in r:
                line = raw.strip()
                if not line:
                    continue
                event = json.loads(line)
                last_msg = event.get("message", {})
                if (c := last_msg.get("content", "")):
                    content_parts.append(c)
                    if on_text_chunk is not None:
                        on_text_chunk(c)
                if (t := last_msg.get("thinking", "")):
                    reasoning_parts.append(t)
                if event.get("done"):
                    final_payload = event
                    break
        return ChatResponse(
            content="".join(content_parts),
            model=final_payload.get("model", body["model"]),
            usage={
                "input_tokens": final_payload.get("prompt_eval_count", 0),
                "output_tokens": final_payload.get("eval_count", 0),
                "total_tokens": (final_payload.get("prompt_eval_count", 0)
                                 + final_payload.get("eval_count", 0)),
            },
            finish_reason=final_payload.get("done_reason", "stop"),
            reasoning_content="".join(reasoning_parts) or None,
            tool_uses=last_msg.get("tool_calls"),
        )
