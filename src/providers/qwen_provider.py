"""Qwen 3.5 provider backed by a Tencent TI-ONE OpenAI-compatible service."""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

try:
    from openai import OpenAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = None

from .openai_compatible import OpenAICompatibleProvider


class QwenProvider(OpenAICompatibleProvider):
    """Qwen provider for the configured Tencent TI-ONE deployment."""

    # SGLang/TI-ONE returns aggregate prompt/completion counts in a final
    # choices=[] stream chunk only when include_usage is requested.
    STREAM_INCLUDE_USAGE = True

    DEFAULT_BASE_URL = (
        "https://ms-mnhdj86z-100034032793-sw.gw.ap-zhongwei.ti.tencentcs.com/"
        "ms-mnhdj86z/v1"
    )
    DEFAULT_MODEL = "ms-mnhdj86z"
    ROUTING_HEADER = "X-Clawd-Route-Key"
    ROUTING_KEY_ENV = "QWEN_ROUTING_KEY"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        routing_key: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
    ) -> None:
        resolved_routing_key = routing_key or os.environ.get(self.ROUTING_KEY_ENV)
        self.routing_key = resolved_routing_key or uuid.uuid4().hex
        if not self.routing_key.strip() or any(
            character in self.routing_key for character in "\r\n"
        ):
            raise ValueError("Qwen routing key must be a non-empty HTTP header value")
        self.enable_thinking = enable_thinking
        super().__init__(
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            model=model or self.DEFAULT_MODEL,
        )

    def _create_client(self) -> Any:
        if OpenAI is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "openai package is not installed. Install project dependencies to use QwenProvider."
            )
        # TI-ONE's public gateway expects the AuthToken as the complete
        # Authorization header value. A value beginning with "Bearer " is also
        # preserved, so either gateway authentication form can be configured.
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={
                "Authorization": self.api_key,
                self.ROUTING_HEADER: self.routing_key,
            },
        )

    def _build_request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        options = super()._build_request_kwargs(kwargs)
        enable_thinking = self.enable_thinking
        if enable_thinking is None:
            thinking_value = os.environ.get("QWEN_ENABLE_THINKING", "0")
            enable_thinking = thinking_value.strip().casefold() in {
                "1",
                "true",
                "yes",
                "on",
            }
        options.setdefault(
            "extra_body",
            {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
        return options

    def get_available_models(self) -> list[str]:
        return [self.DEFAULT_MODEL]
