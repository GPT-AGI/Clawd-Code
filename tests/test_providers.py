"""Tests for LLM providers."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.providers import get_provider_class
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.glm_provider import GLMProvider
from src.providers.openai_provider import OpenAIProvider
from src.providers.qwen_provider import QwenProvider
from src.providers.base import ChatMessage, ChatResponse


class TestChatMessage(unittest.TestCase):
    """Test ChatMessage dataclass."""

    def test_create_message(self):
        """Test creating a chat message."""
        msg = ChatMessage(role="user", content="Hello")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello")

    def test_to_dict(self):
        """Test converting message to dict."""
        msg = ChatMessage(role="user", content="Hello")
        result = msg.to_dict()
        self.assertEqual(result, {"role": "user", "content": "Hello"})


class TestChatResponse(unittest.TestCase):
    """Test ChatResponse dataclass."""

    def test_create_response(self):
        """Test creating a chat response."""
        response = ChatResponse(
            content="Hello!",
            model="gpt-4",
            usage={"input_tokens": 10, "output_tokens": 5},
            finish_reason="stop",
        )
        self.assertEqual(response.content, "Hello!")
        self.assertEqual(response.model, "gpt-4")
        self.assertIsNone(response.reasoning_content)


    def test_response_with_reasoning(self):
        """Test response with reasoning content."""
        response = ChatResponse(
            content="Answer",
            model="glm-4.5",
            usage={"input_tokens": 10, "output_tokens": 5},
            finish_reason="stop",
            reasoning_content="Reasoning process...",
        )
        self.assertEqual(response.reasoning_content, "Reasoning process...")


class TestOpenAICompatibleUsage(unittest.TestCase):
    def test_cached_prompt_tokens_are_preserved(self):
        provider = OpenAIProvider(api_key="test-key")
        usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details={"cached_tokens": 80},
        )

        actual = provider._build_usage_dict(usage)

        self.assertEqual(actual["cache_read_input_tokens"], 80)


class TestAnthropicProvider(unittest.TestCase):
    """Test Anthropic provider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = AnthropicProvider(api_key="test_key")
        self.assertEqual(provider.model, "claude-sonnet-4-6")
        self.assertEqual(provider.api_key, "test_key")

    def test_custom_model(self):
        """Test provider with custom model."""
        provider = AnthropicProvider(api_key="test_key", model="claude-3-opus-20240229")
        self.assertEqual(provider.model, "claude-3-opus-20240229")

    def test_get_available_models(self):
        """Test getting available models."""
        provider = AnthropicProvider(api_key="test_key")
        models = provider.get_available_models()
        self.assertIn("claude-sonnet-4-20250514", models)
        self.assertIn("claude-3-5-sonnet-20241022", models)

    @patch("src.providers.anthropic_provider.anthropic.Anthropic")
    def test_chat(self, mock_anthropic):
        """Test synchronous chat."""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Mock text block with type and text attributes
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello!"
        mock_response.content = [mock_text_block]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Test
        provider = AnthropicProvider(api_key="test_key")
        messages = [ChatMessage(role="user", content="Hi")]
        response = provider.chat(messages)

        self.assertEqual(response.content, "Hello!")
        self.assertEqual(response.model, "claude-sonnet-4-20250514")
        self.assertEqual(response.finish_reason, "end_turn")

    @patch("src.providers.anthropic_provider.anthropic.Anthropic")
    def test_chat_accepts_dict_messages(self, mock_anthropic):
        """Test synchronous chat with dict messages."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Mock text block with type and text attributes
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello!"
        mock_response.content = [mock_text_block]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(api_key="test_key")
        messages = [{"role": "user", "content": "Hi"}]
        response = provider.chat(messages)

        self.assertEqual(response.content, "Hello!")
        mock_client.messages.create.assert_called_once()
        self.assertEqual(
            mock_client.messages.create.call_args.kwargs["messages"], messages
        )

    @patch("src.providers.anthropic_provider.anthropic.Anthropic")
    def test_chat_stream_response_with_tool_use(self, mock_anthropic):
        """Structured streaming returns final text and tool uses."""
        mock_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_stream
        mock_stream.__exit__.return_value = False
        mock_stream.text_stream = iter(["Hello", " world"])

        final_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello world"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_1"
        tool_block.name = "Read"
        tool_block.input = {"file_path": "README.md"}
        final_response.content = [text_block, tool_block]
        final_response.model = "claude-sonnet-4-20250514"
        final_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        final_response.stop_reason = "tool_use"
        mock_stream.get_final_message.return_value = final_response

        mock_client.messages.stream.return_value = mock_stream
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(api_key="test_key")
        chunks: list[str] = []
        response = provider.chat_stream_response(
            [ChatMessage(role="user", content="Hi")],
            tools=[{"name": "Read", "description": "", "input_schema": {"type": "object"}}],
            on_text_chunk=chunks.append,
        )

        self.assertEqual("".join(chunks), "Hello world")
        self.assertEqual(response.content, "Hello world")
        self.assertEqual(response.finish_reason, "tool_use")
        self.assertEqual(response.tool_uses[0]["name"], "Read")


class TestOpenAIProvider(unittest.TestCase):
    """Test OpenAI provider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = OpenAIProvider(api_key="test_key")
        self.assertEqual(provider.model, "gpt-5.4")

    def test_custom_model(self):
        """Test provider with custom model."""
        provider = OpenAIProvider(api_key="test_key", model="gpt-4-turbo")
        self.assertEqual(provider.model, "gpt-4-turbo")

    def test_get_available_models(self):
        """Test getting available models."""
        provider = OpenAIProvider(api_key="test_key")
        models = provider.get_available_models()
        self.assertIn("gpt-4", models)
        self.assertIn("gpt-4o", models)

    @patch("src.providers.openai_provider.OpenAI")
    def test_chat(self, mock_openai):
        """Test synchronous chat."""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        mock_response.choices[0].finish_reason = "stop"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Test
        provider = OpenAIProvider(api_key="test_key")
        messages = [ChatMessage(role="user", content="Hi")]
        response = provider.chat(messages)

        self.assertEqual(response.content, "Hello!")
        self.assertEqual(response.model, "gpt-4")
        self.assertEqual(response.usage["total_tokens"], 15)

    @patch("src.providers.openai_provider.OpenAI")
    def test_chat_accepts_dict_messages(self, mock_openai):
        """Test synchronous chat with dict messages."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        mock_response.choices[0].finish_reason = "stop"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="test_key")
        messages = [{"role": "user", "content": "Hi"}]
        response = provider.chat(messages)

        self.assertEqual(response.content, "Hello!")
        mock_client.chat.completions.create.assert_called_once()
        self.assertEqual(
            mock_client.chat.completions.create.call_args.kwargs["messages"], messages
        )

    @patch("src.providers.openai_provider.OpenAI")
    def test_chat_stream_response_rebuilds_tool_calls(self, mock_openai):
        """Streaming chunks are rebuilt into a final response with tool calls."""
        mock_client = MagicMock()

        chunk1 = MagicMock()
        chunk1.model = "gpt-4"
        chunk1.usage = None
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].finish_reason = None
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].delta.reasoning_content = None
        chunk1.choices[0].delta.tool_calls = []

        tool_call_delta = MagicMock()
        tool_call_delta.index = 0
        tool_call_delta.id = "call_1"
        tool_call_delta.function = MagicMock(name="function")
        tool_call_delta.function.name = "Read"
        tool_call_delta.function.arguments = '{"file_path":"README.md"}'

        chunk2 = MagicMock()
        chunk2.model = "gpt-4"
        chunk2.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].finish_reason = "tool_calls"
        chunk2.choices[0].delta.content = None
        chunk2.choices[0].delta.reasoning_content = None
        chunk2.choices[0].delta.tool_calls = [tool_call_delta]

        mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="test_key")
        chunks: list[str] = []
        response = provider.chat_stream_response(
            [ChatMessage(role="user", content="Hi")],
            tools=[{"name": "Read", "description": "", "input_schema": {"type": "object"}}],
            on_text_chunk=chunks.append,
        )

        self.assertEqual("".join(chunks), "Hello")
        self.assertEqual(response.content, "Hello")
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.tool_uses[0]["name"], "Read")
        self.assertEqual(response.usage["total_tokens"], 15)


class TestGLMProvider(unittest.TestCase):
    """Test GLM provider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = GLMProvider(api_key="test_key")
        self.assertEqual(provider.model, "zai/glm-5")

    def test_custom_model(self):
        """Test provider with custom model."""
        provider = GLMProvider(api_key="test_key", model="glm-4")
        self.assertEqual(provider.model, "glm-4")

    def test_get_available_models(self):
        """Test getting available models."""
        provider = GLMProvider(api_key="test_key")
        models = provider.get_available_models()
        self.assertIn("zai/glm-4.5", models)
        self.assertIn("zai/glm-4", models)

    @patch("src.providers.glm_provider.ZhipuAI")
    def test_chat(self, mock_zhipu):
        """Test synchronous chat."""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.reasoning_content = None
        mock_response.model = "glm-4.5"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        mock_response.choices[0].finish_reason = "stop"
        mock_client.chat.completions.create.return_value = mock_response
        mock_zhipu.return_value = mock_client

        # Test
        provider = GLMProvider(api_key="test_key")
        messages = [ChatMessage(role="user", content="Hi")]
        response = provider.chat(messages)

        self.assertEqual(response.content, "Hello!")
        self.assertEqual(response.model, "glm-4.5")
        self.assertIsNone(response.reasoning_content)

    @patch("src.providers.glm_provider.ZhipuAI")
    def test_chat_with_reasoning(self, mock_zhipu):
        """Test chat with reasoning content."""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer"
        mock_response.choices[0].message.reasoning_content = "Thinking..."
        mock_response.model = "glm-4.5"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        mock_response.choices[0].finish_reason = "stop"
        mock_client.chat.completions.create.return_value = mock_response
        mock_zhipu.return_value = mock_client

        # Test
        provider = GLMProvider(api_key="test_key")
        messages = [ChatMessage(role="user", content="Complex question")]
        response = provider.chat(messages)

        self.assertEqual(response.content, "Answer")
        self.assertEqual(response.reasoning_content, "Thinking...")


class TestQwenProvider(unittest.TestCase):
    def test_tencent_defaults(self):
        provider = QwenProvider(api_key="test-token")
        self.assertEqual(provider.model, "ms-mnhdj86z")
        self.assertTrue(provider.base_url.endswith("/ms-mnhdj86z/v1"))

    @patch("src.providers.qwen_provider.OpenAI")
    def test_client_uses_tione_authorization_value(self, mock_openai):
        provider = QwenProvider(api_key="test-token", routing_key="agent-123")
        _ = provider.client

        mock_openai.assert_called_once_with(
            api_key="test-token",
            base_url=provider.base_url,
            default_headers={
                "Authorization": "test-token",
                "X-Clawd-Route-Key": "agent-123",
            },
        )

    @patch("src.providers.qwen_provider.OpenAI")
    def test_routing_key_is_stable_per_provider_and_distinct_between_agents(
        self, mock_openai
    ):
        first = QwenProvider(api_key="test-token")
        second = QwenProvider(api_key="test-token")

        _ = first.client
        _ = first.client
        _ = second.client

        self.assertNotEqual(first.routing_key, second.routing_key)
        self.assertEqual(len(first.routing_key), 32)
        self.assertEqual(mock_openai.call_count, 2)
        self.assertEqual(
            mock_openai.call_args_list[0].kwargs["default_headers"][
                "X-Clawd-Route-Key"
            ],
            first.routing_key,
        )

    @patch.dict("os.environ", {"QWEN_ROUTING_KEY": "fixed-rollout"})
    def test_routing_key_can_be_overridden_from_environment(self):
        provider = QwenProvider(api_key="test-token")

        self.assertEqual(provider.routing_key, "fixed-rollout")

    @patch("src.providers.qwen_provider.OpenAI")
    def test_chat_disables_thinking_by_default(self, mock_openai):
        mock_client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "QWEN_OK"
        response.choices[0].message.reasoning_content = None
        response.choices[0].message.tool_calls = None
        response.choices[0].finish_reason = "stop"
        response.model = "ms-mnhdj86z"
        response.usage = None
        mock_client.chat.completions.create.return_value = response
        mock_openai.return_value = mock_client

        provider = QwenProvider(api_key="test-token")
        actual = provider.chat([ChatMessage(role="user", content="Hi")])

        self.assertEqual(actual.content, "QWEN_OK")
        request = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(
            request["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    @patch.dict("os.environ", {"QWEN_ENABLE_THINKING": "1"})
    @patch("src.providers.qwen_provider.OpenAI")
    def test_chat_can_enable_thinking_from_environment(self, mock_openai):
        mock_client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "QWEN_OK"
        response.choices[0].message.reasoning_content = "thinking"
        response.choices[0].message.tool_calls = None
        response.choices[0].finish_reason = "stop"
        response.model = "ms-rns547kc"
        response.usage = None
        mock_client.chat.completions.create.return_value = response
        mock_openai.return_value = mock_client

        provider = QwenProvider(api_key="test-token")
        provider.chat([ChatMessage(role="user", content="Hi")])

        request = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(
            request["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": True}},
        )

    @patch.dict("os.environ", {"QWEN_ENABLE_THINKING": "0"})
    @patch("src.providers.qwen_provider.OpenAI")
    def test_explicit_thinking_setting_is_isolated_from_process_environment(
        self, mock_openai
    ):
        mock_client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "QWEN_OK"
        response.choices[0].message.reasoning_content = "thinking"
        response.choices[0].message.tool_calls = None
        response.choices[0].finish_reason = "stop"
        response.model = "ms-rns547kc"
        response.usage = None
        mock_client.chat.completions.create.return_value = response
        mock_openai.return_value = mock_client

        provider = QwenProvider(api_key="test-token", enable_thinking=True)
        provider.chat([ChatMessage(role="user", content="Hi")])

        request = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(
            request["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": True}},
        )

    @patch("src.providers.qwen_provider.OpenAI")
    def test_stream_requests_and_collects_terminal_usage_chunk(self, mock_openai):
        mock_client = MagicMock()

        content_chunk = MagicMock()
        content_chunk.model = "ms-rns547kc"
        content_chunk.usage = None
        content_chunk.choices = [MagicMock()]
        content_chunk.choices[0].finish_reason = "stop"
        content_chunk.choices[0].delta.content = "QWEN_OK"
        content_chunk.choices[0].delta.reasoning_content = None
        content_chunk.choices[0].delta.tool_calls = []

        usage_chunk = MagicMock()
        usage_chunk.model = "ms-rns547kc"
        usage_chunk.usage = MagicMock(
            prompt_tokens=120,
            completion_tokens=8,
            total_tokens=128,
            prompt_tokens_details={"cached_tokens": 100},
        )
        usage_chunk.choices = []

        mock_client.chat.completions.create.return_value = iter(
            [content_chunk, usage_chunk]
        )
        mock_openai.return_value = mock_client

        provider = QwenProvider(api_key="test-token")
        actual = provider.chat_stream_response(
            [ChatMessage(role="user", content="Hi")]
        )

        request = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["stream_options"], {"include_usage": True})
        self.assertEqual(actual.content, "QWEN_OK")
        self.assertEqual(actual.usage["input_tokens"], 120)
        self.assertEqual(actual.usage["output_tokens"], 8)
        self.assertEqual(actual.usage["total_tokens"], 128)
        self.assertEqual(actual.usage["cache_read_input_tokens"], 100)


class TestGetProviderClass(unittest.TestCase):
    """Test get_provider_class function."""

    def test_get_anthropic_provider(self):
        """Test getting Anthropic provider class."""
        cls = get_provider_class("anthropic")
        self.assertEqual(cls, AnthropicProvider)

    def test_get_openai_provider(self):
        """Test getting OpenAI provider class."""
        cls = get_provider_class("openai")
        self.assertEqual(cls, OpenAIProvider)

    def test_get_glm_provider(self):
        """Test getting GLM provider class."""
        cls = get_provider_class("glm")
        self.assertEqual(cls, GLMProvider)

    def test_get_qwen_provider_and_alias(self):
        self.assertEqual(get_provider_class("qwen"), QwenProvider)
        self.assertEqual(get_provider_class("qwen3.5"), QwenProvider)

    def test_get_unknown_provider(self):
        """Test getting unknown provider."""
        with self.assertRaises(ValueError) as context:
            get_provider_class("unknown")

        self.assertIn("Unknown provider", str(context.exception))


if __name__ == "__main__":
    unittest.main()
