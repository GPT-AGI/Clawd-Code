import json
from unittest.mock import MagicMock

from src.agent.conversation import Conversation
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import ChatResponse
from src.tool_system.agent_loop import _build_openai_tool_result_content, run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry


def test_skill_schema_is_anthropic_compatible():
    """Skill uses an Anthropic-compatible top-level object schema."""
    specs = list(build_default_registry().list_specs())
    skill = next(spec for spec in specs if spec.name == "Skill")

    assert skill.input_schema["type"] == "object"
    assert "anyOf" not in skill.input_schema
    assert "oneOf" not in skill.input_schema
    assert "allOf" not in skill.input_schema
    assert skill.input_schema["additionalProperties"] is False

    assert set(skill.input_schema["properties"]) == {
        "skill",
        "args",
        "name",
        "input",
    }


def test_anthropic_tool_result_serialises_structured_output():
    """Structured tool results are JSON strings before provider submission."""
    result = _build_openai_tool_result_content(
        {
            "success": True,
            "value": 123,
        }
    )

    assert isinstance(result, str)
    assert json.loads(result) == {
        "success": True,
        "value": 123,
    }

    text_result = _build_openai_tool_result_content("hello")

    assert isinstance(text_result, str)
    assert text_result == "hello"

def test_send_user_message_has_strict_required_schema():
    """SendUserMessage requires schema-valid Anthropic tool input."""
    specs = list(build_default_registry().list_specs())

    send_user_message = next(
        spec for spec in specs if spec.name == "SendUserMessage"
    )

    assert send_user_message.strict is True
    assert send_user_message.input_schema["type"] == "object"
    assert send_user_message.input_schema["additionalProperties"] is False

    assert set(send_user_message.input_schema["required"]) == {
        "message",
        "status",
    }

    assert (
        send_user_message.input_schema["properties"]["message"]["type"]
        == "string"
    )

    assert send_user_message.input_schema["properties"]["status"]["enum"] == [
        "normal",
        "proactive",
    ]


def _tools_sent_to(provider, workspace):
    """Run one agent-loop turn and return the tool schemas the provider got."""
    provider.chat_stream_response = MagicMock(side_effect=NotImplementedError())
    provider.chat = MagicMock(
        return_value=ChatResponse(
            content="done",
            model="test-model",
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="stop",
            tool_uses=None,
        )
    )
    conversation = Conversation()
    conversation.add_user_message("hi")

    run_agent_loop(
        conversation=conversation,
        provider=provider,
        tool_registry=build_default_registry(),
        tool_context=ToolContext(workspace_root=workspace),
    )

    return provider.chat.call_args.kwargs["tools"]


def test_agent_loop_forwards_strict_tools_to_anthropic(tmp_path):
    """Anthropic receives SendUserMessage, and only it, marked strict."""
    tools = _tools_sent_to(AnthropicProvider(api_key="test_key"), tmp_path)

    assert [tool["name"] for tool in tools if tool.get("strict")] == [
        "SendUserMessage"
    ]


def test_agent_loop_sends_no_strict_flag_to_other_providers(tmp_path):
    """OpenAI-style providers never receive the Anthropic-only strict key."""
    tools = _tools_sent_to(MagicMock(), tmp_path)

    assert tools
    assert not any("strict" in tool for tool in tools)
