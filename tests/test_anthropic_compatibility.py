import json

from src.tool_system.agent_loop import _build_openai_tool_result_content
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


def test_agent_loop_forwards_strict_tools_to_anthropic():
    """Anthropic tool definitions must preserve ToolSpec.strict."""
    from pathlib import Path

    source = Path("src/tool_system/agent_loop.py").read_text(
        encoding="utf-8"
    )

    assert "_is_anthropic_provider(provider)" in source
    assert 'spec.name == "SendUserMessage"' in source
    assert "and spec.strict" in source
    assert 'tool_schema["strict"] = True' in source
