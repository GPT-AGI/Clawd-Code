"""Test agent loop with mocked provider to verify tool invocation."""

import unittest
from unittest.mock import MagicMock
from pathlib import Path
import tempfile

from src.agent.conversation import Conversation
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import ChatResponse
from src.tool_system.defaults import build_default_registry
from src.tool_system.context import ToolContext
from src.tool_system.agent_loop import (
    AgentLoopResult,
    _team_lifecycle_warning,
    run_agent_loop,
)
from src.tool_system.tools import TaskCreateTool, TeamCreateTool, TeammateCreateTool


class TestAgentLoop(unittest.TestCase):
    """Test agent loop logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_agent_loop_calls_tool(self):
        """Test agent loop correctly dispatches a tool call from mocked LLM."""
        conversation = Conversation()
        conversation.add_user_message("Create a file hello.py with content print('hello world')")

        # Mock provider
        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()

        # First response: tool use Write
        mock_tool_use = {
            "id": "toolu_123",
            "name": "Write",
            "input": {
                "file_path": str(self.workspace / "hello.py"),
                "content": "print('hello world')"
            }
        }
        mock_response1 = ChatResponse(
            content="I will create the file.",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20},
            finish_reason="tool_use",
            tool_uses=[mock_tool_use],
        )

        # Second response: final text after tool result
        mock_response2 = ChatResponse(
            content="File created successfully!",
            model="test-model",
            usage={"input_tokens": 30, "output_tokens": 10},
            finish_reason="stop",
            tool_uses=None,
        )

        mock_provider.chat.side_effect = [mock_response1, mock_response2]

        result = run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
            max_output_tokens=8192,
            verbose=False,
        )

        # Verify final response
        self.assertIsInstance(result, AgentLoopResult)
        self.assertEqual(result.response_text, "File created successfully!")

        # Verify provider was called twice
        self.assertEqual(mock_provider.chat.call_count, 2)
        self.assertTrue(
            all(call.kwargs["max_tokens"] == 8192 for call in mock_provider.chat.call_args_list)
        )

        # Verify file was created
        hello_py = self.workspace / "hello.py"
        self.assertTrue(hello_py.exists())
        self.assertEqual(hello_py.read_text(), "print('hello world')")

    def test_agent_loop_creates_hello_world(self):
        """Test agent loop creates hello.py and writes print('hello world')."""
        conversation = Conversation()
        conversation.add_user_message("Create a file hello.py with content print('hello world')")

        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()

        # First response: tool use Write
        hello_path = self.workspace / "hello.py"
        mock_tool_write = {
            "id": "toolu_123",
            "name": "Write",
            "input": {
                "file_path": str(hello_path),
                "content": "print('hello world')"
            }
        }
        mock_response1 = ChatResponse(
            content="I will create the file.",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20},
            finish_reason="tool_use",
            tool_uses=[mock_tool_write],
        )

        # Second response: final
        mock_response2 = ChatResponse(
            content="File created successfully!",
            model="test-model",
            usage={"input_tokens": 30, "output_tokens": 10},
            finish_reason="stop",
            tool_uses=None,
        )

        mock_provider.chat.side_effect = [mock_response1, mock_response2]

        result = run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
            verbose=False,
        )

        self.assertIsInstance(result, AgentLoopResult)
        self.assertEqual(result.response_text, "File created successfully!")
        self.assertTrue(hello_path.exists())
        self.assertEqual(hello_path.read_text(), "print('hello world')")

    def test_anthropic_tool_results_are_serialized_as_text(self):
        requirements = self.workspace / "requirements.md"
        requirements.write_text("First pricing rule.\n", encoding="utf-8")
        conversation = Conversation()
        conversation.add_user_message("Read requirements.md")
        provider = AnthropicProvider(api_key="test", model="test-model")
        provider.chat = MagicMock(side_effect=[
            ChatResponse(
                content="",
                model="test-model",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="tool_use",
                tool_uses=[{
                    "id": "toolu_read",
                    "name": "Read",
                    "input": {"file_path": "requirements.md"},
                }],
            ),
            ChatResponse(
                content="The first pricing rule is present.",
                model="test-model",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="stop",
                tool_uses=None,
            ),
        ])

        run_agent_loop(
            conversation=conversation,
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
        )

        tool_result = conversation.messages[2].content[0]
        self.assertIsInstance(tool_result.content, str)
        self.assertIn("First pricing rule.", tool_result.content)

    def test_anthropic_groups_and_compacts_write_results(self):
        conversation = Conversation(max_history=3)
        conversation.add_user_message("Create two files")
        provider = AnthropicProvider(api_key="test", model="test-model")
        provider.chat = MagicMock(side_effect=[
            ChatResponse(
                content="",
                model="test-model",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="tool_use",
                tool_uses=[
                    {
                        "id": "toolu_one",
                        "name": "Write",
                        "input": {"file_path": "one.txt", "content": "secret-one"},
                    },
                    {
                        "id": "toolu_two",
                        "name": "Write",
                        "input": {"file_path": "two.txt", "content": "secret-two"},
                    },
                ],
            ),
            ChatResponse(
                content="done",
                model="test-model",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="stop",
                tool_uses=None,
            ),
        ])

        run_agent_loop(
            conversation=conversation,
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
            max_turns=4,
        )

        self.assertGreaterEqual(conversation.max_history, 10)
        self.assertEqual(conversation.messages[0].content, "Create two files")
        result_blocks = conversation.messages[2].content
        self.assertEqual(len(result_blocks), 2)
        self.assertNotIn("secret-one", result_blocks[0].content)
        self.assertNotIn("structuredPatch", result_blocks[0].content)
        self.assertIn('"success": true', result_blocks[0].content)

    def test_agent_loop_stream_emits_final_text_chunks(self):
        """Streaming mode emits final response chunks without changing the result."""
        conversation = Conversation()
        conversation.add_user_message("Say hello")

        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()
        mock_provider.chat.return_value = ChatResponse(
            content="Hello from Clawd!",
            model="test-model",
            usage={"input_tokens": 3, "output_tokens": 4},
            finish_reason="stop",
            tool_uses=None,
        )

        chunks: list[str] = []
        result = run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
            stream=True,
            verbose=False,
            on_text_chunk=chunks.append,
        )

        self.assertEqual("".join(chunks), "Hello from Clawd!")
        self.assertEqual(result.response_text, "Hello from Clawd!")
        self.assertEqual(mock_provider.chat.call_count, 1)
        self.assertEqual(len(conversation.messages), 2)
        self.assertEqual(conversation.messages[-1].role, "assistant")
        self.assertEqual(conversation.messages[-1].content, "Hello from Clawd!")

    def test_agent_loop_stream_only_emits_final_turn_text(self):
        """Streaming mode skips interim tool-planning text and emits the final answer only."""
        conversation = Conversation()
        conversation.add_user_message("Create a file hello.py with content print('hello world')")

        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()
        hello_path = self.workspace / "hello.py"
        mock_response1 = ChatResponse(
            content="I will create the file.",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20},
            finish_reason="tool_use",
            tool_uses=[{
                "id": "toolu_123",
                "name": "Write",
                "input": {
                    "file_path": str(hello_path),
                    "content": "print('hello world')",
                },
            }],
        )
        mock_response2 = ChatResponse(
            content="File created successfully!",
            model="test-model",
            usage={"input_tokens": 30, "output_tokens": 10},
            finish_reason="stop",
            tool_uses=None,
        )
        mock_provider.chat.side_effect = [mock_response1, mock_response2]

        chunks: list[str] = []
        result = run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
            stream=True,
            verbose=False,
            on_text_chunk=chunks.append,
        )

        self.assertEqual("".join(chunks), "File created successfully!")
        self.assertEqual(result.response_text, "File created successfully!")
        self.assertTrue(hello_path.exists())

    def test_agent_loop_stream_uses_structured_provider_streaming_for_tool_turns(self):
        """Structured provider streaming can emit pre-tool text and final text across turns."""
        conversation = Conversation()
        conversation.add_user_message("Create hello.py")

        provider = MagicMock()
        hello_path = self.workspace / "hello.py"

        stream_responses = [
            ChatResponse(
                content="I will create the file.",
                model="test-model",
                usage={"input_tokens": 10, "output_tokens": 20},
                finish_reason="tool_use",
                tool_uses=[{
                    "id": "toolu_123",
                    "name": "Write",
                    "input": {
                        "file_path": str(hello_path),
                        "content": "print('hello world')",
                    },
                }],
            ),
            ChatResponse(
                content="File created successfully!",
                model="test-model",
                usage={"input_tokens": 30, "output_tokens": 10},
                finish_reason="stop",
                tool_uses=None,
            ),
        ]

        def stream_side_effect(messages, tools=None, on_text_chunk=None, **kwargs):
            response = stream_responses.pop(0)
            if on_text_chunk is not None and response.content:
                on_text_chunk(response.content)
            return response

        provider.chat_stream_response.side_effect = stream_side_effect
        provider.chat.side_effect = AssertionError("chat() should not be used when structured streaming is available")

        chunks: list[str] = []
        result = run_agent_loop(
            conversation=conversation,
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
            stream=True,
            verbose=False,
            on_text_chunk=chunks.append,
        )

        self.assertEqual("".join(chunks), "I will create the file.File created successfully!")
        self.assertEqual(result.response_text, "File created successfully!")
        self.assertEqual(provider.chat_stream_response.call_count, 2)
        self.assertTrue(hello_path.exists())

    def test_agent_loop_stream_falls_back_when_structured_streaming_is_unavailable(self):
        """If the provider lacks structured streaming, the stable synchronous path still works."""
        conversation = Conversation()
        conversation.add_user_message("Say hello")

        provider = MagicMock()
        provider.chat_stream_response.side_effect = NotImplementedError()
        provider.chat.return_value = ChatResponse(
            content="Hello from fallback!",
            model="test-model",
            usage={"input_tokens": 2, "output_tokens": 3},
            finish_reason="stop",
            tool_uses=None,
        )

        chunks: list[str] = []
        result = run_agent_loop(
            conversation=conversation,
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
            stream=True,
            verbose=False,
            on_text_chunk=chunks.append,
        )

        self.assertEqual("".join(chunks), "Hello from fallback!")
        self.assertEqual(result.response_text, "Hello from fallback!")
        provider.chat.assert_called_once()

    def test_empty_response_without_tools_is_corrected_and_retried(self):
        conversation = Conversation()
        conversation.add_user_message("Create done.txt")
        provider = MagicMock()
        output_path = self.workspace / "done.txt"
        provider.chat_stream_response.side_effect = NotImplementedError()
        provider.chat.side_effect = [
            ChatResponse(
                content="",
                reasoning_content="I should create the file.",
                model="test-model",
                usage={},
                finish_reason="stop",
                tool_uses=None,
            ),
            ChatResponse(
                content="Continuing with the implementation.",
                model="test-model",
                usage={},
                finish_reason="tool_calls",
                tool_uses=[{
                    "id": "write-after-empty",
                    "name": "Write",
                    "input": {"file_path": str(output_path), "content": "done"},
                }],
            ),
            ChatResponse(
                content="Implementation complete.",
                model="test-model",
                usage={},
                finish_reason="stop",
                tool_uses=None,
            ),
        ]
        events = []

        result = run_agent_loop(
            conversation=conversation,
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
            max_turns=6,
            on_event=events.append,
        )

        self.assertEqual(result.response_text, "Implementation complete.")
        self.assertEqual(provider.chat.call_count, 3)
        self.assertEqual(output_path.read_text(encoding="utf-8"), "done")
        retries = [event for event in events if event.kind == "empty_response_retry"]
        self.assertEqual(len(retries), 1)
        corrective_messages = [
            message.content
            for message in conversation.messages
            if message.role == "user"
            and isinstance(message.content, str)
            and message.content.startswith("Your previous response contained neither")
        ]
        self.assertEqual(len(corrective_messages), 1)

    def test_repeated_empty_responses_fail_instead_of_scoring_as_success(self):
        conversation = Conversation()
        conversation.add_user_message("Do the task")
        provider = MagicMock()
        provider.chat_stream_response.side_effect = NotImplementedError()
        provider.chat.return_value = ChatResponse(
            content="",
            reasoning_content="thinking only",
            model="test-model",
            usage={},
            finish_reason="stop",
            tool_uses=None,
        )
        events = []

        with self.assertRaisesRegex(RuntimeError, "corrective retries"):
            run_agent_loop(
                conversation=conversation,
                provider=provider,
                tool_registry=self.registry,
                tool_context=self.context,
                max_turns=6,
                on_event=events.append,
            )

        self.assertEqual(provider.chat.call_count, 4)
        self.assertEqual(
            [event.kind for event in events].count("empty_response_retry"), 3
        )
        self.assertEqual([event.kind for event in events].count("run_failed"), 1)

    def test_incomplete_team_blocks_final_answer_until_lead_cleans_up(self):
        conversation = Conversation()
        conversation.add_user_message("Review the implementation")
        TeamCreateTool().run({"team_name": "unfinished"}, self.context)
        TeammateCreateTool().run(
            {
                "name": "reviewer",
                "role": "review",
                "instructions": "Review the implementation",
                "tools": ["Read"],
            },
            self.context,
        )

        provider = MagicMock()
        provider.chat_stream_response.side_effect = NotImplementedError()
        provider.chat.side_effect = [
            ChatResponse(
                content="The task is complete.",
                model="test-model",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="stop",
                tool_uses=None,
            ),
            ChatResponse(
                content="I need to settle the team first.",
                model="test-model",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="tool_use",
                tool_uses=[{
                    "id": "toolu_delete",
                    "name": "TeamDelete",
                    "input": {},
                }],
            ),
            ChatResponse(
                content="The task is complete after team cleanup.",
                model="test-model",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="stop",
                tool_uses=None,
            ),
        ]

        result = run_agent_loop(
            conversation=conversation,
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
            max_turns=5,
        )

        self.assertEqual(result.response_text, "The task is complete after team cleanup.")
        self.assertEqual(provider.chat.call_count, 3)
        self.assertIsNone(self.context.team)
        warnings = [
            message.content
            for message in conversation.messages
            if message.role == "user"
            and isinstance(message.content, str)
            and message.content.startswith("Team lifecycle guard:")
        ]
        self.assertEqual(len(warnings), 1)
        self.assertIn("TaskCreate", warnings[0])

    def test_completed_team_with_late_pending_task_is_not_settled(self):
        created = TeamCreateTool().run({"team_name": "reopen"}, self.context).output
        TeammateCreateTool().run(
            {
                "name": "worker",
                "role": "implementation",
                "instructions": "Implement the assigned task.",
                "tools": ["Read"],
            },
            self.context,
        )
        team = self.context.team_store.load_team(created["team_id"])
        team.transition_to("running")
        team.transition_to("completed")
        self.context.team_store.save_team(team)
        self.context.reload_team_state()
        TaskCreateTool().run(
            {
                "key": "late",
                "subject": "Late work",
                "description": "Complete work discovered after the first batch.",
                "owner": "worker",
            },
            self.context,
        )

        warning = _team_lifecycle_warning(self.context)

        self.assertIsNotNone(warning)
        self.assertIn("TeamRun", warning)
        self.assertIn("pending=1", warning)

    def test_strict_team_with_finished_tasks_requests_team_verify(self):
        created = TeamCreateTool().run(
            {"team_name": "strict", "quality_gates": True}, self.context
        ).output
        TeammateCreateTool().run(
            {
                "name": "worker",
                "role": "implementation",
                "instructions": "Implement the assigned task.",
                "tools": ["Read"],
            },
            self.context,
        )
        task_id = TaskCreateTool().run(
            {
                "key": "done",
                "subject": "Done",
                "description": "Already completed",
                "owner": "worker",
            },
            self.context,
        ).output["task"]["id"]
        self.context.tasks[task_id]["status"] = "completed"
        self.context.persist_tasks()
        team = self.context.team_store.load_team(created["team_id"])
        team.transition_to("running")
        self.context.team_store.save_team(team)

        warning = _team_lifecycle_warning(self.context)

        self.assertIsNotNone(warning)
        self.assertIn("TeamVerify", warning)
        self.assertIn("validation", warning)


if __name__ == "__main__":
    unittest.main()
