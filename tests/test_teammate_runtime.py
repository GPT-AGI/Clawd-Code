from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.providers.base import ChatResponse
from src.teammate.runtime import TeammateRuntime
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.errors import ToolInputError
from src.tool_system.tools import TaskCreateTool, TeamCreateTool, TeammateCreateTool, TeamRunTool


class ScriptedProvider:
    model = "test-model"

    def __init__(self) -> None:
        self.responses = [
            self._tool("research-message", "SendMessage", {"to": "coder", "summary": "rules", "message": "Shipping is not discountable."}),
            self._final("Research complete."),
            self._tool("coder-message", "SendMessage", {"to": "reviewer", "summary": "implementation", "message": "Pricing fixed; tests pass."}),
            self._final("Implementation complete."),
            self._tool("review-message", "SendMessage", {"to": "lead", "summary": "approval", "message": "Reviewed and approved."}),
            self._final("Review complete."),
        ]
        self.calls: list[list[dict]] = []

    @staticmethod
    def _tool(call_id: str, name: str, tool_input: dict) -> ChatResponse:
        return ChatResponse(
            content="",
            model="test-model",
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="tool_use",
            tool_uses=[{"id": call_id, "name": name, "input": tool_input}],
        )

    @staticmethod
    def _final(content: str) -> ChatResponse:
        return ChatResponse(
            content=content,
            model="test-model",
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="stop",
            tool_uses=None,
        )

    def chat(self, messages, tools=None, **kwargs):
        self.calls.append(messages)
        return self.responses.pop(0)


class TestTeammateRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.registry = build_default_registry(include_user_tools=False)
        self.context = ToolContext(workspace_root=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_runs_dependency_chain_with_independent_sessions_and_messages(self) -> None:
        created = TeamCreateTool().run({"team_name": "order-repair"}, self.context).output
        provider = ScriptedProvider()
        self.context.teammate_runtime = TeammateRuntime(provider, self.registry)
        with self.assertRaises(ToolInputError):
            TeammateCreateTool().run(
                {
                    "name": "invalid",
                    "role": "invalid",
                    "instructions": "Use a missing tool",
                    "tools": ["DoesNotExist"],
                },
                self.context,
            )
        tool_sets = {
            "researcher": ["Read", "Glob", "Grep"],
            "coder": ["Read", "Glob", "Grep", "Write", "Edit", "Bash"],
            "reviewer": ["Read", "Glob", "Grep", "Bash"],
        }
        teammates = {}
        for name, tools in tool_sets.items():
            teammates[name] = TeammateCreateTool().run(
                {
                    "name": name,
                    "role": name,
                    "instructions": f"Act as the {name}.",
                    "tools": tools,
                },
                self.context,
            ).output

        analysis = TaskCreateTool().run(
            {"key": "analysis", "subject": "Analysis", "description": "Inspect rules", "owner": "researcher"},
            self.context,
        ).output["task"]
        implementation = TaskCreateTool().run(
            {
                "key": "implementation",
                "subject": "Implementation",
                "description": "Repair pricing",
                "owner": "coder",
                "blockedBy": ["analysis"],
            },
            self.context,
        ).output["task"]
        review = TaskCreateTool().run(
            {
                "key": "review",
                "subject": "Review",
                "description": "Review and test",
                "owner": "reviewer",
                "blockedBy": ["implementation"],
            },
            self.context,
        ).output["task"]

        result = TeamRunTool().run({}, self.context)

        self.assertFalse(result.is_error)
        self.assertEqual(result.output["status"], "completed")
        self.assertEqual(
            result.output["executed_task_ids"],
            [analysis["id"], implementation["id"], review["id"]],
        )
        self.assertEqual({task["status"] for task in self.context.tasks.values()}, {"completed"})
        agents = self.context.team_store.list_agents(created["team_id"])
        self.assertEqual({agent.status for agent in agents}, {"completed"})

        messages = self.context.team_store.list_messages(created["team_id"])
        self.assertEqual(len(messages), 3)
        self.assertEqual([message.status for message in messages], ["consumed", "consumed", "delivered"])
        names = {agent.agent_id: agent.name for agent in agents}
        names[created["lead_agent_id"]] = "lead"
        self.assertEqual(
            [(names[message.sender_id], names[message.recipient_id]) for message in messages],
            [("researcher", "coder"), ("coder", "reviewer"), ("reviewer", "lead")],
        )
        for teammate in teammates.values():
            session = self.context.team_store.load_session(created["team_id"], teammate["session_id"])
            self.assertGreaterEqual(len(session["conversation"]["messages"]), 3)
        trace_events = self.context.team_store.list_events(created["team_id"])
        trace_types = {event["type"] for event in trace_events}
        self.assertTrue({
            "run.started",
            "model.started",
            "model.response",
            "tool.started",
            "tool.completed",
            "run.completed",
        }.issubset(trace_types))
        send_message_events = [
            event for event in trace_events
            if event["type"] == "tool.started"
            and event["data"].get("tool_name") == "SendMessage"
        ]
        self.assertEqual(
            [event["data"].get("actor_name") for event in send_message_events],
            ["researcher", "coder", "reviewer"],
        )
        self.assertEqual(provider.responses, [])

        acceptance = {
            "required_agents": ["researcher", "coder", "reviewer"],
            "required_tasks": [
                {"name": "analysis", "owner": "researcher", "blocked_by": []},
                {"name": "implementation", "owner": "coder", "blocked_by": ["analysis"]},
                {"name": "review", "owner": "reviewer", "blocked_by": ["implementation"]},
            ],
            "required_messages": [
                {"from": "researcher", "to": "coder"},
                {"from": "coder", "to": "reviewer"},
                {"from": "reviewer", "to": "lead"},
            ],
            "required_test_command": "python -m unittest checks.order_acceptance -v",
            "required_final_status": "completed",
        }
        (self.root / "acceptance.json").write_text(json.dumps(acceptance), encoding="utf-8")
        evaluator = Path(__file__).parents[1] / "teammate-evals" / "order-discount" / "evaluate.py"
        checked = subprocess.run(
            [sys.executable, str(evaluator), "--workspace", str(self.root), "--collaboration-only"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)


if __name__ == "__main__":
    unittest.main()
