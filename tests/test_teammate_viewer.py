from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from src.agent.conversation import Conversation, TextContentBlock, ToolUseContentBlock
from src.teammate.models import AgentRecord, Message
from src.teammate.trace import redact_trace_value
from src.teammate.viewer import build_trace_snapshot, create_trace_server
from src.tool_system.context import ToolContext
from src.tool_system.tools import TaskCreateTool, TeamCreateTool


class TestTeammateTraceViewer(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.context = ToolContext(workspace_root=self.root)
        self.team = TeamCreateTool().run({"team_name": "trace-demo"}, self.context).output
        self.agent = AgentRecord(
            agent_id="researcher-1",
            team_id=self.team["team_id"],
            name="researcher",
            role="research",
            session_id="session-1",
            model="test-model",
        )
        self.context.team_store.save_agent(self.agent)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_reconstructs_legacy_tool_events_from_session(self) -> None:
        task = TaskCreateTool().run(
            {"key": "analysis", "subject": "Inspect", "description": "Inspect files", "owner": self.agent.agent_id},
            self.context,
        ).output["task"]
        conversation = Conversation()
        conversation.add_user_message("Inspect requirements")
        conversation.add_assistant_message([
            TextContentBlock(text="I will inspect the file."),
            ToolUseContentBlock(id="read-1", name="Read", input={"file_path": "requirements.md"}),
        ])
        conversation.add_tool_result_message("read-1", json.dumps({"content": "rule one"}))
        self.context.team_store.save_session(
            self.team["team_id"],
            self.agent.session_id,
            {
                "session_id": self.agent.session_id,
                "team_id": self.team["team_id"],
                "agent_id": self.agent.agent_id,
                "model": "test-model",
                "conversation": conversation.to_dict(),
            },
        )
        message = Message(
            message_id="message-1",
            team_id=self.team["team_id"],
            sender_id=self.agent.agent_id,
            recipient_id=self.team["lead_agent_id"],
            content="Rules inspected",
            summary="analysis",
        )
        message.transition_to("delivered")
        self.context.team_store.save_message(message)

        snapshot = build_trace_snapshot(self.root)

        self.assertTrue(snapshot["historical_reconstruction"])
        self.assertEqual(snapshot["tasks"][0]["id"], task["id"])
        self.assertEqual(snapshot["messages"][0]["sender_name"], "researcher")
        trace_types = {event["type"] for event in snapshot["events"]}
        self.assertTrue({"model.response", "tool.started", "tool.completed"}.issubset(trace_types))
        tool_event = next(event for event in snapshot["events"] if event["type"] == "tool.started")
        self.assertEqual(tool_event["data"]["tool_input"]["file_path"], "requirements.md")
        self.assertEqual(tool_event["actor_name"], "researcher")

    def test_redacts_credentials_without_hiding_token_usage(self) -> None:
        value = redact_trace_value({
            "api_key": "top-secret",
            "input_tokens": 42,
            "command": "ANTHROPIC_AUTH_TOKEN=exposed curl -H 'Authorization: Bearer abc123'",
        })

        self.assertEqual(value["api_key"], "[REDACTED]")
        self.assertEqual(value["input_tokens"], 42)
        self.assertNotIn("exposed", value["command"])
        self.assertNotIn("abc123", value["command"])

    def test_http_api_serves_snapshot(self) -> None:
        try:
            server = create_trace_server(self.root, port=0)
        except PermissionError:
            self.skipTest("sandbox does not permit binding a localhost test server")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload["team"]["team_name"], "trace-demo")
            self.assertEqual(response.status, 200)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
