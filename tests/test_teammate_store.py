from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.teammate.models import AgentRecord, Message, Team, TeamTask
from src.tool_system.context import ToolContext
from src.tool_system.errors import ToolInputError
from src.tool_system.tools import (
    ReadMessagesTool,
    SendMessageTool,
    TaskCreateTool,
    TaskGetTool,
    TaskOutputTool,
    TaskRetryTool,
    TaskUpdateTool,
    TeamCancelTool,
    TeamCreateTool,
    TeamDeleteTool,
    TeamIntegrateTool,
    TeammateCreateTool,
    TeammateResumeTool,
    TeammateStopTool,
    TeamRunTool,
    TeamResumeTool,
)


class TestTeammateModels(unittest.TestCase):
    def test_team_state_machine_allows_reopen_but_rejects_invalid_transition(self) -> None:
        team = Team(team_id="team-1", team_name="demo", lead_agent_id="lead-1")
        team.transition_to("running")
        team.transition_to("completed")
        team.transition_to("running")

        with self.assertRaises(ValueError):
            team.transition_to("created")

    def test_agent_and_message_validate_status(self) -> None:
        with self.assertRaises(ValueError):
            AgentRecord(
                agent_id="agent-1",
                team_id="team-1",
                name="researcher",
                role="research",
                session_id="session-1",
                status="unknown",
            )

        agent = AgentRecord(
            agent_id="agent-2",
            team_id="team-1",
            name="coder",
            role="implementation",
            session_id="session-2",
        )
        agent.transition_to("running")
        agent.transition_to("stopping")
        agent.transition_to("cancelled")
        self.assertEqual(agent.status, "cancelled")
        with self.assertRaises(ValueError):
            Message(
                message_id="message-1",
                team_id="team-1",
                sender_id="lead-1",
                recipient_id="agent-1",
                content="hello",
                status="unknown",
            )

    def test_task_and_message_state_machines(self) -> None:
        task = TeamTask(id="task-1", subject="Inspect", description="Inspect runtime")
        task.transition_to("in_progress")
        task.transition_to("completed")
        with self.assertRaises(ValueError):
            task.transition_to("in_progress")

        message = Message(
            message_id="message-1",
            team_id="team-1",
            sender_id="lead-1",
            recipient_id="agent-1",
            content="hello",
        )
        message.transition_to("delivered")
        message.transition_to("consumed")
        self.assertIsNotNone(message.delivered_at)
        self.assertIsNotNone(message.consumed_at)


class TestTeamPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_team_layout_and_task_state_survive_context_restart(self) -> None:
        context = ToolContext(workspace_root=self.root)
        created = TeamCreateTool().run(
            {"team_name": "demo", "description": "persistent team"}, context
        ).output
        team_id = created["team_id"]
        team_dir = self.root / ".clawd" / "teams" / team_id

        self.assertTrue((self.root / ".clawd" / "team.json").is_file())
        self.assertTrue((team_dir / "team.json").is_file())
        self.assertTrue((team_dir / "tasks.json").is_file())
        self.assertTrue((team_dir / "events.jsonl").is_file())
        self.assertTrue((team_dir / "agents").is_dir())
        self.assertTrue((team_dir / "sessions").is_dir())
        self.assertTrue((team_dir / "messages").is_dir())
        self.assertEqual(context.team_store.list_events(team_id)[0]["type"], "team.created")

        task = TaskCreateTool().run(
            {"subject": "Inspect", "description": "Inspect the runtime"}, context
        ).output["task"]
        TaskUpdateTool().run(
            {"taskId": task["id"], "status": "in_progress", "owner": "lead"}, context
        )

        restored = ToolContext(workspace_root=self.root)
        self.assertEqual(restored.team["team_id"], team_id)
        self.assertEqual(restored.tasks[task["id"]]["status"], "in_progress")
        self.assertEqual(restored.tasks[task["id"]]["owner"], created["lead_agent_id"])

    def test_disband_removes_active_pointer_and_preserves_history(self) -> None:
        context = ToolContext(workspace_root=self.root)
        team_id = TeamCreateTool().run({"team_name": "demo"}, context).output["team_id"]

        result = TeamDeleteTool().run({}, context).output

        self.assertTrue(result["success"])
        self.assertFalse((self.root / ".clawd" / "team.json").exists())
        archived_path = self.root / ".clawd" / "teams" / team_id / "team.json"
        archived = json.loads(archived_path.read_text(encoding="utf-8"))
        self.assertEqual(archived["status"], "cancelled")

    def test_agent_records_roundtrip_through_team_store(self) -> None:
        context = ToolContext(workspace_root=self.root)
        team_id = TeamCreateTool().run({"team_name": "demo"}, context).output["team_id"]
        agent = AgentRecord(
            agent_id="researcher-1",
            team_id=team_id,
            name="researcher",
            role="research",
            session_id="session-1",
        )

        context.team_store.save_agent(agent)

        restored = context.team_store.load_agent(team_id, agent.agent_id)
        self.assertEqual(restored, agent)
        self.assertEqual(context.team_store.list_agents(team_id), [agent])

    def test_messages_and_sessions_roundtrip_through_team_store(self) -> None:
        context = ToolContext(workspace_root=self.root)
        created = TeamCreateTool().run({"team_name": "demo"}, context).output
        teammate = TeammateCreateTool().run(
            {
                "name": "researcher",
                "role": "research",
                "instructions": "Inspect only",
                "tools": ["Read"],
            },
            context,
        ).output

        result = SendMessageTool().run(
            {"to": "researcher", "summary": "start", "message": "Inspect requirements"},
            context,
        ).output

        stored = context.team_store.load_message(created["team_id"], result["message_id"])
        self.assertEqual(stored.sender_id, created["lead_agent_id"])
        self.assertEqual(stored.recipient_id, teammate["agent_id"])
        self.assertEqual(stored.status, "delivered")
        session = context.team_store.load_session(created["team_id"], teammate["session_id"])
        self.assertEqual(session["agent_id"], teammate["agent_id"])

    def test_creation_tools_explain_how_to_start_worker_execution(self) -> None:
        context = ToolContext(workspace_root=self.root)

        created = TeamCreateTool().run({"team_name": "guided"}, context).output
        self.assertFalse(created["team_started"])
        self.assertEqual(
            [item["tool"] for item in created["next_required_actions"]],
            ["TeammateCreate", "TaskCreate", "TeamRun"],
        )

        teammate = TeammateCreateTool().run(
            {
                "name": "reviewer",
                "role": "review",
                "instructions": "Review the implementation",
                "tools": ["Read"],
            },
            context,
        ).output
        self.assertFalse(teammate["worker_started"])
        self.assertEqual(
            [item["tool"] for item in teammate["next_required_actions"]],
            ["TaskCreate", "TeamRun"],
        )

        task = TaskCreateTool().run(
            {
                "key": "review",
                "subject": "Review",
                "description": "Review the implementation",
                "owner": "reviewer",
            },
            context,
        ).output
        self.assertFalse(task["task_started"])
        self.assertEqual(task["next_required_actions"][0]["tool"], "TeamRun")

    def test_lead_does_not_wait_for_a_worker_that_has_not_started(self) -> None:
        context = ToolContext(workspace_root=self.root)
        TeamCreateTool().run({"team_name": "idle"}, context)
        TeammateCreateTool().run(
            {
                "name": "reviewer",
                "role": "review",
                "instructions": "Review the implementation",
                "tools": ["Read", "SendMessage"],
            },
            context,
        )

        output = ReadMessagesTool().run({"wait_s": 1}, context).output

        self.assertEqual(output["messages"], [])
        self.assertTrue(output["wait_skipped"])
        self.assertEqual(output["next_required_actions"][0]["tool"], "TaskCreate")

    def test_teammates_can_exchange_and_poll_direct_messages(self) -> None:
        lead_context = ToolContext(workspace_root=self.root)
        created = TeamCreateTool().run({"team_name": "dynamic"}, lead_context).output
        first = TeammateCreateTool().run(
            {
                "name": "frontend",
                "role": "frontend contract",
                "instructions": "Coordinate the API contract",
                "tools": ["Read"],
            },
            lead_context,
        ).output
        second = TeammateCreateTool().run(
            {
                "name": "backend",
                "role": "backend contract",
                "instructions": "Coordinate the API contract",
                "tools": ["Read"],
            },
            lead_context,
        ).output
        sender = ToolContext(workspace_root=self.root, actor_id=first["agent_id"])
        receiver = ToolContext(workspace_root=self.root, actor_id=second["agent_id"])

        SendMessageTool().run(
            {
                "to": "backend",
                "summary": "interface",
                "message": {"path": "/orders", "method": "POST"},
            },
            sender,
        )
        inbox = ReadMessagesTool().run({"wait_s": 0}, receiver).output["messages"]

        self.assertEqual(len(inbox), 1)
        self.assertEqual(inbox[0]["from"], "frontend")
        self.assertEqual(inbox[0]["message"]["path"], "/orders")
        self.assertEqual(ReadMessagesTool().run({}, receiver).output["messages"], [])
        events = lead_context.team_store.list_events(created["team_id"])
        self.assertTrue(any(event["type"] == "message.consumed" for event in events))

    def test_task_tools_accept_stable_keys_and_done_alias(self) -> None:
        context = ToolContext(workspace_root=self.root)
        TeamCreateTool().run({"team_name": "dynamic"}, context)
        task = TaskCreateTool().run(
            {
                "key": "backend-contract",
                "subject": "Backend contract",
                "description": "Implement the agreed contract",
            },
            context,
        ).output["task"]

        updated = TaskUpdateTool().run(
            {"taskId": "backend-contract", "status": "done", "output": "ready"}, context
        ).output

        self.assertEqual(updated["taskId"], task["id"])
        self.assertEqual(updated["statusChange"]["to"], "completed")
        self.assertEqual(
            TaskGetTool().run({"taskId": "backend-contract"}, context).output["task"]["status"],
            "completed",
        )
        self.assertEqual(
            TaskOutputTool().run({"task_id": "backend-contract"}, context).output["task"]["output"],
            "ready",
        )

    def test_task_tool_enforces_state_transitions_without_partial_update(self) -> None:
        context = ToolContext(workspace_root=self.root)
        TeamCreateTool().run({"team_name": "demo"}, context)
        task_id = TaskCreateTool().run(
            {"subject": "Inspect", "description": "Inspect runtime"}, context
        ).output["task"]["id"]
        TaskUpdateTool().run({"taskId": task_id, "status": "completed"}, context)

        with self.assertRaises(ToolInputError):
            TaskUpdateTool().run(
                {"taskId": task_id, "status": "in_progress", "subject": "Changed"}, context
            )

        self.assertEqual(context.tasks[task_id]["subject"], "Inspect")

    def test_mutating_tools_are_not_marked_read_only(self) -> None:
        self.assertFalse(TeamCreateTool().spec().is_read_only)
        self.assertFalse(TeamCancelTool().spec().is_read_only)
        self.assertFalse(TeamDeleteTool().spec().is_read_only)
        self.assertFalse(TeamIntegrateTool().spec().is_read_only)
        self.assertFalse(TaskCreateTool().spec().is_read_only)
        self.assertFalse(TaskUpdateTool().spec().is_read_only)
        self.assertFalse(TaskRetryTool().spec().is_read_only)
        self.assertFalse(ReadMessagesTool().spec().is_read_only)
        self.assertFalse(SendMessageTool().spec().is_read_only)
        self.assertFalse(TeammateCreateTool().spec().is_read_only)
        self.assertFalse(TeammateStopTool().spec().is_read_only)
        self.assertFalse(TeammateResumeTool().spec().is_read_only)
        self.assertFalse(TeamRunTool().spec().is_read_only)
        self.assertFalse(TeamResumeTool().spec().is_read_only)


if __name__ == "__main__":
    unittest.main()
