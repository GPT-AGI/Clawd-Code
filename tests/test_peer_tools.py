from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.peer.models import PeerParticipant, PeerRunRecord
from src.peer.policy import CommunicationPolicy
from src.peer.store import PeerStore
from src.peer.tools import (
    PeerBroadcastTool,
    PeerListTool,
    PeerReadMessagesTool,
    PeerSendMessageTool,
)
from src.tool_system.context import ToolContext


class PeerToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = PeerStore(self.root / "runs")
        self.run_id = "tool-run"
        self.ids = ("tool-run-p1", "tool-run-p2", "tool-run-p3")
        self.store.create_run(
            PeerRunRecord(
                run_id=self.run_id,
                mission="mission",
                repo_path=str(self.root),
                base_revision="0" * 40,
                peer_count=3,
                communication="p2p",
                workspace_mode="shared",
                provider="scripted",
                model=None,
                timeout_seconds=10,
                max_turns=5,
                max_output_tokens=1024,
                token_budget=None,
                turn_budget=None,
                output_dir=str(self.root / "runs"),
            ),
            {"schema_version": 1},
        )
        for index, peer_id in enumerate(self.ids, start=1):
            self.store.save_participant(
                PeerParticipant(
                    peer_id=peer_id,
                    run_id=self.run_id,
                    name=f"peer-{index}",
                    session_id=f"session-{index}",
                    workspace_mode="shared",
                    workspace_path=str(self.root),
                )
            )
        policy = CommunicationPolicy("p2p", self.ids)
        self.control = SimpleNamespace(policy=policy, stop_event=threading.Event())

    def tearDown(self) -> None:
        self.temp.cleanup()

    def context(self, peer_id: str) -> ToolContext:
        return ToolContext(
            workspace_root=self.root,
            actor_id=peer_id,
            peer_store=self.store,
            peer_run_id=self.run_id,
            peer_id=peer_id,
            peer_control=self.control,
        )

    def test_all_peers_see_identical_roster_without_privilege_fields(self) -> None:
        rosters = [
            PeerListTool().run({}, self.context(peer_id)).output["peers"]
            for peer_id in self.ids
        ]
        self.assertEqual(rosters[0], rosters[1])
        self.assertEqual(rosters[1], rosters[2])
        self.assertTrue(all(set(item) == {"peer_id", "name", "status", "session_id"} for item in rosters[0]))

    def test_read_messages_waits_for_notification_without_polling(self) -> None:
        receiver = self.context(self.ids[1])
        sender = self.context(self.ids[0])

        def delayed_send() -> None:
            time.sleep(0.08)
            PeerSendMessageTool().run(
                {"to": "peer-2", "message": {"ready": True}}, sender
            )

        thread = threading.Thread(target=delayed_send)
        thread.start()
        started = time.monotonic()
        output = PeerReadMessagesTool().run(
            {"wait_seconds": 1}, receiver
        ).output
        elapsed = time.monotonic() - started
        thread.join(timeout=1)
        self.assertGreaterEqual(elapsed, 0.06)
        self.assertLess(elapsed, 0.5)
        self.assertEqual(output["messages"][0]["message"], {"ready": True})
        self.assertEqual(
            PeerReadMessagesTool().run({}, receiver).output["messages"], []
        )

    def test_broadcast_tool_returns_per_recipient_deliveries(self) -> None:
        output = PeerBroadcastTool().run(
            {"message": "hello", "idempotency_key": "hello-1"},
            self.context(self.ids[0]),
        ).output
        self.assertEqual(set(output["recipients"]), set(self.ids[1:]))
        self.assertEqual(len(output["message_ids"]), 2)
        self.assertEqual(len(self.store.list_messages(self.run_id)), 2)


if __name__ == "__main__":
    unittest.main()
