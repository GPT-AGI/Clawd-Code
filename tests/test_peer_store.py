from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from src.peer.models import PeerParticipant, PeerRunRecord
from src.peer.policy import CommunicationPolicy, PolicyRejected
from src.peer.store import PeerStore


class PeerStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = PeerStore(self.root / "runs")
        self.run_id = "run-one"
        self.peer_ids = ("run-one-p1", "run-one-p2", "run-one-p3")
        run = PeerRunRecord(
            run_id=self.run_id,
            mission="same mission",
            repo_path=str(self.root),
            base_revision="0" * 40,
            peer_count=3,
            communication="p2p",
            workspace_mode="shared",
            provider="scripted",
            model=None,
            timeout_seconds=30,
            max_turns=10,
            max_output_tokens=1024,
            token_budget=None,
            turn_budget=None,
            output_dir=str(self.root / "runs"),
        )
        self.store.create_run(run, {"schema_version": 1, "run_id": self.run_id})
        for index, peer_id in enumerate(self.peer_ids, start=1):
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
        self.policy = CommunicationPolicy("p2p", self.peer_ids)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_roster_is_stable_and_contains_no_lead(self) -> None:
        first = [peer.to_dict() for peer in self.store.list_participants(self.run_id)]
        second = [peer.to_dict() for peer in PeerStore(self.root / "runs").list_participants(self.run_id)]
        self.assertEqual(first, second)
        self.assertEqual({peer["name"] for peer in first}, {"peer-1", "peer-2", "peer-3"})
        self.assertNotIn("lead_agent_id", json.dumps(first))

    def test_direct_message_persists_and_consumes_exactly_once(self) -> None:
        message = self.store.send_message(
            self.run_id,
            self.peer_ids[0],
            "peer-2",
            {"contract": "v1"},
            summary="interface",
            policy=self.policy,
        )
        reloaded = PeerStore(self.root / "runs")
        unread = reloaded.list_messages(
            self.run_id, recipient_id=self.peer_ids[1], status="delivered"
        )
        self.assertEqual([item.message_id for item in unread], [message.message_id])
        consumed = reloaded.consume_messages(self.run_id, self.peer_ids[1])
        self.assertEqual([item.payload for item in consumed], [{"contract": "v1"}])
        self.assertEqual(reloaded.consume_messages(self.run_id, self.peer_ids[1]), [])
        persisted = reloaded.list_messages(self.run_id)[0]
        self.assertEqual(persisted.status, "consumed")
        self.assertIsNotNone(persisted.consumed_at)

    def test_unknown_self_cross_run_and_illegal_payload_are_rejected(self) -> None:
        with self.assertRaises(PolicyRejected):
            self.store.send_message(
                self.run_id,
                self.peer_ids[0],
                self.peer_ids[0],
                "self",
                policy=self.policy,
            )
        with self.assertRaises(PolicyRejected):
            self.store.send_message(
                self.run_id,
                self.peer_ids[0],
                "unknown-peer",
                "unknown",
                policy=self.policy,
            )
        foreign = CommunicationPolicy("p2p", ("other-p1", "other-p2"))
        with self.assertRaises(PolicyRejected):
            self.store.send_message(
                self.run_id, "other-p1", "other-p2", "cross", policy=foreign
            )
        with self.assertRaises(ValueError):
            self.store.send_message(
                self.run_id,
                self.peer_ids[0],
                self.peer_ids[1],
                object(),
                policy=self.policy,
            )
        rejected = [event for event in self.store.list_events(self.run_id) if event["type"] == "policy.rejected"]
        self.assertEqual(len(rejected), 3)

    def test_broadcast_exactly_once_excludes_sender_and_is_idempotent(self) -> None:
        first = self.store.broadcast(
            self.run_id,
            self.peer_ids[0],
            {"decision": 7},
            policy=self.policy,
            idempotency_key="decision-7",
        )
        retry = self.store.broadcast(
            self.run_id,
            self.peer_ids[0],
            {"decision": 7},
            policy=self.policy,
            idempotency_key="decision-7",
        )
        self.assertEqual(first.broadcast_id, retry.broadcast_id)
        self.assertEqual(set(first.recipients), set(self.peer_ids[1:]))
        self.assertNotIn(self.peer_ids[0], first.recipients)
        messages = self.store.list_messages(self.run_id)
        self.assertEqual(len(messages), 2)
        self.assertEqual(len({item.recipient_id for item in messages}), 2)

    def test_star_acl_allows_only_edges_touching_coordinator(self) -> None:
        star = CommunicationPolicy("star", self.peer_ids, self.peer_ids[0])
        self.assertTrue(star.can_send(self.peer_ids[1], self.peer_ids[0]))
        self.assertTrue(star.can_send(self.peer_ids[0], self.peer_ids[2]))
        self.assertFalse(star.can_send(self.peer_ids[1], self.peer_ids[2]))
        with self.assertRaises(PolicyRejected):
            self.store.send_message(
                self.run_id,
                self.peer_ids[1],
                self.peer_ids[2],
                "forbidden",
                policy=star,
            )
        worker_broadcast = self.store.broadcast(
            self.run_id,
            self.peer_ids[1],
            "to hub",
            policy=star,
        )
        self.assertEqual(worker_broadcast.recipients, [self.peer_ids[0]])

    def test_independent_and_artifact_only_have_no_edges(self) -> None:
        for condition in ("solo", "independent", "artifact-only"):
            policy = CommunicationPolicy(condition, self.peer_ids)
            self.assertFalse(policy.exposes_message_tools())
            self.assertFalse(policy.can_send(self.peer_ids[0], self.peer_ids[1]))

    def test_concurrent_consumers_do_not_duplicate_delivery(self) -> None:
        count = 40
        for index in range(count):
            self.store.send_message(
                self.run_id,
                self.peer_ids[0],
                self.peer_ids[1],
                {"index": index},
                policy=self.policy,
            )
        outputs: list[str] = []
        lock = threading.Lock()

        def consume() -> None:
            items = PeerStore(self.root / "runs").consume_messages(
                self.run_id, self.peer_ids[1]
            )
            with lock:
                outputs.extend(item.message_id for item in items)

        threads = [threading.Thread(target=consume) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(len(outputs), count)
        self.assertEqual(len(set(outputs)), count)
        self.assertTrue(all(not thread.is_alive() for thread in threads))

    def test_concurrent_events_remain_valid_jsonl(self) -> None:
        def append(worker: int) -> None:
            for index in range(25):
                self.store.append_event(
                    self.run_id, "test.concurrent", {"worker": worker, "index": index}
                )

        threads = [threading.Thread(target=append, args=(index,)) for index in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        path = self.store.run_dir(self.run_id) / "events.jsonl"
        parsed = [json.loads(line) for line in path.read_text().splitlines()]
        concurrent = [item for item in parsed if item["type"] == "test.concurrent"]
        self.assertEqual(len(concurrent), 150)
        self.assertTrue(all("created_at" in item and "monotonic_ns" in item for item in concurrent))

    def test_simultaneous_broadcasts_have_exact_delivery_counts(self) -> None:
        broadcasts: list[str] = []
        lock = threading.Lock()

        def send(sender_id: str) -> None:
            item = self.store.broadcast(
                self.run_id,
                sender_id,
                {"sender": sender_id},
                policy=self.policy,
                idempotency_key=f"broadcast-{sender_id}",
            )
            with lock:
                broadcasts.append(item.broadcast_id)

        threads = [threading.Thread(target=send, args=(peer_id,)) for peer_id in self.peer_ids]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(len(set(broadcasts)), 3)
        self.assertEqual(len(self.store.list_messages(self.run_id)), 6)
        self.assertEqual(len(self.store.list_broadcasts(self.run_id)), 3)


if __name__ == "__main__":
    unittest.main()
