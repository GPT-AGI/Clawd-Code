from __future__ import annotations

import threading
import time
import unittest

from src.webhooks import WebhookEvent, WebhookProcessor


class WebhookAcceptance(unittest.TestCase):
    def test_duplicate_success_is_not_reprocessed(self) -> None:
        processor = WebhookProcessor()
        event = WebhookEvent("a", "evt-1", 1, {})
        calls: list[str] = []
        self.assertEqual(processor.process(event, lambda item: calls.append(item.event_id)), "processed")
        self.assertEqual(processor.process(event, lambda item: calls.append(item.event_id)), "duplicate")
        self.assertEqual(calls, ["evt-1"])

    def test_handler_failure_is_retryable(self) -> None:
        processor = WebhookProcessor()
        event = WebhookEvent("a", "evt-1", 1, {})
        with self.assertRaisesRegex(RuntimeError, "temporary"):
            processor.process(event, lambda _: (_ for _ in ()).throw(RuntimeError("temporary")))
        calls: list[int] = []
        self.assertEqual(processor.process(event, lambda _: calls.append(1)), "processed")
        self.assertEqual(calls, [1])

    def test_stale_sequence_is_ignored(self) -> None:
        processor = WebhookProcessor()
        processor.process(WebhookEvent("a", "new", 5, {}), lambda _: None)
        called: list[bool] = []
        result = processor.process(WebhookEvent("a", "old", 4, {}), lambda _: called.append(True))
        self.assertEqual(result, "stale")
        self.assertEqual(called, [])

    def test_equal_sequence_with_new_id_is_stale(self) -> None:
        processor = WebhookProcessor()
        processor.process(WebhookEvent("a", "first", 5, {}), lambda _: None)
        self.assertEqual(
            processor.process(WebhookEvent("a", "second", 5, {}), lambda _: None),
            "stale",
        )

    def test_tenant_state_is_isolated(self) -> None:
        processor = WebhookProcessor()
        calls: list[str] = []
        for tenant in ("a", "b"):
            result = processor.process(
                WebhookEvent(tenant, "same-id", 1, {}),
                lambda event: calls.append(event.tenant_id),
            )
            self.assertEqual(result, "processed")
        self.assertEqual(calls, ["a", "b"])

    def test_concurrent_duplicate_invokes_handler_once(self) -> None:
        processor = WebhookProcessor()
        event = WebhookEvent("a", "race", 1, {})
        barrier = threading.Barrier(4)
        calls: list[int] = []
        results: list[str] = []
        lock = threading.Lock()

        def handler(_: WebhookEvent) -> None:
            time.sleep(0.03)
            with lock:
                calls.append(1)

        def run() -> None:
            barrier.wait(timeout=1)
            result = processor.process(event, handler)
            with lock:
                results.append(result)

        threads = [threading.Thread(target=run) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
        self.assertEqual(calls, [1])
        self.assertEqual(sorted(results), ["duplicate", "duplicate", "duplicate", "processed"])

    def test_validation_happens_before_handler(self) -> None:
        processor = WebhookProcessor()
        called: list[bool] = []
        for event in (
            WebhookEvent("", "id", 1, {}),
            WebhookEvent("a", "", 1, {}),
            WebhookEvent("a", "id", 0, {}),
            WebhookEvent("a", "id", True, {}),
        ):
            with self.assertRaises(ValueError):
                processor.process(event, lambda _: called.append(True))
        self.assertEqual(called, [])

    def test_snapshot_is_sorted_and_detached(self) -> None:
        processor = WebhookProcessor()
        processor.process(WebhookEvent("a", "z", 1, {}), lambda _: None)
        processor.process(WebhookEvent("a", "a", 2, {}), lambda _: None)
        snapshot = processor.snapshot()
        self.assertEqual(snapshot["a"], {"processed_ids": ["a", "z"], "last_sequence": 2})
        snapshot["a"]["processed_ids"].append("injected")
        self.assertNotIn("injected", processor.snapshot()["a"]["processed_ids"])


if __name__ == "__main__":
    unittest.main()
