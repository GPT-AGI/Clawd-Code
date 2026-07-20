from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

from .models import (
    PeerBroadcast,
    PeerMessage,
    PeerParticipant,
    PeerRunRecord,
    PeerSubmission,
    utc_now,
)
from .policy import CommunicationPolicy, PolicyRejected


_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _locked_path(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock(path):
        with path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class PeerSignalBus:
    """In-process event delivery for the first thread-based peer backend."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._generations: dict[tuple[str, str], int] = {}

    def generation(self, run_id: str, peer_id: str) -> int:
        with self._condition:
            return self._generations.get((run_id, peer_id), 0)

    def notify(self, run_id: str, peer_id: str) -> None:
        with self._condition:
            key = (run_id, peer_id)
            self._generations[key] = self._generations.get(key, 0) + 1
            self._condition.notify_all()

    def notify_run(self, run_id: str) -> None:
        with self._condition:
            for key in list(self._generations):
                if key[0] == run_id:
                    self._generations[key] += 1
            self._condition.notify_all()

    def wait(
        self,
        run_id: str,
        peer_id: str,
        generation: int,
        timeout: float | None,
        stop_event: threading.Event | None = None,
    ) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._generations.get((run_id, peer_id), 0) == generation:
                if stop_event is not None and stop_event.is_set():
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True


class PeerStore:
    """Filesystem-backed peer state with run-wide atomic mutations."""

    MAX_PAYLOAD_BYTES = 64 * 1024

    def __init__(self, base_dir: str | Path, *, signal_bus: PeerSignalBus | None = None):
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.signal_bus = signal_bus or PeerSignalBus()

    def run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id

    def _state_lock(self, run_id: str) -> Path:
        return self.run_dir(run_id) / ".state.lock"

    def create_run(self, run: PeerRunRecord, manifest: dict[str, Any]) -> None:
        directory = self.run_dir(run.run_id)
        if directory.exists():
            raise ValueError(f"peer run already exists: {run.run_id}")
        for name in ("participants", "sessions", "messages", "broadcasts", "submissions"):
            (directory / name).mkdir(parents=True, exist_ok=True)
        self._write_json(directory / "run.json", run.to_dict())
        self._write_json(directory / "manifest.json", manifest)
        (directory / "events.jsonl").touch()
        self.append_event(run.run_id, "peer_run.created", {"run": run.to_dict()})

    def load_run(self, run_id: str) -> PeerRunRecord | None:
        path = self.run_dir(run_id) / "run.json"
        if not path.is_file():
            return None
        return PeerRunRecord.from_dict(self._read_json(path))

    def save_run(self, run: PeerRunRecord) -> None:
        with _locked_path(self._state_lock(run.run_id)):
            self._write_json_unlocked(self.run_dir(run.run_id) / "run.json", run.to_dict())

    def mutate_run(
        self, run_id: str, mutator: Callable[[PeerRunRecord], Any]
    ) -> tuple[PeerRunRecord, Any]:
        with _locked_path(self._state_lock(run_id)):
            run = self._load_run_unlocked(run_id)
            result = mutator(run)
            self._write_json_unlocked(self.run_dir(run_id) / "run.json", run.to_dict())
        return run, result

    def save_participant(self, participant: PeerParticipant) -> None:
        with _locked_path(self._state_lock(participant.run_id)):
            self._write_json_unlocked(
                self._participant_path(participant.run_id, participant.peer_id),
                participant.to_dict(),
            )

    def load_participant(self, run_id: str, peer_id: str) -> PeerParticipant | None:
        path = self._participant_path(run_id, peer_id)
        if not path.is_file():
            return None
        return PeerParticipant.from_dict(self._read_json(path))

    def list_participants(self, run_id: str) -> list[PeerParticipant]:
        directory = self.run_dir(run_id) / "participants"
        if not directory.is_dir():
            return []
        peers = [
            PeerParticipant.from_dict(self._read_json(path))
            for path in sorted(directory.glob("*.json"))
        ]
        peers.sort(key=lambda peer: (peer.name, peer.peer_id))
        return peers

    def find_participant(self, run_id: str, identity: str) -> PeerParticipant | None:
        normalized = identity.strip().casefold()
        for peer in self.list_participants(run_id):
            if peer.peer_id == identity or peer.name.casefold() == normalized:
                return peer
        return None

    def mutate_participant(
        self,
        run_id: str,
        peer_id: str,
        mutator: Callable[[PeerParticipant], Any],
    ) -> tuple[PeerParticipant, Any]:
        with _locked_path(self._state_lock(run_id)):
            path = self._participant_path(run_id, peer_id)
            if not path.is_file():
                raise ValueError(f"unknown peer: {peer_id}")
            participant = PeerParticipant.from_dict(self._read_json(path))
            result = mutator(participant)
            self._write_json_unlocked(path, participant.to_dict())
        return participant, result

    def save_session(self, run_id: str, session_id: str, value: dict[str, Any]) -> None:
        with _locked_path(self._state_lock(run_id)):
            self._write_json_unlocked(
                self.run_dir(run_id) / "sessions" / f"{session_id}.json", value
            )

    def load_session(self, run_id: str, session_id: str) -> dict[str, Any] | None:
        path = self.run_dir(run_id) / "sessions" / f"{session_id}.json"
        return self._read_json(path) if path.is_file() else None

    def append_event(
        self,
        run_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
        *,
        created_at: str | None = None,
        monotonic_ns: int | None = None,
    ) -> dict[str, Any]:
        event = self._new_event(
            run_id,
            event_type,
            data,
            created_at=created_at,
            monotonic_ns=monotonic_ns,
        )
        with _locked_path(self._state_lock(run_id)):
            self._append_event_unlocked(run_id, event)
        return event

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        path = self.run_dir(run_id) / "events.jsonl"
        if not path.is_file():
            return []
        result: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    result.append(value)
        return result

    def record_policy_rejection(
        self,
        run_id: str,
        sender_id: str,
        recipient_id: str | None,
        operation: str,
        reason: str,
    ) -> None:
        self.append_event(
            run_id,
            "policy.rejected",
            {
                "peer_id": sender_id,
                "recipient_id": recipient_id,
                "operation": operation,
                "reason": reason,
            },
        )

    def send_message(
        self,
        run_id: str,
        sender_id: str,
        recipient: str,
        payload: Any,
        *,
        policy: CommunicationPolicy,
        summary: str | None = None,
        idempotency_key: str | None = None,
        broadcast_id: str | None = None,
    ) -> PeerMessage:
        encoded = self._validate_payload(payload)
        if summary is not None and (not isinstance(summary, str) or len(summary) > 1_000):
            raise ValueError("summary must be a string of at most 1000 characters")
        sender_peer = self.find_participant(run_id, sender_id)
        recipient_peer = self.find_participant(run_id, recipient)
        recipient_id = recipient_peer.peer_id if recipient_peer is not None else recipient
        if sender_peer is None:
            reason = f"sender is not a participant in this run: {sender_id}"
            self.record_policy_rejection(run_id, sender_id, recipient_id, "send", reason)
            raise PolicyRejected(reason)
        if recipient_peer is None:
            reason = f"recipient is not a participant in this run: {recipient_id}"
            self.record_policy_rejection(run_id, sender_id, recipient_id, "send", reason)
            raise PolicyRejected(reason)
        try:
            policy.require_send(sender_id, recipient_id)
        except PolicyRejected as exc:
            self.record_policy_rejection(
                run_id, sender_id, recipient_id, "send", str(exc)
            )
            raise

        with _locked_path(self._state_lock(run_id)):
            if idempotency_key:
                previous = self._find_idempotent_message_unlocked(
                    run_id, sender_id, recipient_id, idempotency_key, broadcast_id
                )
                if previous is not None:
                    return previous
            now = utc_now()
            message = PeerMessage(
                message_id=uuid.uuid4().hex,
                run_id=run_id,
                sender_id=sender_id,
                recipient_id=recipient_id,
                payload=payload,
                payload_size_bytes=len(encoded),
                summary=summary,
                idempotency_key=idempotency_key,
                broadcast_id=broadcast_id,
                created_at=now,
                delivered_at=now,
            )
            self._write_json_unlocked(self._message_path(run_id, message.message_id), message.to_dict())
            metadata = self._message_metadata(message)
            self._append_event_unlocked(
                run_id, self._new_event(run_id, "message.created", metadata)
            )
            self._append_event_unlocked(
                run_id, self._new_event(run_id, "message.delivered", metadata)
            )
        self.signal_bus.notify(run_id, recipient_id)
        return message

    def broadcast(
        self,
        run_id: str,
        sender_id: str,
        payload: Any,
        *,
        policy: CommunicationPolicy,
        summary: str | None = None,
        idempotency_key: str | None = None,
    ) -> PeerBroadcast:
        encoded = self._validate_payload(payload)
        if self.find_participant(run_id, sender_id) is None:
            reason = f"sender is not a participant in this run: {sender_id}"
            self.record_policy_rejection(run_id, sender_id, None, "broadcast", reason)
            raise PolicyRejected(reason)
        recipients = policy.broadcast_recipients(sender_id)
        if not recipients:
            reason = f"broadcast has no allowed recipients under {policy.condition} policy"
            self.record_policy_rejection(run_id, sender_id, None, "broadcast", reason)
            raise PolicyRejected(reason)

        created_messages: list[PeerMessage] = []
        with _locked_path(self._state_lock(run_id)):
            if idempotency_key:
                previous = self._find_broadcast_unlocked(run_id, sender_id, idempotency_key)
                if previous is not None:
                    return previous
            broadcast_id = uuid.uuid4().hex
            for recipient_id in recipients:
                now = utc_now()
                message = PeerMessage(
                    message_id=uuid.uuid4().hex,
                    run_id=run_id,
                    sender_id=sender_id,
                    recipient_id=recipient_id,
                    payload=payload,
                    payload_size_bytes=len(encoded),
                    summary=summary,
                    broadcast_id=broadcast_id,
                    idempotency_key=idempotency_key,
                    created_at=now,
                    delivered_at=now,
                )
                self._write_json_unlocked(
                    self._message_path(run_id, message.message_id), message.to_dict()
                )
                metadata = self._message_metadata(message)
                self._append_event_unlocked(
                    run_id, self._new_event(run_id, "message.created", metadata)
                )
                self._append_event_unlocked(
                    run_id, self._new_event(run_id, "message.delivered", metadata)
                )
                created_messages.append(message)
            broadcast = PeerBroadcast(
                broadcast_id=broadcast_id,
                run_id=run_id,
                sender_id=sender_id,
                recipients=recipients,
                message_ids=[message.message_id for message in created_messages],
                payload_size_bytes=len(encoded),
                idempotency_key=idempotency_key,
            )
            self._write_json_unlocked(
                self._broadcast_path(run_id, broadcast_id), broadcast.to_dict()
            )
            self._append_event_unlocked(
                run_id,
                self._new_event(
                    run_id,
                    "broadcast.created",
                    {
                        "broadcast_id": broadcast_id,
                        "sender_id": sender_id,
                        "recipients": recipients,
                        "message_ids": broadcast.message_ids,
                        "payload_size_bytes": len(encoded),
                    },
                ),
            )
        for recipient_id in recipients:
            self.signal_bus.notify(run_id, recipient_id)
        return broadcast

    def consume_messages(self, run_id: str, recipient_id: str) -> list[PeerMessage]:
        consumed: list[PeerMessage] = []
        with _locked_path(self._state_lock(run_id)):
            for message in self._list_messages_unlocked(run_id):
                if message.recipient_id != recipient_id or message.status != "delivered":
                    continue
                message.consume()
                self._write_json_unlocked(
                    self._message_path(run_id, message.message_id), message.to_dict()
                )
                self._append_event_unlocked(
                    run_id,
                    self._new_event(
                        run_id,
                        "message.consumed",
                        {
                            **self._message_metadata(message),
                            "consumed_at": message.consumed_at,
                        },
                    ),
                )
                consumed.append(message)
        return consumed

    def list_messages(
        self,
        run_id: str,
        *,
        recipient_id: str | None = None,
        status: str | None = None,
    ) -> list[PeerMessage]:
        messages = self._list_messages_unlocked(run_id)
        if recipient_id is not None:
            messages = [item for item in messages if item.recipient_id == recipient_id]
        if status is not None:
            messages = [item for item in messages if item.status == status]
        return messages

    def has_unread(self, run_id: str, recipient_id: str) -> bool:
        return bool(self.list_messages(run_id, recipient_id=recipient_id, status="delivered"))

    def wait_for_unread(
        self,
        run_id: str,
        recipient_id: str,
        timeout: float | None,
        *,
        stop_event: threading.Event | None = None,
    ) -> bool:
        generation = self.signal_bus.generation(run_id, recipient_id)
        if self.has_unread(run_id, recipient_id):
            return True
        self.signal_bus.wait(run_id, recipient_id, generation, timeout, stop_event)
        return self.has_unread(run_id, recipient_id)

    def attempt_submission(
        self,
        run_id: str,
        peer_id: str,
        revision: str,
        summary: str,
        validation: dict[str, Any],
    ) -> tuple[PeerSubmission, dict[str, Any] | None]:
        with _locked_path(self._state_lock(run_id)):
            run = self._load_run_unlocked(run_id)
            if not self._participant_path(run_id, peer_id).is_file():
                raise ValueError(f"submitting peer is not a participant in this run: {peer_id}")
            attempt_id = uuid.uuid4().hex
            already = run.accepted_submission
            if already is not None:
                submission = PeerSubmission(
                    attempt_id=attempt_id,
                    run_id=run_id,
                    peer_id=peer_id,
                    revision=revision,
                    summary=summary,
                    status="already_submitted",
                    validation=validation,
                )
                accepted = dict(already)
            elif not validation.get("valid"):
                submission = PeerSubmission(
                    attempt_id=attempt_id,
                    run_id=run_id,
                    peer_id=peer_id,
                    revision=revision,
                    summary=summary,
                    status="rejected",
                    validation=validation,
                )
                accepted = None
            else:
                submission = PeerSubmission(
                    attempt_id=attempt_id,
                    run_id=run_id,
                    peer_id=peer_id,
                    revision=str(validation.get("resolved_revision") or revision),
                    summary=summary,
                    status="accepted",
                    validation=validation,
                )
                accepted = submission.to_dict()
                run.accepted_submission = accepted
                run.set_status("submitted", reason="accepted peer submission")
                self._write_json_unlocked(self.run_dir(run_id) / "run.json", run.to_dict())
            self._write_json_unlocked(
                self.run_dir(run_id) / "submissions" / f"{attempt_id}.json",
                submission.to_dict(),
            )
            event_type = {
                "accepted": "submit.accepted",
                "rejected": "submit.rejected",
                "already_submitted": "submit.already_submitted",
            }[submission.status]
            self._append_event_unlocked(
                run_id, self._new_event(run_id, "submit.attempted", submission.to_dict())
            )
            self._append_event_unlocked(
                run_id, self._new_event(run_id, event_type, submission.to_dict())
            )
        return submission, accepted

    def list_submissions(self, run_id: str) -> list[PeerSubmission]:
        directory = self.run_dir(run_id) / "submissions"
        values = [
            PeerSubmission.from_dict(self._read_json(path))
            for path in sorted(directory.glob("*.json"))
        ] if directory.is_dir() else []
        values.sort(key=lambda item: (item.created_at, item.attempt_id))
        return values

    def list_broadcasts(self, run_id: str) -> list[PeerBroadcast]:
        directory = self.run_dir(run_id) / "broadcasts"
        values = [
            PeerBroadcast.from_dict(self._read_json(path))
            for path in sorted(directory.glob("*.json"))
        ] if directory.is_dir() else []
        values.sort(key=lambda item: (item.created_at, item.broadcast_id))
        return values

    def update_usage(
        self,
        run_id: str,
        peer_id: str,
        usage_delta: dict[str, int],
    ) -> tuple[dict[str, int], dict[str, int]]:
        keys = (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "total_tokens",
            "turns",
            "model_calls",
            "tool_calls",
        )
        with _locked_path(self._state_lock(run_id)):
            run = self._load_run_unlocked(run_id)
            participant_path = self._participant_path(run_id, peer_id)
            participant = PeerParticipant.from_dict(self._read_json(participant_path))
            for key in keys:
                value = int(usage_delta.get(key, 0) or 0)
                run.usage[key] = int(run.usage.get(key, 0) or 0) + value
                participant.usage[key] = int(participant.usage.get(key, 0) or 0) + value
            run.usage["total_tokens"] = sum(
                int(run.usage.get(key, 0) or 0)
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            )
            participant.usage["total_tokens"] = sum(
                int(participant.usage.get(key, 0) or 0)
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            )
            run.updated_at = utc_now()
            participant.updated_at = utc_now()
            self._write_json_unlocked(self.run_dir(run_id) / "run.json", run.to_dict())
            self._write_json_unlocked(participant_path, participant.to_dict())
        return dict(run.usage), dict(participant.usage)

    def save_result(self, run_id: str, result: dict[str, Any]) -> Path:
        path = self.run_dir(run_id) / "result.json"
        self._write_json(path, result)
        return path

    def load_manifest(self, run_id: str) -> dict[str, Any]:
        return self._read_json(self.run_dir(run_id) / "manifest.json")

    def _load_run_unlocked(self, run_id: str) -> PeerRunRecord:
        path = self.run_dir(run_id) / "run.json"
        if not path.is_file():
            raise ValueError(f"unknown peer run: {run_id}")
        return PeerRunRecord.from_dict(self._read_json(path))

    def _participant_path(self, run_id: str, peer_id: str) -> Path:
        return self.run_dir(run_id) / "participants" / f"{peer_id}.json"

    def _message_path(self, run_id: str, message_id: str) -> Path:
        return self.run_dir(run_id) / "messages" / f"{message_id}.json"

    def _broadcast_path(self, run_id: str, broadcast_id: str) -> Path:
        return self.run_dir(run_id) / "broadcasts" / f"{broadcast_id}.json"

    def _list_messages_unlocked(self, run_id: str) -> list[PeerMessage]:
        directory = self.run_dir(run_id) / "messages"
        values = [
            PeerMessage.from_dict(self._read_json(path))
            for path in directory.glob("*.json")
        ] if directory.is_dir() else []
        values.sort(key=lambda message: (message.created_at, message.message_id))
        return values

    def _find_idempotent_message_unlocked(
        self,
        run_id: str,
        sender_id: str,
        recipient_id: str,
        key: str,
        broadcast_id: str | None,
    ) -> PeerMessage | None:
        for message in self._list_messages_unlocked(run_id):
            if (
                message.sender_id == sender_id
                and message.recipient_id == recipient_id
                and message.idempotency_key == key
                and message.broadcast_id == broadcast_id
            ):
                return message
        return None

    def _find_broadcast_unlocked(
        self, run_id: str, sender_id: str, key: str
    ) -> PeerBroadcast | None:
        directory = self.run_dir(run_id) / "broadcasts"
        if not directory.is_dir():
            return None
        for path in directory.glob("*.json"):
            value = PeerBroadcast.from_dict(self._read_json(path))
            if value.sender_id == sender_id and value.idempotency_key == key:
                return value
        return None

    @staticmethod
    def _validate_payload(payload: Any) -> bytes:
        if payload is None:
            raise ValueError("message payload cannot be null")
        try:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("message payload must be JSON serializable") from exc
        if len(encoded) > PeerStore.MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"message payload exceeds {PeerStore.MAX_PAYLOAD_BYTES} bytes"
            )
        return encoded

    @staticmethod
    def _message_metadata(message: PeerMessage) -> dict[str, Any]:
        return {
            "message_id": message.message_id,
            "sender_id": message.sender_id,
            "recipient_id": message.recipient_id,
            "broadcast_id": message.broadcast_id,
            "payload_size_bytes": message.payload_size_bytes,
            "status": message.status,
            "created_at": message.created_at,
            "delivered_at": message.delivered_at,
        }

    @staticmethod
    def _new_event(
        run_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
        *,
        created_at: str | None = None,
        monotonic_ns: int | None = None,
    ) -> dict[str, Any]:
        return {
            "event_id": uuid.uuid4().hex,
            "run_id": run_id,
            "type": event_type,
            "created_at": created_at or utc_now(),
            "monotonic_ns": monotonic_ns if monotonic_ns is not None else time.monotonic_ns(),
            "data": data or {},
        }

    def _append_event_unlocked(self, run_id: str, event: dict[str, Any]) -> None:
        path = self.run_dir(run_id) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object in {path}")
        return value

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        with _locked_path(path.with_name(f".{path.name}.lock")):
            PeerStore._write_json_unlocked(path, value)

    @staticmethod
    def _write_json_unlocked(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
