from __future__ import annotations

from dataclasses import dataclass


class PolicyRejected(ValueError):
    pass


@dataclass(frozen=True)
class CommunicationPolicy:
    condition: str
    peer_ids: tuple[str, ...]
    coordinator_peer_id: str | None = None

    MESSAGE_CONDITIONS = {"star", "p2p"}

    def exposes_message_tools(self) -> bool:
        return self.condition in self.MESSAGE_CONDITIONS

    def can_send(self, sender_id: str, recipient_id: str) -> bool:
        if sender_id not in self.peer_ids or recipient_id not in self.peer_ids:
            return False
        if sender_id == recipient_id:
            return False
        if self.condition == "p2p":
            return True
        if self.condition == "star":
            coordinator = self.coordinator_peer_id
            return coordinator is not None and coordinator in {sender_id, recipient_id}
        return False

    def require_send(self, sender_id: str, recipient_id: str) -> None:
        if sender_id not in self.peer_ids:
            raise PolicyRejected(f"sender is not a participant in this run: {sender_id}")
        if recipient_id not in self.peer_ids:
            raise PolicyRejected(f"recipient is not a participant in this run: {recipient_id}")
        if sender_id == recipient_id:
            raise PolicyRejected("peers cannot send messages to themselves")
        if not self.can_send(sender_id, recipient_id):
            raise PolicyRejected(
                f"communication edge rejected by {self.condition} policy: "
                f"{sender_id} -> {recipient_id}"
            )

    def broadcast_recipients(self, sender_id: str) -> list[str]:
        if sender_id not in self.peer_ids:
            raise PolicyRejected(f"sender is not a participant in this run: {sender_id}")
        return [
            peer_id
            for peer_id in self.peer_ids
            if peer_id != sender_id and self.can_send(sender_id, peer_id)
        ]
