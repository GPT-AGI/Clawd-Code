"""Peer-native collaboration runtime without a privileged LLM lead."""

from .models import PeerParticipant, PeerRunConfig, PeerRunRecord
from .store import PeerStore

__all__ = [
    "PeerParticipant",
    "PeerRunConfig",
    "PeerRunRecord",
    "PeerRuntime",
    "PeerStore",
]


def __getattr__(name: str):
    if name == "PeerRuntime":
        from .runtime import PeerRuntime

        return PeerRuntime
    raise AttributeError(name)
