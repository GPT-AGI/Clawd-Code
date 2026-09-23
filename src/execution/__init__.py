"""Execution backends for local and sandboxed agent workspaces."""

from .backend import CommandOutcome, RemoteStat, WorkspaceBackend

__all__ = ["CommandOutcome", "RemoteStat", "WorkspaceBackend"]
