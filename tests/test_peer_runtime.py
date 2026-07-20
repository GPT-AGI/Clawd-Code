from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from src.peer.backend import PeerBoundaryResult, ScriptedPeerBackend
from src.peer.models import PeerRunConfig
from src.peer.runtime import PeerRuntime
from src.tool_system.defaults import build_default_registry
from src.tool_system.protocol import ToolCall


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


class PeerRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.name", "Peer Tests")
        git(self.repo, "config", "user.email", "peer-tests@example.invalid")
        (self.repo / "README.md").write_text("fixture\n", encoding="utf-8")
        git(self.repo, "add", "README.md")
        git(self.repo, "commit", "-qm", "initial")
        self.output = Path(self.temp.name) / "runs"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config(self, **overrides: object) -> PeerRunConfig:
        values: dict[str, object] = {
            "repo_path": str(self.repo),
            "mission": "Implement the repository and submit a verified revision.",
            "peers": 2,
            "communication": "p2p",
            "workspace_mode": "shared",
            "provider": "scripted",
            "model": "deterministic",
            "timeout_seconds": 3.0,
            "max_turns": 10,
            "output_dir": str(self.output),
        }
        values.update(overrides)
        return PeerRunConfig(**values)  # type: ignore[arg-type]

    def runtime(self, handler: object) -> PeerRuntime:
        return PeerRuntime(
            ScriptedPeerBackend(handler),  # type: ignore[arg-type]
            build_default_registry(include_user_tools=False),
        )

    def test_peers_overlap_and_sessions_are_independent(self) -> None:
        starts: dict[str, float] = {}
        initial_prompts: dict[str, str] = {}
        lock = threading.Lock()

        def handler(session, prompt, registry, context):
            with lock:
                starts[session.spec.peer_id] = time.monotonic()
                initial_prompts[session.spec.peer_id] = prompt
            time.sleep(0.15)
            if session.spec.peer_name == "peer-1":
                head = git(Path(session.spec.workspace_path), "rev-parse", "HEAD")
                registry.dispatch(
                    ToolCall(
                        "PeerSubmit",
                        {"revision": head, "summary": "base fixture is valid"},
                    ),
                    context,
                )
            return PeerBoundaryResult(response_text="done", num_turns=1)

        result = self.runtime(handler).run(self.config())
        self.assertEqual(result["status"], "completed")
        self.assertLess(max(starts.values()) - min(starts.values()), 0.1)
        sessions = {peer["session_id"] for peer in result["participants"]}
        self.assertEqual(len(sessions), 2)
        self.assertEqual(set(initial_prompts.values()), {self.config().mission})
        self.assertEqual(result["orphan_threads"], [])

    def test_idle_peer_is_event_driven_woken_and_sees_message(self) -> None:
        peer_two_ready = threading.Event()
        wake_prompts: list[str] = []

        def handler(session, prompt, registry, context):
            if session.spec.peer_name == "peer-2" and session.boundary_index == 1:
                peer_two_ready.set()
                return PeerBoundaryResult(response_text="available", num_turns=1)
            if session.spec.peer_name == "peer-1" and session.boundary_index == 1:
                self.assertTrue(peer_two_ready.wait(1))
                sent = registry.dispatch(
                    ToolCall(
                        "SendMessage",
                        {
                            "to": "peer-2",
                            "summary": "contract",
                            "message": {"endpoint": "/v1/items"},
                        },
                    ),
                    context,
                )
                self.assertFalse(sent.is_error)
                return PeerBoundaryResult(response_text="sent", num_turns=1)
            if session.spec.peer_name == "peer-2" and session.boundary_index == 2:
                wake_prompts.append(prompt)
                head = git(Path(session.spec.workspace_path), "rev-parse", "HEAD")
                registry.dispatch(
                    ToolCall(
                        "PeerSubmit",
                        {"revision": head, "summary": "contract received"},
                    ),
                    context,
                )
            return PeerBoundaryResult(response_text="idle", num_turns=1)

        result = self.runtime(handler).run(self.config())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(wake_prompts), 1)
        self.assertIn("/v1/items", wake_prompts[0])
        events = [json.loads(line) for line in Path(result["events_path"]).read_text().splitlines()]
        types = [event["type"] for event in events]
        self.assertIn("peer.idle", types)
        self.assertIn("peer.woken", types)
        consumed = [event for event in events if event["type"] == "message.consumed"]
        self.assertEqual(len(consumed), 1)

    def test_first_valid_concurrent_submit_wins_atomically(self) -> None:
        submit_barrier = threading.Barrier(2)

        def handler(session, prompt, registry, context):
            head = git(Path(session.spec.workspace_path), "rev-parse", "HEAD")
            submit_barrier.wait(timeout=1)
            registry.dispatch(
                ToolCall(
                    "PeerSubmit",
                    {"revision": head, "summary": session.spec.peer_name},
                ),
                context,
            )
            return PeerBoundaryResult(response_text="submitted", num_turns=1)

        result = self.runtime(handler).run(self.config())
        statuses = [submission["status"] for submission in result["submissions"]]
        self.assertEqual(statuses.count("accepted"), 1)
        self.assertEqual(statuses.count("already_submitted"), 1)
        self.assertIn(
            result["accepted_submission"]["peer_id"],
            {peer["peer_id"] for peer in result["participants"]},
        )

    def test_submit_stops_later_tool_dispatch(self) -> None:
        after_submit: list[dict[str, object]] = []

        def handler(session, prompt, registry, context):
            if session.spec.peer_name == "peer-1":
                head = git(Path(session.spec.workspace_path), "rev-parse", "HEAD")
                accepted = registry.dispatch(
                    ToolCall("PeerSubmit", {"revision": head, "summary": "done"}), context
                )
                blocked = registry.dispatch(ToolCall("PeerList", {}), context)
                after_submit.append(
                    {"accepted": accepted.output["status"], "blocked": blocked.is_error}
                )
            return PeerBoundaryResult(response_text="done", num_turns=1)

        result = self.runtime(handler).run(self.config())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(after_submit, [{"accepted": "accepted", "blocked": True}])

    def test_timeout_budget_and_peer_crash_leave_no_orphans(self) -> None:
        def idle_handler(session, prompt, registry, context):
            return PeerBoundaryResult(response_text="idle", num_turns=1)

        timeout = self.runtime(idle_handler).run(
            self.config(timeout_seconds=0.25, max_turns=5)
        )
        self.assertEqual(timeout["status"], "timed_out")
        self.assertEqual(timeout["orphan_threads"], [])

        def budget_handler(session, prompt, registry, context):
            return PeerBoundaryResult(
                response_text="spent",
                usage={"input_tokens": 3, "output_tokens": 2},
                num_turns=1,
            )

        budget = self.runtime(budget_handler).run(
            self.config(token_budget=5, timeout_seconds=1)
        )
        self.assertEqual(budget["status"], "budget_exhausted")
        self.assertGreaterEqual(budget["usage"]["total_tokens"], 5)
        self.assertEqual(budget["orphan_threads"], [])

        def crash_handler(session, prompt, registry, context):
            if session.spec.peer_name == "peer-1":
                raise RuntimeError("scripted crash")
            head = git(Path(session.spec.workspace_path), "rev-parse", "HEAD")
            registry.dispatch(
                ToolCall("PeerSubmit", {"revision": head, "summary": "survived"}), context
            )
            return PeerBoundaryResult(response_text="done", num_turns=1)

        crash = self.runtime(crash_handler).run(self.config())
        self.assertEqual(crash["status"], "completed")
        self.assertIn("failed", {peer["status"] for peer in crash["participants"]})
        self.assertEqual(crash["orphan_threads"], [])

    def test_worktree_peer_commit_can_be_integrated_and_submitted_by_other_peer(self) -> None:
        commit_ready = threading.Event()
        commit_holder: list[str] = []

        def handler(session, prompt, registry, context):
            workspace = Path(session.spec.workspace_path)
            if session.spec.peer_name == "peer-1":
                (workspace / "peer_one.py").write_text("VALUE = 1\n", encoding="utf-8")
                git(workspace, "add", "peer_one.py")
                git(workspace, "commit", "-qm", "peer one")
                commit_holder.append(git(workspace, "rev-parse", "HEAD"))
                commit_ready.set()
            else:
                self.assertTrue(commit_ready.wait(1))
                git(workspace, "cherry-pick", commit_holder[0])
                (workspace / "peer_two.py").write_text("VALUE = 2\n", encoding="utf-8")
                git(workspace, "add", "peer_two.py")
                git(workspace, "commit", "-qm", "peer two integrates peer one")
                head = git(workspace, "rev-parse", "HEAD")
                registry.dispatch(
                    ToolCall(
                        "PeerSubmit",
                        {"revision": head, "summary": "integrated both commits"},
                    ),
                    context,
                )
            return PeerBoundaryResult(response_text="done", num_turns=1)

        result = self.runtime(handler).run(
            self.config(workspace_mode="worktree", cleanup_worktrees=True)
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["retained_worktrees"], [])
        accepted = result["accepted_submission"]["revision"]
        names = git(self.repo, "show", "--format=", "--name-only", accepted)
        self.assertIn("peer_two.py", names)
        parent_names = git(self.repo, "show", "--format=", "--name-only", f"{accepted}^")
        self.assertIn("peer_one.py", parent_names)

    def test_external_cancel_wakes_idle_peers_and_stops_cleanly(self) -> None:
        idle = threading.Barrier(3)

        def handler(session, prompt, registry, context):
            idle.wait(timeout=2)
            return PeerBoundaryResult(response_text="available", num_turns=1)

        runtime = self.runtime(handler)
        holder: list[dict[str, object]] = []

        def execute() -> None:
            holder.append(runtime.run(self.config(timeout_seconds=5), run_id="cancel-run"))

        thread = threading.Thread(target=execute)
        thread.start()
        idle.wait(timeout=2)
        time.sleep(0.05)
        self.assertTrue(runtime.cancel("cancel-run", "user_cancelled"))
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(holder[0]["status"], "cancelled")
        self.assertEqual(holder[0]["orphan_threads"], [])

    def test_orphan_threads_take_precedence_over_accepted_submission_status(self) -> None:
        from src.peer.models import PeerRunRecord

        run = PeerRunRecord(
            run_id="run",
            mission="mission",
            repo_path=str(self.repo),
            base_revision=git(self.repo, "rev-parse", "HEAD"),
            peer_count=1,
            communication="solo",
            workspace_mode="shared",
            provider="scripted",
            model=None,
            timeout_seconds=1,
            max_turns=1,
            max_output_tokens=1,
            token_budget=None,
            turn_budget=None,
            output_dir=str(self.output),
            accepted_submission={"revision": git(self.repo, "rev-parse", "HEAD")},
        )
        self.assertEqual(
            PeerRuntime._terminal_status(run, "submitted", ["stuck-worker"]),
            "failed",
        )

    def test_default_control_state_is_git_excluded_from_shared_commits(self) -> None:
        def handler(session, prompt, registry, context):
            if session.spec.peer_name == "peer-1":
                workspace = Path(session.spec.workspace_path)
                (workspace / "solution.py").write_text("VALUE = 1\n", encoding="utf-8")
                git(workspace, "add", "-A")
                git(workspace, "commit", "-qm", "solution without control state")
                head = git(workspace, "rev-parse", "HEAD")
                registry.dispatch(
                    ToolCall("PeerSubmit", {"revision": head, "summary": "done"}), context
                )
            return PeerBoundaryResult(response_text="done", num_turns=1)

        config = self.config()
        config = PeerRunConfig(
            **{**config.to_dict(), "output_dir": None}
        )
        result = self.runtime(handler).run(config)
        names = git(
            self.repo,
            "show",
            "--format=",
            "--name-only",
            result["accepted_submission"]["revision"],
        ).splitlines()
        self.assertIn("solution.py", names)
        self.assertFalse(any(name.startswith(".clawd/") for name in names))


if __name__ == "__main__":
    unittest.main()
