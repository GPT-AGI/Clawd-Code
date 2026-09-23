from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.peer.backend import PeerBoundaryResult, ScriptedPeerBackend
from src.peer.models import PeerRunConfig
from src.peer.runtime import PeerRuntime
from src.tool_system.defaults import build_default_registry
from src.tool_system.protocol import ToolCall


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


class PeerProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.name", "Protocol Tests")
        git(self.repo, "config", "user.email", "protocol@example.invalid")
        (self.repo / "TASK.md").write_text("fixture\n", encoding="utf-8")
        git(self.repo, "add", "TASK.md")
        git(self.repo, "commit", "-qm", "initial")
        self.registry = build_default_registry(include_user_tools=False)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_all_conditions_keep_noncommunication_tools_and_budget_comparable(self) -> None:
        observed: dict[str, set[str]] = {}
        manifests: dict[str, dict] = {}

        for condition in ("solo", "independent", "artifact-only", "star", "p2p"):
            peers = 1 if condition == "solo" else 2

            def handler(session, prompt, registry, context, condition=condition):
                observed.setdefault(condition, {spec.name for spec in registry.list_specs()})
                if session.spec.peer_name == "peer-1":
                    head = git(Path(session.spec.workspace_path), "rev-parse", "HEAD")
                    registry.dispatch(
                        ToolCall(
                            "PeerSubmit",
                            {"revision": head, "summary": f"{condition} result"},
                        ),
                        context,
                    )
                return PeerBoundaryResult(response_text="done", num_turns=1)

            runtime = PeerRuntime(ScriptedPeerBackend(handler), self.registry)
            output = self.root / f"runs-{condition}"
            result = runtime.run(
                PeerRunConfig(
                    repo_path=str(self.repo),
                    mission="identical mission",
                    peers=peers,
                    communication=condition,
                    workspace_mode="shared",
                    provider="scripted",
                    model="same-model",
                    timeout_seconds=2,
                    max_turns=7,
                    token_budget=100,
                    output_dir=str(output),
                )
            )
            manifests[condition] = json.loads(Path(result["manifest_path"]).read_text())
            self.assertEqual(result["status"], "completed")

        for condition in ("independent", "artifact-only", "solo"):
            self.assertNotIn("SendMessage", observed[condition])
            self.assertNotIn("ReadMessages", observed[condition])
            self.assertNotIn("Broadcast", observed[condition])
        for condition in ("star", "p2p"):
            self.assertIn("SendMessage", observed[condition])
            self.assertIn("ReadMessages", observed[condition])
            self.assertIn("Broadcast", observed[condition])
        stripped = {
            condition: tools - {"SendMessage", "ReadMessages", "Broadcast"}
            for condition, tools in observed.items()
        }
        self.assertEqual(len({frozenset(value) for value in stripped.values()}), 1)
        self.assertTrue(all(item["config"]["max_turns"] == 7 for item in manifests.values()))
        self.assertTrue(all(item["config"]["token_budget"] == 100 for item in manifests.values()))
        self.assertTrue(all(item["config"]["model"] == "same-model" for item in manifests.values()))

    def test_peer_prompt_is_neutral_and_has_no_task_dag_or_fixed_professions(self) -> None:
        from src.peer.models import PeerParticipant

        participant = PeerParticipant(
            peer_id="run-p1",
            run_id="run",
            name="peer-1",
            session_id="session",
            workspace_mode="shared",
            workspace_path=str(self.repo),
        )
        prompt = PeerRuntime.peer_system_context(participant)
        lowered = prompt.casefold()
        for forbidden in ("planner", "coder", "reviewer", "owned task", "task dag"):
            self.assertNotIn(forbidden, lowered)
        self.assertIn("equal coding peer", lowered)
        self.assertIn("no participant has supervisory authority", lowered)
        self.assertNotIn("lead_agent_id", lowered)

    def test_peer_tool_search_indexes_peer_submit_not_legacy_team_tools(self) -> None:
        observed: list[dict] = []

        def handler(session, prompt, registry, context):
            if session.spec.peer_name == "peer-1":
                searched = registry.dispatch(
                    ToolCall("ToolSearch", {"query": "PeerSubmit"}), context
                )
                observed.append(searched.output)
                head = git(Path(session.spec.workspace_path), "rev-parse", "HEAD")
                registry.dispatch(
                    ToolCall("PeerSubmit", {"revision": head, "summary": "valid"}), context
                )
            return PeerBoundaryResult(response_text="done", num_turns=1)

        result = PeerRuntime(ScriptedPeerBackend(handler), self.registry).run(
            PeerRunConfig(
                repo_path=str(self.repo),
                mission="mission",
                peers=2,
                communication="p2p",
                workspace_mode="shared",
                provider="scripted",
                timeout_seconds=2,
                max_turns=3,
                output_dir=str(self.root / "search-runs"),
            )
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(observed[0]["matches"], ["PeerSubmit"])
        self.assertNotIn("TeamRun", observed[0]["matches"])

    def test_invalid_revision_is_rejected_then_another_peer_can_submit(self) -> None:
        attempts: list[str] = []

        def handler(session, prompt, registry, context):
            if session.spec.peer_name == "peer-1":
                bad = registry.dispatch(
                    ToolCall(
                        "PeerSubmit",
                        {"revision": "definitely-not-a-commit", "summary": "invalid"},
                    ),
                    context,
                )
                attempts.append(bad.output["status"])
            else:
                head = git(Path(session.spec.workspace_path), "rev-parse", "HEAD")
                good = registry.dispatch(
                    ToolCall("PeerSubmit", {"revision": head, "summary": "valid"}), context
                )
                attempts.append(good.output["status"])
            return PeerBoundaryResult(response_text="done", num_turns=1)

        result = PeerRuntime(ScriptedPeerBackend(handler), self.registry).run(
            PeerRunConfig(
                repo_path=str(self.repo),
                mission="mission",
                peers=2,
                communication="p2p",
                workspace_mode="shared",
                provider="scripted",
                timeout_seconds=2,
                max_turns=3,
                output_dir=str(self.root / "invalid-runs"),
            )
        )
        self.assertIn("rejected", attempts)
        self.assertIn("accepted", attempts)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            [item["status"] for item in result["submissions"]].count("rejected"), 1
        )

    def test_message_call_in_independent_condition_is_rejected_and_traced(self) -> None:
        rejection_outputs: list[dict] = []

        def handler(session, prompt, registry, context):
            if session.spec.peer_name == "peer-1":
                rejected = registry.dispatch(
                    ToolCall(
                        "SendMessage",
                        {"to": "peer-2", "message": "forbidden"},
                    ),
                    context,
                )
                rejection_outputs.append(rejected.output)
                head = git(Path(session.spec.workspace_path), "rev-parse", "HEAD")
                registry.dispatch(
                    ToolCall("PeerSubmit", {"revision": head, "summary": "done"}), context
                )
            return PeerBoundaryResult(response_text="done", num_turns=1)

        result = PeerRuntime(ScriptedPeerBackend(handler), self.registry).run(
            PeerRunConfig(
                repo_path=str(self.repo),
                mission="mission",
                peers=2,
                communication="independent",
                workspace_mode="shared",
                provider="scripted",
                timeout_seconds=2,
                max_turns=3,
                output_dir=str(self.root / "independent-rejection"),
            )
        )
        self.assertIn("unavailable under independent", rejection_outputs[0]["error"])
        events = [json.loads(line) for line in Path(result["events_path"]).read_text().splitlines()]
        self.assertEqual(
            [event["type"] for event in events].count("policy.rejected"), 1
        )


if __name__ == "__main__":
    unittest.main()
