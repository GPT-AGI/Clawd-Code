from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.providers.base import ChatResponse
from src.teammate.runtime import TeammateRuntime
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.tools import (
    TaskCreateTool,
    TeamConfigureTool,
    TeamCreateTool,
    TeamRunTool,
    TeamVerifyTool,
    TeammateCreateTool,
)


class FinalProvider:
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        return ChatResponse(
            content="implemented and checked",
            model=self.model,
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="stop",
            tool_uses=None,
        )


class TestTeamQualityGates(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.registry = build_default_registry(include_user_tools=False)
        self.provider = FinalProvider()
        self.context = ToolContext(workspace_root=self.root)
        self.context.teammate_runtime = TeammateRuntime(self.provider, self.registry)
        TeamCreateTool().run(
            {"team_name": "strict", "quality_gates": True}, self.context
        )
        TeamConfigureTool().run(
            {
                "architecture_contract": "samplepkg owns the public VALUE contract",
                "install_command": (
                    "python -m pip install -e . --no-deps --no-build-isolation"
                ),
                "import_command": "python -c \"import samplepkg\"",
                "integration_command": (
                    "python -c \"import samplepkg; assert samplepkg.VALUE == 1\""
                ),
            },
            self.context,
        )
        for name in ("one", "two"):
            TeammateCreateTool().run(
                {
                    "name": name,
                    "role": "implementation",
                    "instructions": f"Implement the {name} partition.",
                    "tools": ["Read", "Write", "Bash"],
                },
                self.context,
            )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _task(
        self,
        key: str,
        owner: str,
        path: str,
        *,
        blocked_by: list[str] | None = None,
        provides: list[str] | None = None,
        depends: list[str] | None = None,
    ) -> str:
        payload = {
            "key": key,
            "subject": key,
            "description": f"Implement {key}",
            "owner": owner,
            "ownedFiles": [path],
            "acceptanceChecks": [f"python -m py_compile {path}"],
            "providesInterfaces": provides or [],
            "dependsOnInterfaces": depends or [],
        }
        if blocked_by:
            payload["blockedBy"] = blocked_by
        return TaskCreateTool().run(payload, self.context).output["task"]["id"]

    def _write_package(self) -> None:
        (self.root / "samplepkg").mkdir()
        (self.root / "samplepkg" / "__init__.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        (self.root / "helper.py").write_text("HELPER = True\n", encoding="utf-8")
        (self.root / "pyproject.toml").write_text(
            "[build-system]\n"
            "requires = ['setuptools']\n"
            "build-backend = 'setuptools.build_meta'\n"
            "[project]\n"
            "name = 'strict-team-fixture'\n"
            "version = '0.0.1'\n",
            encoding="utf-8",
        )

    def test_rejects_single_worker_and_overlapping_ownership_before_model_calls(self) -> None:
        self._task("first", "one", "samplepkg")
        self._task("second", "two", "samplepkg/__init__.py")

        result = TeamRunTool().run({"max_workers": 2}, self.context)

        self.assertTrue(result.is_error)
        self.assertEqual(result.output["status"], "blocked")
        self.assertIn("ownedFiles overlap", result.output["error"])
        self.assertEqual(self.provider.calls, 0)

    def test_completed_tasks_require_clean_validation_before_team_completion(self) -> None:
        self._write_package()
        self._task("package", "one", "samplepkg")
        self._task("helper", "two", "helper.py")

        rollout = TeamRunTool().run({"max_workers": 2}, self.context)

        self.assertFalse(rollout.is_error)
        self.assertEqual(rollout.output["status"], "verification_required")
        self.assertEqual(self.context.team_store.load_active_team().status, "running")

        verified = TeamVerifyTool().run({"timeout_s": 120}, self.context)

        self.assertFalse(verified.is_error, verified.output)
        self.assertEqual(verified.output["status"], "completed")
        self.assertEqual(verified.output["validation"]["status"], "passed")
        self.assertEqual(
            [stage["stage"] for stage in verified.output["validation"]["stages"]],
            ["bootstrap", "install", "import", "integration"],
        )

    def test_interface_dependency_requires_peer_coordination(self) -> None:
        self._task("provider", "one", "provider.py", provides=["public-api"])
        self._task("independent", "two", "independent.py")
        self._task(
            "consumer",
            "two",
            "consumer.py",
            blocked_by=["provider"],
            depends=["public-api"],
        )

        result = TeamRunTool().run({"max_workers": 2}, self.context)

        self.assertTrue(result.is_error)
        self.assertEqual(result.output["status"], "blocked")
        self.assertIn("peer message", result.output["error"])


if __name__ == "__main__":
    unittest.main()
