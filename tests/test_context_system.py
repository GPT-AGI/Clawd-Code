from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.agent.conversation import Conversation
from src.context_system import build_context_prompt
from src.context_system.git_context import collect_git_context
from src.providers.base import ChatResponse
from src.tool_system.agent_loop import _build_effective_system_prompt, run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry


class TestContextSystem(unittest.TestCase):
    def test_build_context_prompt_includes_workspace_and_claude_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CLAUDE.md").write_text("Project rule: always add tests.", encoding="utf-8")
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_app.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

            prompt = build_context_prompt(root)

            self.assertIn("## Runtime Context", prompt)
            self.assertIn("## Project Instructions", prompt)
            self.assertIn("Project rule: always add tests.", prompt)
            self.assertIn("README.md", prompt)
            self.assertIn("pyproject.toml", prompt)

    def test_collect_git_context_handles_non_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = collect_git_context(tmp)
            self.assertFalse(ctx.available)

    def test_agent_loop_injects_context_prompt_for_non_anthropic(self) -> None:
        registry = build_default_registry(include_user_tools=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CLAUDE.md").write_text("Follow the CLAUDE instructions.", encoding="utf-8")
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")

            ctx = ToolContext(workspace_root=root)

            conversation = Conversation()
            conversation.add_user_message("hello")

            provider = MagicMock()
            provider.chat.return_value = ChatResponse(
                content="ok",
                model="test",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="stop",
                tool_uses=None,
            )

            out = run_agent_loop(conversation, provider, registry, ctx, verbose=False)
            self.assertEqual(out.response_text, "ok")
            system_message = provider.chat.call_args.args[0][0]
            self.assertEqual(system_message["role"], "system")
            self.assertIn("## Runtime Context", system_message["content"])
            self.assertIn("## Local Engineering Tools", system_message["content"])
            self.assertIn("## Adaptive Team Orchestration", system_message["content"])
            self.assertIn("valid to complete the task without creating a team", system_message["content"])
            self.assertIn("Never send local paths", system_message["content"])
            self.assertIn("## Project Instructions", system_message["content"])
            self.assertIn("Follow the CLAUDE instructions.", system_message["content"])

    def test_teammate_prompt_does_not_receive_leader_orchestration_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = ToolContext(workspace_root=Path(tmp), actor_id="worker-1")

            prompt = _build_effective_system_prompt("style", context)

        self.assertNotIn("## Adaptive Team Orchestration", prompt)


if __name__ == "__main__":
    unittest.main()
