from __future__ import annotations

import io
import json
import os
import socket
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.protocol import ToolCall
from src.tool_system.registry import ToolRegistry
from src.tool_system.tools import (
    AskUserQuestionTool,
    BashTool,
    BriefTool,
    ConfigTool,
    CronCreateTool,
    CronDeleteTool,
    CronListTool,
    FileEditTool,
    FileReadTool,
    FileWriteTool,
    GlobTool,
    GrepTool,
    LSPTool,
    MCPTool,
    ListMcpResourcesTool,
    ReadMcpResourceTool,
    SkillTool,
    SleepTool,
    TodoWriteTool,
    StructuredOutputTool,
    TaskStopTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskOutputTool,
    TaskUpdateTool,
    ToolSearchTool,
    WebFetchTool,
    WebSearchTool,
    TeamCreateTool,
    TeamDeleteTool,
    EnterWorktreeTool,
    ExitWorktreeTool,
    EnterPlanModeTool,
    ExitPlanModeTool,
)


class ToolSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()


class TestReadTool(ToolSystemTests):
    def test_read_returns_cat_n_format(self) -> None:
        p = self.root / "a.txt"
        p.write_text("line1\nline2\nline3\n", encoding="utf-8")
        tool = FileReadTool()
        out = tool.run({"file_path": str(p), "offset": 2, "limit": 2}, self.ctx).output
        self.assertEqual(out["type"], "text")
        self.assertEqual(out["file"]["content"], "2\tline2\n3\tline3")

    def test_read_allows_relative_path_under_workspace(self) -> None:
        p = self.root / "a.txt"
        p.write_text("x\n", encoding="utf-8")
        tool = FileReadTool()
        out = tool.run({"file_path": "a.txt", "limit": 10}, self.ctx).output
        self.assertEqual(out["type"], "text")
        self.assertIn("1\tx", out["file"]["content"])

    def test_read_returns_file_unchanged_stub(self) -> None:
        p = self.root / "same.txt"
        p.write_text("line\n", encoding="utf-8")
        tool = FileReadTool()
        first = tool.run({"file_path": str(p), "limit": 10}, self.ctx).output
        self.assertEqual(first["type"], "text")
        second = tool.run({"file_path": str(p)}, self.ctx).output
        self.assertEqual(second["type"], "file_unchanged")

    def test_read_notebook(self) -> None:
        p = self.root / "nb.ipynb"
        p.write_text('{"cells":[{"cell_type":"markdown","source":["hi"]}]}', encoding="utf-8")
        out = FileReadTool().run({"file_path": str(p)}, self.ctx).output
        self.assertEqual(out["type"], "notebook")
        self.assertEqual(len(out["file"]["cells"]), 1)

    def test_read_pdf(self) -> None:
        p = self.root / "x.pdf"
        p.write_bytes(b"%PDF-1.4\n1 0 obj\n")
        out = FileReadTool().run({"file_path": str(p)}, self.ctx).output
        self.assertEqual(out["type"], "pdf")

    def test_read_blocks_device_paths(self) -> None:
        with self.assertRaises(Exception):
            FileReadTool().run({"file_path": "/dev/zero"}, self.ctx)


class TestWriteTool(ToolSystemTests):
    def test_write_creates_file(self) -> None:
        tool = FileWriteTool()
        p = self.root / "b.txt"
        out = tool.run({"file_path": str(p), "content": "hello"}, self.ctx).output
        self.assertTrue(p.exists())
        self.assertEqual(out["type"], "create")
        self.assertEqual(out["filePath"], str(p))

    def test_write_requires_read_before_overwrite(self) -> None:
        p = self.root / "c.txt"
        p.write_text("old", encoding="utf-8")
        tool = FileWriteTool()
        with self.assertRaises(Exception):
            tool.run({"file_path": str(p), "content": "new"}, self.ctx)

        FileReadTool().run({"file_path": str(p), "limit": 10}, self.ctx)
        tool.run({"file_path": str(p), "content": "new"}, self.ctx)
        self.assertEqual(p.read_text(encoding="utf-8"), "new")

    def test_write_blocks_docs_by_default(self) -> None:
        """Writing .md files should require permission when allow_docs is False."""
        tool = FileWriteTool()
        p = self.root / "README.md"
        # Permission check should return 'ask' behavior
        result = tool.check_permissions({"file_path": str(p), "content": "x"}, self.ctx)
        self.assertEqual(result.behavior.value, "ask")
        # But run() itself should NOT raise - it just proceeds (permission is checked elsewhere)
        # Note: run() will still succeed because permission checking moved to check_permissions()


class TestEditTool(ToolSystemTests):
    def test_edit_requires_read(self) -> None:
        p = self.root / "d.txt"
        p.write_text("hello world", encoding="utf-8")
        tool = FileEditTool()
        with self.assertRaises(Exception):
            tool.run({"file_path": str(p), "old_string": "world", "new_string": "you"}, self.ctx)

    def test_edit_replaces_unique(self) -> None:
        p = self.root / "e.txt"
        p.write_text("hello world", encoding="utf-8")
        FileReadTool().run({"file_path": str(p), "limit": 10}, self.ctx)
        out = FileEditTool().run({"file_path": str(p), "old_string": "world", "new_string": "you"}, self.ctx).output
        self.assertEqual(out["filePath"], str(p))
        self.assertEqual(out["replaceAll"], False)
        self.assertEqual(p.read_text(encoding="utf-8"), "hello you")

    def test_edit_requires_replace_all_for_non_unique(self) -> None:
        p = self.root / "f.txt"
        p.write_text("a a a", encoding="utf-8")
        FileReadTool().run({"file_path": str(p), "limit": 10}, self.ctx)
        with self.assertRaises(Exception):
            FileEditTool().run({"file_path": str(p), "old_string": "a", "new_string": "b"}, self.ctx)
        FileEditTool().run({"file_path": str(p), "old_string": "a", "new_string": "b", "replace_all": True}, self.ctx)
        self.assertEqual(p.read_text(encoding="utf-8"), "b b b")


class TestGlobTool(ToolSystemTests):
    def test_glob_sorts_by_mtime(self) -> None:
        a = self.root / "x1.py"
        b = self.root / "x2.py"
        a.write_text("a", encoding="utf-8")
        time.sleep(0.01)
        b.write_text("b", encoding="utf-8")
        out = GlobTool().run({"pattern": "*.py", "path": str(self.root), "limit": 10}, self.ctx).output
        self.assertEqual(out["filenames"][0], str(b))
        self.assertEqual(out["filenames"][1], str(a))


class TestGrepTool(ToolSystemTests):
    def test_grep_files_with_matches(self) -> None:
        (self.root / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
        (self.root / "b.txt").write_text("nope\n", encoding="utf-8")
        out = GrepTool().run({"pattern": "hello", "path": str(self.root)}, self.ctx).output
        self.assertEqual(out["mode"], "files_with_matches")
        self.assertEqual(out["numFiles"], 1)
        self.assertIn("a.txt", out["filenames"][0])

    def test_grep_content_mode_with_line_numbers(self) -> None:
        (self.root / "a.txt").write_text("hello\nhello\n", encoding="utf-8")
        out = GrepTool().run({"pattern": "hello", "path": str(self.root), "output_mode": "content", "-n": True}, self.ctx).output
        self.assertIn(":1:", out["content"])


class TestBashTool(ToolSystemTests):
    def _enable_strict_team(self) -> None:
        self.ctx.team = {
            "protocol_version": 2,
            "settings": {
                "protocol_version": 2,
                "quality_gates": {"strict": True, "protocol_version": 2},
            },
        }

    def test_bash_echo(self) -> None:
        out = BashTool().run({"command": "echo hello"}, self.ctx).output
        self.assertEqual(out["exit_code"], 0)
        self.assertIn("hello", out["stdout"])

    def test_bash_blocks_sudo(self) -> None:
        with self.assertRaises(Exception):
            BashTool().run({"command": "sudo echo nope"}, self.ctx)

    def test_strict_team_blocks_recursive_source_deletion(self) -> None:
        self._enable_strict_team()

        with self.assertRaisesRegex(Exception, "preserves the best workspace"):
            BashTool().run(
                {"command": "cd /workspace && rm -rf src/package"}, self.ctx
            )

    def test_strict_team_allows_recursive_generated_cache_cleanup(self) -> None:
        self._enable_strict_team()
        cache = self.root / ".pytest_cache"
        cache.mkdir()
        (cache / "state").write_text("x", encoding="utf-8")

        result = BashTool().run(
            {"command": "rm -rf .pytest_cache", "cwd": str(self.root)}, self.ctx
        )

        self.assertEqual(result.output["exit_code"], 0)
        self.assertFalse(cache.exists())

    def test_strict_team_blocks_common_nested_destructive_forms(self) -> None:
        self._enable_strict_team()
        package = self.root / "pkg"
        package.mkdir()
        source = package / "module.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        commands = [
            "sh -c 'rm -rf pkg'",
            "find pkg -depth -delete",
            "git clean -fdx",
            (
                f"{sys.executable} -c \"import shutil; "
                "shutil.rmtree('pkg')\""
            ),
        ]

        for command in commands:
            with self.subTest(command=command):
                with self.assertRaisesRegex(
                    Exception, "preserves the best workspace"
                ):
                    BashTool().run(
                        {"command": command, "cwd": str(self.root)}, self.ctx
                    )
                self.assertEqual(
                    source.read_text(encoding="utf-8"), "VALUE = 1\n"
                )

    def test_strict_destructive_guard_ignores_source_text_and_heredoc_data(self) -> None:
        self._enable_strict_team()

        printf_result = BashTool().run(
            {
                "command": "printf '%s' 'rm -rf pkg; git clean -fdx' > note.txt",
                "cwd": str(self.root),
            },
            self.ctx,
        )
        heredoc_result = BashTool().run(
            {
                "command": (
                    "cat > cleanup.py <<'PY'\n"
                    "# example only\n"
                    "shutil.rmtree('pkg')\n"
                    "find pkg -delete\n"
                    "PY"
                ),
                "cwd": str(self.root),
            },
            self.ctx,
        )
        python_result = BashTool().run(
            {
                "command": (
                    f"{sys.executable} -c \"print("
                    "\\\"shutil.rmtree('pkg')\\\")\""
                ),
                "cwd": str(self.root),
            },
            self.ctx,
        )

        self.assertEqual(printf_result.output["exit_code"], 0)
        self.assertEqual(heredoc_result.output["exit_code"], 0)
        self.assertEqual(python_result.output["exit_code"], 0)
        self.assertIn("rm -rf pkg", (self.root / "note.txt").read_text())
        self.assertIn("find pkg -delete", (self.root / "cleanup.py").read_text())

    def test_strict_lead_cannot_mutate_clawd_with_file_or_bash_tools(self) -> None:
        self._enable_strict_team()
        control = self.root / ".clawd"
        control.mkdir()
        state = control / "state.json"
        state.write_text('{"status": "original"}\n', encoding="utf-8")
        FileReadTool().run({"file_path": str(state)}, self.ctx)

        with self.assertRaisesRegex(Exception, "protects .clawd control state"):
            FileWriteTool().run(
                {
                    "file_path": str(control / "forged.json"),
                    "content": "{}\n",
                },
                self.ctx,
            )
        with self.assertRaisesRegex(Exception, "protects .clawd control state"):
            FileEditTool().run(
                {
                    "file_path": str(state),
                    "old_string": "original",
                    "new_string": "forged",
                },
                self.ctx,
            )
        with self.assertRaisesRegex(Exception, "ownership violation"):
            BashTool().run(
                {
                    "command": "printf '{\\\"status\\\": \\\"forged\\\"}\\n' > .clawd/state.json",
                    "cwd": str(self.root),
                },
                self.ctx,
            )

        self.assertFalse((control / "forged.json").exists())
        self.assertEqual(
            state.read_text(encoding="utf-8"), '{"status": "original"}\n'
        )

    def test_bash_runs_a_trailing_command_after_cd(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()

        out = BashTool().run(
            {"command": f"cd {nested} && printf 'ran-here:%s' \"$PWD\""},
            self.ctx,
        ).output

        self.assertEqual(out["exit_code"], 0)
        self.assertEqual(out["stdout"], f"ran-here:{nested}")
        self.assertEqual(self.ctx.cwd, self.root)

    def test_bash_exposes_the_active_python_interpreter(self) -> None:
        out = BashTool().run(
            {
                "command": (
                    '"$CLAWD_PYTHON" -c "import sys; print(sys.executable)"'
                )
            },
            self.ctx,
        ).output

        self.assertEqual(out["exit_code"], 0)
        self.assertEqual(
            Path(out["stdout"].strip()).resolve(),
            Path(sys.executable).resolve(),
        )


class TestWebFetchTool(ToolSystemTests):
    def test_web_fetch_blocks_file_scheme(self) -> None:
        with self.assertRaises(Exception):
            WebFetchTool().run({"url": "file:///etc/passwd"}, self.ctx)

    def test_web_fetch_extracts_text(self) -> None:
        html_doc = "<html><body><h1>Title</h1><p>Hello <b>world</b></p></body></html>"

        class _Resp(io.BytesIO):
            headers = {"Content-Type": "text/html"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with patch.object(socket, "getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 0))]):
            with patch.object(urllib.request, "urlopen", return_value=_Resp(html_doc.encode("utf-8"))):
                out = WebFetchTool().run({"url": "https://example.com/"}, self.ctx).output
                self.assertIn("Title", out["content"])
                self.assertIn("Hello world", out["content"])


class TestWebSearchTool(ToolSystemTests):
    def test_web_search_parses_results(self) -> None:
        html_doc = """
        <a class="result__a" href="https://example.com/">Example</a>
        <a class="result__snippet">Snippet</a>
        """

        class _Resp(io.BytesIO):
            headers = {"Content-Type": "text/html"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with patch.object(urllib.request, "urlopen", return_value=_Resp(html_doc.encode("utf-8"))):
            out = WebSearchTool().run({"query": "example", "num": 1}, self.ctx).output
            self.assertEqual(len(out["results"]), 1)
            self.assertEqual(out["results"][0]["url"], "https://example.com/")


class TestSleepTool(ToolSystemTests):
    def test_sleep_short(self) -> None:
        start = time.time()
        SleepTool().run({"seconds": 0.01}, self.ctx)
        self.assertGreaterEqual(time.time() - start, 0.0)


class TestTaskStopTool(ToolSystemTests):
    def test_task_stop(self) -> None:
        def target(stop_event):
            while not stop_event.is_set():
                time.sleep(0.01)

        task = self.ctx.task_manager.start(name="loop", target=target)
        out = TaskStopTool().run({"task_id": task.task_id}, self.ctx).output
        self.assertTrue(out["stopped"])


class TestConfigTool(ToolSystemTests):
    def test_config_get_set_roundtrip(self) -> None:
        from src import config as config_mod

        cfg_path = self.root / "config.json"
        cfg_path.write_text(json.dumps(config_mod.get_default_config()), encoding="utf-8")
        with patch("src.config.get_config_path", return_value=cfg_path):
            get_out = ConfigTool().run({"setting": "default_provider"}, self.ctx).output
            self.assertEqual(get_out["operation"], "get")
            set_out = ConfigTool().run({"setting": "default_provider", "value": "openai"}, self.ctx).output
            self.assertEqual(set_out["operation"], "set")
            self.assertEqual(ConfigTool().run({"setting": "default_provider"}, self.ctx).output["value"], "openai")


class TestMCPTool(ToolSystemTests):
    def test_mcp_calls_client(self) -> None:
        class Client:
            def call_tool(self, tool_name: str, args: dict) -> Any:
                return {"tool": tool_name, "args": args}

            def list_tools(self) -> list[str]:
                return ["x"]

        self.ctx.mcp_clients["srv"] = Client()
        out = MCPTool().run({"server": "srv", "tool": "x", "input": {"a": 1}}, self.ctx).output
        self.assertEqual(out["output"]["args"]["a"], 1)


class TestLSPTool(ToolSystemTests):
    def test_lsp_requires_client(self) -> None:
        out = LSPTool().run({"method": "initialize", "params": {}}, self.ctx)
        self.assertTrue(out.is_error)

    def test_lsp_calls_client(self) -> None:
        class Client:
            def request(self, method: str, params=None) -> Any:
                return {"method": method, "params": params}

        self.ctx.lsp_client = Client()
        out = LSPTool().run({"method": "hover", "params": {"x": 1}}, self.ctx).output
        self.assertEqual(out["response"]["params"]["x"], 1)


class TestSkillTool(ToolSystemTests):
    def test_skill_runs_markdown_skill(self) -> None:
        from src.skills.create import create_skill

        skills_dir = self.root / "skills"
        create_skill(
            directory=skills_dir,
            name="hello",
            description="say hello",
            body="Hello $ARGUMENTS[0]!",
            arguments=["name"],
        )
        with patch.dict(os.environ, {"CLAWD_SKILLS_DIR": str(skills_dir)}):
            out = SkillTool().run({"skill": "hello", "args": "bob"}, self.ctx).output
            self.assertTrue(out["success"])
            self.assertIn("Hello bob!", out["prompt"])
            self.assertEqual(out["loadedFrom"], "user")

    def test_skill_runs_legacy_python_skill(self) -> None:
        skills_dir = self.root / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "legacy.py").write_text(
            "def run(input, context):\n    return 'hi ' + input.get('name','world')\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"CLAWD_SKILLS_DIR": str(skills_dir)}):
            out = SkillTool().run({"name": "legacy", "input": {"name": "bob"}}, self.ctx).output
            self.assertEqual(out["output"], "hi bob")


class TestNewParityTools(ToolSystemTests):
    def test_ask_user_question_uses_handler(self) -> None:
        self.ctx.ask_user = lambda questions: {questions[0]["question"]: "Option A"}
        out = AskUserQuestionTool().run(
            {
                "questions": [
                    {
                        "question": "Choose?",
                        "header": "Choice",
                        "options": [
                            {"label": "Option A", "description": "A"},
                            {"label": "Option B", "description": "B"},
                        ],
                    }
                ]
            },
            self.ctx,
        ).output
        self.assertEqual(out["answers"]["Choose?"], "Option A")

    def test_todo_write(self) -> None:
        out = TodoWriteTool().run(
            {"todos": [{"content": "x", "status": "pending", "activeForm": "Doing x"}]},
            self.ctx,
        ).output
        self.assertEqual(out["newTodos"][0]["content"], "x")

    def test_task_tools_roundtrip(self) -> None:
        created = TaskCreateTool().run({"subject": "T1", "description": "D1"}, self.ctx).output
        task_id = created["task"]["id"]
        listed = TaskListTool().run({}, self.ctx).output
        self.assertEqual(len(listed["tasks"]), 1)
        TaskUpdateTool().run({"taskId": task_id, "status": "completed"}, self.ctx)
        got = TaskGetTool().run({"taskId": task_id}, self.ctx).output
        self.assertEqual(got["task"]["status"], "completed")
        task_out = TaskOutputTool().run({"task_id": task_id}, self.ctx).output
        self.assertEqual(task_out["task"]["task_id"], task_id)

    def test_tool_search(self) -> None:
        reg = build_default_registry(include_user_tools=False)
        out = ToolSearchTool(reg).run({"query": "read"}, self.ctx).output
        self.assertIn("Read", out["matches"])

    def test_tool_search_matches_natural_language_and_returns_schema(self) -> None:
        reg = build_default_registry(include_user_tools=False)
        out = ToolSearchTool(reg).run({"query": "read file content local"}, self.ctx).output
        self.assertEqual(out["matches"][0], "Read")
        read = next(tool for tool in out["tools"] if tool["name"] == "Read")
        self.assertIn("file_path", read["input_schema"]["properties"])

    def test_tool_search_supports_wildcard_and_synonyms(self) -> None:
        reg = build_default_registry(include_user_tools=False)
        search = ToolSearchTool(reg)
        all_tools = search.run({"query": "*", "max_results": 50}, self.ctx).output
        self.assertIn("Read", all_tools["matches"])
        self.assertIn("Bash", all_tools["matches"])
        self.assertGreaterEqual(all_tools["total_matches"], len(all_tools["matches"]))

        shell = search.run({"query": "execute shell command"}, self.ctx).output
        self.assertEqual(shell["matches"][0], "Bash")

    def test_cron_tools_roundtrip(self) -> None:
        created = CronCreateTool().run({"cron": "*/5 * * * *", "prompt": "ping"}, self.ctx).output
        cron_id = created["id"]
        listed = CronListTool().run({}, self.ctx).output
        self.assertEqual(len(listed["jobs"]), 1)
        deleted = CronDeleteTool().run({"id": cron_id}, self.ctx).output
        self.assertTrue(deleted["success"])

    def test_structured_output(self) -> None:
        out = StructuredOutputTool().run({"ok": True}, self.ctx).output
        self.assertTrue(out["structured_output"]["ok"])

    def test_mcp_resource_tools(self) -> None:
        class Client:
            def list_resources(self):
                return [{"uri": "x://1", "name": "r1", "mimeType": "text/plain"}]

            def read_resource(self, uri: str):
                return {"contents": [{"uri": uri, "text": "hello"}]}

        self.ctx.mcp_clients["srv"] = Client()
        listed = ListMcpResourcesTool().run({"server": "srv"}, self.ctx).output
        self.assertEqual(listed[0]["uri"], "x://1")
        read = ReadMcpResourceTool().run({"server": "srv", "uri": "x://1"}, self.ctx).output
        self.assertEqual(read["contents"][0]["text"], "hello")


class TestRegistryAndHelloWorldTool(ToolSystemTests):
    def test_can_load_user_tool_hello_world(self) -> None:
        user_dir = self.root / "tools"
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "hello.py").write_text(
            "tool_spec = {\n"
            "  'name': 'HelloWorld',\n"
            "  'description': 'hello world tool',\n"
            "  'input_schema': { 'type': 'object', 'properties': { 'name': { 'type': 'string' } } },\n"
            "}\n"
            "def run(tool_input, context):\n"
            "  return { 'message': 'hello ' + tool_input.get('name','world') }\n",
            encoding="utf-8",
        )

        from src.tool_system.loader import load_tools_from_dir

        tools = load_tools_from_dir(user_dir)
        self.assertEqual(len(tools), 1)
        reg = ToolRegistry(tools=tools)
        result = reg.dispatch(ToolCall(name="HelloWorld", input={"name": "alice"}), self.ctx)
        self.assertEqual(result.output["message"], "hello alice")


class TestBriefAndAgentTools(ToolSystemTests):
    def test_brief_tool(self) -> None:
        out = BriefTool().run({"text": "abc", "max_chars": 2}, self.ctx).output
        self.assertEqual(out["preview"], "ab…")

    def test_agent_tool_sequences_calls(self) -> None:
        reg = build_default_registry(include_user_tools=False)
        ctx = ToolContext(workspace_root=self.root)
        p = self.root / "x.txt"
        p.write_text("hi", encoding="utf-8")
        call = {"name": "Read", "input": {"file_path": str(p), "limit": 10}}
        out = reg.get("Agent").run({"calls": [call]}, ctx).output  # type: ignore[union-attr]
        self.assertEqual(out["results"][0]["name"], "Read")


class TestTeamTools(ToolSystemTests):
    def test_team_create_roundtrip(self) -> None:
        """Test creating and deleting a team."""
        # Create team
        create_out = TeamCreateTool().run(
            {"team_name": "test-team", "description": "A test team"},
            self.ctx,
        ).output
        self.assertEqual(create_out["team_name"], "test-team")
        self.assertIsNotNone(create_out["lead_agent_id"])
        self.assertEqual(self.ctx.team["team_name"], "test-team")

        # Verify team file was created
        team_file = self.root / ".clawd" / "team.json"
        self.assertTrue(team_file.exists())

        # Delete team
        delete_out = TeamDeleteTool().run({}, self.ctx).output
        self.assertTrue(delete_out["success"])
        self.assertEqual(delete_out["team_name"], "test-team")
        self.assertIsNone(self.ctx.team)

        # Verify team file was deleted
        self.assertFalse(team_file.exists())

    def test_team_delete_no_team(self) -> None:
        """Test deleting when no team exists."""
        out = TeamDeleteTool().run({}, self.ctx).output
        self.assertFalse(out["success"])
        self.assertEqual(out["message"], "No active team")

    def test_team_create_requires_name(self) -> None:
        """Test team name validation."""
        from src.tool_system.errors import ToolInputError

        with self.assertRaises(ToolInputError):
            TeamCreateTool().run({"team_name": ""}, self.ctx)


class TestWorktreeTools(ToolSystemTests):
    def test_worktree_roundtrip(self) -> None:
        """Test entering and exiting a worktree."""
        # Enter worktree
        enter_out = EnterWorktreeTool().run({"name": "test-tree"}, self.ctx).output
        self.assertIn("test-tree", enter_out["worktreePath"])
        self.assertIsNotNone(self.ctx.worktree_root)
        self.assertEqual(self.ctx.cwd, self.ctx.worktree_root)

        # Verify worktree directory exists
        worktree_dir = self.root / ".clawd" / "worktrees" / "test-tree"
        self.assertTrue(worktree_dir.exists())

        # Exit worktree
        exit_out = ExitWorktreeTool().run({}, self.ctx).output
        self.assertIn("Exited worktree", exit_out["message"])
        self.assertIsNone(self.ctx.worktree_root)
        self.assertEqual(self.ctx.cwd, self.root)

    def test_worktree_enter_already_in(self) -> None:
        """Test entering worktree when already in one."""
        from src.tool_system.errors import ToolPermissionError

        EnterWorktreeTool().run({"name": "first"}, self.ctx)
        with self.assertRaises(ToolPermissionError):
            EnterWorktreeTool().run({"name": "second"}, self.ctx)

    def test_worktree_exit_not_in(self) -> None:
        """Test exiting worktree when not in one."""
        from src.tool_system.errors import ToolPermissionError

        with self.assertRaises(ToolPermissionError):
            ExitWorktreeTool().run({}, self.ctx)

    def test_worktree_name_validation(self) -> None:
        """Test worktree name validation."""
        from src.tool_system.errors import ToolInputError

        # Invalid empty name
        with self.assertRaises(ToolInputError):
            EnterWorktreeTool().run({"name": ""}, self.ctx)

        # Invalid characters
        with self.assertRaises(ToolInputError):
            EnterWorktreeTool().run({"name": "invalid name!"}, self.ctx)

        # Too long
        with self.assertRaises(ToolInputError):
            EnterWorktreeTool().run({"name": "a" * 65}, self.ctx)


class TestPlanModeTools(ToolSystemTests):
    def test_plan_mode_roundtrip(self) -> None:
        """Test entering and exiting plan mode."""
        # Enter plan mode
        enter_out = EnterPlanModeTool().run({}, self.ctx).output
        self.assertTrue(self.ctx.plan_mode)
        self.assertIn("Entered plan mode", enter_out["message"])

        # Exit plan mode
        exit_out = ExitPlanModeTool().run({}, self.ctx).output
        self.assertFalse(self.ctx.plan_mode)
        self.assertFalse(exit_out["isAgent"])
        self.assertTrue(exit_out["hasTaskTool"])

    def test_plan_mode_exit_with_plan(self) -> None:
        """Test exiting plan mode with a plan."""
        EnterPlanModeTool().run({}, self.ctx)

        plan_content = "# My Plan\n\n- Do something\n- Do something else"
        exit_out = ExitPlanModeTool().run({"plan": plan_content}, self.ctx).output

        self.assertEqual(exit_out["plan"], plan_content)
        self.assertIsNotNone(exit_out["filePath"])

        # Verify plan file was created
        plan_file = self.root / ".clawd" / "plan.md"
        self.assertTrue(plan_file.exists())
        self.assertEqual(plan_file.read_text(encoding="utf-8"), plan_content)

    def test_plan_mode_exit_with_custom_path(self) -> None:
        """Test exiting plan mode with custom plan file path."""
        EnterPlanModeTool().run({}, self.ctx)

        custom_path = self.root / "my-plan.md"
        plan_content = "# Custom Plan"
        exit_out = ExitPlanModeTool().run(
            {"plan": plan_content, "planFilePath": str(custom_path)},
            self.ctx,
        ).output

        self.assertEqual(exit_out["filePath"], str(custom_path))
        self.assertTrue(custom_path.exists())

    def test_plan_mode_exit_not_in_mode(self) -> None:
        """Test exiting plan mode when not in it."""
        from src.tool_system.errors import ToolPermissionError

        with self.assertRaises(ToolPermissionError):
            ExitPlanModeTool().run({}, self.ctx)

    def test_plan_mode_plan_validation(self) -> None:
        """Test plan input validation."""
        from src.tool_system.errors import ToolInputError

        EnterPlanModeTool().run({}, self.ctx)

        # Plan must be string
        with self.assertRaises(ToolInputError):
            ExitPlanModeTool().run({"plan": 123}, self.ctx)

        # Plan file path must be string
        with self.assertRaises(ToolInputError):
            ExitPlanModeTool().run({"plan": "x", "planFilePath": 123}, self.ctx)


if __name__ == "__main__":
    unittest.main()
