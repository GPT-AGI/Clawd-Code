"""Regression tests for Windows bash discovery and workspace security.

All platform/subprocess interactions are mocked. These tests run correctly on
any OS without requiring bash, WSL, Git Bash, cmd.exe, or PowerShell.

Covered behaviours
------------------
 1. Bash discovered via shutil.which on Windows.
 2. Bash missing on Windows raises ToolExecutionError (not FileNotFoundError).
 3. subprocess FileNotFoundError at launch is translated to ToolExecutionError.
 4. Non-zero process exit preserves normal ToolResult error semantics (not an
    executable-not-found error).
 5. POSIX absolute cwd (/home/project) recognised.
 6. Windows backslash absolute cwd (D:\\project) recognised.
 7. Windows forward-slash absolute cwd (D:/project) recognised.
 8. Relative cwd is rejected with ToolInputError.
 9. Drive-letter-only cwd (D:, C:file) is rejected.
10. Path inside workspace is allowed by ensure_allowed_path.
11. Sibling path is rejected by ensure_allowed_path.
12. Traversal outside workspace is rejected.
13. POSIX shell discovery returns ["bash", "-lc"] unchanged.
14. BashTool dispatches using _find_shell result, not a hardcoded list.
15. SendUserMessage required message/status behaviour is preserved.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.tool_system.context import ToolContext
from src.tool_system.errors import ToolExecutionError, ToolInputError, ToolPermissionError
from src.tool_system.protocol import ToolCall
from src.tool_system.tools.bash import (
    BashTool,
    _find_shell,
    _is_absolute_path,
)


# ---------------------------------------------------------------------------
# 1-2. Windows bash discovery via shutil.which
# ---------------------------------------------------------------------------

class TestFindShellWindows(unittest.TestCase):
    """_find_shell() on Windows uses shutil.which('bash') only."""

    def _windows_patch(self):
        return patch("src.tool_system.tools.bash.platform.system", return_value="Windows")

    def test_bash_discovered_via_which(self):
        # When shutil.which finds bash, _find_shell returns it with -lc.
        fake_bash = r"C:\tools\git\bin\bash.exe"
        with self._windows_patch(), \
             patch("src.tool_system.tools.bash.shutil.which",
                   side_effect=lambda name: fake_bash if name == "bash" else None):
            result = _find_shell()
        self.assertEqual(result, [fake_bash, "-lc"])

    def test_bash_missing_raises_tool_execution_error(self):
        # When shutil.which returns None, ToolExecutionError is raised.
        # The message must be human-readable, not contain '[WinError 2]'.
        with self._windows_patch(), \
             patch("src.tool_system.tools.bash.shutil.which", return_value=None):
            with self.assertRaises(ToolExecutionError) as cm:
                _find_shell()
        msg = str(cm.exception)
        self.assertIn("Bash", msg)
        self.assertNotIn("WinError", msg)

    def test_no_cmd_fallback(self):
        # cmd.exe must never be returned as a fallback.
        with self._windows_patch(), \
             patch("src.tool_system.tools.bash.shutil.which", return_value=None):
            with self.assertRaises(ToolExecutionError):
                _find_shell()
            # If no exception were raised we would check the result here;
            # since ToolExecutionError is always raised when bash is absent,
            # cmd.exe cannot be in the result.


# ---------------------------------------------------------------------------
# 3. subprocess FileNotFoundError translates to ToolExecutionError
# ---------------------------------------------------------------------------

class TestSubprocessFileNotFoundError(unittest.TestCase):
    """FileNotFoundError from subprocess.run must become ToolExecutionError."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_file_not_found_becomes_tool_execution_error(self):
        fake_shell = ["/usr/local/bin/bash", "-lc"]
        with patch("src.tool_system.tools.bash._find_shell", return_value=fake_shell), \
             patch("src.tool_system.tools.bash.subprocess.run",
                   side_effect=FileNotFoundError(2, "No such file or directory")):
            with self.assertRaises(ToolExecutionError) as cm:
                BashTool().run({"command": "echo hi"}, self.ctx)
        msg = str(cm.exception)
        # Must be actionable, not a raw WinError
        self.assertIn("not found", msg.lower())
        self.assertNotIn("WinError", msg)


# ---------------------------------------------------------------------------
# 4. Non-zero exit preserves normal ToolResult error semantics
# ---------------------------------------------------------------------------

class TestNonZeroExitSemantics(unittest.TestCase):
    """A non-zero exit code must produce is_error=True ToolResult, not an exception."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_nonzero_exit_returns_tool_result_with_is_error(self):
        fake_shell = ["/usr/bin/bash", "-lc"]
        mock_proc = MagicMock()
        mock_proc.stdout = ""
        mock_proc.stderr = "command not found"
        mock_proc.returncode = 127

        with patch("src.tool_system.tools.bash._find_shell", return_value=fake_shell), \
             patch("src.tool_system.tools.bash.subprocess.run", return_value=mock_proc):
            result = BashTool().run({"command": "notacommand"}, self.ctx)

        # Must NOT raise; must return a ToolResult with is_error=True
        self.assertTrue(result.is_error)
        self.assertEqual(result.output["exit_code"], 127)

    def test_zero_exit_returns_tool_result_without_error(self):
        fake_shell = ["/usr/bin/bash", "-lc"]
        mock_proc = MagicMock()
        mock_proc.stdout = "hello\n"
        mock_proc.stderr = ""
        mock_proc.returncode = 0

        with patch("src.tool_system.tools.bash._find_shell", return_value=fake_shell), \
             patch("src.tool_system.tools.bash.subprocess.run", return_value=mock_proc):
            result = BashTool().run({"command": "echo hello"}, self.ctx)

        self.assertFalse(result.is_error)
        self.assertEqual(result.output["exit_code"], 0)


# ---------------------------------------------------------------------------
# 5-9. _is_absolute_path validator
# ---------------------------------------------------------------------------

class TestIsAbsolutePath(unittest.TestCase):
    """_is_absolute_path uses PureWindowsPath/PurePosixPath for correctness."""

    # --- accepted ---

    def test_posix_root_accepted(self):
        self.assertTrue(_is_absolute_path("/"))

    def test_posix_path_accepted(self):
        self.assertTrue(_is_absolute_path("/home/user/project"))

    def test_windows_backslash_accepted(self):
        self.assertTrue(_is_absolute_path(r"D:\project"))
        self.assertTrue(_is_absolute_path(r"C:\Users\dev\project"))

    def test_windows_forward_slash_accepted(self):
        self.assertTrue(_is_absolute_path("D:/project"))
        self.assertTrue(_is_absolute_path("C:/foo/bar"))

    # --- rejected ---

    def test_empty_string_rejected(self):
        self.assertFalse(_is_absolute_path(""))

    def test_relative_path_rejected(self):
        self.assertFalse(_is_absolute_path("relative/path"))
        self.assertFalse(_is_absolute_path("file.txt"))

    def test_drive_letter_only_rejected(self):
        # "D:" alone is drive-relative on Windows, not absolute.
        self.assertFalse(_is_absolute_path("D:"))

    def test_drive_letter_with_filename_rejected(self):
        # "C:file" is drive-relative, not absolute.
        self.assertFalse(_is_absolute_path("C:file"))


# ---------------------------------------------------------------------------
# 8. BashTool raises ToolInputError for relative cwd
# ---------------------------------------------------------------------------

class TestRelativeCwdRejected(unittest.TestCase):
    """BashTool.run must raise ToolInputError when cwd is not absolute."""

    def test_relative_cwd_raises_input_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = ToolContext(workspace_root=Path(tmp))
            with self.assertRaises(ToolInputError) as cm:
                BashTool().run({"command": "echo hi", "cwd": "relative/dir"}, ctx)
        self.assertIn("absolute", str(cm.exception).lower())


# ---------------------------------------------------------------------------
# 10-12. Workspace security (ensure_allowed_path)
# ---------------------------------------------------------------------------

class TestWorkspaceSecurity(unittest.TestCase):
    """The allow-list sandbox must be enforced by ensure_allowed_path."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_path_inside_workspace_allowed(self):
        inner = self.root / "subdir" / "file.txt"
        result = self.ctx.ensure_allowed_path(inner)
        self.assertEqual(result.resolve(), inner.resolve())

    def test_sibling_path_rejected(self):
        sibling = self.root.parent / "sibling-workspace"
        with self.assertRaises(ToolPermissionError) as cm:
            self.ctx.ensure_allowed_path(sibling)
        self.assertIn("outside allowed working directories", str(cm.exception))

    def test_traversal_outside_workspace_rejected(self):
        # e.g. /tmp/workspace/../../etc/passwd resolves outside the root
        traversal = self.root / ".." / ".." / "etc" / "passwd"
        with self.assertRaises(ToolPermissionError):
            self.ctx.ensure_allowed_path(traversal)


# ---------------------------------------------------------------------------
# 13. POSIX shell discovery returns ["bash", "-lc"]
# ---------------------------------------------------------------------------

class TestFindShellPosix(unittest.TestCase):
    """_find_shell on non-Windows must return ["bash", "-lc"] unchanged."""

    def test_linux_returns_bash_lc(self):
        with patch("src.tool_system.tools.bash.platform.system", return_value="Linux"):
            result = _find_shell()
        self.assertEqual(result, ["bash", "-lc"])

    def test_darwin_returns_bash_lc(self):
        with patch("src.tool_system.tools.bash.platform.system", return_value="Darwin"):
            result = _find_shell()
        self.assertEqual(result, ["bash", "-lc"])


# ---------------------------------------------------------------------------
# 14. BashTool dispatches with _find_shell result
# ---------------------------------------------------------------------------

class TestBashToolDispatch(unittest.TestCase):
    """BashTool.run must use the argv prefix from _find_shell, not hardcoded list."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_uses_find_shell_result(self):
        fake_shell = ["/usr/local/bin/bash", "-lc"]
        mock_proc = MagicMock()
        mock_proc.stdout = "hello\n"
        mock_proc.stderr = ""
        mock_proc.returncode = 0

        with patch("src.tool_system.tools.bash._find_shell", return_value=fake_shell), \
             patch("src.tool_system.tools.bash.subprocess.run",
                   return_value=mock_proc) as mock_run:
            BashTool().run({"command": "echo hello"}, self.ctx)

        called_argv = mock_run.call_args[0][0]
        self.assertEqual(called_argv, ["/usr/local/bin/bash", "-lc", "echo hello"])

    def test_find_shell_error_propagates(self):
        with patch("src.tool_system.tools.bash._find_shell",
                   side_effect=ToolExecutionError("no bash found")):
            with self.assertRaises(ToolExecutionError) as cm:
                BashTool().run({"command": "echo hello"}, self.ctx)
        self.assertIn("no bash found", str(cm.exception))


# ---------------------------------------------------------------------------
# 15. SendUserMessage required fields preserved
# ---------------------------------------------------------------------------

class TestSendUserMessageRegression(unittest.TestCase):
    """SendUserMessage dispatch must be unaffected by changes to bash.py."""

    def setUp(self):
        from src.tool_system.defaults import build_default_registry
        self.tmp = tempfile.TemporaryDirectory()
        self.registry = build_default_registry()
        self.ctx = ToolContext(workspace_root=Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_accepted_with_message_and_status(self):
        result = self.registry.dispatch(
            ToolCall(
                name="SendUserMessage",
                input={"message": "hello", "status": "normal"},
                tool_use_id="toolu_reg_1",
            ),
            self.ctx,
        )
        self.assertFalse(result.is_error)
        self.assertEqual(result.output["message"], "hello")

    def test_rejected_with_missing_both_fields(self):
        with self.assertRaises(ToolInputError):
            self.registry.dispatch(
                ToolCall(
                    name="SendUserMessage",
                    input={},
                    tool_use_id="toolu_reg_2",
                ),
                self.ctx,
            )

    def test_rejected_with_missing_status(self):
        # 'status' is required; omitting it must raise ToolInputError.
        with self.assertRaises(ToolInputError):
            self.registry.dispatch(
                ToolCall(
                    name="SendUserMessage",
                    input={"message": "hello"},
                    tool_use_id="toolu_reg_3",
                ),
                self.ctx,
            )


# ---------------------------------------------------------------------------
# CD-01 to CD-09. Standalone-cd parser regression tests
# ---------------------------------------------------------------------------

class TestTryExtractCd(unittest.TestCase):
    """_try_extract_cd must distinguish standalone cd from compound commands.

    CD-01  cd /valid/path               -> recognised as standalone
    CD-02  cd "/valid/path with spaces" -> recognised as standalone
    CD-03  cd /some/path && git status  -> NOT standalone
    CD-04  cd /some/path ; pwd          -> NOT standalone
    CD-05  cd /some/path || exit        -> NOT standalone
    CD-06  cd /d "D:\\Clawd-Code" && .  -> NOT standalone (3+ shlex tokens)
    CD-07  cd /d "D:\\Clawd-Code"       -> NOT standalone (CMD /d flag)
    CD-08  standalone cd outside root  -> workspace rejection enforced
    CD-09  git status --short           -> unaffected (not a cd command)
    """

    def setUp(self):
        from src.tool_system.tools.bash import _try_extract_cd
        self._extract = _try_extract_cd

    # CD-01
    def test_cd01_standalone_posix_path(self):
        result = self._extract("cd /valid/path")
        self.assertIsNotNone(result)
        from pathlib import Path
        self.assertEqual(result, Path("/valid/path"))

    # CD-02
    def test_cd02_standalone_quoted_path(self):
        result = self._extract('cd "/valid/path with spaces"')
        self.assertIsNotNone(result)
        from pathlib import Path
        self.assertEqual(result, Path("/valid/path with spaces"))

    # CD-03
    def test_cd03_compound_and_not_standalone(self):
        result = self._extract("cd /some/path && git status")
        self.assertIsNone(result, "compound && must not be treated as standalone cd")

    # CD-04
    def test_cd04_compound_semicolon_not_standalone(self):
        result = self._extract("cd /some/path ; pwd")
        self.assertIsNone(result, "semicolon compound must not be treated as standalone cd")

    # CD-05
    def test_cd05_compound_or_not_standalone(self):
        result = self._extract("cd /some/path || exit")
        self.assertIsNone(result, "|| compound must not be treated as standalone cd")

    # CD-06
    def test_cd06_cmd_style_with_compound(self):
        result = self._extract('cd /d "D:\\Clawd-Code" && git status')
        self.assertIsNone(result, "CMD-style cd /d with && must not be treated as standalone cd")

    # CD-07
    def test_cd07_cmd_style_no_cmd_semantics(self):
        # cd /d "D:\path" is CMD syntax (3 shlex tokens: cd, /d, path).
        # Must NOT be recognised as standalone cd, and /d must NOT become D:\.
        result = self._extract('cd /d "D:\\Clawd-Code"')
        self.assertIsNone(result, "CMD /d flag must not be recognised as standalone cd target")

    # CD-08: workspace rejection still enforced for standalone cd
    def test_cd08_standalone_cd_outside_workspace_rejected(self):
        from src.tool_system.errors import ToolPermissionError
        with tempfile.TemporaryDirectory() as tmp:
            ctx = ToolContext(workspace_root=Path(tmp))
            # Use a POSIX-style absolute path that is outside any workspace
            # and has no spaces, so shlex.split(posix=True) parses it cleanly
            # on both Windows and POSIX hosts.
            outside_path = "/outside-workspace-sibling"
            # _try_extract_cd returns the path; BashTool.run then calls
            # context.ensure_allowed_path which must reject it.
            # _find_shell mock is not needed: ensure_allowed_path raises
            # before subprocess is ever reached.
            with self.assertRaises(ToolPermissionError):
                BashTool().run({"command": f"cd {outside_path}"}, ctx)

    # CD-09
    def test_cd09_normal_command_unaffected(self):
        result = self._extract("git status --short")
        self.assertIsNone(result, "non-cd command must not be intercepted")


# ---------------------------------------------------------------------------
# CD integration: compound cd command reaches Bash subprocess, not intercepted
# ---------------------------------------------------------------------------

class TestCompoundCdReachesBash(unittest.TestCase):
    """Compound cd commands must be forwarded to the Bash subprocess."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _run_with_mock_bash(self, command: str):
        """Run BashTool with a mocked subprocess and return (result, mock_run)."""
        fake_shell = ["/usr/bin/bash", "-lc"]
        mock_proc = MagicMock()
        mock_proc.stdout = "mocked"
        mock_proc.stderr = ""
        mock_proc.returncode = 0
        with patch("src.tool_system.tools.bash._find_shell", return_value=fake_shell), \
             patch("src.tool_system.tools.bash.subprocess.run",
                   return_value=mock_proc) as mock_run:
            result = BashTool().run({"command": command}, self.ctx)
        return result, mock_run

    def test_and_compound_reaches_subprocess(self):
        """cd /path && cmd must reach subprocess, not be intercepted."""
        result, mock_run = self._run_with_mock_bash("cd /some/path && git status")
        # subprocess.run must have been called (not short-circuited by cd handler)
        mock_run.assert_called_once()
        called_cmd = mock_run.call_args[0][0][-1]
        self.assertEqual(called_cmd, "cd /some/path && git status")

    def test_cmd_slash_d_compound_reaches_subprocess(self):
        """cd /d path && cmd must reach subprocess without /d interpretation."""
        result, mock_run = self._run_with_mock_bash(
            'cd /d "D:\\Clawd-Code" && git branch --show-current'
        )
        mock_run.assert_called_once()
        called_cmd = mock_run.call_args[0][0][-1]
        self.assertIn("D:\\Clawd-Code", called_cmd)

    def test_semicolon_compound_reaches_subprocess(self):
        """cd /path ; pwd must reach subprocess."""
        _, mock_run = self._run_with_mock_bash("cd /some/path ; pwd")
        mock_run.assert_called_once()

    def test_or_compound_reaches_subprocess(self):
        """cd /path || exit must reach subprocess."""
        _, mock_run = self._run_with_mock_bash("cd /some/path || exit")
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# 16. Windows text-decoding policy regression
# ---------------------------------------------------------------------------

class TestWindowsTextDecoding(unittest.TestCase):
    """Bash subprocess output must use an explicit decoding policy."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_explicit_text_decoding_policy(self):
        fake_shell = [r"C:\tools\git\bin\bash.exe", "-lc"]

        mock_proc = MagicMock()
        mock_proc.stdout = "Ünïcödé output: café résumé naïve\n"
        mock_proc.stderr = ""
        mock_proc.returncode = 0

        with patch(
            "src.tool_system.tools.bash._find_shell",
            return_value=fake_shell,
        ), patch(
            "src.tool_system.tools.bash.subprocess.run",
            return_value=mock_proc,
        ) as mock_run:
            result = BashTool().run(
                {"command": "echo hello"},
                self.ctx,
            )

        kwargs = mock_run.call_args.kwargs

        # text mode must be enabled
        self.assertTrue(kwargs["text"])

        # encoding must be explicitly UTF-8 (not the Windows locale default)
        self.assertEqual(kwargs["encoding"].lower(), "utf-8")

        # errors="replace" prevents decoding failures from terminating the tool
        # and makes undecodable sequences visible as replacement characters
        # rather than silently dropping them.
        self.assertEqual(kwargs["errors"].lower(), "replace")
        self.assertNotEqual(kwargs["errors"].lower(), "ignore")

        # successful ToolResult semantics
        self.assertFalse(result.is_error)
        self.assertEqual(result.output["exit_code"], 0)
        self.assertEqual(result.output["stdout"], mock_proc.stdout)
        self.assertEqual(result.output["stderr"], "")


if __name__ == "__main__":
    unittest.main()
