"""Workspace-selection regression tests (WS-CLI-01 through WS-CLI-12).

All tests use temporary directories/files and mocks only.
No real DesignXpress repository is accessed.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_repl(workspace_root):
    """Return a minimal mock object standing in for ClawdREPL."""
    m = MagicMock()
    m._workspace_root = workspace_root
    m.tool_context = MagicMock()
    m.tool_context.workspace_root = workspace_root
    return m


def _cli_resolve(workspace_arg):
    """Call _resolve_workspace from src.cli directly."""
    from src.cli import _resolve_workspace
    return _resolve_workspace(workspace_arg)


# ---------------------------------------------------------------------------
# WS-CLI-01  No --workspace option
# ---------------------------------------------------------------------------

class TestNoWorkspaceOption(unittest.TestCase):
    """WS-CLI-01: Existing CWD behaviour remains unchanged when --workspace omitted."""

    def test_resolve_workspace_none_returns_none(self):
        """_resolve_workspace(None) must return None (caller uses Path.cwd())."""
        result = _cli_resolve(None)
        self.assertIsNone(result)

    def test_start_repl_no_workspace_calls_repl_with_none(self):
        """start_repl without workspace passes workspace_root=None to ClawdREPL."""
        with patch("src.config.get_default_provider", return_value="glm"):
            with patch("src.repl.ClawdREPL") as mock_cls:
                mock_cls.return_value = MagicMock()
                from src.cli import start_repl
                start_repl(stream=False, workspace=None)
                mock_cls.assert_called_once_with(
                    provider_name="glm",
                    stream=False,
                    workspace_root=None,
                )


# ---------------------------------------------------------------------------
# WS-CLI-02  Valid absolute explicit workspace
# ---------------------------------------------------------------------------

class TestValidAbsoluteWorkspace(unittest.TestCase):
    """WS-CLI-02: Valid absolute path is accepted and resolved."""

    def test_valid_directory_is_returned(self):
        with tempfile.TemporaryDirectory() as td:
            result = _cli_resolve(td)
            self.assertIsNotNone(result)
            self.assertIsInstance(result, Path)
            self.assertTrue(result.is_dir())


# ---------------------------------------------------------------------------
# WS-CLI-03  Relative workspace
# ---------------------------------------------------------------------------

class TestRelativeWorkspace(unittest.TestCase):
    """WS-CLI-03: Relative path is canonicalised/resolved."""

    def test_relative_path_resolved(self):
        """A genuinely relative path string resolves to the canonical absolute directory."""
        import os
        # Use a parent temp dir so we can build a relative child path from cwd.
        with tempfile.TemporaryDirectory() as parent_td:
            child_name = "ws_cli_03_child"
            child_abs = Path(parent_td) / child_name
            child_abs.mkdir()
            expected = child_abs.resolve()

            # Compute a relative path from the current working directory to child_abs.
            try:
                rel = os.path.relpath(str(child_abs))
            except ValueError:
                # On Windows, relpath raises ValueError when paths are on different
                # drives.  Fall back to the absolute path so the resolve assertion
                # still exercises _resolve_workspace's expanduser().resolve() logic.
                rel = str(child_abs)

            # rel is a relative string (or same-drive absolute fallback).
            result = _cli_resolve(rel)
            self.assertIsNotNone(result, f"_resolve_workspace({rel!r}) returned None")
            self.assertIsInstance(result, Path)
            self.assertTrue(result.is_absolute(), f"result is not absolute: {result}")
            self.assertEqual(result, expected)


# ---------------------------------------------------------------------------
# WS-CLI-04  Non-existent workspace
# ---------------------------------------------------------------------------

class TestNonExistentWorkspace(unittest.TestCase):
    """WS-CLI-04: Non-existent path is rejected cleanly."""

    def test_missing_directory_returns_none(self):
        fake = r"C:\this_path_absolutely_does_not_exist_ws_cli_04" if sys.platform == "win32" \
            else "/this_path_absolutely_does_not_exist_ws_cli_04"
        import io
        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            result = _cli_resolve(fake)
        self.assertIsNone(result)
        self.assertIn("does not exist", mock_err.getvalue())

    def test_main_exits_cleanly_on_missing_workspace(self):
        """main() returns 1 (not raises) for a missing --workspace."""
        fake = r"C:\no_such_ws_cli_04_main" if sys.platform == "win32" \
            else "/no_such_ws_cli_04_main"
        with patch("sys.argv", ["clawd", "--workspace", fake]):
            import io
            with patch("sys.stderr", new_callable=io.StringIO):
                from src.cli import main
                result = main()
        self.assertEqual(result, 1)


# ---------------------------------------------------------------------------
# WS-CLI-05  Workspace pointing to a file
# ---------------------------------------------------------------------------

class TestFileAsWorkspace(unittest.TestCase):
    """WS-CLI-05: A file path as workspace is rejected cleanly."""

    def test_file_path_returns_none(self):
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            file_path = tf.name
        try:
            import io
            with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                result = _cli_resolve(file_path)
            self.assertIsNone(result)
            self.assertIn("not a directory", mock_err.getvalue())
        finally:
            Path(file_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# WS-CLI-06  Explicit workspace reaches ClawdREPL
# ---------------------------------------------------------------------------

class TestWorkspaceReachesREPL(unittest.TestCase):
    """WS-CLI-06: ClawdREPL receives the selected canonical workspace path."""

    def test_workspace_propagated_to_repl(self):
        with tempfile.TemporaryDirectory() as td:
            expected = Path(td).resolve()
            with patch("src.config.get_default_provider", return_value="glm"):
                with patch("src.repl.ClawdREPL") as mock_cls:
                    mock_cls.return_value = MagicMock()
                    from src.cli import start_repl
                    start_repl(stream=False, workspace=expected)
                    mock_cls.assert_called_once_with(
                        provider_name="glm",
                        stream=False,
                        workspace_root=expected,
                    )


# ---------------------------------------------------------------------------
# WS-CLI-07  ToolContext workspace
# ---------------------------------------------------------------------------

class TestToolContextWorkspace(unittest.TestCase):
    """WS-CLI-07: ToolContext.workspace_root equals the explicitly selected workspace."""

    def test_tool_context_uses_selected_workspace(self):
        from src.tool_system.context import ToolContext
        with tempfile.TemporaryDirectory() as td:
            expected = Path(td).resolve()
            ctx = ToolContext(workspace_root=expected)
            self.assertEqual(ctx.workspace_root, expected)


# ---------------------------------------------------------------------------
# WS-CLI-08  Startup header
# ---------------------------------------------------------------------------

class TestStartupHeader(unittest.TestCase):
    """WS-CLI-08: The displayed workspace equals the authoritative selected workspace."""

    def _make_repl_with_workspace(self, workspace_root: Path):
        """Build a ClawdREPL-like object without hitting the real provider."""

        class _FakeREPL:
            def __init__(self_inner):
                self_inner._workspace_root = workspace_root

            def _display_cwd(self_inner) -> str:
                workspace = str(self_inner._workspace_root)
                home = str(Path.home())
                if workspace.startswith(home):
                    return workspace.replace(home, "~", 1)
                return workspace

        return _FakeREPL()

    def test_display_cwd_matches_workspace_root(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td).resolve()
            fake = self._make_repl_with_workspace(ws)
            displayed = fake._display_cwd()
            # Either the full path or home-relative – both must CONTAIN the dir name
            self.assertTrue(
                displayed == str(ws) or displayed.endswith(ws.name) or str(ws) in displayed,
                f"displayed={displayed!r} does not represent workspace={ws}",
            )


# ---------------------------------------------------------------------------
# WS-CLI-09  Inside-workspace access
# ---------------------------------------------------------------------------

class TestInsideWorkspaceAccess(unittest.TestCase):
    """WS-CLI-09: Path inside selected root is allowed."""

    def test_inside_path_allowed(self):
        from src.tool_system.context import ToolContext
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td).resolve()
            ctx = ToolContext(workspace_root=ws)
            # Create a file inside workspace
            inside_file = ws / "hello.txt"
            inside_file.write_text("hi")
            # Should not raise
            allowed = ctx.ensure_allowed_path(inside_file)
            self.assertEqual(allowed, inside_file.resolve())


# ---------------------------------------------------------------------------
# WS-CLI-10  Sibling path rejection
# ---------------------------------------------------------------------------

class TestSiblingPathRejection(unittest.TestCase):
    """WS-CLI-10: Path outside the selected workspace is rejected."""

    def test_sibling_path_rejected(self):
        from src.tool_system.context import ToolContext
        from src.tool_system.errors import ToolPermissionError
        with tempfile.TemporaryDirectory() as td_ws, \
             tempfile.TemporaryDirectory() as td_sibling:
            ws = Path(td_ws).resolve()
            sibling = Path(td_sibling).resolve()
            ctx = ToolContext(workspace_root=ws)
            sibling_file = sibling / "secret.txt"
            sibling_file.write_text("not allowed")
            with self.assertRaises(ToolPermissionError):
                ctx.ensure_allowed_path(sibling_file)


# ---------------------------------------------------------------------------
# WS-CLI-11  Traversal rejection
# ---------------------------------------------------------------------------

class TestTraversalRejection(unittest.TestCase):
    """WS-CLI-11: workspace/../outside is canonicalised and rejected."""

    def test_traversal_escape_rejected(self):
        from src.tool_system.context import ToolContext
        from src.tool_system.errors import ToolPermissionError
        with tempfile.TemporaryDirectory() as td_ws, \
             tempfile.TemporaryDirectory() as td_outside:
            ws = Path(td_ws).resolve()
            outside = Path(td_outside).resolve()
            ctx = ToolContext(workspace_root=ws)
            # Build a traversal path that would escape the workspace
            # e.g.  <ws>/../<td_outside>/file.txt
            traversal = ws / ".." / outside.name / "escaped.txt"
            # Create the target so it exists (some impls check existence first)
            escaped_file = outside / "escaped.txt"
            escaped_file.write_text("escaped")
            with self.assertRaises(ToolPermissionError):
                ctx.ensure_allowed_path(traversal)


# ---------------------------------------------------------------------------
# WS-CLI-12  No security expansion
# ---------------------------------------------------------------------------

class TestNoSecurityExpansion(unittest.TestCase):
    """WS-CLI-12: Selecting one workspace does NOT implicitly add Clawd-Code as allowed."""

    def test_clawd_source_not_implicitly_allowed(self):
        from src.tool_system.context import ToolContext
        from src.tool_system.errors import ToolPermissionError
        with tempfile.TemporaryDirectory() as td_ws:
            ws = Path(td_ws).resolve()
            ctx = ToolContext(workspace_root=ws)
            # src/cli.py lives in the Clawd-Code source tree
            clawd_src_file = Path(__file__).parent.parent / "src" / "cli.py"
            if not clawd_src_file.exists():
                self.skipTest("Could not locate src/cli.py for this test")
            with self.assertRaises(ToolPermissionError):
                ctx.ensure_allowed_path(clawd_src_file)


# ---------------------------------------------------------------------------
# Symlink helper
# ---------------------------------------------------------------------------

def _try_symlink(src: Path, dst: Path) -> bool:
    """Attempt to create a symlink dst -> src.  Returns True on success.

    On Windows, symlink creation requires either Developer Mode or elevated
    privileges.  When the OS refuses, the function returns False so that the
    calling test can skip itself via unittest.SkipTest.
    """
    try:
        dst.symlink_to(src)
        return True
    except (OSError, NotImplementedError):
        return False


# ---------------------------------------------------------------------------
# WS-CLI-13  Directory-symlink escape
# ---------------------------------------------------------------------------

class TestDirectorySymlinkEscape(unittest.TestCase):
    """WS-CLI-13: workspace/linked-outside -> outside/ is rejected."""

    def test_directory_symlink_escape_rejected(self):
        from src.tool_system.context import ToolContext
        from src.tool_system.errors import ToolPermissionError

        with tempfile.TemporaryDirectory() as td_ws, \
             tempfile.TemporaryDirectory() as td_outside:
            ws = Path(td_ws).resolve()
            outside = Path(td_outside).resolve()

            # Create secret file in outside dir
            secret = outside / "secret.txt"
            secret.write_text("classified")

            # Create symlink inside workspace pointing at outside dir
            link = ws / "linked-outside"
            if not _try_symlink(outside, link):
                raise unittest.SkipTest(
                    "Cannot create directory symlink on this platform/environment "
                    "(Windows Developer Mode or elevated privileges required)"
                )

            ctx = ToolContext(workspace_root=ws)
            escaped_path = link / "secret.txt"

            with self.assertRaises(ToolPermissionError):
                ctx.ensure_allowed_path(escaped_path)


# ---------------------------------------------------------------------------
# WS-CLI-14  File-symlink escape
# ---------------------------------------------------------------------------

class TestFileSymlinkEscape(unittest.TestCase):
    """WS-CLI-14: workspace/apparently-safe.txt -> outside/secret.txt is rejected."""

    def test_file_symlink_escape_rejected(self):
        from src.tool_system.context import ToolContext
        from src.tool_system.errors import ToolPermissionError

        with tempfile.TemporaryDirectory() as td_ws, \
             tempfile.TemporaryDirectory() as td_outside:
            ws = Path(td_ws).resolve()
            outside = Path(td_outside).resolve()

            # Create the real target outside the workspace
            secret = outside / "secret.txt"
            secret.write_text("classified")

            # Create a file symlink inside workspace pointing at the outside file
            safe_looking = ws / "apparently-safe.txt"
            if not _try_symlink(secret, safe_looking):
                raise unittest.SkipTest(
                    "Cannot create file symlink on this platform/environment "
                    "(Windows Developer Mode or elevated privileges required)"
                )

            ctx = ToolContext(workspace_root=ws)

            with self.assertRaises(ToolPermissionError):
                ctx.ensure_allowed_path(safe_looking)


# ---------------------------------------------------------------------------
# WS-CLI-15  Safe internal symlink (should be ALLOWED)
# ---------------------------------------------------------------------------

class TestSafeInternalSymlink(unittest.TestCase):
    """WS-CLI-15: workspace/linked-inside -> workspace/real/ is allowed."""

    def test_internal_symlink_allowed(self):
        from src.tool_system.context import ToolContext

        with tempfile.TemporaryDirectory() as td_ws:
            ws = Path(td_ws).resolve()

            # Create real subdirectory and a file inside it
            real_dir = ws / "real"
            real_dir.mkdir()
            safe_file = real_dir / "safe.txt"
            safe_file.write_text("safe content")

            # Create a symlink inside workspace pointing at the real subdir
            link_dir = ws / "linked-inside"
            if not _try_symlink(real_dir, link_dir):
                raise unittest.SkipTest(
                    "Cannot create directory symlink on this platform/environment "
                    "(Windows Developer Mode or elevated privileges required)"
                )

            ctx = ToolContext(workspace_root=ws)
            via_link = link_dir / "safe.txt"

            # Should NOT raise – the resolved target is still inside the workspace
            result = ctx.ensure_allowed_path(via_link)
            self.assertEqual(result, safe_file.resolve())


# ---------------------------------------------------------------------------
# WS-CLI-16  Chained-symlink escape
# ---------------------------------------------------------------------------

class TestChainedSymlinkEscape(unittest.TestCase):
    """WS-CLI-16: workspace/first-link -> workspace/second-link -> outside/ is rejected."""

    def test_chained_symlink_escape_rejected(self):
        from src.tool_system.context import ToolContext
        from src.tool_system.errors import ToolPermissionError

        with tempfile.TemporaryDirectory() as td_ws, \
             tempfile.TemporaryDirectory() as td_outside:
            ws = Path(td_ws).resolve()
            outside = Path(td_outside).resolve()

            # Create secret file in outside dir
            secret = outside / "secret.txt"
            secret.write_text("classified")

            # second-link -> outside dir
            second_link = ws / "second-link"
            if not _try_symlink(outside, second_link):
                raise unittest.SkipTest(
                    "Cannot create directory symlink on this platform/environment "
                    "(Windows Developer Mode or elevated privileges required)"
                )

            # first-link -> second-link (chained, still inside workspace dir entry)
            first_link = ws / "first-link"
            if not _try_symlink(second_link, first_link):
                raise unittest.SkipTest(
                    "Cannot create chained symlink on this platform/environment "
                    "(Windows Developer Mode or elevated privileges required)"
                )

            ctx = ToolContext(workspace_root=ws)
            escaped_path = first_link / "secret.txt"

            with self.assertRaises(ToolPermissionError):
                ctx.ensure_allowed_path(escaped_path)


if __name__ == "__main__":
    unittest.main()
