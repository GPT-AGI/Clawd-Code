from __future__ import annotations

import platform
import re
import shlex
import shutil
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from ..context import ToolContext
from ..errors import ToolExecutionError, ToolInputError, ToolPermissionError
from ..protocol import ToolResult
from ..registry import ToolSpec


_DANGEROUS_PATTERNS = [
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd\b\s+if=", re.IGNORECASE),
    re.compile(r"\brm\b.*\s+-rf\s+/\s*$", re.IGNORECASE),
    re.compile(r"\brm\b.*\s+-rf\s+/\s+"),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE),
]


def _find_shell() -> list[str]:
    """Return the shell argv prefix to use for command execution.

    On POSIX systems the existing behaviour is preserved: ["bash", "-lc"].

    On Windows, shutil.which("bash") is used for deterministic discovery.
    If bash is not found on Windows, ToolExecutionError is raised with a
    concise actionable message rather than propagating a raw [WinError 2].
    """
    if platform.system().lower() != "windows":
        # Non-Windows: keep the original behaviour exactly.
        return ["bash", "-lc"]

    # Windows: deterministic discovery via PATH only.
    bash_on_path = shutil.which("bash")
    if bash_on_path:
        return [bash_on_path, "-lc"]

    raise ToolExecutionError(
        "Bash is not available on this system. "
        "Install Git for Windows (https://gitforwindows.org/) and ensure "
        "bash.exe is on the PATH, or enable WSL."
    )


def _is_absolute_path(p: str) -> bool:
    """Return True if *p* is an absolute path on any supported OS.

    Uses stdlib PureWindowsPath and PurePosixPath so the check is correct
    for both Windows (drive-letter) and POSIX (leading slash) paths without
    relying on the host OS's native Path interpretation.

    Accepts:
        D:\\project   D:/project   /home/project   /

    Rejects:
        relative/path   D:   C:file   (empty string)
    """
    if not p:
        return False

    # POSIX absolute: starts with /
    if PurePosixPath(p).is_absolute():
        return True

    # Windows absolute: drive letter + separator, e.g. C:\\ or C:/
    # PureWindowsPath("D:") is NOT absolute (drive-relative), so this is safe.
    # PureWindowsPath("D:\\foo") and PureWindowsPath("D:/foo") ARE absolute.
    # PureWindowsPath("C:file") is drive-relative, not absolute.
    try:
        win_path = PureWindowsPath(p)
        if win_path.is_absolute():
            return True
    except Exception:
        pass

    return False


def _truncate(s: str, limit: int = 20000) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + "\n\n... [truncated] ..."


def _try_extract_cd(command: str) -> Path | None:
    """Return the cd target only when *command* is a standalone Bash cd.

    Returns ``None`` for any compound or ambiguous input so that the command
    is forwarded to the real Bash executable for parsing.

    Accepted (exactly two shlex tokens: ``cd <target>``):
        cd /some/path
        cd "some directory with spaces"

    Rejected (three or more tokens — compound, pipe, or CMD-style):
        cd /some/path && git status
        cd /some/path ; pwd
        cd /some/path || exit
        cd /d "D:\\Clawd-Code" && ...      (CMD /d flag -> 3+ tokens)
        cd /d "D:\\Clawd-Code"             (CMD /d flag -> 3 tokens)
    """
    stripped = command.strip()
    if not stripped.startswith("cd "):
        return None
    try:
        parts = shlex.split(stripped, posix=True)
    except ValueError:
        return None
    # Exactly two tokens required: ['cd', '<target>'].
    # Three or more tokens means a compound command, shell operator, or
    # extra flags (e.g. CMD /d) -- none of which are standalone Bash cd.
    if len(parts) == 2 and parts[0] == "cd":
        return Path(parts[1])
    return None


class BashTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Bash",
            description="Execute a shell command.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout_s": {"type": "integer"},
                },
                "required": ["command"],
            },
            is_destructive=True,
            max_result_size_chars=50_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        command = tool_input["command"]
        if not isinstance(command, str) or not command.strip():
            raise ToolInputError("command must be a non-empty string")
        if "\x00" in command:
            raise ToolInputError("command contains NUL byte")

        for pat in _DANGEROUS_PATTERNS:
            if pat.search(command):
                raise ToolPermissionError("refusing to run potentially dangerous command")

        explicit_cwd = tool_input.get("cwd")
        if explicit_cwd is not None:
            if not isinstance(explicit_cwd, str) or not _is_absolute_path(explicit_cwd):
                raise ToolInputError("cwd must be an absolute path when provided")
            cwd = context.ensure_allowed_path(explicit_cwd)
        else:
            cwd = context.cwd or context.workspace_root

        cd_target = _try_extract_cd(command)
        if cd_target is not None:
            next_dir = (cwd / cd_target).expanduser().resolve() if not cd_target.is_absolute() else cd_target.expanduser().resolve()
            next_dir = context.ensure_allowed_path(next_dir)
            if not next_dir.exists() or not next_dir.is_dir():
                return ToolResult(name="Bash", output={"error": f"directory does not exist: {next_dir}"}, is_error=True)
            context.cwd = next_dir
            return ToolResult(name="Bash", output={"cwd": str(context.cwd), "stdout": "", "stderr": ""})

        timeout_s = tool_input.get("timeout_s", 60)
        if not isinstance(timeout_s, int) or timeout_s < 1 or timeout_s > 600:
            raise ToolInputError("timeout_s must be an integer between 1 and 600")

        # Discover the shell on each call so that tests can mock _find_shell
        # or platform.system without module-level caching complications.
        shell_argv = _find_shell()

        try:
            completed = subprocess.run(
                [*shell_argv, command],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )
        except FileNotFoundError as exc:
            # The shell executable was not found at launch time.
            # Translate to ToolExecutionError so [WinError 2] never escapes
            # as a raw exception to callers.
            raise ToolExecutionError(
                f"Bash executable not found: {shell_argv[0]!r}. "
                "Ensure bash is installed and available on PATH."
            ) from exc

        stdout = _truncate(completed.stdout or "")
        stderr = _truncate(completed.stderr or "")
        output: dict[str, Any] = {
            "cwd": str(cwd),
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
        return ToolResult(name="Bash", output=output, is_error=completed.returncode != 0)
