from __future__ import annotations

import ast
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..ownership import (
    audit_changed_paths,
    bash_audit_required,
    control_state_guard,
    snapshot_local_workspace,
)
from ..protocol import ToolResult
from ..registry import ToolSpec


_DANGEROUS_PATTERNS = [
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd\b\s+if=", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE),
]

_SAFE_RECURSIVE_DELETE_NAMES = {
    ".eggs",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


def _strict_protocol_v2_team(context: ToolContext) -> bool:
    team = context.team or {}
    settings = team.get("settings") if isinstance(team.get("settings"), dict) else {}
    quality = (
        settings.get("quality_gates")
        if isinstance(settings.get("quality_gates"), dict)
        else {}
    )
    versions = [1]
    for raw in (
        team.get("protocol_version"),
        settings.get("protocol_version"),
        quality.get("protocol_version"),
    ):
        try:
            versions.append(int(raw))
        except (TypeError, ValueError):
            continue
    return bool(quality.get("strict") and max(versions) >= 2)


_SHELL_NAMES = {"bash", "dash", "ksh", "sh", "zsh"}
_SHELL_CONTROL_PREFIXES = {
    "!",
    "do",
    "elif",
    "else",
    "if",
    "then",
    "time",
    "until",
    "while",
}
_COMMAND_WRAPPERS = {"command", "exec", "nohup"}
_HEREDOC_PATTERN = re.compile(
    r"(?<!<)<<(?P<strip>-)?\s*(?:'(?P<single>[^']+)'|\"(?P<double>[^\"]+)\"|(?P<plain>[A-Za-z0-9_]+))"
)


def _strip_heredoc_bodies(source: str) -> str:
    """Keep shell command lines while removing heredoc data bodies.

    A destructive-looking string written to a source file is data, not an executed
    command.  Interpreter stdin heredocs are intentionally outside this lexical
    guard; teammate post-execution rollback remains the final safety boundary.
    """

    kept: list[str] = []
    pending: list[tuple[str, bool]] = []
    for line in source.splitlines(keepends=True):
        if pending:
            delimiter, strip_tabs = pending[0]
            candidate = line.rstrip("\r\n")
            if strip_tabs:
                candidate = candidate.lstrip("\t")
            if candidate == delimiter:
                pending.pop(0)
            continue
        kept.append(line)
        for match in _HEREDOC_PATTERN.finditer(line):
            delimiter = (
                match.group("single")
                or match.group("double")
                or match.group("plain")
            )
            pending.append((delimiter, bool(match.group("strip"))))
    return "".join(kept)


def _normalize_shell_newlines(source: str) -> str:
    """Turn executable newlines into command separators without touching quotes."""

    out: list[str] = []
    quote: str | None = None
    escaped = False
    comment = False
    for index, char in enumerate(source):
        if comment:
            if char == "\n":
                comment = False
                out.append(" ; ")
            continue
        if escaped:
            escaped = False
            if char != "\n":
                out.extend(("\\", char))
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote:
            out.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            out.append(char)
            continue
        if char == "#" and (
            index == 0 or source[index - 1].isspace() or source[index - 1] in ";|&()"
        ):
            comment = True
            continue
        out.append(" ; " if char == "\n" else char)
    return "".join(out)


def _shell_command_argvs(source: str) -> list[list[str]]:
    normalized = _normalize_shell_newlines(_strip_heredoc_bodies(source))
    lexer = shlex.shlex(
        normalized,
        posix=True,
        punctuation_chars=";&|()",
    )
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return []
    commands: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token and all(character in ";&|()" for character in token):
            if current:
                commands.append(current)
                current = []
            continue
        current.append(token)
    if current:
        commands.append(current)
    return commands


def _unwrap_command(argv: list[str]) -> list[str]:
    remaining = list(argv)
    while remaining:
        first = remaining[0]
        if first in _SHELL_CONTROL_PREFIXES or re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*=.*", first
        ):
            remaining.pop(0)
            continue
        name = PurePosixPath(first).name
        if name in _COMMAND_WRAPPERS:
            remaining.pop(0)
            continue
        if name == "env":
            remaining.pop(0)
            while remaining and (
                remaining[0].startswith("-")
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", remaining[0])
            ):
                remaining.pop(0)
            continue
        break
    return remaining


def _safe_delete_target(target: str) -> bool:
    normalized = target.strip().replace("\\", "/").rstrip("/") or "/"
    if normalized == "/" or any(marker in normalized for marker in "$`*?[]{}"):
        return False
    parts = PurePosixPath(normalized).parts
    return bool(
        any(part in _SAFE_RECURSIVE_DELETE_NAMES for part in parts)
        or any(part.endswith(".egg-info") for part in parts)
        or ".clawd/task-tests/" in normalized.lstrip("/") + "/"
    )


def _rm_delete_targets(argv: list[str]) -> list[str]:
    recursive = False
    targets: list[str] = []
    options_done = False
    for argument in argv[1:]:
        if not options_done and argument == "--":
            options_done = True
            continue
        if not options_done and argument.startswith("-"):
            if argument == "--recursive" or (
                not argument.startswith("--")
                and any(flag in argument[1:] for flag in "rR")
            ):
                recursive = True
            continue
        targets.append(argument)
    return targets if recursive else []


def _find_delete_targets(argv: list[str]) -> list[str]:
    if "-delete" not in argv[1:]:
        return []
    roots: list[str] = []
    for argument in argv[1:]:
        if argument == "-delete" or argument.startswith("-") or argument in {"!", "("}:
            if roots or argument == "-delete":
                break
            continue
        roots.append(argument)
    return roots or ["."]


def _git_clean_targets(argv: list[str]) -> list[str]:
    index = 1
    git_root = "."
    while index < len(argv):
        argument = argv[index]
        if argument == "-C" and index + 1 < len(argv):
            git_root = argv[index + 1]
            index += 2
            continue
        if argument.startswith("-"):
            index += 1
            continue
        break
    if index >= len(argv) or argv[index] != "clean":
        return []
    clean_arguments = argv[index + 1 :]
    if any(
        argument == "--dry-run"
        or (
            argument.startswith("-")
            and not argument.startswith("--")
            and "n" in argument[1:]
        )
        for argument in clean_arguments
    ):
        return []
    return [git_root]


def _python_rmtree_targets(argv: list[str]) -> list[str]:
    code: str | None = None
    for index, argument in enumerate(argv[1:]):
        if argument == "-c" and index + 2 < len(argv):
            code = argv[index + 2]
            break
    if code is None:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    module_aliases = {"shutil"}
    function_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "shutil":
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "shutil":
            for alias in node.names:
                if alias.name == "rmtree":
                    function_aliases.add(alias.asname or alias.name)
    targets: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        matched = bool(
            isinstance(function, ast.Attribute)
            and function.attr == "rmtree"
            and isinstance(function.value, ast.Name)
            and function.value.id in module_aliases
        ) or bool(isinstance(function, ast.Name) and function.id in function_aliases)
        if not matched:
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(
            node.args[0].value, str
        ):
            targets.append(node.args[0].value)
        else:
            targets.append("<dynamic shutil.rmtree target>")
    return targets


def _destructive_delete_targets(command: str, *, depth: int = 0) -> list[str]:
    if depth > 4:
        return ["<nested shell command depth exceeded>"]
    targets: list[str] = []
    for raw_argv in _shell_command_argvs(command):
        argv = _unwrap_command(raw_argv)
        if not argv:
            continue
        name = PurePosixPath(argv[0]).name.casefold()
        if name == "rm":
            targets.extend(_rm_delete_targets(argv))
        elif name == "find":
            targets.extend(_find_delete_targets(argv))
        elif name == "git":
            targets.extend(_git_clean_targets(argv))
        elif name in _SHELL_NAMES:
            for index, argument in enumerate(argv[1:]):
                if argument.startswith("-") and "c" in argument[1:] and index + 2 < len(argv):
                    targets.extend(
                        _destructive_delete_targets(argv[index + 2], depth=depth + 1)
                    )
                    break
        elif re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", name):
            targets.extend(_python_rmtree_targets(argv))
    return targets


def _unsafe_recursive_delete_target(command: str) -> str | None:
    """Return the first non-generated target of a destructive command."""

    return next(
        (
            target
            for target in _destructive_delete_targets(command)
            if not _safe_delete_target(target)
        ),
        None,
    )


def _truncate(s: str, limit: int = 20000) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + "\n\n... [truncated] ..."


def _try_extract_cd(command: str) -> Path | None:
    stripped = command.strip()
    if not stripped.startswith("cd "):
        return None
    try:
        parts = shlex.split(stripped, posix=True)
    except ValueError:
        return None
    if len(parts) == 2 and parts[0] == "cd":
        return Path(parts[1])
    return None


class BashTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Bash",
            description=(
                "Execute a shell command. The active Clawd Python interpreter is available as "
                "$CLAWD_PYTHON and its directory is prepended to PATH."
            ),
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

        delete_targets = _destructive_delete_targets(command)
        if any(target.strip().rstrip("/") in {"", "/"} for target in delete_targets):
            raise ToolPermissionError("refusing to run potentially dangerous command")
        if _strict_protocol_v2_team(context):
            delete_target = next(
                (
                    target
                    for target in delete_targets
                    if not _safe_delete_target(target)
                ),
                None,
            )
            if delete_target is not None:
                raise ToolPermissionError(
                    "strict protocol v2 preserves the best workspace and refuses "
                    f"recursive deletion of deliverable path {delete_target!r}; edit "
                    "the owned files in place or use TeamReplan for a recoverable plan "
                    "replacement. TeamAbort is terminal and is not a restart operation"
                )

        explicit_cwd = tool_input.get("cwd")
        if explicit_cwd is not None:
            if not isinstance(explicit_cwd, str) or not explicit_cwd.startswith("/"):
                raise ToolInputError("cwd must be an absolute path when provided")
            cwd = context.ensure_allowed_path(explicit_cwd)
        else:
            cwd = context.cwd or context.workspace_root

        cd_target = _try_extract_cd(command)
        if cd_target is not None and command.strip().startswith("cd ") and len(command.strip().splitlines()) == 1:
            next_dir = (cwd / cd_target).expanduser().resolve() if not cd_target.is_absolute() else cd_target.expanduser().resolve()
            next_dir = context.ensure_allowed_path(next_dir)
            if not next_dir.exists() or not next_dir.is_dir():
                return ToolResult(name="Bash", output={"error": f"directory does not exist: {next_dir}"}, is_error=True)
            context.cwd = next_dir
            return ToolResult(name="Bash", output={"cwd": str(context.cwd), "stdout": "", "stderr": ""})

        timeout_s = tool_input.get("timeout_s", 60)
        if not isinstance(timeout_s, int) or timeout_s < 1 or timeout_s > 600:
            raise ToolInputError("timeout_s must be an integer between 1 and 600")

        python_executable = str(Path(sys.executable).absolute())
        python_bin = str(Path(python_executable).parent)
        environment_prefix = (
            f"export PATH={shlex.quote(python_bin)}:$PATH\n"
            f"export CLAWD_PYTHON={shlex.quote(python_executable)}\n"
        )
        with context.mutation_lock:
            with control_state_guard(context) as control_backup:
                before = (
                    snapshot_local_workspace(context)
                    if bash_audit_required(context)
                    else None
                )
                try:
                    completed = subprocess.run(
                        ["bash", "-lc", environment_prefix + command],
                        cwd=str(cwd),
                        capture_output=True,
                        text=True,
                        timeout=timeout_s,
                    )
                finally:
                    if before is not None:
                        audit_changed_paths(
                            context,
                            tool_name="Bash",
                            before=before,
                            after=snapshot_local_workspace(context),
                            control_backup=control_backup,
                        )

        stdout = _truncate(completed.stdout or "")
        stderr = _truncate(completed.stderr or "")
        output: dict[str, Any] = {
            "cwd": str(cwd),
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
        return ToolResult(name="Bash", output=output, is_error=completed.returncode != 0)
