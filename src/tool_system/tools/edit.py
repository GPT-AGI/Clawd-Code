from __future__ import annotations

import difflib
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..permission_handler import PermissionResult
from ..protocol import ToolResult
from ..diff_utils import unified_diff_hunks
from ..ownership import require_owned_path
from ..registry import ToolSpec


class FileEditTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Edit",
            description="Performs exact string replacements in files.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
            is_destructive=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def check_permissions(
        self, tool_input: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        """Check if edit permission is allowed for this file."""
        file_path = tool_input.get("file_path")
        if not isinstance(file_path, str):
            return PermissionResult.allow()  # Input validation happens in run()

        try:
            path = context.ensure_allowed_path(file_path)
        except ToolPermissionError:
            return PermissionResult.allow()  # Path validation happens in run()

        if path.suffix.lower() in {".md", ".markdown"} and not context.permission_context.allow_docs:
            return PermissionResult.ask(
                message="Editing documentation files is blocked unless allow_docs is enabled",
                suggestion="Enable allow_docs to edit .md files",
            )
        return PermissionResult.allow()

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = tool_input["file_path"]
        old = tool_input["old_string"]
        new = tool_input["new_string"]
        replace_all = bool(tool_input.get("replace_all", False))

        if not isinstance(file_path, str):
            raise ToolInputError("file_path must be a string")
        if not isinstance(old, str) or not isinstance(new, str):
            raise ToolInputError("old_string/new_string must be strings")

        path = context.ensure_allowed_path(file_path)

        with context.mutation_lock:
            require_owned_path(context, path, tool_name="Edit")
            if not path.exists():
                raise ToolInputError(f"file does not exist: {path}")
            if not context.was_file_read_and_unchanged(path):
                raise ToolInputError(
                    "refusing to edit: file must be read first and unchanged since last read"
                )

            original_file = path.read_text(encoding="utf-8", errors="replace")
            count = original_file.count(old)
            if count == 0:
                raise ToolInputError("old_string not found in file")
            if count > 1 and not replace_all:
                raise ToolInputError(
                    "old_string is not unique; provide a larger old_string or set replace_all=true"
                )

            if replace_all:
                updated = original_file.replace(old, new)
            else:
                updated = original_file.replace(old, new, 1)

            path.write_text(updated, encoding="utf-8")
            context.mark_file_read(path)
        before_lines = original_file.splitlines(keepends=True)
        after_lines = updated.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=str(path),
                tofile=str(path),
                n=3,
                lineterm="",
            )
        )
        hunks = unified_diff_hunks(diff_lines)
        return ToolResult(
            name="Edit",
            output={
                "filePath": str(path),
                "oldString": old,
                "newString": new,
                "originalFile": original_file,
                "structuredPatch": hunks,
                "userModified": False,
                "replaceAll": bool(replace_all),
            },
        )
