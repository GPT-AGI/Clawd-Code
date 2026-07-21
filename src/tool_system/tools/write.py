from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..permission_handler import PermissionResult
from ..protocol import ToolResult
from ..diff_utils import unified_diff_hunks
from ..ownership import require_owned_path
from ..registry import ToolSpec


class FileWriteTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Write",
            description="Write a file to the local filesystem.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
            },
            is_destructive=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def check_permissions(
        self, tool_input: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        """Check if write permission is allowed for this file."""
        file_path = tool_input.get("file_path")
        if not isinstance(file_path, str):
            return PermissionResult.allow()  # Input validation happens in run()

        try:
            path = context.ensure_allowed_path(file_path)
        except ToolPermissionError:
            return PermissionResult.allow()  # Path validation happens in run()

        if path.suffix.lower() in {".md", ".markdown"} and not context.permission_context.allow_docs:
            return PermissionResult.ask(
                message="Writing documentation files is blocked unless allow_docs is enabled",
                suggestion="Enable allow_docs to write .md files",
            )
        return PermissionResult.allow()

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = tool_input["file_path"]
        content = tool_input["content"]
        if not isinstance(file_path, str):
            raise ToolInputError("file_path must be a string")
        if not isinstance(content, str):
            raise ToolInputError("content must be a string")

        path = context.ensure_allowed_path(file_path)

        with context.mutation_lock:
            require_owned_path(context, path, tool_name="Write")
            original_file: str | None = None
            if path.exists():
                if not context.was_file_read_and_unchanged(path):
                    raise ToolInputError(
                        "refusing to overwrite: file must be read first and unchanged since last read"
                    )
                original_file = path.read_text(encoding="utf-8", errors="replace")

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            context.mark_file_read(path)
        before_lines = (original_file or "").splitlines(keepends=True)
        after_lines = content.splitlines(keepends=True)
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
            name="Write",
            output={
                "type": "update" if original_file is not None else "create",
                "filePath": str(path),
                "content": content,
                "structuredPatch": hunks,
                "originalFile": original_file,
            },
        )
