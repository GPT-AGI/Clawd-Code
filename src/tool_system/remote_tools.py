from __future__ import annotations

import base64
import difflib
import json
import mimetypes
from pathlib import PurePosixPath
from typing import Any

from .context import ToolContext
from .diff_utils import unified_diff_hunks
from .errors import ToolExecutionError, ToolInputError, ToolPermissionError
from .ownership import (
    audit_changed_paths,
    bash_audit_required,
    control_state_guard,
    require_owned_path,
    snapshot_remote_workspace,
)
from .protocol import ToolResult
from .registry import ToolSpec
from .tools.bash import (
    _DANGEROUS_PATTERNS,
    _destructive_delete_targets,
    _safe_delete_target,
    _strict_protocol_v2_team,
    _truncate,
    _try_extract_cd,
)
from .tools.edit import FileEditTool
from .tools.glob import GlobTool
from .tools.grep import GrepTool
from .tools.read import FileReadTool
from .tools.write import FileWriteTool


def _backend(context: ToolContext) -> Any:
    if context.workspace_backend is None:
        raise ToolExecutionError("remote workspace backend is unavailable")
    return context.workspace_backend


class RemoteBashTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Bash",
            description="Execute a shell command inside the remote sandbox workspace.",
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
        for pattern in _DANGEROUS_PATTERNS:
            if pattern.search(command):
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

        backend = _backend(context)
        explicit_cwd = tool_input.get("cwd")
        if explicit_cwd is not None:
            if not isinstance(explicit_cwd, str) or not explicit_cwd.startswith("/"):
                raise ToolInputError("cwd must be an absolute path when provided")
            try:
                cwd = context.resolve_execution_path(explicit_cwd)
            except ValueError as exc:
                raise ToolPermissionError(str(exc)) from exc
        else:
            cwd = context.execution_cwd or context.execution_workspace_root or "/workspace"

        cd_target = _try_extract_cd(command)
        if cd_target is not None and len(command.strip().splitlines()) == 1:
            try:
                next_dir = backend.resolve_path(
                    str(cd_target), cwd=cwd, local_root=context.workspace_root
                )
            except ValueError as exc:
                raise ToolPermissionError(str(exc)) from exc
            stat = backend.stat(next_dir)
            if not stat.exists or not stat.is_dir:
                return ToolResult(
                    name="Bash",
                    output={"error": f"directory does not exist: {next_dir}"},
                    is_error=True,
                )
            context.execution_cwd = next_dir
            return ToolResult(
                name="Bash",
                output={"cwd": next_dir, "stdout": "", "stderr": ""},
            )

        timeout_s = tool_input.get("timeout_s", 60)
        if not isinstance(timeout_s, int) or timeout_s < 1 or timeout_s > 600:
            raise ToolInputError("timeout_s must be an integer between 1 and 600")
        with context.mutation_lock:
            with control_state_guard(context) as control_backup:
                before = (
                    snapshot_remote_workspace(context)
                    if bash_audit_required(context)
                    else None
                )
                try:
                    result = backend.exec(command, cwd=cwd, timeout_s=timeout_s)
                finally:
                    if before is not None:
                        audit_changed_paths(
                            context,
                            tool_name="Bash",
                            before=before,
                            after=snapshot_remote_workspace(context),
                            control_backup=control_backup,
                        )
        return ToolResult(
            name="Bash",
            output={
                "cwd": cwd,
                "exit_code": result.exit_code,
                "stdout": _truncate(result.stdout or ""),
                "stderr": _truncate(result.stderr or ""),
            },
            is_error=result.exit_code != 0,
        )


class RemoteFileReadTool:
    def spec(self) -> ToolSpec:
        spec = FileReadTool().spec()
        return ToolSpec(
            name=spec.name,
            description="Read a file from the remote sandbox workspace.",
            input_schema=spec.input_schema,
            aliases=spec.aliases,
            is_read_only=spec.is_read_only,
            max_result_size_chars=spec.max_result_size_chars,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = tool_input["file_path"]
        if not isinstance(file_path, str):
            raise ToolInputError("file_path must be a string")
        if file_path.startswith(("http://", "https://")):
            return ToolResult(
                name="Read",
                output={"error": "The 'Read' tool is for sandbox files; use WebFetch for URLs"},
                is_error=True,
            )
        limit = tool_input.get("limit", 2000)
        offset = tool_input.get("offset", 1)
        pages = tool_input.get("pages")
        if not isinstance(limit, int) or limit < 1 or limit > 2000:
            raise ToolInputError("limit must be an integer between 1 and 2000")
        if not isinstance(offset, int) or offset < 1:
            raise ToolInputError("offset must be an integer >= 1")
        if pages is not None and not isinstance(pages, str):
            raise ToolInputError("pages must be a string when provided")
        try:
            path = context.resolve_execution_path(file_path)
        except ValueError as exc:
            raise ToolPermissionError(str(exc)) from exc
        backend = _backend(context)
        stat = backend.stat(path)
        if not stat.exists:
            return ToolResult(name="Read", output={"error": f"file not found: {path}"}, is_error=True)
        if stat.is_dir:
            return ToolResult(name="Read", output={"error": f"path is a directory: {path}"}, is_error=True)
        if (
            context.was_remote_file_read_and_unchanged(path)
            and pages is None
            and "offset" not in tool_input
            and "limit" not in tool_input
        ):
            return ToolResult(
                name="Read", output={"type": "file_unchanged", "file": {"filePath": path}}
            )

        suffix = PurePosixPath(path).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf"}:
            data = backend.read_bytes(path)
            if len(data) > 5 * 1024 * 1024:
                return ToolResult(
                    name="Read",
                    output={"error": f"file too large to inline: {path} ({len(data)} bytes)"},
                    is_error=True,
                )
            context.mark_remote_file_read(path)
            if suffix == ".pdf":
                if pages is not None and pages.strip():
                    return ToolResult(
                        name="Read",
                        output={"error": "PDF page-range reads are not supported; omit pages"},
                        is_error=True,
                    )
                return ToolResult(
                    name="Read",
                    output={
                        "type": "pdf",
                        "file": {
                            "filePath": path,
                            "base64": base64.b64encode(data).decode("ascii"),
                            "originalSize": len(data),
                        },
                    },
                )
            mime, _ = mimetypes.guess_type(path)
            return ToolResult(
                name="Read",
                output={
                    "type": "image",
                    "file": {
                        "base64": base64.b64encode(data).decode("ascii"),
                        "type": mime or "image/png",
                        "originalSize": len(data),
                        "filePath": path,
                    },
                },
            )

        try:
            text = backend.read_text(path)
        except Exception as exc:
            raise ToolExecutionError(str(exc)) from exc
        context.mark_remote_file_read(path)
        if suffix == ".ipynb":
            try:
                cells = json.loads(text).get("cells")
            except Exception as exc:
                return ToolResult(
                    name="Read",
                    output={"error": f"failed to parse notebook: {exc}"},
                    is_error=True,
                )
            if not isinstance(cells, list):
                return ToolResult(name="Read", output={"error": "no cells found"}, is_error=True)
            return ToolResult(
                name="Read", output={"type": "notebook", "file": {"filePath": path, "cells": cells}}
            )

        lines = text.splitlines()
        sliced = lines[offset - 1 : offset - 1 + limit]
        numbered = "\n".join(f"{index + offset}\t{line}" for index, line in enumerate(sliced))
        return ToolResult(
            name="Read",
            output={
                "type": "text",
                "file": {
                    "filePath": path,
                    "content": numbered,
                    "numLines": len(sliced),
                    "startLine": offset,
                    "totalLines": len(lines),
                },
            },
        )


class RemoteFileWriteTool:
    def spec(self) -> ToolSpec:
        return FileWriteTool().spec()

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = tool_input["file_path"]
        content = tool_input["content"]
        if not isinstance(file_path, str):
            raise ToolInputError("file_path must be a string")
        if not isinstance(content, str):
            raise ToolInputError("content must be a string")
        try:
            path = context.resolve_execution_path(file_path)
        except ValueError as exc:
            raise ToolPermissionError(str(exc)) from exc
        backend = _backend(context)
        with context.mutation_lock:
            require_owned_path(
                context, path, tool_name="Write", execution_path=True
            )
            stat = backend.stat(path)
            original: str | None = None
            if stat.exists:
                if stat.is_dir:
                    raise ToolInputError(f"path is a directory: {path}")
                if not context.was_remote_file_read_and_unchanged(path):
                    raise ToolInputError(
                        "refusing to overwrite: file must be read first and unchanged since last read"
                    )
                original = backend.read_text(path)
            backend.write_text(path, content)
            context.mark_remote_file_read(path)
        diff = list(
            difflib.unified_diff(
                (original or "").splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=path,
                tofile=path,
                n=3,
                lineterm="",
            )
        )
        return ToolResult(
            name="Write",
            output={
                "type": "update" if original is not None else "create",
                "filePath": path,
                "content": content,
                "structuredPatch": unified_diff_hunks(diff),
                "originalFile": original,
            },
        )


class RemoteFileEditTool:
    def spec(self) -> ToolSpec:
        return FileEditTool().spec()

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = tool_input["file_path"]
        old = tool_input["old_string"]
        new = tool_input["new_string"]
        replace_all = bool(tool_input.get("replace_all", False))
        if not isinstance(file_path, str):
            raise ToolInputError("file_path must be a string")
        if not isinstance(old, str) or not isinstance(new, str):
            raise ToolInputError("old_string/new_string must be strings")
        try:
            path = context.resolve_execution_path(file_path)
        except ValueError as exc:
            raise ToolPermissionError(str(exc)) from exc
        backend = _backend(context)
        with context.mutation_lock:
            require_owned_path(
                context, path, tool_name="Edit", execution_path=True
            )
            stat = backend.stat(path)
            if not stat.exists or not stat.is_file:
                raise ToolInputError(f"file does not exist: {path}")
            if not context.was_remote_file_read_and_unchanged(path):
                raise ToolInputError(
                    "refusing to edit: file must be read first and unchanged since last read"
                )
            original = backend.read_text(path)
            count = original.count(old)
            if count == 0:
                raise ToolInputError("old_string not found in file")
            if count > 1 and not replace_all:
                raise ToolInputError(
                    "old_string is not unique; provide a larger old_string or set replace_all=true"
                )
            updated = original.replace(old, new) if replace_all else original.replace(old, new, 1)
            backend.write_text(path, updated)
            context.mark_remote_file_read(path)
        diff = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=path,
                tofile=path,
                n=3,
                lineterm="",
            )
        )
        return ToolResult(
            name="Edit",
            output={
                "filePath": path,
                "oldString": old,
                "newString": new,
                "originalFile": original,
                "structuredPatch": unified_diff_hunks(diff),
                "userModified": False,
                "replaceAll": replace_all,
            },
        )


class RemoteGlobTool:
    def spec(self) -> ToolSpec:
        return GlobTool().spec()

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = tool_input["pattern"]
        base = tool_input.get("path")
        limit = tool_input.get("limit", 100)
        if not isinstance(pattern, str) or not pattern:
            raise ToolInputError("pattern must be a non-empty string")
        if base is not None and (not isinstance(base, str) or not base):
            raise ToolInputError("path must be a non-empty string when provided")
        if not isinstance(limit, int) or limit < 1 or limit > 10_000:
            raise ToolInputError("limit must be an integer between 1 and 10000")
        try:
            root = context.resolve_execution_path(base) if base else (context.execution_cwd or "/workspace")
        except ValueError as exc:
            raise ToolPermissionError(str(exc)) from exc
        script = """
import base64,glob,json,os,sys
p=json.loads(base64.b64decode(sys.argv[1])); root=p['root']; pattern=p['pattern']; limit=p['limit']
matches=[x for x in glob.glob(os.path.join(root,pattern),recursive=True) if os.path.isfile(x)]
matches.sort(key=lambda x: os.stat(x).st_mtime_ns,reverse=True)
print(json.dumps({'filenames':matches[:limit],'numFiles':min(len(matches),limit),'truncated':len(matches)>limit}))
""".strip()
        output = _backend(context).run_json_helper(
            script, {"root": root, "pattern": pattern, "limit": limit}
        )
        return ToolResult(name="Glob", output=output)


class RemoteGrepTool:
    def spec(self) -> ToolSpec:
        return GrepTool().spec()

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = tool_input["pattern"]
        if not isinstance(pattern, str) or pattern == "":
            raise ToolInputError("pattern must be a non-empty string")
        base = tool_input.get("path")
        glob_pattern = tool_input.get("glob")
        type_name = tool_input.get("type")
        output_mode = tool_input.get("output_mode", "files_with_matches")
        head_limit = tool_input.get("head_limit")
        offset = tool_input.get("offset", 0)
        if base is not None and (not isinstance(base, str) or not base):
            raise ToolInputError("path must be a non-empty string when provided")
        if glob_pattern is not None and not isinstance(glob_pattern, str):
            raise ToolInputError("glob must be a string when provided")
        if type_name is not None and not isinstance(type_name, str):
            raise ToolInputError("type must be a string when provided")
        if output_mode not in {"content", "files_with_matches", "count"}:
            raise ToolInputError("invalid output_mode")
        if head_limit is not None and (not isinstance(head_limit, int) or head_limit < 0):
            raise ToolInputError("head_limit must be an integer >= 0")
        if not isinstance(offset, int) or offset < 0:
            raise ToolInputError("offset must be an integer >= 0")
        try:
            root = context.resolve_execution_path(base) if base else (context.execution_cwd or "/workspace")
        except ValueError as exc:
            raise ToolPermissionError(str(exc)) from exc
        stat = _backend(context).stat(root)
        if not stat.exists:
            raise ToolInputError(f"path does not exist: {root}")
        payload = dict(tool_input)
        payload["root"] = root
        script = r"""
import base64,fnmatch,json,os,re,sys
p=json.loads(base64.b64decode(sys.argv[1])); root=p['root']; pattern=p['pattern']
mode=p.get('output_mode','files_with_matches'); flags=re.MULTILINE
if p.get('-i'): flags|=re.IGNORECASE
if p.get('multiline'): flags|=re.DOTALL
try: regex=re.compile(pattern,flags)
except re.error as e:
 print(json.dumps({'__error__':f'invalid regex: {e}'})); raise SystemExit
paths=[]
if os.path.isfile(root): paths=[root]
else:
 for d,dirs,files in os.walk(root):
  dirs[:]=[x for x in dirs if x not in {'.git','.svn','.hg','.bzr','.jj','.sl'}]
  paths.extend(os.path.join(d,x) for x in files)
globpat=p.get('glob'); typename=p.get('type')
if globpat: paths=[x for x in paths if fnmatch.fnmatch(os.path.basename(x),globpat) or fnmatch.fnmatch(x,globpat)]
if typename: paths=[x for x in paths if os.path.splitext(x)[1].lower().lstrip('.')==typename.lower()]
matched=[]; lines=[]; total=0
for path in paths:
 try:
  text=open(path,encoding='utf-8',errors='replace').read()
 except Exception: continue
 if not regex.search(text): continue
 matched.append(path)
 if mode=='content':
  for n,line in enumerate(text.splitlines(),1):
   found=list(regex.finditer(line))
   if not found: continue
   total+=len(found); prefix=f'{path}:{n}:' if p.get('-n') else f'{path}:'; lines.append(prefix+line)
 elif mode=='count': total+=len(list(regex.finditer(text)))
offset=p.get('offset',0); head=p.get('head_limit'); default=250 if head is None else head
def page(xs):
 if head==0: return xs[offset:],None
 sliced=xs[offset:offset+default]; return sliced,default if len(xs)-offset>default else None
if mode=='content':
 items,lim=page(lines); out={'mode':mode,'numFiles':len(matched),'filenames':matched,'content':'\n'.join(items),'numLines':len(items),'appliedOffset':offset}
elif mode=='count':
 items,lim=page(matched); out={'mode':mode,'numFiles':len(matched),'filenames':items,'numMatches':total,'appliedOffset':offset}
else:
 items,lim=page(matched); out={'mode':mode,'numFiles':len(matched),'filenames':items,'appliedOffset':offset}
if lim is not None: out['appliedLimit']=lim
print(json.dumps(out))
""".strip()
        output = _backend(context).run_json_helper(script, payload)
        if "__error__" in output:
            raise ToolInputError(output["__error__"])
        return ToolResult(name="Grep", output=output)


REMOTE_WORKSPACE_TOOLS = (
    RemoteBashTool,
    RemoteFileReadTool,
    RemoteFileWriteTool,
    RemoteFileEditTool,
    RemoteGlobTool,
    RemoteGrepTool,
)
