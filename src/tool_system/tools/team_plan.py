from __future__ import annotations

import ast
import hashlib
import json
import posixpath
import re
import shlex
import uuid
from typing import Any

from ...teammate.models import AgentRecord, TeamTask
from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


_DEFAULT_TOOLS = ["Read", "Write", "Edit", "Bash"]
_MODULE_NAME = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_EXECUTION_BOUNDS: dict[str, tuple[int, int]] = {
    "max_workers": (1, 16),
    "timeout_s": (1, 86_400),
    "token_budget": (1, 100_000_000),
    "turn_budget": (1, 100_000),
    "max_retries": (0, 10),
    "lease_timeout_s": (5, 86_400),
    "verify_timeout_s": (1, 86_400),
}
_EXECUTION_DEFAULTS: dict[str, int | float | bool | None] = {
    "timeout_s": None,
    "token_budget": None,
    "turn_budget": None,
    "max_retries": 0,
    "lease_timeout_s": 900,
    "verify_timeout_s": 900,
    "auto_verify": True,
}
_JSON_COMPATIBLE_FIELDS: dict[str, type[Any]] = {
    "contract": dict,
    "workers": list,
    "tasks": list,
    "validation": dict,
}
_SOURCE_SUFFIXES = {
    ".py",
    ".pyi",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".rs",
    ".go",
    ".java",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
}
_CEREMONIAL_ROOTS = {
    ".github",
    ".circleci",
    "ci",
    "docs",
    "doc",
    "examples",
    "example",
    "tests",
    "test",
}
_CEREMONIAL_FILES = {
    "readme",
    "license",
    "copying",
    "changelog",
    "authors",
    "contributors",
    "code_of_conduct",
    "contributing",
    "security",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "tox.ini",
    "noxfile.py",
    "mkdocs.yml",
    "mkdocs.yaml",
    "manifest.in",
}


def _issue(
    code: str,
    path: str,
    message: str,
    suggestion: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "code": code,
        "path": path,
        "message": message,
        "suggestion": suggestion,
        **details,
    }


def _non_empty_string(
    value: Any,
    path: str,
    issues: list[dict[str, Any]],
    *,
    required: bool = True,
) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        issues.append(
            _issue(
                "INVALID_STRING",
                path,
                f"{path} must be a non-empty string",
                "Provide a concise non-empty value.",
            )
        )
        return None
    return value.strip()


def _string_list(
    value: Any,
    path: str,
    issues: list[dict[str, Any]],
    *,
    required: bool = False,
) -> list[str]:
    if value is None:
        if required:
            issues.append(
                _issue(
                    "MISSING_LIST",
                    path,
                    f"{path} is required",
                    "Provide an array of non-empty strings.",
                )
            )
        return []
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        issues.append(
            _issue(
                "INVALID_LIST",
                path,
                f"{path} must be an array of non-empty strings",
                "Remove empty entries and submit an array of strings.",
            )
        )
        return []
    return list(dict.fromkeys(item.strip() for item in value))


def _normalize_owned_path(value: str) -> str | None:
    normalized = value.strip().replace("\\", "/")
    if normalized == "/workspace":
        normalized = ""
    elif normalized.startswith("/workspace/"):
        normalized = normalized[len("/workspace/") :]
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = posixpath.normpath(normalized).rstrip("/")
    if normalized in {"", "."}:
        return None
    if normalized.startswith("/") or ".." in normalized.split("/"):
        return None
    if any(mark in normalized for mark in "*?[]"):
        return None
    return normalized


def _paths_overlap(left: str, right: str) -> bool:
    return bool(
        left == right
        or left.startswith(right + "/")
        or right.startswith(left + "/")
    )


def _task_id() -> str:
    return uuid.uuid4().hex[:12]


def _agent_id() -> str:
    return uuid.uuid4().hex[:12]


def _session_id() -> str:
    return uuid.uuid4().hex


def _is_trivial_check(command: str) -> bool:
    normalized = " ".join(command.strip().lower().split())
    if bool(
        normalized in {"true", ":", "ls", "pwd"}
        or re.fullmatch(r"(?:echo|printf)(?:\s+.*)?", normalized)
        or re.fullmatch(r"(?:/bin/)?ls(?:\s+.*)?", normalized)
        or re.fullmatch(r"test\s+-[efd]\s+.*", normalized)
        or re.fullmatch(r"\[\s+-[efd]\s+.*\s+\]", normalized)
    ):
        return True
    shell_control = _unquoted_shell_control(command)
    if "||" in shell_control:
        return True
    if re.search(
        r"(?:;|\n)\s*(?::|true\b|exit\s+0\b)\s*$",
        shell_control,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(r"[;&|\n]", shell_control):
        return False
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    code: str | None = None
    for index, part in enumerate(parts):
        executable = posixpath.basename(part).lower()
        if not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable):
            continue
        try:
            code_flag = parts.index("-c", index + 1)
        except ValueError:
            continue
        code = parts[code_flag + 1] if code_flag + 1 < len(parts) else ""
        break
    return code is not None and _is_trivial_python(code)


def _decode_json_compatible_field(
    value: Any,
    path: str,
    expected_type: type[Any],
    issues: list[dict[str, Any]],
) -> Any:
    """Decode a model-stringified complex field before normal validation.

    This is deliberately only a transport compatibility shim.  Parsed values are
    still passed through every existing semantic validator below.
    """

    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        issues.append(
            _issue(
                "INVALID_JSON_STRING",
                path,
                f"{path} contains invalid JSON: {exc.msg}",
                f"Pass a native {expected_type.__name__} or valid JSON encoding one.",
                line=exc.lineno,
                column=exc.colno,
            )
        )
        return None
    if not isinstance(decoded, expected_type):
        issues.append(
            _issue(
                "INVALID_JSON_VALUE",
                path,
                (
                    f"decoded {path} must be a {expected_type.__name__}, "
                    f"got {type(decoded).__name__}"
                ),
                f"Encode the same shape accepted by the native {path} field.",
            )
        )
        return None
    return decoded


def _python_inline_code(command: str) -> str | None:
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    for index, part in enumerate(parts):
        executable = posixpath.basename(part).lower()
        if not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable):
            continue
        try:
            code_flag = parts.index("-c", index + 1)
        except ValueError:
            continue
        return parts[code_flag + 1] if code_flag + 1 < len(parts) else ""
    return None


def _is_weak_acceptance_check(command: str) -> bool:
    """Return true for import/existence-only Python API smoke checks.

    Focused test runners, compilation commands, behavioral assertions, and public
    signature assertions remain valid.  The check intentionally targets the common
    ceremonial pattern ``import X; assert hasattr(...); assert callable(...)``.
    """

    normalized = " ".join(command.strip().lower().split())
    if re.search(r"(?:^|\s)(?:pytest|unittest)(?:\s|$)", normalized):
        return False
    if re.search(r"python(?:\d+(?:\.\d+)*)?\s+-m\s+(?:pytest|unittest)\b", normalized):
        return False
    code = _python_inline_code(command)
    if code is None:
        return False
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return False

    weak_calls = {"hasattr", "callable", "getattr"}

    def call_name(call: ast.Call) -> str:
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return ""

    assertions = [node for node in module.body if isinstance(node, ast.Assert)]
    if not assertions:
        return all(
            isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign))
            for node in module.body
        )
    for assertion in assertions:
        calls = [node for node in ast.walk(assertion.test) if isinstance(node, ast.Call)]
        names = {call_name(call) for call in calls}
        if "signature" in names:
            return False
        if any(name and name not in weak_calls for name in names):
            return False
        # A comparison against a value/property is behavioral evidence unless its
        # only observations are the weak existence/introspection calls above.
        if isinstance(assertion.test, (ast.Compare, ast.BoolOp, ast.BinOp)) and not calls:
            return False
        if not calls and any(
            isinstance(node, (ast.Attribute, ast.Subscript))
            for node in ast.walk(assertion.test)
        ):
            return False
    return True


def _is_substantive_source_path(value: str) -> bool:
    normalized = value.strip().replace("\\", "/").strip("/")
    if not normalized:
        return False
    parts = [part.casefold() for part in normalized.split("/") if part]
    first = parts[0] if parts else ""
    name = parts[-1] if parts else ""
    stem = name.rsplit(".", 1)[0] if "." in name else name
    if first in _CEREMONIAL_ROOTS:
        return False
    if name in _CEREMONIAL_FILES or stem in _CEREMONIAL_FILES:
        return False
    if name.startswith("requirements") or name.endswith((".lock", ".md", ".rst")):
        return False
    suffix = posixpath.splitext(name)[1].casefold()
    if suffix in _SOURCE_SUFFIXES and name not in {"setup.py", "noxfile.py"}:
        return True
    # A concrete directory such as ``src/pkg`` or ``package`` denotes a source
    # partition unless it is one of the documentation/test/CI roots above.
    return not suffix and not name.startswith(".")


def _unquoted_shell_control(command: str) -> str:
    """Keep shell control text while blanking quoted payloads."""

    output: list[str] = []
    quote: str | None = None
    escaped = False
    for char in command:
        if escaped:
            output.append(" " if quote is not None else char)
            escaped = False
            continue
        if char == "\\" and quote != "'":
            output.append(" " if quote is not None else char)
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            output.append(" ")
            continue
        if char in {"'", '"'}:
            quote = char
            output.append(" ")
            continue
        output.append(char)
    return "".join(output)


def _is_trivial_python(code: str) -> bool:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return False
    if not module.body:
        return True

    def is_zero(value: ast.AST) -> bool:
        return isinstance(value, ast.Constant) and value.value in {None, 0}

    def is_success_exit(call: ast.Call) -> bool:
        function = call.func
        named_exit = isinstance(function, ast.Name) and function.id in {"exit", "quit"}
        sys_exit = (
            isinstance(function, ast.Attribute)
            and function.attr == "exit"
            and isinstance(function.value, ast.Name)
            and function.value.id == "sys"
        )
        return bool(
            (named_exit or sys_exit)
            and not call.keywords
            and (not call.args or (len(call.args) == 1 and is_zero(call.args[0])))
        )

    def is_trivial_statement(statement: ast.stmt) -> bool:
        if isinstance(statement, ast.Pass):
            return True
        if isinstance(statement, ast.Assert):
            # An assertion containing no runtime name/call can only test a value
            # invented by the check itself (for example ``assert True``).
            return not any(
                isinstance(node, (ast.Name, ast.Attribute, ast.Call, ast.Subscript))
                for node in ast.walk(statement.test)
            )
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if isinstance(call.func, ast.Name) and call.func.id == "print":
                return True
            return is_success_exit(call)
        if isinstance(statement, ast.Raise) and isinstance(statement.exc, ast.Call):
            call = statement.exc
            return bool(
                isinstance(call.func, ast.Name)
                and call.func.id == "SystemExit"
                and not call.keywords
                and (not call.args or (len(call.args) == 1 and is_zero(call.args[0])))
            )
        return False

    return all(is_trivial_statement(statement) for statement in module.body)


def _plan_schema() -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mode": {"type": "string", "enum": ["replace"]},
            "expected_revision": {"type": "integer", "minimum": 0},
            "idempotency_key": {"type": "string"},
            "contract": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {"type": "string"},
                    "interfaces": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "provider_task": {"type": "string"},
                                "consumer_tasks": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "signature": {"type": "string"},
                                "mode": {
                                    "type": "string",
                                    "enum": ["frozen", "handoff"],
                                },
                            },
                            "required": [
                                "name",
                                "provider_task",
                                "consumer_tasks",
                                "signature",
                                "mode",
                            ],
                        },
                    },
                },
                "required": ["summary", "interfaces"],
            },
            "workers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                        "instructions": {"type": "string"},
                        "tools": {"type": "array", "items": {"type": "string"}},
                        "model": {"type": "string"},
                        "workspace_mode": {
                            "type": "string",
                            "enum": ["auto", "shared", "worktree"],
                        },
                        "auto_integrate": {"type": "boolean"},
                    },
                    "required": ["name", "instructions"],
                },
            },
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "key": {"type": "string"},
                        "subject": {"type": "string"},
                        "instructions": {"type": "string"},
                        "description": {"type": "string"},
                        "owner": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": ["implementation", "validation"],
                        },
                        "owned_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Concrete deliverable paths owned by this task; include any "
                                "persistent project test files. Teammates receive a separate "
                                "task-private scratch location for disposable self-tests."
                            ),
                        },
                        "acceptance_checks": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Executable shell commands that assert behavior; wrap Python "
                                "expressions with python -c instead of supplying bare Python."
                            ),
                        },
                        "blocked_by": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "metadata": {"type": "object"},
                    },
                    "required": ["key", "owner"],
                },
            },
            "validation": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "profile": {
                        "type": "string",
                        "enum": ["python-package", "generic"],
                    },
                    "install_command": {"type": "string"},
                    "import_command": {"type": "string"},
                    "integration_command": {"type": "string"},
                    "imports": {"type": "array", "items": {"type": "string"}},
                    "commands": {"type": "array", "items": {"type": "string"}},
                },
            },
            "execution": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    **{
                        name: {
                            "type": "number" if name == "timeout_s" else "integer"
                        }
                        for name in _EXECUTION_BOUNDS
                    },
                    "auto_verify": {"type": "boolean"},
                },
            },
        },
        "required": ["contract", "workers", "tasks", "validation"],
    }
    properties = schema["properties"]
    for field in _JSON_COMPATIBLE_FIELDS:
        native_schema = properties[field]
        properties[field] = {
            "oneOf": [
                native_schema,
                {
                    "type": "string",
                    "description": (
                        f"JSON-encoded {field}; decoded and subjected to the same "
                        "semantic validation as the native value."
                    ),
                },
            ]
        }
    return schema


class TeamPlanTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamPlan",
            description=(
                "Atomically replace a strict team's complete worker/task plan. Submit the "
                "contract, workers, tasks, validation, and execution settings together. "
                "The tool normalizes paths and returns structured needs_plan_fix issues "
                "without leaving partial workers or tasks. Frozen interfaces permit "
                "parallel work; handoff interfaces create task dependencies."
            ),
            input_schema=_plan_schema(),
            is_read_only=False,
            max_result_size_chars=200_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.actor_id is not None:
            raise ToolInputError("only the lead may submit a team plan")
        if context.team is None:
            raise ToolInputError("no active team: create a strict team before TeamPlan")

        team_id = str(context.team["team_id"])
        team = context.team_store.load_team(team_id)
        if team is None:
            raise ToolInputError("active team state is unavailable")
        context.reload_team_state()
        current_plan = team.settings.get("team_plan")
        current_plan = current_plan if isinstance(current_plan, dict) else {}
        current_revision = int(current_plan.get("revision") or 0)
        issues: list[dict[str, Any]] = []
        decoded_fields = {
            field: _decode_json_compatible_field(
                tool_input.get(field), field, expected_type, issues
            )
            for field, expected_type in _JSON_COMPATIBLE_FIELDS.items()
        }

        mode = tool_input.get("mode", "replace")
        if mode != "replace":
            issues.append(
                _issue(
                    "UNSUPPORTED_MODE",
                    "mode",
                    "TeamPlan v2 only supports mode='replace'",
                    "Resubmit the complete plan with mode='replace'.",
                )
            )
        expected_revision = tool_input.get("expected_revision")
        if expected_revision is not None and (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            issues.append(
                _issue(
                    "INVALID_REVISION",
                    "expected_revision",
                    "expected_revision must be a non-negative integer",
                    f"Use the current revision, {current_revision}.",
                    current_revision=current_revision,
                )
            )
        idempotency_key = tool_input.get("idempotency_key")
        if idempotency_key is not None:
            idempotency_key = _non_empty_string(
                idempotency_key, "idempotency_key", issues
            )

        busy_tasks = [
            str(task.get("key") or task_id)
            for task_id, task in context.tasks.items()
            if task.get("status") == "in_progress"
        ]
        busy_agents = [
            agent.name
            for agent in context.team_store.list_agents(team_id)
            if agent.status in {"running", "stopping"}
        ]
        if busy_tasks or busy_agents:
            issues.append(
                _issue(
                    "PLAN_BUSY",
                    "mode",
                    "a running plan cannot be replaced",
                    (
                        "Call TeamCancel, wait for every worker to stop, then call "
                        "TeamReplan before replacing the plan. Do not use TeamAbort; "
                        "it is terminal."
                    ),
                    active_tasks=busy_tasks,
                    active_workers=busy_agents,
                )
            )

        execution = self._execution(tool_input.get("execution"), issues)
        validation = self._validation(decoded_fields["validation"], issues)
        workers, worker_by_name = self._workers(
            decoded_fields["workers"], context, issues
        )
        if "max_workers" not in execution:
            distinct_workers = {worker["name"].lower() for worker in workers}
            execution["max_workers"] = min(max(len(distinct_workers), 1), 16)
        task_specs = self._tasks(decoded_fields["tasks"], worker_by_name, issues)
        contract = self._contract(decoded_fields["contract"], task_specs, issues)
        self._apply_contract(contract, task_specs, issues)
        self._validate_task_graph(task_specs, workers, validation, issues)

        if issues:
            return self._needs_fix(current_revision, issues)

        canonical = {
            "mode": "replace",
            "contract": contract,
            "workers": workers,
            "tasks": task_specs,
            "validation": validation,
            "execution": execution,
        }
        encoded = json.dumps(
            canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        plan_hash = hashlib.sha256(encoded).hexdigest()
        contract_hash = hashlib.sha256(
            json.dumps(
                contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        agents, sessions, tasks = self._materialize(
            team_id, workers, task_specs, plan_hash, contract_hash, validation
        )
        architecture_contract = self._architecture_contract(contract)
        quality = {
            "strict": True,
            "configured": True,
            "protocol_version": 2,
            "architecture_contract": architecture_contract,
            "contract": contract,
            "contract_hash": contract_hash,
            "install_command": validation["install_command"],
            "import_command": validation["import_command"],
            "integration_command": validation["integration_command"],
            "validation_profile": validation,
            "plan_accepted": False,
            "validation": {"status": "pending", "reason": "new plan revision"},
        }
        # Protocol v2 has one authoritative execution source: the immutable plan
        # plus its execution_manifest.  Legacy top-level settings are intentionally
        # not populated, because TeamRun/TeamResume previously drifted them away
        # from the accepted plan and caused completed work to lose protocol credit.
        settings_updates = {"quality_gates": quality}
        plan_record = {
            "hash": plan_hash,
            "contract_hash": contract_hash,
            **canonical,
        }

        try:
            stored_team, changed = context.team_store.replace_team_plan(
                team_id,
                tasks=tasks,
                agents=agents,
                sessions=sessions,
                settings_updates=settings_updates,
                plan_record=plan_record,
                expected_revision=expected_revision,
                idempotency_key=idempotency_key,
            )
        except ValueError as exc:
            message = str(exc)
            code = (
                "TEAM_TERMINAL"
                if "terminal" in message
                else "PLAN_BUSY"
                if "running plan cannot be replaced" in message
                else "REPLAN_REQUIRED"
                if "TeamReplan" in message
                else "IDEMPOTENCY_KEY_REUSE"
                if "idempotency_key" in message
                else "REVISION_CONFLICT"
                if "revision" in message
                else "PLAN_COMMIT_CONFLICT"
            )
            return self._needs_fix(
                current_revision,
                [
                    _issue(
                        code,
                        (
                            "idempotency_key"
                            if code == "IDEMPOTENCY_KEY_REUSE"
                            else "mode"
                            if code
                            in {"TEAM_TERMINAL", "REPLAN_REQUIRED", "PLAN_BUSY"}
                            else "expected_revision"
                        ),
                        message,
                        (
                            "Keep the completed/aborted team unchanged for scoring."
                            if code == "TEAM_TERMINAL"
                            else "Stop active workers, then call TeamReplan before replacing the plan."
                            if code == "PLAN_BUSY"
                            else "Call TeamReplan, then submit one complete replacement plan."
                            if code == "REPLAN_REQUIRED"
                            else "Reload the active plan and resubmit against its current revision."
                        ),
                    )
                ],
            )

        persisted = stored_team.settings.get("team_plan") or {}
        revision = int(persisted.get("revision") or current_revision)
        context.reload_team_state()
        if changed:
            context.team_store.append_event(
                team_id,
                "team.plan_committed",
                {
                    "revision": revision,
                    "plan_hash": plan_hash,
                    "protocol_version": 2,
                    "worker_count": len(agents),
                    "task_count": len(tasks),
                },
            )
            carry_forward = stored_team.settings.get("last_plan_carry_forward")
            if (
                isinstance(carry_forward, dict)
                and int(carry_forward.get("plan_revision") or 0) == revision
                and carry_forward.get("tasks")
            ):
                context.team_store.append_event(
                    team_id,
                    "team.tasks_carried_forward",
                    {
                        "from_revision": carry_forward.get("from_revision"),
                        "plan_revision": revision,
                        "tasks": carry_forward["tasks"],
                        "requires_acceptance": True,
                    },
                )
        carry_forward = stored_team.settings.get("last_plan_carry_forward")
        carry_forward = carry_forward if isinstance(carry_forward, dict) else {}
        return ToolResult(
            name="TeamPlan",
            output={
                "status": "ready",
                "team_id": team_id,
                "protocol_version": 2,
                "revision": revision,
                "plan_hash": str(persisted.get("hash") or plan_hash),
                "idempotent": not changed,
                "workers": [
                    {
                        "name": agent.name,
                        "agent_id": agent.agent_id,
                        "workspace_mode": agent.workspace_mode,
                    }
                    for agent in context.team_store.list_agents(team_id)
                ],
                "tasks": [
                    {
                        "key": task.get("key"),
                        "id": task_id,
                        "owner": task.get("owner"),
                        "blocked_by": list(task.get("blockedBy") or []),
                        "owned_files": list(task.get("owned_files") or []),
                    }
                    for task_id, task in context.tasks.items()
                ],
                "contract": persisted.get("contract") or contract,
                "validation": persisted.get("validation") or validation,
                "execution": persisted.get("execution") or execution,
                "carried_forward_tasks": list(carry_forward.get("tasks") or []),
                "next_required_actions": [
                    {
                        "tool": "TeamRun",
                        "instruction": (
                            "Run the committed plan. The harness owns final verification."
                        ),
                    }
                ],
            },
        )

    @staticmethod
    def _needs_fix(
        revision: int, issues: list[dict[str, Any]]
    ) -> ToolResult:
        return ToolResult(
            name="TeamPlan",
            output={
                "status": "needs_plan_fix",
                "revision": revision,
                "issues": issues,
                "next_required_action": (
                    "Fix every issue and replace the complete plan in one TeamPlan call."
                ),
            },
            is_error=True,
        )

    @staticmethod
    def _execution(
        value: Any, issues: list[dict[str, Any]]
    ) -> dict[str, int | float | bool | None]:
        if value is None:
            return dict(_EXECUTION_DEFAULTS)
        if not isinstance(value, dict):
            issues.append(
                _issue(
                    "INVALID_EXECUTION",
                    "execution",
                    "execution must be an object",
                    "Provide only supported TeamRun numeric options.",
                )
            )
            return dict(_EXECUTION_DEFAULTS)
        output: dict[str, int | float | bool | None] = dict(_EXECUTION_DEFAULTS)
        for name, raw in value.items():
            if name == "auto_verify":
                if raw is not True:
                    issues.append(
                        _issue(
                            "AUTO_VERIFY_REQUIRED",
                            "execution.auto_verify",
                            "protocol v2 always performs harness-owned verification",
                            "Set auto_verify=true or omit it.",
                        )
                    )
                continue
            if name not in _EXECUTION_BOUNDS:
                issues.append(
                    _issue(
                        "UNKNOWN_EXECUTION_OPTION",
                        f"execution.{name}",
                        f"unknown execution option {name!r}",
                        "Remove this option.",
                    )
                )
                continue
            minimum, maximum = _EXECUTION_BOUNDS[name]
            valid_type = isinstance(raw, (int, float)) and not isinstance(raw, bool)
            if name != "timeout_s":
                valid_type = isinstance(raw, int) and not isinstance(raw, bool)
            if not valid_type or raw < minimum or raw > maximum:
                issues.append(
                    _issue(
                        "INVALID_EXECUTION_OPTION",
                        f"execution.{name}",
                        f"{name} must be between {minimum} and {maximum}",
                        "Choose a value within the supported bounds.",
                    )
                )
                continue
            output[name] = raw
        return output

    @staticmethod
    def _validation(value: Any, issues: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(value, dict):
            issues.append(
                _issue(
                    "INVALID_VALIDATION",
                    "validation",
                    "validation must be an object",
                    "Provide a python-package or generic validation profile.",
                )
            )
            value = {}
        profile = value.get("profile", "python-package")
        if profile not in {"python-package", "generic"}:
            issues.append(
                _issue(
                    "INVALID_VALIDATION_PROFILE",
                    "validation.profile",
                    "profile must be 'python-package' or 'generic'",
                    "Use the profile matching the repository.",
                )
            )
            profile = "python-package"
        imports = _string_list(value.get("imports"), "validation.imports", issues)
        commands = _string_list(value.get("commands"), "validation.commands", issues)
        for index, module in enumerate(imports):
            if not _MODULE_NAME.fullmatch(module):
                issues.append(
                    _issue(
                        "INVALID_IMPORT",
                        f"validation.imports[{index}]",
                        f"invalid Python module name {module!r}",
                        "Use a dotted import name such as package.submodule.",
                    )
                )
        for index, command in enumerate(commands):
            if _is_trivial_check(command):
                issues.append(
                    _issue(
                        "TRIVIAL_VALIDATION_CHECK",
                        f"validation.commands[{index}]",
                        f"validation command {command!r} does not verify behavior",
                        "Run a real import, API, integration, or test-suite assertion.",
                    )
                )
        install = value.get("install_command")
        import_command = value.get("import_command")
        integration = value.get("integration_command")
        if profile == "python-package":
            install = install or "python -m pip install -e . --no-deps --no-build-isolation"
            if not import_command and imports:
                import_command = "python -c " + shlex.quote(
                    "; ".join(f"import {module}" for module in imports)
                )
            integration = integration or (
                " && ".join(commands) if commands else "python -m pytest -q"
            )
        else:
            install = install or "true"
            import_command = import_command or "true"
            integration = integration or (" && ".join(commands) if commands else None)
        install = _non_empty_string(
            install, "validation.install_command", issues
        )
        import_command = _non_empty_string(
            import_command, "validation.import_command", issues
        )
        integration = _non_empty_string(
            integration, "validation.integration_command", issues
        )
        if integration and _is_trivial_check(integration):
            issues.append(
                _issue(
                    "TRIVIAL_VALIDATION_CHECK",
                    "validation.integration_command",
                    f"integration command {integration!r} does not verify behavior",
                    "Run a real integration assertion or repository test suite.",
                )
            )
        return {
            "profile": profile,
            "imports": imports,
            "commands": commands,
            "install_command": install or "",
            "import_command": import_command or "",
            "integration_command": integration or "",
        }

    @staticmethod
    def _workers(
        value: Any,
        context: ToolContext,
        issues: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        if not isinstance(value, list):
            issues.append(
                _issue(
                    "INVALID_WORKERS",
                    "workers",
                    "workers must be an array",
                    "Provide at least two task-specific workers.",
                )
            )
            value = []
        if len(value) < 2:
            issues.append(
                _issue(
                    "MIN_WORKERS",
                    "workers",
                    "strict TeamPlan requires at least two workers",
                    "Create at least two workers with distinct owned tasks.",
                )
            )
        workers: list[dict[str, Any]] = []
        by_name: dict[str, dict[str, Any]] = {}
        for index, raw in enumerate(value):
            path = f"workers[{index}]"
            if not isinstance(raw, dict):
                issues.append(
                    _issue(
                        "INVALID_WORKER",
                        path,
                        "worker must be an object",
                        "Replace it with a worker definition.",
                    )
                )
                continue
            name = _non_empty_string(raw.get("name"), f"{path}.name", issues)
            instructions = _non_empty_string(
                raw.get("instructions"), f"{path}.instructions", issues
            )
            role = _non_empty_string(
                raw.get("role", "implementation"), f"{path}.role", issues
            )
            if name is None:
                continue
            normalized_name = name.lower()
            if normalized_name in by_name:
                issues.append(
                    _issue(
                        "DUPLICATE_WORKER",
                        f"{path}.name",
                        f"worker name {name!r} is duplicated",
                        "Use a unique worker name.",
                    )
                )
                continue
            tools = _string_list(raw.get("tools", _DEFAULT_TOOLS), f"{path}.tools", issues)
            validate_tools = getattr(context.teammate_runtime, "validate_tools", None)
            if callable(validate_tools) and tools:
                try:
                    tools = validate_tools(tools)
                except ValueError as exc:
                    issues.append(
                        _issue(
                            "INVALID_WORKER_TOOLS",
                            f"{path}.tools",
                            str(exc),
                            "Use only tools available to teammate workers.",
                        )
                    )
            model = raw.get("model")
            if model is not None:
                model = _non_empty_string(model, f"{path}.model", issues)
            validate_model = getattr(context.teammate_runtime, "validate_model", None)
            if callable(validate_model):
                try:
                    model = validate_model(model)
                except ValueError as exc:
                    issues.append(
                        _issue(
                            "INVALID_WORKER_MODEL",
                            f"{path}.model",
                            str(exc),
                            "Omit model to inherit the lead endpoint model.",
                        )
                    )
            requested_workspace = raw.get("workspace_mode", "auto")
            workspace_mode = "shared" if requested_workspace == "auto" else requested_workspace
            if workspace_mode == "worktree" and context.workspace_backend is not None:
                workspace_mode = "shared"
            elif workspace_mode == "worktree":
                issues.append(
                    _issue(
                        "WORKTREE_REQUIRES_INCREMENTAL_SETUP",
                        f"{path}.workspace_mode",
                        "atomic TeamPlan does not create local git worktrees",
                        "Use workspace_mode='auto' or 'shared'.",
                    )
                )
                workspace_mode = "shared"
            if workspace_mode not in {"shared", "worktree"}:
                issues.append(
                    _issue(
                        "INVALID_WORKSPACE_MODE",
                        f"{path}.workspace_mode",
                        "workspace_mode must be auto, shared, or worktree",
                        "Use auto so the harness selects a supported mode.",
                    )
                )
                workspace_mode = "shared"
            auto_integrate = bool(raw.get("auto_integrate", False))
            if auto_integrate and workspace_mode != "worktree":
                issues.append(
                    _issue(
                        "INVALID_AUTO_INTEGRATE",
                        f"{path}.auto_integrate",
                        "auto_integrate requires a worktree workspace",
                        "Disable auto_integrate for shared/AGS workspaces.",
                    )
                )
            worker = {
                "name": name,
                "role": role or "implementation",
                "instructions": instructions or "",
                "tools": tools,
                "model": model,
                "workspace_mode": workspace_mode,
                "auto_integrate": auto_integrate,
            }
            workers.append(worker)
            by_name[normalized_name] = worker
        return workers, by_name

    @staticmethod
    def _tasks(
        value: Any,
        workers: dict[str, dict[str, Any]],
        issues: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            issues.append(
                _issue(
                    "INVALID_TASKS",
                    "tasks",
                    "tasks must be an array",
                    "Provide at least two implementation tasks.",
                )
            )
            value = []
        task_specs: list[dict[str, Any]] = []
        keys: set[str] = set()
        for index, raw in enumerate(value):
            path = f"tasks[{index}]"
            if not isinstance(raw, dict):
                issues.append(
                    _issue(
                        "INVALID_TASK",
                        path,
                        "task must be an object",
                        "Replace it with a task definition.",
                    )
                )
                continue
            key = _non_empty_string(raw.get("key"), f"{path}.key", issues)
            owner = _non_empty_string(raw.get("owner"), f"{path}.owner", issues)
            description_value = raw.get("instructions", raw.get("description"))
            description = _non_empty_string(
                description_value, f"{path}.instructions", issues
            )
            subject = _non_empty_string(
                raw.get("subject", key or "task"), f"{path}.subject", issues
            )
            if key is None or owner is None:
                continue
            normalized_key = key.lower()
            if normalized_key in keys:
                issues.append(
                    _issue(
                        "DUPLICATE_TASK",
                        f"{path}.key",
                        f"task key {key!r} is duplicated",
                        "Use a stable unique task key.",
                    )
                )
                continue
            keys.add(normalized_key)
            worker = workers.get(owner.lower())
            if worker is None:
                issues.append(
                    _issue(
                        "UNKNOWN_OWNER",
                        f"{path}.owner",
                        f"unknown worker {owner!r}",
                        "Choose a name declared in workers.",
                    )
                )
            raw_owned = _string_list(raw.get("owned_files"), f"{path}.owned_files", issues)
            owned_files: list[str] = []
            for owned_index, owned in enumerate(raw_owned):
                normalized = _normalize_owned_path(owned)
                if normalized is None:
                    issues.append(
                        _issue(
                            "INVALID_OWNED_PATH",
                            f"{path}.owned_files[{owned_index}]",
                            f"owned path {owned!r} is not a concrete workspace path",
                            "Use a relative concrete path without '..' or glob syntax.",
                        )
                    )
                elif normalized not in owned_files:
                    owned_files.append(normalized)
            kind = raw.get("kind", "implementation")
            if kind not in {"implementation", "validation"}:
                issues.append(
                    _issue(
                        "INVALID_TASK_KIND",
                        f"{path}.kind",
                        "kind must be implementation or validation",
                        "Use validation only for a read-only integration task.",
                    )
                )
                kind = "implementation"
            if kind != "validation" and not owned_files:
                issues.append(
                    _issue(
                        "MISSING_OWNED_FILES",
                        f"{path}.owned_files",
                        f"implementation task {key!r} has no owned files",
                        "Declare the concrete files or directories this task owns.",
                    )
                )
            if (
                kind == "implementation"
                and owned_files
                and not any(_is_substantive_source_path(path) for path in owned_files)
            ):
                issues.append(
                    _issue(
                        "CEREMONIAL_IMPLEMENTATION_TASK",
                        f"{path}.owned_files",
                        (
                            f"implementation task {key!r} owns only documentation, tests, "
                            "CI, examples, or packaging metadata"
                        ),
                        (
                            "Assign this worker a real source partition (.py/.pyi or another "
                            "core source file/directory), or mark the task as validation."
                        ),
                    )
                )
            acceptance = _string_list(
                raw.get("acceptance_checks"), f"{path}.acceptance_checks", issues
            )
            if kind == "implementation" and not acceptance:
                issues.append(
                    _issue(
                        "MISSING_ACCEPTANCE_CHECKS",
                        f"{path}.acceptance_checks",
                        f"implementation task {key!r} has no behavioral acceptance check",
                        "Add a focused compile, import, API, or test assertion for this task.",
                    )
                )
            for check_index, check in enumerate(acceptance):
                if _is_trivial_check(check):
                    issues.append(
                        _issue(
                            "TRIVIAL_ACCEPTANCE_CHECK",
                            f"{path}.acceptance_checks[{check_index}]",
                            f"acceptance check {check!r} only confirms a happy path or file presence",
                            "Assert compilation, an API contract, integration behavior, or focused tests.",
                        )
                    )
            if (
                kind == "implementation"
                and acceptance
                and all(_is_weak_acceptance_check(check) for check in acceptance)
            ):
                issues.append(
                    _issue(
                        "WEAK_ACCEPTANCE_CHECK",
                        f"{path}.acceptance_checks",
                        (
                            f"implementation task {key!r} only checks imports or API "
                            "existence (hasattr/callable)"
                        ),
                        (
                            "Add a behavioral assertion, a public API signature assertion, "
                            "or a focused pytest/unittest command."
                        ),
                    )
                )
            metadata = raw.get("metadata", {})
            if not isinstance(metadata, dict):
                issues.append(
                    _issue(
                        "INVALID_METADATA",
                        f"{path}.metadata",
                        "metadata must be an object",
                        "Use a JSON object for optional task metadata.",
                    )
                )
                metadata = {}
            blocked_by = _string_list(
                raw.get("blocked_by"), f"{path}.blocked_by", issues
            )
            task_specs.append(
                {
                    "key": key,
                    "subject": subject or key,
                    "description": description or "",
                    "owner": worker["name"] if worker is not None else owner,
                    "kind": kind,
                    "owned_files": owned_files,
                    "acceptance_checks": acceptance,
                    "blocked_by": blocked_by,
                    "provides_interfaces": [],
                    "depends_on_interfaces": [],
                    "metadata": {**metadata, "task_type": kind},
                }
            )
        return task_specs

    @staticmethod
    def _contract(
        value: Any,
        tasks: list[dict[str, Any]],
        issues: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            issues.append(
                _issue(
                    "INVALID_CONTRACT",
                    "contract",
                    "contract must be an object",
                    "Provide a summary and an interfaces array.",
                )
            )
            value = {}
        summary = _non_empty_string(value.get("summary"), "contract.summary", issues)
        raw_interfaces = value.get("interfaces", [])
        if not isinstance(raw_interfaces, list):
            issues.append(
                _issue(
                    "INVALID_INTERFACES",
                    "contract.interfaces",
                    "interfaces must be an array",
                    "Provide zero or more frozen/handoff interface objects.",
                )
            )
            raw_interfaces = []
        task_keys = {task["key"].lower(): task["key"] for task in tasks}
        interfaces: list[dict[str, Any]] = []
        names: set[str] = set()
        for index, raw in enumerate(raw_interfaces):
            path = f"contract.interfaces[{index}]"
            if not isinstance(raw, dict):
                issues.append(
                    _issue(
                        "INVALID_INTERFACE",
                        path,
                        "interface must be an object",
                        "Provide its name, signature, provider, consumers, and mode.",
                    )
                )
                continue
            name = _non_empty_string(raw.get("name"), f"{path}.name", issues)
            signature = _non_empty_string(
                raw.get("signature"), f"{path}.signature", issues
            )
            provider = _non_empty_string(
                raw.get("provider_task"), f"{path}.provider_task", issues
            )
            consumers = _string_list(
                raw.get("consumer_tasks"), f"{path}.consumer_tasks", issues, required=True
            )
            mode = raw.get("mode")
            if mode not in {"frozen", "handoff"}:
                issues.append(
                    _issue(
                        "INVALID_INTERFACE_MODE",
                        f"{path}.mode",
                        "interface mode must be frozen or handoff",
                        "Use frozen for parallel implementation or handoff for an artifact dependency.",
                    )
                )
                mode = "frozen"
            if name is None or provider is None:
                continue
            normalized_name = name.lower()
            if normalized_name in names:
                issues.append(
                    _issue(
                        "DUPLICATE_INTERFACE",
                        f"{path}.name",
                        f"interface {name!r} is duplicated",
                        "Define each interface exactly once.",
                    )
                )
                continue
            names.add(normalized_name)
            provider_key = task_keys.get(provider.lower())
            if provider_key is None:
                issues.append(
                    _issue(
                        "UNKNOWN_PROVIDER_TASK",
                        f"{path}.provider_task",
                        f"unknown provider task {provider!r}",
                        "Use a key declared in tasks.",
                    )
                )
                provider_key = provider
            canonical_consumers: list[str] = []
            for consumer_index, consumer in enumerate(consumers):
                consumer_key = task_keys.get(consumer.lower())
                if consumer_key is None:
                    issues.append(
                        _issue(
                            "UNKNOWN_CONSUMER_TASK",
                            f"{path}.consumer_tasks[{consumer_index}]",
                            f"unknown consumer task {consumer!r}",
                            "Use a key declared in tasks.",
                        )
                    )
                    continue
                if consumer_key == provider_key:
                    issues.append(
                        _issue(
                            "SELF_INTERFACE_DEPENDENCY",
                            f"{path}.consumer_tasks[{consumer_index}]",
                            f"task {consumer_key!r} cannot consume its own interface",
                            "Remove the provider from consumer_tasks.",
                        )
                    )
                    continue
                if consumer_key not in canonical_consumers:
                    canonical_consumers.append(consumer_key)
            interfaces.append(
                {
                    "name": name,
                    "signature": signature or "",
                    "provider_task": provider_key,
                    "consumer_tasks": canonical_consumers,
                    "mode": mode,
                }
            )
        return {"summary": summary or "", "interfaces": interfaces}

    @staticmethod
    def _apply_contract(
        contract: dict[str, Any],
        tasks: list[dict[str, Any]],
        issues: list[dict[str, Any]],
    ) -> None:
        by_key = {task["key"].lower(): task for task in tasks}
        for interface in contract.get("interfaces", []):
            provider = by_key.get(str(interface["provider_task"]).lower())
            if provider is None:
                continue
            name = str(interface["name"])
            provider["provides_interfaces"].append(name)
            provider["metadata"].setdefault("interface_contracts", {})[name] = {
                "role": "provider",
                "mode": interface["mode"],
                "signature": interface["signature"],
            }
            for consumer_key in interface.get("consumer_tasks", []):
                consumer = by_key.get(str(consumer_key).lower())
                if consumer is None:
                    continue
                consumer["depends_on_interfaces"].append(name)
                consumer["metadata"].setdefault("interface_contracts", {})[name] = {
                    "role": "consumer",
                    "mode": interface["mode"],
                    "signature": interface["signature"],
                    "provider_task": provider["key"],
                }
                if (
                    interface["mode"] == "handoff"
                    and provider["key"] not in consumer["blocked_by"]
                ):
                    consumer["blocked_by"].append(provider["key"])

    @staticmethod
    def _validate_task_graph(
        tasks: list[dict[str, Any]],
        workers: list[dict[str, Any]],
        validation: dict[str, Any],
        issues: list[dict[str, Any]],
    ) -> None:
        by_key = {task["key"].lower(): task for task in tasks}
        implementation = [task for task in tasks if task["kind"] == "implementation"]
        if len(implementation) < 2:
            issues.append(
                _issue(
                    "MIN_IMPLEMENTATION_TASKS",
                    "tasks",
                    "strict TeamPlan requires at least two implementation tasks",
                    "Split real implementation work across at least two workers.",
                )
            )
        implementation_owners = {
            task["owner"].lower()
            for task in implementation
            if task["owned_files"]
            and task["acceptance_checks"]
            and any(
                _is_substantive_source_path(path) for path in task["owned_files"]
            )
            and not all(
                _is_weak_acceptance_check(check)
                for check in task["acceptance_checks"]
            )
        }
        if len(implementation_owners) < 2:
            issues.append(
                _issue(
                    "MIN_IMPLEMENTATION_OWNERS",
                    "tasks",
                    "real implementation work is not owned by two distinct workers",
                    (
                        "Give at least two workers non-overlapping source partitions "
                        "and behavioral acceptance_checks."
                    ),
                )
            )
        worker_owners = {task["owner"].lower() for task in tasks}
        for index, worker in enumerate(workers):
            if worker["name"].lower() not in worker_owners:
                issues.append(
                    _issue(
                        "UNASSIGNED_WORKER",
                        f"workers[{index}].name",
                        f"worker {worker['name']!r} owns no task",
                        "Remove the ceremonial worker or assign it real work.",
                    )
                )
        for index, task in enumerate(tasks):
            canonical_dependencies: list[str] = []
            for dep_index, identity in enumerate(task["blocked_by"]):
                dependency = by_key.get(identity.lower())
                if dependency is None:
                    issues.append(
                        _issue(
                            "UNKNOWN_TASK_DEPENDENCY",
                            f"tasks[{index}].blocked_by[{dep_index}]",
                            f"unknown task dependency {identity!r}",
                            "Use a key declared in tasks.",
                        )
                    )
                    continue
                if dependency["key"] == task["key"]:
                    issues.append(
                        _issue(
                            "SELF_TASK_DEPENDENCY",
                            f"tasks[{index}].blocked_by[{dep_index}]",
                            "a task cannot block on itself",
                            "Remove this dependency.",
                        )
                    )
                    continue
                if dependency["key"] not in canonical_dependencies:
                    canonical_dependencies.append(dependency["key"])
            task["blocked_by"] = canonical_dependencies
            if task["kind"] == "validation" and not task["acceptance_checks"]:
                task["acceptance_checks"] = [validation["integration_command"]]

        paths: list[tuple[int, dict[str, Any], str]] = []
        for index, task in enumerate(tasks):
            for owned in task["owned_files"]:
                paths.append((index, task, owned))
        for position, (left_index, left_task, left_path) in enumerate(paths):
            for right_index, right_task, right_path in paths[position + 1 :]:
                if left_task["key"] == right_task["key"]:
                    continue
                if left_task["owner"].lower() == right_task["owner"].lower():
                    continue
                if _paths_overlap(left_path, right_path):
                    issues.append(
                        _issue(
                            "PATH_OVERLAP",
                            f"tasks[{right_index}].owned_files",
                            (
                                f"cross-owner path overlap: {left_path!r} owned by "
                                f"{left_task['owner']!r}, {right_path!r} owned by "
                                f"{right_task['owner']!r}"
                            ),
                            "Split ownership into non-overlapping concrete paths.",
                            conflicts_with=f"tasks[{left_index}].owned_files",
                        )
                    )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(key: str, trail: list[str]) -> None:
            if key in visited:
                return
            if key in visiting:
                cycle = trail[trail.index(key) :] + [key]
                issues.append(
                    _issue(
                        "TASK_DEPENDENCY_CYCLE",
                        "tasks",
                        "task dependency cycle: " + " -> ".join(cycle),
                        "Change at least one handoff to frozen or remove a dependency.",
                    )
                )
                return
            visiting.add(key)
            task = by_key.get(key)
            if task is not None:
                for dependency in task["blocked_by"]:
                    visit(dependency.lower(), [*trail, dependency.lower()])
            visiting.remove(key)
            visited.add(key)

        for key in by_key:
            visit(key, [key])
        ready_owners = {
            task["owner"].lower()
            for task in implementation
            if not task["blocked_by"]
        }
        if len(ready_owners) < 2:
            issues.append(
                _issue(
                    "INSUFFICIENT_PARALLEL_START",
                    "tasks",
                    "fewer than two distinct workers have initially ready tasks",
                    "Use frozen contracts for work that can start from a shared signature.",
                )
            )

    @staticmethod
    def _materialize(
        team_id: str,
        workers: list[dict[str, Any]],
        task_specs: list[dict[str, Any]],
        plan_hash: str,
        contract_hash: str,
        validation: dict[str, Any],
    ) -> tuple[list[AgentRecord], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        agents: list[AgentRecord] = []
        sessions: dict[str, dict[str, Any]] = {}
        agent_by_name: dict[str, AgentRecord] = {}
        for worker in workers:
            agent = AgentRecord(
                agent_id=_agent_id(),
                team_id=team_id,
                name=worker["name"],
                role=worker["role"],
                session_id=_session_id(),
                model=worker["model"],
                instructions=worker["instructions"],
                tools=list(worker["tools"]),
                workspace_mode=worker["workspace_mode"],
                auto_integrate=worker["auto_integrate"],
            )
            agents.append(agent)
            agent_by_name[agent.name.lower()] = agent
            sessions[agent.session_id] = {
                "session_id": agent.session_id,
                "team_id": team_id,
                "agent_id": agent.agent_id,
                "model": agent.model,
                "conversation": {"messages": [], "max_history": 300},
            }
        task_id_by_key = {task["key"].lower(): _task_id() for task in task_specs}
        tasks: dict[str, dict[str, Any]] = {}
        for task_spec in task_specs:
            task_id = task_id_by_key[task_spec["key"].lower()]
            dependency_ids = [
                task_id_by_key[identity.lower()]
                for identity in task_spec["blocked_by"]
            ]
            task = TeamTask(
                id=task_id,
                key=task_spec["key"],
                subject=task_spec["subject"],
                description=task_spec["description"],
                owner=agent_by_name[task_spec["owner"].lower()].agent_id,
                blockedBy=dependency_ids,
                metadata={
                    **task_spec["metadata"],
                    "plan_hash": plan_hash,
                    "contract_hash": contract_hash,
                    "task_contract_fingerprint": (
                        TeamPlanTool._task_contract_fingerprint(
                            task_spec, contract_hash
                        )
                    ),
                    "validation_profile": validation["profile"],
                },
                owned_files=list(task_spec["owned_files"]),
                provides_interfaces=list(task_spec["provides_interfaces"]),
                depends_on_interfaces=list(task_spec["depends_on_interfaces"]),
                acceptance_checks=list(task_spec["acceptance_checks"]),
            )
            tasks[task_id] = task.to_dict()
        for task in tasks.values():
            for dependency_id in task["blockedBy"]:
                tasks[dependency_id]["blocks"].append(task["id"])
        return agents, sessions, tasks

    @staticmethod
    def _task_contract_fingerprint(
        task_spec: dict[str, Any], contract_hash: str
    ) -> str:
        """Hash only the stable artifact contract, never revision/runtime state."""

        payload = {
            "schema_version": 1,
            "contract_hash": contract_hash,
            "key": str(task_spec["key"]),
            "subject": str(task_spec["subject"]),
            "description": str(task_spec["description"]),
            "kind": str(task_spec["kind"]),
            "owned_files": sorted(str(path) for path in task_spec["owned_files"]),
            "acceptance_checks": [
                str(command) for command in task_spec["acceptance_checks"]
            ],
            "blocked_by": sorted(
                str(identity).lower() for identity in task_spec["blocked_by"]
            ),
            "provides_interfaces": sorted(
                str(name) for name in task_spec["provides_interfaces"]
            ),
            "depends_on_interfaces": sorted(
                str(name) for name in task_spec["depends_on_interfaces"]
            ),
            "metadata": task_spec["metadata"],
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _architecture_contract(contract: dict[str, Any]) -> str:
        lines = [str(contract["summary"])]
        for interface in contract.get("interfaces", []):
            consumers = ", ".join(interface["consumer_tasks"]) or "none"
            lines.append(
                f"- {interface['name']}: {interface['signature']} "
                f"[{interface['mode']}] {interface['provider_task']} -> {consumers}"
            )
        return "\n".join(lines)
