from __future__ import annotations

import os
import posixpath
import re
import shlex
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterator

from .errors import ToolPermissionError

if TYPE_CHECKING:
    from .context import ToolContext


_IGNORED_DIRECTORY_NAMES = {
    ".cache",
    ".eggs",
    ".git",
    ".gradle",
    ".hypothesis",
    ".mypy_cache",
    ".npm",
    ".nyc_output",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "htmlcov",
    "node_modules",
    "venv",
}
_IGNORED_FILE_NAMES = {
    ".coverage",
    ".DS_Store",
    "coverage.xml",
    "lcov.info",
}
_IGNORED_FILE_PREFIXES = {".coverage."}
_IGNORED_FILE_SUFFIXES = {
    ".bak",
    ".db",
    ".db-journal",
    ".db-shm",
    ".db-wal",
    ".log",
    ".orig",
    ".pyc",
    ".pyo",
    ".rej",
    ".sqlite",
    ".sqlite-journal",
    ".sqlite-shm",
    ".sqlite-wal",
    ".sqlite3",
    ".swo",
    ".swp",
    ".temp",
    ".tmp",
    "~",
}
_INTEGRATION_FILE_NAMES = {
    "androidmanifest.xml",
    "cargo.lock",
    "cargo.toml",
    "composer.json",
    "composer.lock",
    "gemfile",
    "gemfile.lock",
    "go.mod",
    "go.sum",
    "manifest.in",
    "manifest.json",
    "manifest.yaml",
    "manifest.yml",
    "package-lock.json",
    "package.json",
    "pnpm-lock.yaml",
    "pom.xml",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "yarn.lock",
}
_INTEGRATOR_ROLES = {
    "integration",
    "integration_owner",
    "integrator",
    "lead_integrator",
}
_TASK_TEST_ROOT = ".clawd/task-tests"
_CONTROL_ROOT = ".clawd"
_GENERATED_TEST_PATHS_ATTRIBUTE = "_ownership_generated_test_paths"


def protocol_version(context: ToolContext) -> int:
    team = context.team or {}
    settings = team.get("settings") if isinstance(team.get("settings"), dict) else {}
    quality = (
        settings.get("quality_gates")
        if isinstance(settings.get("quality_gates"), dict)
        else {}
    )
    versions = [1]
    for value in (
        team.get("protocol_version"),
        settings.get("protocol_version"),
        quality.get("protocol_version"),
    ):
        try:
            versions.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(versions)


def ownership_enforced(context: ToolContext) -> bool:
    """Whether this is a protocol-v2 teammate task with scoped writes."""

    return bool(
        context.actor_id
        and context.current_task_id
        and context.team
        and protocol_version(context) >= 2
    )


def strict_protocol_v2(context: ToolContext) -> bool:
    """Whether the active team uses strict protocol-v2 safety rules."""

    team = context.team or {}
    settings = team.get("settings") if isinstance(team.get("settings"), dict) else {}
    quality = (
        settings.get("quality_gates")
        if isinstance(settings.get("quality_gates"), dict)
        else {}
    )
    return bool(quality.get("strict") and protocol_version(context) >= 2)


def bash_audit_required(context: ToolContext) -> bool:
    """Audit teammate mutations and strict-v2 lead control-state mutations."""

    return ownership_enforced(context) or strict_protocol_v2(context)


def _is_control_path(relative_path: str) -> bool:
    candidate = _normalize_owned_path(relative_path)
    return candidate == _CONTROL_ROOT or candidate.startswith(_CONTROL_ROOT + "/")


def _normalize_owned_path(
    value: str, *, workspace_roots: tuple[str, ...] = ()
) -> str:
    normalized = value.strip().replace("\\", "/")
    roots = [*workspace_roots, "/workspace"]
    if normalized.startswith("/"):
        for root in roots:
            canonical_root = str(root or "").strip().replace("\\", "/").rstrip("/")
            if not canonical_root:
                continue
            if normalized == canonical_root:
                normalized = ""
                break
            if normalized.startswith(canonical_root + "/"):
                normalized = normalized[len(canonical_root) + 1 :]
                break
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def owned_paths(context: ToolContext) -> tuple[str, ...]:
    if not ownership_enforced(context):
        return ()
    task = context.tasks.get(str(context.current_task_id))
    if not isinstance(task, dict):
        return ()
    normalized = [
        _normalize_owned_path(
            str(value),
            workspace_roots=(
                str(context.workspace_root),
                str(context.execution_workspace_root or ""),
            ),
        )
        for value in (task.get("owned_files") or [])
    ]
    return tuple(path for path in normalized if path)


def task_test_scratch_prefix_for_id(task_id: str | None) -> str:
    """Return the task-private location for disposable teammate tests.

    Protocol-v2 tasks must explicitly own every deliverable file.  Focused tests that
    exist only to validate one teammate's implementation are different: forcing the
    lead to predict their names creates needless repair loops, while allowing arbitrary
    ``test_*.py`` writes would let a teammate modify project or evaluator tests.  Each
    task therefore gets one isolated, non-deliverable scratch subtree.
    """

    raw_task_id = str(task_id or "unknown")
    safe_task_id = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_task_id).strip("._-")
    return f"{_TASK_TEST_ROOT}/{safe_task_id or 'unknown'}"


def task_test_scratch_prefix(context: ToolContext) -> str:
    return task_test_scratch_prefix_for_id(context.current_task_id)


class ControlStateBackup:
    """Out-of-workspace backup used to roll back unauthorized Bash mutations.

    Teammate commands back up the auditable workspace, excluding VCS metadata and
    generated/cache directories.  Strict-v2 lead commands need only the ``.clawd``
    control tree because the lead otherwise owns integration.  After the command,
    only unauthorized changed paths are restored, so legal task work is retained.
    """

    def __init__(self, context: ToolContext) -> None:
        self.context = context
        self.remote = context.workspace_backend is not None
        self.full_workspace = ownership_enforced(context)
        self.location = (
            f"/tmp/clawd-mutation-backup-{uuid.uuid4().hex}"
            if self.remote
            else tempfile.mkdtemp(prefix="clawd-mutation-backup-")
        )
        self.backup_root = (
            posixpath.join(str(self.location), "workspace")
            if self.remote
            else str(Path(str(self.location)) / "workspace")
        )
        self._closed = False
        try:
            self._capture()
        except BaseException:
            self.close()
            raise

    def _capture(self) -> None:
        if self.remote:
            backend = self.context.workspace_backend
            assert backend is not None
            root = posixpath.normpath(
                self.context.execution_workspace_root or "/workspace"
            )
            script = r"""
import base64, json, os, shutil, sys
p = json.loads(base64.b64decode(sys.argv[1])); root = p["root"]; backup = p["backup"]
ignored = set(p["ignored_dirs"]); full = bool(p["full_workspace"])
if os.path.lexists(backup):
    if os.path.islink(backup) or os.path.isfile(backup): os.unlink(backup)
    else: shutil.rmtree(backup)
def ignore(directory, names):
    skipped = set()
    for name in names:
        path = os.path.join(directory, name)
        if name in ignored or name.endswith(".egg-info"): skipped.add(name)
        elif not os.path.islink(path) and not os.path.isdir(path) and not os.path.isfile(path): skipped.add(name)
    return skipped
if full:
    if os.path.isdir(root): shutil.copytree(root, backup, symlinks=True, ignore=ignore)
else:
    os.makedirs(backup, exist_ok=True)
    source = os.path.join(root, ".clawd"); target = os.path.join(backup, ".clawd")
    if os.path.islink(source): os.symlink(os.readlink(source), target)
    elif os.path.isdir(source): shutil.copytree(source, target, symlinks=True)
    elif os.path.isfile(source): shutil.copy2(source, target, follow_symlinks=False)
print(json.dumps({"captured": True}))
""".strip()
            result = backend.run_json_helper(
                script,
                {
                    "operation": "capture_workspace_backup",
                    "root": root,
                    "backup": self.backup_root,
                    "full_workspace": self.full_workspace,
                    "ignored_dirs": sorted(_IGNORED_DIRECTORY_NAMES),
                },
                timeout_s=120,
            )
            if not isinstance(result, dict) or not result.get("captured"):
                raise RuntimeError(
                    "failed to back up remote workspace mutation state"
                )
            return

        root = self.context.workspace_root
        backup_root = Path(self.backup_root)

        def ignore(_directory: str, names: list[str]) -> set[str]:
            return {
                name
                for name in names
                if name in _IGNORED_DIRECTORY_NAMES
                or name.endswith(".egg-info")
                or (
                    not (Path(_directory) / name).is_symlink()
                    and not (Path(_directory) / name).is_dir()
                    and not (Path(_directory) / name).is_file()
                )
            }

        if self.full_workspace:
            shutil.copytree(root, backup_root, symlinks=True, ignore=ignore)
            return
        backup_root.mkdir(parents=True, exist_ok=True)
        source = root / _CONTROL_ROOT
        target = backup_root / _CONTROL_ROOT
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        elif source.is_dir():
            shutil.copytree(source, target, symlinks=True)
        elif source.is_file():
            shutil.copy2(source, target, follow_symlinks=False)

    def restore(self, relative_paths: list[str]) -> None:
        """Restore unauthorized paths while retaining legal task mutations."""

        paths = sorted(
            {
                _normalize_owned_path(path)
                for path in relative_paths
                if _normalize_owned_path(path)
            }
        )
        if not paths:
            return
        if self.remote:
            self._restore_remote(paths)
        else:
            self._restore_local(paths)

    @staticmethod
    def _remove_local(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)

    @classmethod
    def _ensure_local_parents(cls, root: Path, relative: PurePosixPath) -> None:
        cursor = root
        for part in relative.parent.parts:
            cursor /= part
            if cursor.is_symlink() or (cursor.exists() and not cursor.is_dir()):
                cls._remove_local(cursor)
            cursor.mkdir(exist_ok=True)

    def _restore_local(self, paths: list[str]) -> None:
        live_root = self.context.workspace_root
        backup_root = Path(self.backup_root)
        if live_root.is_symlink() or (live_root.exists() and not live_root.is_dir()):
            self._remove_local(live_root)
        live_root.mkdir(parents=True, exist_ok=True)
        # Remove shallow paths first.  If a command replaced a control directory with
        # a symlink, deleting children first would follow that attacker-controlled
        # ancestor and could touch files outside the workspace.
        for relative in sorted(paths, key=lambda value: value.count("/")):
            self._remove_local(live_root / PurePosixPath(relative))
        for relative in sorted(paths, key=lambda value: value.count("/")):
            pure = PurePosixPath(relative)
            source = backup_root.joinpath(*pure.parts)
            target = live_root.joinpath(*pure.parts)
            if not (source.exists() or source.is_symlink()):
                continue
            self._ensure_local_parents(live_root, pure)
            if source.is_symlink():
                target.symlink_to(os.readlink(source))
            elif source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                shutil.copystat(source, target, follow_symlinks=False)
            else:
                shutil.copy2(source, target, follow_symlinks=False)

    def _restore_remote(self, paths: list[str]) -> None:
        backend = self.context.workspace_backend
        assert backend is not None
        root = posixpath.normpath(
            self.context.execution_workspace_root or "/workspace"
        )
        script = r"""
import base64, json, os, shutil, sys
p = json.loads(base64.b64decode(sys.argv[1])); root = p["root"]; backup = p["backup"]
paths = sorted(set(p["paths"]), key=lambda value: value.count("/"))
def remove(path):
    if os.path.islink(path) or os.path.isfile(path): os.unlink(path)
    elif os.path.isdir(path): shutil.rmtree(path)
if os.path.islink(root) or (os.path.lexists(root) and not os.path.isdir(root)): remove(root)
os.makedirs(root, exist_ok=True)
def ensure_parents(relative):
    cursor = root
    for part in relative.split("/")[:-1]:
        cursor = os.path.join(cursor, part)
        if os.path.islink(cursor) or (os.path.exists(cursor) and not os.path.isdir(cursor)):
            remove(cursor)
        os.makedirs(cursor, exist_ok=True)
for relative in paths:
    remove(os.path.join(root, *relative.split("/")))
for relative in paths:
    source = os.path.join(backup, *relative.split("/")); target = os.path.join(root, *relative.split("/"))
    if not os.path.lexists(source): continue
    ensure_parents(relative)
    if os.path.islink(source): os.symlink(os.readlink(source), target)
    elif os.path.isdir(source): os.makedirs(target, exist_ok=True); shutil.copystat(source, target, follow_symlinks=False)
    else: shutil.copy2(source, target, follow_symlinks=False)
print(json.dumps({"restored": len(paths)}))
""".strip()
        backend.run_json_helper(
            script,
            {
                "operation": "restore_workspace_backup",
                "root": root,
                "backup": self.backup_root,
                "paths": paths,
            },
            timeout_s=120,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.remote:
            backend = self.context.workspace_backend
            if backend is not None:
                root = posixpath.normpath(
                    self.context.execution_workspace_root or "/workspace"
                )
                result = backend.exec(
                    f"rm -rf -- {shlex.quote(str(self.location))}",
                    cwd=root,
                    timeout_s=120,
                )
                if int(result.exit_code) != 0:
                    raise RuntimeError(
                        "failed to remove remote control-state backup: "
                        + str(result.stderr or result.stdout or "unknown error")
                    )
        else:
            shutil.rmtree(str(self.location), ignore_errors=True)


@contextmanager
def control_state_guard(context: ToolContext) -> Iterator[ControlStateBackup | None]:
    """Freeze local harness writers and provide a Bash rollback point.

    Local orchestration state and local Bash share one filesystem, so the TeamStore
    transaction lock prevents legitimate harness writes from being attributed to the
    command.  AGS orchestration state is local while Bash runs remotely, so the remote
    copy has no legitimate concurrent writer and needs only backup/audit/rollback.
    """

    if not bash_audit_required(context):
        yield None
        return

    def guarded() -> Iterator[ControlStateBackup]:
        backup = ControlStateBackup(context)
        try:
            yield backup
        finally:
            backup.close()

    if context.workspace_backend is not None:
        yield from guarded()
        return

    team_id = str((context.team or {}).get("team_id") or "")
    if not team_id:
        yield from guarded()
        return
    # Import lazily to avoid coupling normal, non-team tool initialization to the
    # teammate persistence module.
    from ..teammate.store import _locked_team_transaction

    with _locked_team_transaction(context.team_store._transaction_path(team_id)):
        yield from guarded()


def allowed_write_paths(context: ToolContext) -> tuple[str, ...]:
    paths = list(owned_paths(context))
    if ownership_enforced(context):
        paths.append(task_test_scratch_prefix(context))
    return tuple(paths)


def _current_task(context: ToolContext) -> dict[str, Any]:
    task = context.tasks.get(str(context.current_task_id))
    return task if isinstance(task, dict) else {}


def _normalized_role(value: Any) -> str:
    return re.sub(r"[\s-]+", "_", str(value or "").strip().casefold())


def _actor_is_integrator(context: ToolContext) -> bool:
    """Return whether the current writer has explicit integration authority.

    The lead is always the integration authority.  A worker only receives the same
    narrow privilege when its task metadata or persisted agent role says so; task
    names such as ``integration`` and read-only validation tasks are deliberately not
    treated as authority declarations.
    """

    team = context.team or {}
    actor_id = str(context.actor_id or "")
    if actor_id and actor_id == str(team.get("lead_agent_id") or ""):
        return True

    metadata = _current_task(context).get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if metadata.get("integration_owner") is True:
        return True
    if _normalized_role(metadata.get("ownership_role")) in _INTEGRATOR_ROLES:
        return True

    team_id = str(team.get("team_id") or "")
    if not (team_id and actor_id):
        return False
    try:
        agent = context.team_store.load_agent(team_id, actor_id)
    except (OSError, ValueError):
        return False
    return bool(agent and _normalized_role(agent.role) in _INTEGRATOR_ROLES)


def _is_integration_path(relative_path: str) -> bool:
    path = PurePosixPath(_normalize_owned_path(relative_path))
    name = path.name.casefold()
    if name == "__init__.py":
        return True
    if name in _INTEGRATION_FILE_NAMES:
        return True
    return name.startswith("requirements") and name.endswith(".txt")


def _is_generated_test_candidate(relative_path: str) -> bool:
    path = PurePosixPath(_normalize_owned_path(relative_path))
    return bool(
        len(path.parts) >= 2
        and path.parts[0] == "tests"
        and path.name.startswith("test_")
        and path.suffix == ".py"
    )


def _generated_test_paths(context: ToolContext) -> set[str]:
    paths = getattr(context, _GENERATED_TEST_PATHS_ATTRIBUTE, None)
    if not isinstance(paths, set):
        paths = set()
        setattr(context, _GENERATED_TEST_PATHS_ATTRIBUTE, paths)
    return paths


def _remember_generated_test(context: ToolContext, relative_path: str) -> None:
    _generated_test_paths(context).add(_normalize_owned_path(relative_path))


def _path_exists(
    context: ToolContext,
    path: str | Path,
    *,
    execution_path: bool,
) -> bool:
    if execution_path:
        backend = context.workspace_backend
        if backend is None:
            return False
        return bool(backend.stat(str(path)).exists)
    candidate = Path(path)
    return candidate.exists() or candidate.is_symlink()


def _declared_task_paths(context: ToolContext) -> list[tuple[str, str]]:
    roots = (
        str(context.workspace_root),
        str(context.execution_workspace_root or ""),
    )
    declared: list[tuple[str, str]] = []
    for task_id, task in context.tasks.items():
        if not isinstance(task, dict):
            continue
        for value in task.get("owned_files") or []:
            normalized = _normalize_owned_path(
                str(value), workspace_roots=roots
            )
            if normalized:
                declared.append((str(task_id), normalized))
    return declared


def _path_in_declared_scope(candidate: str, declared: str) -> bool:
    return candidate == declared or candidate.startswith(declared + "/")


def _path_reserved_by_other_task(
    context: ToolContext, relative_path: str
) -> bool:
    candidate = _normalize_owned_path(relative_path)
    current_task_id = str(context.current_task_id or "")
    return any(
        task_id != current_task_id
        and _path_in_declared_scope(candidate, declared)
        for task_id, declared in _declared_task_paths(context)
    )


def _is_runtime_artifact(relative: str) -> bool:
    path = PurePosixPath(_normalize_owned_path(relative))
    if any(
        component in _IGNORED_DIRECTORY_NAMES or component.endswith(".egg-info")
        for component in path.parts[:-1]
    ):
        return True
    name = path.name
    return bool(
        name in _IGNORED_FILE_NAMES
        or any(name.startswith(prefix) for prefix in _IGNORED_FILE_PREFIXES)
        or any(name.endswith(suffix) for suffix in _IGNORED_FILE_SUFFIXES)
    )


def _relative_local_path(context: ToolContext, path: str | Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (context.cwd or context.workspace_root) / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(context.workspace_root.resolve())
    except ValueError as exc:
        raise ToolPermissionError(
            f"task writes must remain inside the task workspace: {path}"
        ) from exc
    return relative.as_posix()


def _relative_execution_path(context: ToolContext, path: str) -> str:
    root = posixpath.normpath(context.execution_workspace_root or "/workspace")
    candidate = posixpath.normpath(path)
    if candidate == root:
        return ""
    prefix = root.rstrip("/") + "/"
    if not candidate.startswith(prefix):
        raise ToolPermissionError(
            f"task writes must remain inside the task workspace: {path}"
        )
    return candidate[len(prefix) :]


def relative_task_path(
    context: ToolContext,
    path: str | Path,
    *,
    execution_path: bool = False,
) -> str:
    if execution_path:
        return _relative_execution_path(context, str(path))
    return _relative_local_path(context, path)


def path_is_owned(context: ToolContext, relative_path: str) -> bool:
    if not ownership_enforced(context):
        return True
    candidate = _normalize_owned_path(relative_path)
    if any(
        candidate == owned or candidate.startswith(owned + "/")
        for owned in allowed_write_paths(context)
    ):
        return True
    if _is_runtime_artifact(candidate) and not _path_reserved_by_other_task(
        context, candidate
    ):
        return True
    if candidate in _generated_test_paths(context):
        return True
    return _is_integration_path(candidate) and _actor_is_integrator(context)


def require_owned_path(
    context: ToolContext,
    path: str | Path,
    *,
    tool_name: str,
    execution_path: bool = False,
) -> str:
    """Reject a direct write outside the current v2 task's ownership scope."""

    if not ownership_enforced(context):
        relative = relative_task_path(
            context, path, execution_path=execution_path
        )
        if strict_protocol_v2(context) and _is_control_path(relative):
            raise ToolPermissionError(
                "strict protocol v2 protects .clawd control state; use Team tools "
                "instead of Write/Edit"
            )
        return relative
    relative = relative_task_path(context, path, execution_path=execution_path)
    if path_is_owned(context, relative):
        return relative
    # A new test module under tests/ is task-local disposable scratch.  Remembering
    # provenance keeps subsequent edits legal while existing repository tests remain
    # protected.  Persistent project tests should still be declared in owned_files.
    if (
        _is_generated_test_candidate(relative)
        and not _path_reserved_by_other_task(context, relative)
        and not _path_exists(context, path, execution_path=execution_path)
    ):
        _remember_generated_test(context, relative)
        return relative
    violation = record_ownership_violation(context, tool_name, [relative])
    raise ToolPermissionError(violation["error"])


def record_ownership_violation(
    context: ToolContext, tool_name: str, paths: list[str]
) -> dict[str, Any]:
    normalized = sorted(
        {
            _normalize_owned_path(path) or "."
            for path in paths
            if isinstance(path, str)
        }
    )
    allowed = list(allowed_write_paths(context))
    rendered = ", ".join(normalized[:8])
    if len(normalized) > 8:
        rendered += f", ... (+{len(normalized) - 8} more)"
    error = (
        f"protocol v2 task ownership violation in {tool_name}: changed {rendered}; "
        f"allowed owned_files are {allowed or ['<none>']}; package initializers and "
        "manifests additionally require an explicit lead/integrator role"
    )
    violation = {
        "tool": tool_name,
        "task_id": context.current_task_id,
        "actor_id": context.actor_id,
        "paths": normalized,
        "allowed_paths": allowed,
        "error": error,
    }
    context.ownership_violations.append(violation)
    if context.team is not None:
        team_id = str(context.team.get("team_id") or "")
        if team_id:
            context.team_store.append_event(
                team_id, "task.ownership_violation_detected", violation
            )
    return violation


def _ignored_relative_path(relative: str) -> bool:
    return _is_runtime_artifact(relative)


def snapshot_local_workspace(context: ToolContext) -> dict[str, str]:
    """Return filesystem fingerprints for auditable workspace files.

    Deterministic interpreter/test caches are excluded because they are not
    deliverable source.  ``.clawd`` is intentionally included: local Bash holds the
    TeamStore transaction lock while this snapshot is live, so control-state changes
    can only come from the command itself.  The task-private scratch subtree remains
    writable through :func:`allowed_write_paths`.
    """

    root = context.workspace_root.resolve()
    scan_root = root if ownership_enforced(context) else root / _CONTROL_ROOT
    if not (scan_root.exists() or scan_root.is_symlink()):
        return {}
    declared_paths = tuple(path for _, path in _declared_task_paths(context))

    def declared(relative: str, *, include_ancestors: bool = False) -> bool:
        candidate = _normalize_owned_path(relative)
        return any(
            _path_in_declared_scope(candidate, owned)
            or (
                include_ancestors
                and _path_in_declared_scope(owned, candidate)
            )
            for owned in declared_paths
        )

    snapshot: dict[str, str] = {}
    for directory, names, files in os.walk(scan_root, followlinks=False):
        directory_path = Path(directory)
        names[:] = sorted(
            name
            for name in names
            if (
                name not in _IGNORED_DIRECTORY_NAMES
                and not name.endswith(".egg-info")
            )
            or declared(
                (directory_path / name).relative_to(root).as_posix(),
                include_ancestors=True,
            )
        )
        # os.walk reports directory symlinks in ``names`` even with
        # followlinks=False.  Fingerprint them explicitly so a command cannot replace
        # a protected control directory with a symlink without appearing in the diff.
        for name in list(names):
            path = directory_path / name
            if not path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            try:
                stat = path.lstat()
                metadata = (
                    f"{stat.st_mode & 0o777:o}:{stat.st_size}:"
                    f"{stat.st_mtime_ns}:{stat.st_ctime_ns}:{stat.st_ino}"
                )
                snapshot[relative] = f"link:{metadata}:{os.readlink(path)}"
            except FileNotFoundError:
                continue
        for name in sorted(files):
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            if _ignored_relative_path(relative) and not declared(relative):
                continue
            try:
                stat = path.lstat()
                metadata = (
                    f"{stat.st_mode & 0o777:o}:{stat.st_size}:"
                    f"{stat.st_mtime_ns}:{stat.st_ctime_ns}:{stat.st_ino}"
                )
                if path.is_symlink():
                    fingerprint = f"link:{metadata}:{os.readlink(path)}"
                elif path.is_file():
                    fingerprint = f"file:{metadata}"
                else:
                    continue
            except FileNotFoundError:
                # External processes can race a scan; teammate mutations themselves
                # are serialized by ToolContext.mutation_lock.
                continue
            snapshot[relative] = fingerprint
    return snapshot


_REMOTE_SNAPSHOT_SCRIPT = r"""
import base64, json, os, sys
p = json.loads(base64.b64decode(sys.argv[1])); root = p["root"]
scan_root = root if not p["control_only"] else os.path.join(root, ".clawd")
ignored_dirs = set(p["ignored_dirs"]); ignored_files = set(p["ignored_files"])
ignored_prefixes = tuple(p["ignored_prefixes"]); ignored_suffixes = tuple(p["ignored_suffixes"])
declared_paths = tuple(p["declared_paths"]); out = {}
def declared(rel):
    return any(rel == owned or rel.startswith(owned + "/") or owned.startswith(rel + "/") for owned in declared_paths)
for directory, names, files in os.walk(scan_root, followlinks=False):
    names[:] = sorted(n for n in names if (n not in ignored_dirs and not n.endswith(".egg-info")) or declared(os.path.relpath(os.path.join(directory, n), root).replace(os.sep, "/")))
    for name in list(names):
        path = os.path.join(directory, name)
        if not os.path.islink(path): continue
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        try:
            stat = os.lstat(path)
            metadata = "%o:%s:%s:%s:%s" % (stat.st_mode & 0o777, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
            out[rel] = "link:" + metadata + ":" + os.readlink(path)
        except FileNotFoundError: continue
    for name in sorted(files):
        path = os.path.join(directory, name); rel = os.path.relpath(path, root).replace(os.sep, "/")
        parts = rel.split("/")
        ignored_parent = any(part in ignored_dirs or part.endswith(".egg-info") for part in parts[:-1])
        ignored_file = name in ignored_files or name.startswith(ignored_prefixes) or name.endswith(ignored_suffixes)
        if (ignored_parent or ignored_file) and not declared(rel): continue
        try:
            stat = os.lstat(path)
            metadata = "%o:%s:%s:%s:%s" % (stat.st_mode & 0o777, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
            if os.path.islink(path): value = "link:" + metadata + ":" + os.readlink(path)
            elif os.path.isfile(path):
                value = "file:" + metadata
            else: continue
        except FileNotFoundError: continue
        out[rel] = value
print(json.dumps(out, sort_keys=True))
""".strip()


def snapshot_remote_workspace(context: ToolContext) -> dict[str, str]:
    backend = context.workspace_backend
    if backend is None:
        raise RuntimeError("remote ownership audit requires a workspace backend")
    value = backend.run_json_helper(
        _REMOTE_SNAPSHOT_SCRIPT,
        {
            "root": context.execution_workspace_root or "/workspace",
            "ignored_dirs": sorted(_IGNORED_DIRECTORY_NAMES),
            "ignored_files": sorted(_IGNORED_FILE_NAMES),
            "ignored_prefixes": sorted(_IGNORED_FILE_PREFIXES),
            "ignored_suffixes": sorted(_IGNORED_FILE_SUFFIXES),
            "declared_paths": sorted(
                {path for _, path in _declared_task_paths(context)}
            ),
            "control_only": not ownership_enforced(context),
        },
    )
    if not isinstance(value, dict):
        raise RuntimeError("remote ownership audit returned an invalid snapshot")
    return {str(path): str(fingerprint) for path, fingerprint in value.items()}


def changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    )


def audit_changed_paths(
    context: ToolContext,
    *,
    tool_name: str,
    before: dict[str, str],
    after: dict[str, str],
    control_backup: ControlStateBackup | None = None,
) -> list[str]:
    if not bash_audit_required(context):
        return []
    changed = changed_paths(before, after)
    unauthorized: list[str] = []
    for path in changed:
        if not ownership_enforced(context):
            if _is_control_path(path):
                unauthorized.append(path)
            continue
        if path_is_owned(context, path):
            continue
        # Only additions qualify for the convenient tests/test_*.py scratch rule.
        # A repository test present in the pre-command snapshot remains protected,
        # including against edits and deletion.
        if (
            _is_generated_test_candidate(path)
            and path not in before
            and path in after
        ):
            _remember_generated_test(context, path)
            continue
        unauthorized.append(path)
    if unauthorized:
        if control_backup is None:
            raise RuntimeError(
                "refusing to report an unauthorized Bash mutation without a rollback backup"
            )
        # Restore before recording the violation: for .clawd this ensures the audit
        # event is appended to the genuine event stream.  For delivery files it keeps
        # the best-known workspace intact for an explicit repair plan.
        control_backup.restore(unauthorized)
        violation = record_ownership_violation(context, tool_name, unauthorized)
        raise ToolPermissionError(violation["error"])
    return changed
