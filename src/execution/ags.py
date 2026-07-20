from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import os
import posixpath
import re
import shlex
import shutil
import sys
import tarfile
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Coroutine, TypeVar

from .backend import CommandOutcome, RemoteStat


T = TypeVar("T")
TASK_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
DEFAULT_AGS_IMAGE_REPOSITORY = "swebenchdocker.tencentcloudcr.com/swebench/nl2repo"
DEFAULT_AGS_RUNTIME_IMAGE = "swebenchdocker.tencentcloudcr.com/swebench/swehub:swerex-runtime"


def _build_sandbox_command(
    command: str,
    *,
    timeout_s: int,
    runtime_mount_path: str,
) -> list[str]:
    """Run a task command without leaking the mounted SWE-ReX virtualenv.

    The AGS runtime server is launched from ``<mount>/swerex``.  Its virtualenv
    can therefore be the first entry in PATH even though the task image has a
    different Python version.  Keep the image environment, but remove only
    runtime-owned PATH entries before starting the task command.

    GNU timeout is deliberately inside the sandbox.  It owns the task process
    group and terminates descendants as well, so a timed-out pip/apt command
    cannot keep running and block later tool calls.
    """
    runtime_root = posixpath.join(runtime_mount_path.rstrip("/") or "/", "swerex")
    wrapper = r'''
runtime_root=$1
timeout_seconds=$2
user_command=$3

clean_path=
old_ifs=$IFS
IFS=:
for path_entry in ${PATH:-}; do
    case "$path_entry" in
        "$runtime_root"|"$runtime_root"/*) continue ;;
    esac
    if [ -z "$clean_path" ]; then
        clean_path=$path_entry
    else
        clean_path=$clean_path:$path_entry
    fi
done
IFS=$old_ifs
if [ -z "$clean_path" ]; then
    clean_path=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
fi
export PATH=$clean_path

case "${VIRTUAL_ENV:-}" in
    "$runtime_root"|"$runtime_root"/*) unset VIRTUAL_ENV ;;
esac
case "${PYTHONHOME:-}" in
    "$runtime_root"|"$runtime_root"/*) unset PYTHONHOME ;;
esac

if command -v python3 >/dev/null 2>&1; then
    CLAWD_PYTHON=$(command -v python3)
elif command -v python >/dev/null 2>&1; then
    CLAWD_PYTHON=$(command -v python)
else
    CLAWD_PYTHON=
fi
export CLAWD_PYTHON

if [ -x /usr/bin/timeout ]; then
    timeout_bin=/usr/bin/timeout
elif [ -x /bin/timeout ]; then
    timeout_bin=/bin/timeout
else
    echo "Clawd harness error: GNU timeout is unavailable in the sandbox image" >&2
    exit 125
fi

exec "$timeout_bin" --signal=TERM --kill-after=5s "${timeout_seconds}s" \
    /bin/bash -c "$user_command"
'''.strip()
    return [
        "/bin/bash",
        "-c",
        wrapper,
        "clawd-sandbox-command",
        runtime_root,
        str(timeout_s),
        command,
    ]


def nl2repo_ags_image(task: str, *, version: str = "1.0") -> str:
    normalized = task.strip().lower()
    if not TASK_RE.fullmatch(normalized):
        raise ValueError(f"invalid NL2Repo task name: {task!r}")
    return f"{DEFAULT_AGS_IMAGE_REPOSITORY}:{normalized}-{version}"


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return default


def load_env_file(path: str | Path, *, override: bool = False) -> None:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"AGS env file not found: {source}")
    aliases = {
        "secret_id": ("AGS_SECRET_ID", "TENCENTCLOUD_SECRET_ID"),
        "secret_key": ("AGS_SECRET_KEY", "TENCENTCLOUD_SECRET_KEY"),
        "region": ("AGS_REGION",),
    }
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "[")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        for alias in aliases.get(key, ()):
            if override or alias not in os.environ:
                os.environ[alias] = value


def discover_ags_env_file() -> Path | None:
    configured = _first_env("AGS_ENV_FILE", "AGS_ENV_FILE_PATH")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path.cwd() / ".env")
    repo = Path(__file__).resolve().parents[2]
    candidates.append(repo.parent / "sandbox" / "ags" / ".env")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


@dataclass
class AGSSettings:
    secret_id: str = ""
    secret_key: str = ""
    http_endpoint: str = "ags.tencentcloudapi.com"
    region: str = "ap-guangzhou"
    domain: str = "ap-guangzhou.tencentags.com"
    role_arn: str = ""
    skip_ssl_verify: bool = False
    tool_id: str = ""
    image: str = field(default_factory=lambda: nl2repo_ags_image("retrying"))
    image_registry_type: str = "enterprise"
    cpu: str = "2"
    memory: str = "4Gi"
    port: int = 8000
    timeout: str = "3h"
    startup_timeout: float = 600.0
    runtime_timeout: float = 700.0
    network_mode: str = "PUBLIC"
    mount_name: str = "rex"
    mount_image: str = DEFAULT_AGS_RUNTIME_IMAGE
    mount_image_registry_type: str = "enterprise"
    mount_path: str = "/nix"
    image_subpath: str = "/nix"
    mount_readonly: bool = False
    swerex_root: str = ""

    @classmethod
    def from_env(
        cls,
        *,
        image: str,
        env_file: str | Path | None = None,
        timeout: str | None = None,
        cpu: str | None = None,
        memory: str | None = None,
    ) -> "AGSSettings":
        selected_env = Path(env_file).expanduser() if env_file else discover_ags_env_file()
        if selected_env is not None:
            load_env_file(selected_env)
        return cls(
            secret_id=_first_env("AGS_SECRET_ID", "TENCENTCLOUD_SECRET_ID"),
            secret_key=_first_env("AGS_SECRET_KEY", "TENCENTCLOUD_SECRET_KEY"),
            http_endpoint=_first_env("SLIME_AGENT_AGS_HTTP_ENDPOINT", default=cls.http_endpoint),
            region=_first_env("AGS_REGION", "SLIME_AGENT_AGS_REGION", default=cls.region),
            domain=_first_env("AGS_DOMAIN", "SLIME_AGENT_AGS_DOMAIN", default=cls.domain),
            role_arn=_first_env("AGS_ROLE_ARN", "TENCENTCLOUD_ROLE_ARN", default=cls.role_arn),
            skip_ssl_verify=_first_env("SLIME_AGENT_AGS_SKIP_SSL_VERIFY", default="0").lower()
            in {"1", "true", "yes"},
            tool_id=_first_env("AGS_TOOL_ID", "SLIME_AGENT_AGS_TOOL_ID"),
            image=image,
            image_registry_type=_first_env(
                "SLIME_AGENT_AGS_IMAGE_REGISTRY_TYPE", default=cls.image_registry_type
            ),
            cpu=cpu or _first_env("SLIME_AGENT_AGS_CPU", default=cls.cpu),
            memory=memory or _first_env("SLIME_AGENT_AGS_MEMORY", default=cls.memory),
            port=int(_first_env("SLIME_AGENT_AGS_PORT", default=str(cls.port))),
            timeout=timeout or _first_env("SLIME_AGENT_AGS_TIMEOUT", default=cls.timeout),
            startup_timeout=float(
                _first_env("SLIME_AGENT_AGS_BOOT_TIMEOUT_SEC", default=str(cls.startup_timeout))
            ),
            runtime_timeout=float(
                _first_env("SLIME_AGENT_AGS_RUNTIME_TIMEOUT_SEC", default=str(cls.runtime_timeout))
            ),
            network_mode=_first_env(
                "SLIME_AGENT_AGS_NETWORK_MODE", "AGS_NETWORK_MODE", default=cls.network_mode
            ).upper(),
            mount_name=_first_env("SLIME_AGENT_AGS_MOUNT_NAME", default=cls.mount_name),
            mount_image=_first_env("SLIME_AGENT_AGS_MOUNT_IMAGE", default=cls.mount_image),
            mount_image_registry_type=_first_env(
                "SLIME_AGENT_AGS_MOUNT_IMAGE_REGISTRY_TYPE",
                default=cls.mount_image_registry_type,
            ),
            mount_path=_first_env("SLIME_AGENT_AGS_MOUNT_PATH", default=cls.mount_path),
            image_subpath=_first_env("SLIME_AGENT_AGS_IMAGE_SUBPATH", default=cls.image_subpath),
            mount_readonly=_first_env("SLIME_AGENT_AGS_MOUNT_READONLY", default="0").lower()
            in {"1", "true", "yes"},
            swerex_root=_first_env("SWE_REX_ROOT"),
        )

    def validate(self) -> None:
        missing = [name for name in ("secret_id", "secret_key") if not getattr(self, name)]
        if missing:
            raise RuntimeError(
                "missing AGS credentials: "
                + ", ".join(missing)
                + "; configure AGS_SECRET_ID and AGS_SECRET_KEY or --ags-env-file"
            )
        if self.network_mode not in {"PUBLIC", "SANDBOX", "INTERNAL_SERVICE"}:
            raise RuntimeError(f"invalid AGS network mode: {self.network_mode!r}")

    def deployment_kwargs(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("swerex_root", None)
        return data


def ensure_swerex_importable(settings: AGSSettings | None = None) -> None:
    candidates: list[Path] = []
    if settings is not None and settings.swerex_root:
        candidates.append(Path(settings.swerex_root).expanduser())
    configured = _first_env("SWE_REX_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    repo = Path(__file__).resolve().parents[2]
    candidates.append(repo.parent / "sandbox" / "SWE-ReX" / "src")
    for candidate in reversed(candidates):
        resolved = candidate.resolve()
        if resolved.is_dir() and str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))
    try:
        from swerex.deployment.ags import TencentAGSDeployment  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name and not exc.name.startswith("swerex"):
            raise RuntimeError(
                f"SWE-ReX AGS dependency {exc.name!r} is unavailable; "
                "install Clawd with the ags extra"
            ) from exc
        raise RuntimeError(
            "SWE-ReX with Tencent AGS support is unavailable; install the customized checkout "
            "or set SWE_REX_ROOT to its src directory"
        ) from exc


class AGSWorkspaceBackend:
    """Long-lived synchronous facade over SWE-ReX's async AGS deployment."""

    workspace_root = "/workspace"

    def __init__(self, settings: AGSSettings) -> None:
        settings.validate()
        self.settings = settings
        self.sandbox_id = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._deployment: Any = None
        self._started = False
        self._closed = False

    def start(self) -> "AGSWorkspaceBackend":
        if self._started:
            return self
        ensure_swerex_importable(self.settings)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="clawd-ags", daemon=True)
        self._thread.start()
        try:
            self._submit(self._start_async(), timeout=self.settings.startup_timeout + 90)
        except BaseException:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=10)
            if self._loop is not None:
                self._loop.close()
            self._loop = None
            self._thread = None
            raise
        self._started = True
        return self

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _start_async(self) -> None:
        from swerex.deployment.ags import TencentAGSDeployment

        deployment = TencentAGSDeployment(**self.settings.deployment_kwargs())
        self._deployment = deployment
        try:
            await deployment.start()
        except BaseException:
            try:
                await asyncio.shield(deployment.stop())
            finally:
                self._deployment = None
            raise
        self.sandbox_id = str(deployment.instance_id or "")

    def _submit(
        self,
        coroutine: Coroutine[Any, Any, T],
        *,
        timeout: float | None = None,
        operation: str = "AGS operation",
    ) -> T:
        if self._loop is None:
            raise RuntimeError("AGS backend is not started")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            # Since Python 3.11, concurrent.futures.TimeoutError aliases the
            # built-in TimeoutError.  A completed coroutine may itself have
            # raised a remote CommandTimeoutError; preserve that exception and
            # only cancel when result() actually exhausted its wait deadline.
            if future.done():
                raise
            future.cancel()
            if timeout is None:
                detail = "the configured deadline"
            else:
                detail = f"{timeout:g}s"
            raise TimeoutError(
                f"{operation} timed out after {detail}; the pending request was cancelled"
            ) from exc

    def resolve_path(self, path: str, *, cwd: str, local_root: Path) -> str:
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        normalized_input = path
        local = str(local_root.resolve())
        if normalized_input.startswith("/") and not normalized_input.startswith(
            self.workspace_root
        ):
            normalized_input = str(Path(normalized_input).expanduser().resolve())
        if normalized_input == local or normalized_input.startswith(local + os.sep):
            suffix = normalized_input[len(local) :].replace(os.sep, "/")
            normalized_input = self.workspace_root + suffix
        if not normalized_input.startswith("/"):
            normalized_input = posixpath.join(cwd, normalized_input)
        resolved = posixpath.normpath(normalized_input)
        if resolved != self.workspace_root and not resolved.startswith(self.workspace_root + "/"):
            raise ValueError(f"path is outside the AGS workspace: {path}")
        return resolved

    def exec(
        self,
        command: str,
        *,
        cwd: str,
        timeout_s: int,
        env: dict[str, str] | None = None,
    ) -> CommandOutcome:
        if not self._started or self._deployment is None:
            raise RuntimeError("AGS backend is not started")

        sandbox_command = _build_sandbox_command(
            command,
            timeout_s=timeout_s,
            runtime_mount_path=self.settings.mount_path,
        )

        async def execute() -> Any:
            from swerex.runtime.abstract import Command

            return await self._deployment.runtime.execute(
                Command(
                    command=sandbox_command,
                    shell=False,
                    check=False,
                    timeout=timeout_s + 10,
                    cwd=cwd,
                    env=env,
                    merge_output_streams=False,
                )
            )

        try:
            result = self._submit(
                execute(),
                timeout=timeout_s + 20,
                operation=f"sandbox command ({timeout_s}s limit)",
            )
        except TimeoutError as exc:
            return CommandOutcome(
                exit_code=124,
                stderr=str(exc) or f"Sandbox command timed out after {timeout_s}s",
            )
        stderr = result.stderr or ""
        if int(result.exit_code or 0) == 124 and "timed out" not in stderr.lower():
            timeout_message = f"Clawd sandbox command timed out after {timeout_s}s and was terminated"
            stderr = f"{stderr.rstrip()}\n{timeout_message}".lstrip()
        return CommandOutcome(
            exit_code=int(result.exit_code or 0),
            stdout=result.stdout or "",
            stderr=stderr,
        )

    def run_json_helper(
        self, script: str, payload: dict[str, Any], *, timeout_s: int = 120
    ) -> Any:
        encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
        command = f"python3 -c {shlex.quote(script)} {shlex.quote(encoded)}"
        result = self.exec(command, cwd=self.workspace_root, timeout_s=timeout_s)
        if result.exit_code != 0:
            raise RuntimeError(result.stderr or result.stdout or "remote Python helper failed")
        return json.loads(result.stdout)

    def stat(self, path: str) -> RemoteStat:
        script = (
            "import base64,json,os,sys; p=json.loads(base64.b64decode(sys.argv[1]))['path']; "
            "e=os.path.exists(p); s=os.stat(p) if e else None; "
            "print(json.dumps({'path':p,'exists':e,'is_file':os.path.isfile(p),"
            "'is_dir':os.path.isdir(p),'size':s.st_size if s else 0,"
            "'mtime_ns':s.st_mtime_ns if s else 0}))"
        )
        return RemoteStat(**self.run_json_helper(script, {"path": path}))

    def read_text(self, path: str) -> str:
        async def read() -> Any:
            from swerex.runtime.abstract import ReadFileRequest

            return await self._deployment.runtime.read_file(
                ReadFileRequest(path=path, encoding="utf-8", errors="replace")
            )

        return str(self._submit(read(), timeout=self.settings.runtime_timeout).content)

    def read_bytes(self, path: str) -> bytes:
        result = self.exec(
            f"base64 < {shlex.quote(path)}",
            cwd=self.workspace_root,
            timeout_s=120,
        )
        if result.exit_code != 0:
            raise RuntimeError(result.stderr or f"failed to read {path}")
        return base64.b64decode("".join(result.stdout.splitlines()))

    def write_text(self, path: str, content: str) -> None:
        temporary = f"{path}.clawd-tmp-{uuid.uuid4().hex}"

        async def write() -> Any:
            from swerex.runtime.abstract import WriteFileRequest

            return await self._deployment.runtime.write_file(
                WriteFileRequest(path=temporary, content=content)
            )

        self._submit(write(), timeout=self.settings.runtime_timeout)
        result = self.exec(
            f"mkdir -p {shlex.quote(posixpath.dirname(path))} && mv -f {shlex.quote(temporary)} {shlex.quote(path)}",
            cwd=self.workspace_root,
            timeout_s=120,
        )
        if result.exit_code != 0:
            raise RuntimeError(result.stderr or f"failed to install {path}")

    def upload_tree(self, local_path: Path, remote_path: str) -> None:
        source = local_path.resolve()

        async def upload() -> Any:
            from swerex.runtime.abstract import UploadRequest

            return await self._deployment.runtime.upload(
                UploadRequest(source_path=str(source), target_path=remote_path)
            )

        self._submit(upload(), timeout=max(self.settings.runtime_timeout, 600))

    def download_tree(self, remote_path: str, local_path: Path) -> None:
        destination = local_path.resolve()
        destination.mkdir(parents=True, exist_ok=True)
        archive = f"/tmp/clawd-export-{uuid.uuid4().hex}.tar.gz"
        parent = posixpath.dirname(remote_path)
        name = posixpath.basename(remote_path)
        create = self.exec(
            f"tar -C {shlex.quote(parent)} -czf {shlex.quote(archive)} {shlex.quote(name)}",
            cwd=self.workspace_root,
            timeout_s=600,
        )
        if create.exit_code != 0:
            raise RuntimeError(create.stderr or "failed to archive remote workspace")
        try:
            size = self.stat(archive).size
            chunk_size = 384 * 1024
            packed = bytearray()
            for offset in range(0, size, chunk_size):
                command = (
                    f"tail -c +{offset + 1} {shlex.quote(archive)} | "
                    f"head -c {min(chunk_size, size - offset)} | base64"
                )
                result = self.exec(command, cwd=self.workspace_root, timeout_s=120)
                if result.exit_code != 0:
                    raise RuntimeError(result.stderr or "failed to download remote workspace")
                packed.extend(base64.b64decode("".join(result.stdout.splitlines())))
            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as handle:
                handle.write(packed)
                handle.flush()
                with tarfile.open(handle.name, "r:gz") as tar:
                    self._safe_extract(tar, destination)
            extracted = destination / name
            if extracted.is_dir():
                for item in extracted.iterdir():
                    target = destination / item.name
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    item.rename(target)
                extracted.rmdir()
        finally:
            self.exec(f"rm -f {shlex.quote(archive)}", cwd=self.workspace_root, timeout_s=60)

    @staticmethod
    def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
        root = destination.resolve()
        for member in archive.getmembers():
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"unsafe entry in sandbox archive: {member.name}")
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe path in sandbox archive: {member.name}")
        archive.extractall(destination)

    def reset_workspace(self) -> None:
        result = self.exec(
            (
                "mkdir -p /workspace && "
                "rm -rf -- /workspace/* /workspace/.[!.]* /workspace/..?*"
            ),
            cwd="/",
            timeout_s=120,
        )
        if result.exit_code != 0:
            raise RuntimeError(result.stderr or "failed to reset AGS workspace")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._deployment is not None and self._loop is not None:
                # Under a large reward fan-out AGS stop requests can queue for
                # longer than the old fixed 120-second deadline. Give cleanup
                # enough time to finish, while keeping shutdown bounded.
                cleanup_timeout = min(max(self.settings.runtime_timeout, 120), 600)
                self._submit(
                    self._deployment.stop(),
                    timeout=cleanup_timeout,
                    operation="AGS sandbox cleanup",
                )
        finally:
            self._deployment = None
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=10)
            if self._loop is not None:
                self._loop.close()
            self._loop = None
            self._thread = None

    def __enter__(self) -> "AGSWorkspaceBackend":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
