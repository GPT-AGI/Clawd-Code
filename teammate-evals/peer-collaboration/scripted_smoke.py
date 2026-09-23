from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
FIXTURE = ROOT / "fixture"


def _load_schema_module() -> Any:
    path = ROOT / "schema_validation.py"
    spec = importlib.util.spec_from_file_location("peer_schema_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load schema validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def prepare_workspace(destination: Path) -> Path:
    workspace = destination / "workspace"
    shutil.copytree(FIXTURE, workspace)
    _run(["git", "init", "-q"], workspace)
    _run(["git", "config", "user.name", "Clawd Peer Smoke"], workspace)
    _run(["git", "config", "user.email", "peer-smoke@example.invalid"], workspace)
    _run(["git", "add", "."], workspace)
    _run(["git", "commit", "-qm", "peer smoke fixture"], workspace)
    return workspace


def run_smoke(output_dir: Path) -> dict[str, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.peer.backend import PeerBoundaryResult, ScriptedPeerBackend
    from src.peer.runner import run_peer_collaboration
    from src.tool_system.defaults import build_default_registry
    from src.tool_system.protocol import ToolCall

    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = prepare_workspace(output_dir)
    peer_two_initially_idle = threading.Event()

    protocol_source = '''from __future__ import annotations

from typing import Any


def normalize_item(raw: dict[str, Any]) -> dict[str, Any]:
    tags = tuple(str(tag).strip() for tag in raw.get("tags", ()) if str(tag).strip())
    return {"id": str(raw["id"]), "tags": tags}
'''
    client_source = '''from __future__ import annotations

from typing import Any

from .protocol import normalize_item


def build_request(raw: dict[str, Any]) -> dict[str, Any]:
    return {"item": normalize_item(raw)}
'''

    def handler(session, prompt, registry, context):
        if session.spec.peer_name == "peer-2" and session.boundary_index == 1:
            peer_two_initially_idle.set()
            return PeerBoundaryResult(response_text="waiting for interface", num_turns=1)
        if session.spec.peer_name == "peer-1" and session.boundary_index == 1:
            if not peer_two_initially_idle.wait(2):
                raise RuntimeError("peer-2 did not reach its idle boundary")
            written = registry.dispatch(
                ToolCall(
                    "Write",
                    {"file_path": "src/protocol.py", "content": protocol_source},
                ),
                context,
            )
            if written.is_error:
                raise RuntimeError(str(written.output))
            sent = registry.dispatch(
                ToolCall(
                    "SendMessage",
                    {
                        "to": "peer-2",
                        "summary": "normalization interface",
                        "message": {
                            "function": "normalize_item(raw)",
                            "returns": {"id": "str", "tags": "tuple[str, ...]"},
                        },
                    },
                ),
                context,
            )
            if sent.is_error:
                raise RuntimeError(str(sent.output))
            return PeerBoundaryResult(response_text="interface sent", num_turns=1)
        if session.spec.peer_name == "peer-2" and session.boundary_index == 2:
            if "normalize_item" not in prompt or "tuple[str, ...]" not in prompt:
                raise RuntimeError("peer-2 did not receive the interface contract")
            written = registry.dispatch(
                ToolCall(
                    "Write",
                    {"file_path": "src/client.py", "content": client_source},
                ),
                context,
            )
            if written.is_error:
                raise RuntimeError(str(written.output))
            tested = registry.dispatch(
                ToolCall(
                    "Bash",
                    {
                        "command": (
                            f"{sys.executable} -m unittest discover -s tests -v && "
                            "git add src/protocol.py src/client.py && "
                            "git commit -m 'implement negotiated item contract'"
                        )
                    },
                ),
                context,
            )
            if tested.is_error:
                raise RuntimeError(str(tested.output))
            revision = _run(["git", "rev-parse", "HEAD"], Path(session.spec.workspace_path))
            submitted = registry.dispatch(
                ToolCall(
                    "PeerSubmit",
                    {
                        "revision": revision,
                        "summary": "Negotiated, implemented, and tested the shared interface.",
                    },
                ),
                context,
            )
            if submitted.is_error:
                raise RuntimeError(str(submitted.output))
            return PeerBoundaryResult(response_text="submitted", num_turns=1)
        return PeerBoundaryResult(response_text="available", num_turns=1)

    mission = (workspace / "TASK.md").read_text(encoding="utf-8")
    result = run_peer_collaboration(
        mission,
        repo=workspace,
        peers=2,
        communication="p2p",
        workspace_mode="shared",
        timeout_seconds=10,
        max_turns=8,
        output_dir=output_dir / "runs",
        acceptance_command=[
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-v",
        ],
        backend=ScriptedPeerBackend(handler),
        base_registry=build_default_registry(include_user_tools=False),
        run_id="scripted-smoke",
    )
    run_dir = Path(result["result_path"]).parent
    _load_schema_module().validate_run(run_dir)
    if result["status"] != "completed":
        raise RuntimeError(f"scripted smoke did not complete: {result['status']}")
    if result.get("acceptance", {}).get("exit_code") != 0:
        raise RuntimeError("scripted smoke acceptance failed")
    if not result["messages"] or not result["accepted_submission"]:
        raise RuntimeError("scripted smoke did not exercise messaging and submission")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="clawd-peer-smoke-")
        output = Path(temporary.name)
    else:
        temporary = None
        output = args.output_dir.expanduser().resolve()
    try:
        result = run_smoke(output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
