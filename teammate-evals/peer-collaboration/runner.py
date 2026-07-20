from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real-model peer collaboration pilot.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--peers", type=int, required=True)
    parser.add_argument(
        "--communication",
        required=True,
        choices=("solo", "independent", "none", "artifact-only", "star", "p2p"),
    )
    parser.add_argument("--workspace-mode", choices=("shared", "worktree"), required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--turn-budget", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coordinator-peer")
    parser.add_argument("--acceptance-command")
    args = parser.parse_args()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.peer.runner import run_peer_collaboration

    repo = args.repo.expanduser().resolve()
    prompt = args.prompt_file.expanduser()
    if not prompt.is_absolute():
        prompt = repo / prompt
    result = run_peer_collaboration(
        prompt.read_text(encoding="utf-8"),
        repo=repo,
        peers=args.peers,
        communication=args.communication,
        workspace_mode=args.workspace_mode,
        provider_name=args.provider,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        max_turns=args.max_turns,
        max_output_tokens=args.max_output_tokens,
        token_budget=args.token_budget,
        turn_budget=args.turn_budget,
        output_dir=args.output_dir,
        coordinator_peer=args.coordinator_peer,
        acceptance_command=(
            shlex.split(args.acceptance_command) if args.acceptance_command else None
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "completed":
        return 2
    if result.get("acceptance") and result["acceptance"]["exit_code"] != 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
