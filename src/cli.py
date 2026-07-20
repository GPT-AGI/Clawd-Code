"""CLI entry point for Clawd Codex."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table


def main():
    """CLI main entry point."""
    # Quick path for --version
    if len(sys.argv) == 2 and sys.argv[1] in ['--version', '-v', '-V']:
        from src import __version__
        print(f"clawd-codex version {__version__} (Python)")
        return 0

    parser = argparse.ArgumentParser(
        description="Clawd Codex - Claude Code Python Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  clawd --version          Show version
  clawd login              Configure API keys
  clawd config             Show current configuration
  clawd run --prompt-file TASK.md
  clawd trace .             Inspect teammate tool calls and messages
  clawd team status         Inspect the active teammate team
  clawd team stop coder     Stop one worker and requeue its work
  clawd peer run --repo . --prompt-file TASK.md --peers 3 --communication p2p
  clawd --stream           Start REPL with live response rendering
  clawd                    Start interactive REPL
"""
    )

    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version information'
    )
    parser.add_argument(
        '--config',
        action='store_true',
        help='Show current configuration'
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Enable live response rendering in the REPL'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # login subcommand
    subparsers.add_parser('login', help='Configure API keys')

    # config subcommand
    config_parser = subparsers.add_parser('config', help='Show or switch configuration')
    config_parser.add_argument(
        '--use',
        choices=('glm', 'glm5', 'qwen', 'qwen3.5'),
        help='Switch the default model profile',
    )

    run_parser = subparsers.add_parser('run', help='Run one prompt non-interactively')
    run_parser.add_argument('prompt', nargs='?', help='Prompt text (reads stdin when omitted)')
    run_parser.add_argument('-f', '--prompt-file', type=Path, help='Read the prompt from a file')
    run_parser.add_argument('-C', '--workspace', type=Path, default=Path('.'), help='Workspace root for local tools')
    run_parser.add_argument('--provider', help='Configured provider to use')
    run_parser.add_argument('--model', help='Model override for this run')
    run_parser.add_argument('--max-turns', type=int, default=100, help='Maximum model turns')
    run_parser.add_argument('--stream', dest='run_stream', action='store_true', help='Stream final text')
    run_parser.add_argument('--quiet', action='store_true', help='Hide tool progress')

    trace_parser = subparsers.add_parser('trace', help='Open the teammate trace viewer')
    trace_parser.add_argument('workspace', nargs='?', default='.', help='Workspace containing .clawd state')
    trace_parser.add_argument('--host', default='127.0.0.1', help='Viewer bind host')
    trace_parser.add_argument('--port', type=int, default=8765, help='Viewer bind port (0 chooses a free port)')
    trace_parser.add_argument('--team', help='Team ID to select initially')
    trace_parser.add_argument('--open', action='store_true', help='Open the viewer in the default browser')

    team_parser = subparsers.add_parser('team', help='Inspect and control teammate workers')
    team_subparsers = team_parser.add_subparsers(dest='team_command', required=True)

    def add_team_workspace(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            '-C', '--workspace', type=Path, default=Path('.'),
            help='Workspace containing persistent teammate state',
        )

    team_list_parser = team_subparsers.add_parser('list', help='List persisted teams')
    add_team_workspace(team_list_parser)

    team_status_parser = team_subparsers.add_parser('status', help='Show team, worker, and task status')
    add_team_workspace(team_status_parser)
    team_status_parser.add_argument('--team-id', help='Historical team ID; defaults to the active team')

    team_stop_parser = team_subparsers.add_parser('stop', help='Stop one worker without cancelling the team')
    add_team_workspace(team_stop_parser)
    team_stop_parser.add_argument('teammate', help='Worker name or ID')
    team_stop_parser.add_argument('--reason', help='Reason recorded in the trace')
    team_stop_parser.add_argument(
        '--task-policy', choices=('requeue', 'cancel'), default='requeue',
        help='How to handle unfinished tasks',
    )

    worker_resume_parser = team_subparsers.add_parser('resume-worker', help='Make a stopped worker available again')
    add_team_workspace(worker_resume_parser)
    worker_resume_parser.add_argument('teammate', help='Worker name or ID')

    reassign_parser = team_subparsers.add_parser('reassign', help='Assign a stopped or pending task to a worker')
    add_team_workspace(reassign_parser)
    reassign_parser.add_argument('task', help='Task ID or stable key')
    reassign_parser.add_argument('teammate', help='Replacement worker name or ID')

    team_cancel_parser = team_subparsers.add_parser('cancel', help='Cancel the entire active team')
    add_team_workspace(team_cancel_parser)
    team_cancel_parser.add_argument('--reason', help='Reason recorded in the trace')

    team_resume_parser = team_subparsers.add_parser('resume', help='Resume active persisted team execution')
    add_team_workspace(team_resume_parser)
    team_resume_parser.add_argument('--provider', help='Configured provider to use')
    team_resume_parser.add_argument('--model', help='Model override for worker runs')
    team_resume_parser.add_argument('--max-turns', type=int, default=30, help='Maximum turns per worker task')
    team_resume_parser.add_argument('--max-workers', type=int, help='Maximum concurrent workers')
    team_resume_parser.add_argument('--timeout', type=float, help='Team timeout in seconds')
    team_resume_parser.add_argument('--token-budget', type=int, help='Aggregate token budget')
    team_resume_parser.add_argument('--turn-budget', type=int, help='Aggregate model turn budget')
    team_resume_parser.add_argument('--max-retries', type=int, help='Automatic retries per task')
    team_resume_parser.add_argument('--lease-timeout', type=int, help='Task lease timeout in seconds')
    team_resume_parser.add_argument('--no-retry-failed', action='store_true', help='Do not retry failed tasks')
    team_resume_parser.add_argument('--no-retry-cancelled', action='store_true', help='Do not retry cancelled tasks')

    peer_parser = subparsers.add_parser('peer', help='Run peer-native collaboration')
    peer_subparsers = peer_parser.add_subparsers(dest='peer_command', required=True)
    peer_run_parser = peer_subparsers.add_parser(
        'run', help='Run a fixed-size peer-native collaboration experiment'
    )
    peer_run_parser.add_argument('--repo', type=Path, required=True, help='Git repository root')
    peer_run_parser.add_argument('--prompt-file', type=Path, required=True, help='Top-level mission file')
    peer_run_parser.add_argument('--peers', type=int, required=True, help='Number of equal peers')
    peer_run_parser.add_argument(
        '--communication',
        required=True,
        choices=('solo', 'independent', 'none', 'artifact-only', 'star', 'p2p'),
    )
    peer_run_parser.add_argument(
        '--workspace-mode', required=True, choices=('shared', 'worktree')
    )
    peer_run_parser.add_argument('--provider', help='Configured provider to use')
    peer_run_parser.add_argument('--model', help='Model override for every peer')
    peer_run_parser.add_argument('--timeout-seconds', type=float, default=300.0)
    peer_run_parser.add_argument('--max-turns', type=int, default=30)
    peer_run_parser.add_argument('--max-output-tokens', type=int, default=4096)
    peer_run_parser.add_argument('--token-budget', type=int)
    peer_run_parser.add_argument('--turn-budget', type=int)
    peer_run_parser.add_argument('--output-dir', type=Path)
    peer_run_parser.add_argument('--coordinator-peer', help='Star coordinator ID/name')
    peer_run_parser.add_argument(
        '--acceptance-command',
        help='Shell-like argv string run against the accepted revision',
    )
    peer_run_parser.add_argument(
        '--retain-worktrees', action='store_true', help='Do not remove peer worktrees after the run'
    )

    args = parser.parse_args()

    # Handle --version
    if args.version:
        from src import __version__
        print(f"clawd-codex version {__version__} (Python)")
        return 0

    # Handle --config
    if args.config:
        return show_config()

    # Handle commands
    if args.command == 'login':
        return handle_login()
    elif args.command == 'config':
        return show_config(use_provider=args.use)
    elif args.command == 'run':
        try:
            prompt = _read_run_prompt(args.prompt, args.prompt_file, args.workspace)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        return run_once(
            prompt,
            workspace=args.workspace,
            provider_name=args.provider,
            model=args.model,
            max_turns=args.max_turns,
            stream=args.stream or args.run_stream,
            quiet=args.quiet,
        )
    elif args.command == 'trace':
        from src.teammate.viewer import serve_trace_viewer

        return serve_trace_viewer(
            Path(args.workspace),
            host=args.host,
            port=args.port,
            team_id=args.team,
            open_browser=args.open,
        )
    elif args.command == 'team':
        return handle_team_command(args)
    elif args.command == 'peer':
        return handle_peer_command(args)

    # Default: start REPL
    return start_repl(stream=args.stream)


def _read_run_prompt(
    prompt: str | None,
    prompt_file: Path | None,
    workspace: Path,
) -> str:
    if prompt is not None and prompt_file is not None:
        raise ValueError("provide either prompt text or --prompt-file, not both")
    if prompt_file is not None:
        path = prompt_file.expanduser()
        if not path.is_absolute():
            path = workspace.expanduser().resolve() / path
        text = path.read_text(encoding="utf-8")
    elif prompt is not None:
        text = prompt
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        raise ValueError("provide prompt text, --prompt-file, or piped stdin")
    if not text.strip():
        raise ValueError("prompt must be non-empty")
    return text


def handle_peer_command(args: argparse.Namespace) -> int:
    if args.peer_command != 'run':
        return 1
    from src.peer.runner import run_peer_collaboration

    repo = args.repo.expanduser().resolve()
    prompt_path = args.prompt_file.expanduser()
    if not prompt_path.is_absolute():
        prompt_path = repo / prompt_path
    try:
        mission = prompt_path.read_text(encoding='utf-8')
        acceptance = (
            shlex.split(args.acceptance_command)
            if args.acceptance_command
            else None
        )
        result = run_peer_collaboration(
            mission,
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
            acceptance_command=acceptance,
            cleanup_worktrees=not args.retain_worktrees,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        Console(stderr=True).print(f"[red]Peer run failed:[/red] {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get('status') != 'completed':
        return 2
    acceptance_result = result.get('acceptance')
    if isinstance(acceptance_result, dict) and acceptance_result.get('exit_code') != 0:
        return 3
    return 0


def run_once(
    prompt: str,
    *,
    workspace: Path,
    provider_name: str | None,
    model: str | None,
    max_turns: int,
    stream: bool,
    quiet: bool,
) -> int:
    from src.runner import run_prompt
    from src.tool_system.agent_loop import ToolEvent, summarize_tool_result, summarize_tool_use

    output = Console()
    progress = Console(stderr=True)

    def on_event(event: ToolEvent) -> None:
        if quiet:
            return
        if event.kind == "tool_use":
            summary = summarize_tool_use(event.tool_name, event.tool_input or {})
            suffix = f" ({summary})" if summary else ""
            progress.print(f"[cyan]{event.tool_name}[/cyan]{suffix}")
        elif event.kind == "tool_result" and event.is_error:
            summary = summarize_tool_result(event.tool_name or "Tool", event.tool_output)
            progress.print(f"[red]{summary}[/red]")
        elif event.kind == "tool_error":
            progress.print(f"[red]{event.tool_name or 'Tool'}: {event.error or 'failed'}[/red]")

    def on_text_chunk(chunk: str) -> None:
        output.print(chunk, end="", markup=False, highlight=False, soft_wrap=True)

    try:
        result = run_prompt(
            prompt,
            workspace=workspace,
            provider_name=provider_name,
            model=model,
            max_turns=max_turns,
            stream=stream,
            on_event=on_event,
            on_text_chunk=on_text_chunk if stream else None,
        )
    except Exception as exc:
        progress.print(f"[red]Run failed: {exc}[/red]")
        return 1

    if stream:
        output.print()
    else:
        output.print(result.response_text, markup=False, highlight=False)
    return 2 if result.response_text == "[Max tool turns reached]" else 0


def handle_team_command(args: argparse.Namespace) -> int:
    from src.teammate.control import (
        cancel_team,
        list_teams,
        reassign_task,
        resume_teammate,
        stop_teammate,
        team_status,
    )
    from src.teammate.store import TeamStore

    output = Console()
    errors = Console(stderr=True)
    workspace = Path(args.workspace).expanduser().resolve()

    try:
        if args.team_command == 'list':
            teams = list_teams(workspace)
            table = Table(title="Teammate Teams")
            table.add_column("Active")
            table.add_column("Team")
            table.add_column("ID")
            table.add_column("Status")
            table.add_column("Updated")
            for team in teams:
                table.add_row(
                    "*" if team["active"] else "",
                    str(team["team_name"]),
                    str(team["team_id"]),
                    str(team["status"]),
                    str(team["updated_at"]),
                )
            output.print(table)
            return 0

        if args.team_command == 'status':
            snapshot = team_status(workspace, args.team_id)
            team = snapshot["team"]
            output.print(
                f"[bold]{team['team_name']}[/bold] ({team['team_id']}) "
                f"status=[cyan]{team['status']}[/cyan]"
            )
            agents = Table(title="Workers")
            agents.add_column("Name")
            agents.add_column("Role")
            agents.add_column("Status")
            agents.add_column("Model")
            for agent in snapshot["agents"]:
                agents.add_row(
                    str(agent["name"]),
                    str(agent["role"]),
                    str(agent["status"]),
                    str(agent.get("model") or "default"),
                )
            output.print(agents)
            tasks = Table(title="Tasks")
            tasks.add_column("Key")
            tasks.add_column("Status")
            tasks.add_column("Owner")
            tasks.add_column("Subject")
            names = {agent["agent_id"]: agent["name"] for agent in snapshot["agents"]}
            for task in snapshot["tasks"]:
                owner = task.get("owner")
                tasks.add_row(
                    str(task.get("key") or task["id"]),
                    str(task["status"]),
                    str(names.get(owner, owner or "unassigned")),
                    str(task["subject"]),
                )
            output.print(tasks)
            output.print(
                f"Messages: {snapshot['message_count']}  Events: {snapshot['event_count']}"
            )
            return 0

        store = TeamStore(workspace)
        if args.team_command == 'stop':
            result = stop_teammate(
                store,
                args.teammate,
                task_policy=args.task_policy,
                reason=args.reason,
            )
        elif args.team_command == 'resume-worker':
            result = resume_teammate(store, args.teammate)
        elif args.team_command == 'reassign':
            result = reassign_task(store, args.task, args.teammate)
        elif args.team_command == 'cancel':
            result = cancel_team(store, args.reason)
        elif args.team_command == 'resume':
            from src.runner import resume_team

            result = resume_team(
                workspace=workspace,
                provider_name=args.provider,
                model=args.model,
                max_turns=args.max_turns,
                max_workers=args.max_workers,
                timeout_s=args.timeout,
                token_budget=args.token_budget,
                turn_budget=args.turn_budget,
                max_retries=args.max_retries,
                lease_timeout_s=args.lease_timeout,
                retry_failed=not args.no_retry_failed,
                retry_cancelled=not args.no_retry_cancelled,
            )
        else:  # pragma: no cover - argparse enforces known commands
            raise ValueError(f"unknown team command: {args.team_command}")
        output.print(json.dumps(result, indent=2, ensure_ascii=False), markup=False)
        return 0 if result.get("status") not in {"failed", "blocked"} else 2
    except (OSError, ValueError) as exc:
        errors.print(f"[red]Team command failed: {exc}[/red]")
        return 1


def _show_provider_defaults_table() -> None:
    """Print a table showing available providers and their defaults."""
    from src.providers import PROVIDER_INFO

    console = Console()
    table = Table(title="Available Providers & Defaults", show_header=True, header_style="bold")
    table.add_column("Provider", style="cyan")
    table.add_column("Default Model", style="magenta")
    table.add_column("Base URL", style="green")

    for name, info in PROVIDER_INFO.items():
        table.add_row(
            f"{name} ({info['label']})",
            info["default_model"],
            info["default_base_url"],
        )

    console.print(table)
    console.print()


def handle_login():
    """Interactive API configuration."""
    console = Console()
    console.print("\n[bold blue]Clawd Codex - API Configuration[/bold blue]\n")

    # Show available providers and their defaults
    _show_provider_defaults_table()

    # Select provider
    from src.providers import PROVIDER_INFO
    provider_names = list(PROVIDER_INFO.keys())

    provider = Prompt.ask(
        "Select LLM provider",
        choices=provider_names,
        default="anthropic"
    )

    info = PROVIDER_INFO[provider]

    # Input API Key
    api_key = Prompt.ask(
        f"Enter {provider.upper()} API Key",
        password=True
    )

    if not api_key:
        console.print("\n[red]Error: API Key cannot be empty[/red]")
        return 1

    # Optional: Base URL (show default)
    console.print(f"\n[dim]Default:[/dim] {info['default_base_url']}")
    base_url = Prompt.ask(
        f"{provider.upper()} Base URL",
        default=info["default_base_url"]
    )

    # Optional: Default Model (show available options)
    console.print(f"\n[dim]Available models:[/dim] {', '.join(info['available_models'])}")
    console.print(f"[dim]Default:[/dim] [bold]{info['default_model']}[/bold]")
    default_model = Prompt.ask(
        f"{provider.upper()} Default Model",
        default=info["default_model"]
    )

    # Save configuration
    from src.config import set_api_key, set_default_provider

    set_api_key(provider, api_key=api_key, base_url=base_url, default_model=default_model)
    set_default_provider(provider)

    console.print(f"\n[green]✓ {provider.upper()} API Key saved successfully![/green]")
    console.print(f"[green]✓ Default provider set to: {provider}[/green]\n")
    return 0


def show_config(use_provider: str | None = None):
    """Show current configuration and optionally switch the default provider."""
    console = Console()

    try:
        from src.config import get_config_path, load_config, use_model_profile

        if use_provider is not None:
            profile = use_model_profile(use_provider)
            console.print(
                f"\n[green]✓ Active model profile switched to: {profile['name']} "
                f"({profile['provider']}/{profile['default_model']})[/green]"
            )

        config = load_config()
        config_path = get_config_path()

        console.print(f"\n[bold]Configuration File:[/bold] {config_path}\n")
        console.print("[bold]Current Configuration:[/bold]\n")

        # Show default provider
        console.print(f"[cyan]Default Provider:[/cyan] {config.get('default_provider', 'Not set')}")
        console.print(f"[cyan]Active Profile:[/cyan] {config.get('active_profile') or 'custom'}")

        # Show providers (without showing full API keys)
        console.print("\n[cyan]Configured Providers:[/cyan]")
        for provider_name, provider_config in config.get("providers", {}).items():
            api_key = provider_config.get("api_key", "")
            masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "Not set"

            console.print(f"\n  [yellow]{provider_name.upper()}:[/yellow]")
            console.print(f"    API Key: {masked_key}")
            console.print(f"    Base URL: {provider_config.get('base_url', 'Not set')}")
            console.print(f"    Default Model: {provider_config.get('default_model', 'Not set')}")

        console.print()

    except Exception as e:
        console.print(f"\n[red]Error loading configuration: {e}[/red]\n")
        return 1

    return 0


def start_repl(stream: bool = False):
    """Start interactive REPL."""
    from src.config import get_default_provider
    from src.repl import ClawdREPL

    provider = get_default_provider()
    repl = ClawdREPL(provider_name=provider, stream=stream)
    repl.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
