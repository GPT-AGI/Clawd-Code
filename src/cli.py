"""CLI entry point for Clawd Codex."""

from __future__ import annotations

import argparse
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
    login_parser = subparsers.add_parser('login', help='Configure API keys')

    # config subcommand
    config_parser = subparsers.add_parser('config', help='Show current configuration')

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
        return show_config()
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


def show_config():
    """Show current configuration."""
    console = Console()

    try:
        from src.config import load_config, get_config_path

        config = load_config()
        config_path = get_config_path()

        console.print(f"\n[bold]Configuration File:[/bold] {config_path}\n")
        console.print("[bold]Current Configuration:[/bold]\n")

        # Show default provider
        console.print(f"[cyan]Default Provider:[/cyan] {config.get('default_provider', 'Not set')}")

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
