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
    parser.add_argument(
        '--workspace',
        metavar='PATH',
        default=None,
        help='Workspace/project root directory for the REPL (default: current working directory)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # login subcommand
    login_parser = subparsers.add_parser('login', help='Configure API keys')

    # config subcommand
    config_parser = subparsers.add_parser('config', help='Show current configuration')

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

    # Default: start REPL
    workspace = _resolve_workspace(args.workspace)
    if workspace is None and args.workspace is not None:
        # _resolve_workspace already printed the error
        return 1
    return start_repl(stream=args.stream, workspace=workspace)


def _resolve_workspace(workspace_arg: str | None) -> Path | None:
    """Resolve and validate the workspace path argument.

    Returns the resolved canonical Path on success, or None on failure (error
    is printed to stderr before returning).  Returns None also when
    workspace_arg is None (caller must distinguish via the original arg).
    """
    if workspace_arg is None:
        return None

    try:
        p = Path(workspace_arg).expanduser().resolve()
    except Exception as exc:
        print(f"Error: invalid workspace path '{workspace_arg}': {exc}", file=sys.stderr)
        return None

    if not p.exists():
        print(f"Error: workspace does not exist: {p}", file=sys.stderr)
        return None

    if not p.is_dir():
        print(f"Error: workspace is not a directory: {p}", file=sys.stderr)
        return None

    return p


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


def start_repl(stream: bool = False, workspace: Path | None = None):
    """Start interactive REPL.

    Args:
        stream: Enable live response rendering.
        workspace: Explicit workspace root directory.  When None the REPL uses
                   its default (Path.cwd() at construction time).
    """
    from src.config import get_default_provider
    from src.repl import ClawdREPL

    provider = get_default_provider()
    repl = ClawdREPL(provider_name=provider, stream=stream, workspace_root=workspace)
    repl.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
