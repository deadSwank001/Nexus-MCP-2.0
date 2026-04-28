"""
Nexus MCP CLI — serve, install, status commands.

Usage:
    nexus serve              Start the MCP server
    nexus install cursor     Install for Cursor IDE
    nexus install claude     Install for Claude Code
    nexus status             Check installation status
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="nexus-mcp")
def cli():
    """Nexus MCP — Ollama-powered intelligent project context management."""
    pass


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def serve(verbose: bool):
    """Start the Nexus MCP server."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    from nexus_mcp.mcp.server import NexusMcpServer

    server = NexusMcpServer()
    asyncio.run(server.run())


@cli.command()
@click.argument("target", type=click.Choice(["cursor", "claude", "claude-code"]))
def install(target: str):
    """Install Nexus MCP server for an AI development environment."""
    nexus_path = _find_nexus_binary()

    if target == "cursor":
        _install_cursor(nexus_path)
    elif target in ("claude", "claude-code"):
        _install_claude_code(nexus_path)


@cli.command()
@click.argument("target", type=click.Choice(["cursor", "claude", "claude-code"]))
def uninstall(target: str):
    """Uninstall Nexus MCP server from an AI development environment."""
    if target == "cursor":
        _uninstall_cursor()
    elif target in ("claude", "claude-code"):
        _uninstall_claude_code()


@cli.command()
def status():
    """Show MCP server installation status."""
    nexus_path = _find_nexus_binary()

    table = Table(title="Nexus MCP Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Path", style="dim")

    # Binary
    table.add_row(
        "Binary",
        "✅ Found" if nexus_path else "❌ Not found",
        str(nexus_path) if nexus_path else "Not in PATH",
    )

    # Database
    db_path = Path.home() / ".nexus" / "nexus.db"
    table.add_row(
        "Database",
        "✅ Exists" if db_path.exists() else "⚪ Not created yet",
        str(db_path),
    )

    # Cursor config
    cursor_config = _get_cursor_config_path()
    cursor_installed = _check_cursor_installed(cursor_config)
    table.add_row(
        "Cursor",
        "✅ Installed" if cursor_installed else "⚪ Not installed",
        str(cursor_config),
    )

    # Claude Code config
    claude_config = _get_claude_config_path()
    claude_installed = _check_claude_installed(claude_config)
    table.add_row(
        "Claude Code",
        "✅ Installed" if claude_installed else "⚪ Not installed",
        str(claude_config),
    )

    # Ollama
    ollama_status = _check_ollama()
    table.add_row(
        "Ollama",
        "✅ Running" if ollama_status else "⚠️ Not running (AI features disabled)",
        "http://localhost:11434",
    )

    console.print(table)


# ---------------------------------------------------------------------------
# Installation helpers
# ---------------------------------------------------------------------------

def _find_nexus_binary() -> Path | None:
    """Find the nexus binary in PATH or common locations."""
    import shutil
    path = shutil.which("nexus")
    if path:
        return Path(path)
    # Check common pip install locations
    for candidate in [
        Path(sys.executable).parent / "nexus",
        Path(sys.executable).parent / "nexus.exe",
        Path(sys.executable).parent / "Scripts" / "nexus.exe",
    ]:
        if candidate.exists():
            return candidate
    return None


def _get_cursor_config_path() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", "")) / "Cursor" / "User" / "globalStorage" / "cursor.mcp" / "mcp.json"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Cursor" / "User" / "globalStorage" / "cursor.mcp" / "mcp.json"
    return Path.home() / ".config" / "cursor" / "mcp.json"


def _get_claude_config_path() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config" / "claude" / "config.json"


def _install_cursor(nexus_path: Path | None):
    config_path = _get_cursor_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    config["mcpServers"]["nexus-mcp"] = {
        "command": str(nexus_path) if nexus_path else "nexus",
        "args": ["serve"],
    }

    config_path.write_text(json.dumps(config, indent=2))
    console.print(Panel(
        f"[green]✅ Nexus MCP installed for Cursor[/green]\n"
        f"Config: {config_path}",
        title="Installation Complete",
    ))


def _install_claude_code(nexus_path: Path | None):
    config_path = _get_claude_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    config["mcpServers"]["nexus-mcp"] = {
        "command": str(nexus_path) if nexus_path else "nexus",
        "args": ["serve"],
    }

    config_path.write_text(json.dumps(config, indent=2))
    console.print(Panel(
        f"[green]✅ Nexus MCP installed for Claude Code[/green]\n"
        f"Config: {config_path}",
        title="Installation Complete",
    ))


def _uninstall_cursor():
    config_path = _get_cursor_config_path()
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if "mcpServers" in config and "nexus-mcp" in config["mcpServers"]:
            del config["mcpServers"]["nexus-mcp"]
            config_path.write_text(json.dumps(config, indent=2))
            console.print("[green]✅ Nexus MCP uninstalled from Cursor[/green]")
        else:
            console.print("[yellow]Nexus MCP was not installed in Cursor[/yellow]")
    else:
        console.print("[yellow]Cursor config not found[/yellow]")


def _uninstall_claude_code():
    config_path = _get_claude_config_path()
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if "mcpServers" in config and "nexus-mcp" in config["mcpServers"]:
            del config["mcpServers"]["nexus-mcp"]
            config_path.write_text(json.dumps(config, indent=2))
            console.print("[green]✅ Nexus MCP uninstalled from Claude Code[/green]")
        else:
            console.print("[yellow]Nexus MCP was not installed in Claude Code[/yellow]")
    else:
        console.print("[yellow]Claude config not found[/yellow]")


def _check_cursor_installed(config_path: Path) -> bool:
    if config_path.exists():
        config = json.loads(config_path.read_text())
        return "nexus-mcp" in config.get("mcpServers", {})
    return False


def _check_claude_installed(config_path: Path) -> bool:
    if config_path.exists():
        config = json.loads(config_path.read_text())
        return "nexus-mcp" in config.get("mcpServers", {})
    return False


def _check_ollama() -> bool:
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


if __name__ == "__main__":
    cli()
