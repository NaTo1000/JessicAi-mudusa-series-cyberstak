"""
JessicAi-mudusa-series-cyberstak — CLI entry point.

Usage examples::

    python main.py --mode coding
    python main.py --mode debug
    python main.py --mode design
    python main.py sandbox --model hf:microsoft/phi-2
    python main.py toolkit --list --category web_application
    python main.py vps --scale-up --cpu 8 --ram 32
"""

from __future__ import annotations

import sys
from typing import Any

try:
    import click
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    import yaml
except ImportError as exc:
    print(
        f"Missing dependency: {exc}\n"
        "Run: pip install -r requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)

from aicodex import AicodeXAssistant, Mode
from portmanai import PortmanDebugger, PortmanToolchain
from sandbox import SandboxEnvironment, ModelRunner
from sandbox.environment import SandboxConfig
from sandbox.linux_toolkit import LinuxToolkit, ToolCategory
from vps import VPSScaler, ResourceAllocation

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict[str, Any]:
    try:
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


# ---------------------------------------------------------------------------
# CLI groups and commands
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True)
@click.option(
    "--mode",
    type=click.Choice(["coding", "design", "debug"], case_sensitive=False),
    default="coding",
    show_default=True,
    help="Initial AicodeX mode.",
)
@click.pass_context
def cli(ctx: click.Context, mode: str) -> None:
    """JessicAi cybersecurity and AI development suite."""
    ctx.ensure_object(dict)
    ctx.obj["mode"] = mode

    if ctx.invoked_subcommand is None:
        cfg = _load_config("config/aicodex_config.yaml")
        assistant = AicodeXAssistant(config=cfg)
        assistant.set_mode(mode)

        console.print(
            Panel(
                f"[bold green]AicodeX[/bold green] started in "
                f"[bold cyan]{mode.upper()}[/bold cyan] mode.\n"
                "Type your prompt and press Enter. "
                "Type [bold]exit[/bold] to quit, "
                "[bold]/toggle[/bold] to cycle modes.",
                title="JessicAi Suite",
            )
        )

        while True:
            try:
                prompt = console.input(
                    f"[bold yellow][{assistant.mode.label()}][/bold yellow] > "
                )
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/dim]")
                break

            prompt = prompt.strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", "q"):
                console.print("[dim]Goodbye.[/dim]")
                break
            if prompt.lower() == "/toggle":
                new_mode = assistant.toggle_mode()
                console.print(f"[cyan]Switched to {new_mode.label()} mode.[/cyan]")
                continue

            try:
                result = assistant.run(prompt)
                console.print(Panel(result, border_style="green"))
            except Exception as exc:
                console.print(f"[red]Error:[/red] {exc}")


# ---------------------------------------------------------------------------
# sandbox sub-command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--model", "model_source", default=None, help="Model source (hf:/ gh:/ or path)")
@click.option("--image", default=None, help="Docker image override")
@click.option("--run", "run_cmd", default=None, help="Command to run inside the sandbox")
def sandbox(model_source: str | None, image: str | None, run_cmd: str | None) -> None:
    """Launch the sandboxed Linux environment."""
    cfg_data = _load_config("config/sandbox_config.yaml").get("sandbox", {})

    sb_cfg = SandboxConfig(
        image=image or cfg_data.get("default_image", "kalilinux/kali-rolling"),
        memory_limit=cfg_data.get("memory_limit", "2g"),
        cpu_quota=float(cfg_data.get("cpu_quota", 1.0)),
        network_mode=cfg_data.get("network_mode", "none"),
    )
    sb = SandboxEnvironment(config=sb_cfg)

    console.print(
        Panel(
            f"[bold]Sandbox[/bold]\n"
            f"Image : {sb.config.image}\n"
            f"Docker: {'[green]available[/green]' if sb.docker_available else '[red]not found[/red]'}",
            title="Sandbox Environment",
        )
    )

    if model_source:
        runner = ModelRunner()
        console.print(f"[cyan]Loading model:[/cyan] {model_source}")
        try:
            info = runner.load(model_source)
            console.print(f"[green]Model loaded:[/green] {info.identifier} ({info.framework})")
        except Exception as exc:
            console.print(f"[red]Failed to load model:[/red] {exc}")

    if run_cmd:
        result = sb.run(run_cmd.split())
        console.print(
            Panel(
                result.stdout or result.stderr or "(no output)",
                title=f"Exit code: {result.returncode}",
                border_style="green" if result.success else "red",
            )
        )


# ---------------------------------------------------------------------------
# toolkit sub-command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--list", "do_list", is_flag=True, default=False, help="List available tools")
@click.option(
    "--category",
    default=None,
    help="Filter by category (e.g. web_application, exploitation)",
)
@click.option("--distro", default=None, help="Filter by distro (kali or blackarch)")
@click.option("--installed-only", is_flag=True, default=False)
@click.option("--summary", "do_summary", is_flag=True, default=False)
def toolkit(
    do_list: bool,
    category: str | None,
    distro: str | None,
    installed_only: bool,
    do_summary: bool,
) -> None:
    """Browse the Kali-Linux / Black Arch tool catalogue."""
    tk = LinuxToolkit()

    if do_summary:
        s = tk.summary()
        console.print(Panel(
            f"Total tools : {s['total']}\n"
            f"Installed   : {s['installed']}\n"
            f"Kali tools  : {s['kali_tools']}\n"
            f"BlackArch   : {s['blackarch_tools']}",
            title="Toolkit Summary",
        ))
        return

    cat_enum: ToolCategory | None = None
    if category:
        try:
            cat_enum = ToolCategory[category.upper()]
        except KeyError:
            console.print(f"[red]Unknown category:[/red] {category}")
            console.print("Valid categories: " + ", ".join(c.name.lower() for c in ToolCategory))
            return

    tools = tk.list_tools(category=cat_enum, distro=distro, installed_only=installed_only)

    table = Table(title="Security Tools", show_lines=True)
    table.add_column("Name", style="bold cyan")
    table.add_column("Category")
    table.add_column("Distros")
    table.add_column("Installed", justify="center")

    for t in tools:
        table.add_row(
            t.name,
            t.category.name.replace("_", " ").title(),
            ", ".join(t.distros),
            "[green]✓[/green]" if t.is_installed() else "[red]✗[/red]",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# debug sub-command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_text", required=False)
@click.option("--file", "input_file", default=None, help="Read error/code from file")
def debug(input_text: str | None, input_file: str | None) -> None:
    """Analyse an error or code snippet with PortmanAI."""
    if input_file:
        try:
            with open(input_file) as fh:
                text = fh.read()
        except OSError as exc:
            console.print(f"[red]Cannot read file:[/red] {exc}")
            return
    elif input_text:
        text = input_text
    else:
        console.print("[yellow]Enter the error / code to analyse (Ctrl-D to finish):[/yellow]")
        lines = []
        try:
            while True:
                lines.append(input())
        except EOFError:
            pass
        text = "\n".join(lines)

    if not text.strip():
        console.print("[red]No input provided.[/red]")
        return

    debugger = PortmanDebugger()
    report = debugger.analyse(text)

    severity_colours = {
        "critical": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "green",
    }
    colour = severity_colours.get(report.severity, "white")

    console.print(
        Panel(
            f"[bold]Issue:[/bold] {report.issue}\n\n"
            f"[bold]Severity:[/bold] [{colour}]{report.severity.upper()}[/{colour}]\n"
            f"[bold]Root Cause:[/bold] {report.root_cause}\n\n"
            f"[bold]Suggested Fix:[/bold]\n{report.suggested_fix}\n\n"
            f"[dim]Confidence: {report.confidence:.0%}[/dim]",
            title="PortmanAI Debug Report",
            border_style=colour,
        )
    )


# ---------------------------------------------------------------------------
# vps sub-command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--scale-up", "do_scale_up", is_flag=True)
@click.option("--scale-down", "do_scale_down", is_flag=True)
@click.option("--cpu", "cpu_cores", default=None, type=int, help="Target CPU cores")
@click.option("--ram", "ram_gb", default=None, type=float, help="Target RAM in GB")
@click.option("--gpu", "gpu_count", default=None, type=int, help="Target GPU count")
@click.option("--migrate-to", "provider", default=None, help="Provider to migrate to")
@click.option("--region", default=None, help="Cloud region")
@click.option("--status", "do_status", is_flag=True)
def vps(
    do_scale_up: bool,
    do_scale_down: bool,
    cpu_cores: int | None,
    ram_gb: float | None,
    gpu_count: int | None,
    provider: str | None,
    region: str | None,
    do_status: bool,
) -> None:
    """Manage VPS resource scaling."""
    cfg = _load_config("config/sandbox_config.yaml")
    vps_cfg = cfg.get("sandbox", {}).get("vps", {})
    init = vps_cfg.get("initial_allocation", {})

    scaler = VPSScaler(
        initial=ResourceAllocation(
            cpu_cores=init.get("cpu_cores", 2),
            ram_gb=init.get("ram_gb", 4.0),
            gpu_count=init.get("gpu_count", 0),
            storage_gb=init.get("storage_gb", 20.0),
        )
    )

    if do_scale_up:
        alloc = scaler.scale_up(cpu_cores=cpu_cores, ram_gb=ram_gb, gpu_count=gpu_count)
        console.print(f"[green]Scaled up:[/green] {alloc.to_dict()}")
    elif do_scale_down:
        alloc = scaler.scale_down(cpu_cores=cpu_cores, ram_gb=ram_gb, gpu_count=gpu_count)
        console.print(f"[yellow]Scaled down:[/yellow] {alloc.to_dict()}")
    elif provider:
        alloc = scaler.migrate(provider=provider, region=region or "default")
        console.print(f"[cyan]Migrated:[/cyan] {alloc.to_dict()}")
    elif do_status:
        stats = scaler.local_resource_stats()
        console.print(
            Panel(
                "\n".join(f"{k}: {v}" for k, v in stats.items()),
                title="Local Resource Stats",
            )
        )
    else:
        alloc = scaler.current_allocation.to_dict()
        console.print(Panel(
            "\n".join(f"{k}: {v}" for k, v in alloc.items()),
            title="Current VPS Allocation",
        ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli(obj={})
