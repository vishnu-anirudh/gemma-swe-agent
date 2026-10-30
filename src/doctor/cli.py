"""RepoGraph Doctor CLI Interface.

Allows developers to inspect code-graphs, analyze blast-radius breakages,
synthesize SRI patches, and launch the interactive visual studio.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from src.doctor.graph_engine import CodeGraphEngine
from src.doctor.repair_agent import RepoGraphRepairAgent
from src.doctor.scenarios import SCENARIOS
from src.doctor.server import start_server


def main():
    parser = argparse.ArgumentParser(
        description="RepoGraph Doctor: Autonomous Code-Graph Blast-Radius & Repair Studio"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="scenario_1",
        help="Pre-configured scenario ID (scenario_1, scenario_2, scenario_3)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the interactive web-based visual studio",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the interactive web server (default: 8765)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="",
        help="Path to local directory to scan for code-graph dependencies",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="",
        help="Target function/class symbol to analyze for blast radius",
    )

    args = parser.parse_args()
    console = Console()

    if args.serve:
        console.print(
            Panel.fit(
                f"[bold cyan]RepoGraph Doctor Interactive Studio[/bold cyan]\n"
                f"URL: [bold green]http://127.0.0.1:{args.port}[/bold green]\n"
                f"[dim]Sub-3B Post-Trained Gemma-2-2B Local Agent[/dim]",
                title="🚀 Server Launched",
            )
        )
        server = start_server(port=args.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down server...[/yellow]")
            server.server_close()
        return

    # Scenario Mode
    sc_id = args.scenario
    if sc_id not in SCENARIOS:
        console.print(f"[red]Scenario '{sc_id}' not found. Choose from {list(SCENARIOS.keys())}[/red]")
        sys.exit(1)

    sc = SCENARIOS[sc_id]
    console.print(
        Panel.fit(
            f"[bold magenta]{sc.title}[/bold magenta]\n"
            f"[dim]{sc.description}[/dim]\n"
            f"Target: [bold yellow]{sc.target_node}[/bold yellow]\n"
            f"Breaking Change: [italic]{sc.breaking_change}[/italic]",
            title="🔍 Scenario Selected",
        )
    )

    engine = CodeGraphEngine()
    for fpath, content in sc.files.items():
        engine.add_file(fpath, content)

    engine.compute_node_embeddings()
    blast_items = engine.compute_blast_radius(sc.target_node, sc.breaking_change)

    table = Table(title=f"Blast Radius Analysis ({len(blast_items)} affected callers)")
    table.add_column("Caller Symbol", style="cyan", no_wrap=True)
    table.add_column("Severity", style="bold red")
    table.add_column("Hops", justify="right")
    table.add_column("File Path", style="dim")
    table.add_column("Impact Diagnostic", style="white")

    for item in blast_items:
        color = "red" if item.severity == "CRITICAL" else "yellow"
        table.add_row(
            item.node_name,
            f"[{color}]{item.severity}[/{color}]",
            str(item.distance),
            item.file_path,
            item.reason,
        )

    console.print(table)

    console.print("\n[bold cyan]Synthesizing and Verifying SRI Patches with Gemma-2-2B...[/bold cyan]")
    agent = RepoGraphRepairAgent(engine)
    report = agent.repair_blast_radius(
        target_node_name=sc.target_node,
        breaking_change_desc=sc.breaking_change,
        blast_items=blast_items,
        test_verifier=sc.test_verifier,
    )

    console.print(
        Panel.fit(
            f"Total Affected Callers: [bold]{report.total_affected_callers}[/bold]\n"
            f"Clean SRI Patches Applied: [bold green]{report.repaired_count}[/bold green]\n"
            f"Dual-Suite Test Verification: [{'green' if report.tests_passed else 'red'}]{report.test_output}[/]",
            title="🎯 Autonomous Repair Summary",
        )
    )

    for rep in report.repairs:
        console.print(f"\n[bold green]✓ Repaired:[/bold green] [cyan]{rep.node_id}[/cyan]")
        syntax = Syntax(rep.sri_block, "diff", theme="monokai", line_numbers=False)
        console.print(syntax)


if __name__ == "__main__":
    main()
