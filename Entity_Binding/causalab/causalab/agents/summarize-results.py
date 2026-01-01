#!/usr/bin/env -S uv run python
"""Summarize results from a SLURM array job run."""

import argparse
import json
from pathlib import Path
from io import StringIO
import numpy as np

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


def summarize_job(job_id: str, expected_count: int):
    """Generate summary for a completed job."""

    # Read all result files
    job_dir = Path(f"/mnt/polished-lake/data/causal_agent_runs/{job_id}")

    if not job_dir.exists():
        print(f"Error: Job directory not found: {job_dir}")
        return

    result_files = list(job_dir.glob("*.json"))
    if not result_files:
        print(f"No results found in {job_dir}")
        return

    # Parse results
    results = []
    for result_file in result_files:
        with open(result_file) as f:
            results.append(json.load(f))

    actual_count = len(results)

    # Compute statistics
    total_runs = len(results)
    successful = sum(1 for r in results if not r.get("issues"))
    with_issues = total_runs - successful

    graded_results = [r for r in results if r.get("grade") is not None]
    grades = [r["grade"] for r in graded_results]

    # Compute cost statistics
    costed_results = [r for r in results if r.get("total_cost_usd") is not None]
    costs = [r["total_cost_usd"] for r in costed_results]
    total_cost = sum(costs)

    # Create console that captures to string with ANSI codes
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=102)

    # Generate rich formatted summary
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]SUMMARY FOR JOB {job_id}[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")

    console.print(f"[bold]Total runs:[/bold]           {actual_count}/{expected_count}")
    missing_count = expected_count - actual_count
    if missing_count > 0:
        console.print(
            f"[bold]Missing runs:[/bold]         [red]{missing_count}[/red] [dim](crashed before writing results)[/dim]"
        )

    console.print(
        f"[bold]Successful:[/bold]           [green]{successful}[/green] ({100 * successful / total_runs:.0f}%)"
    )
    console.print(
        f"[bold]With issues:[/bold]          [red]{with_issues}[/red] ({100 * with_issues / total_runs:.0f}%)"
    )
    console.print()

    avg_cost = total_cost / len(costed_results) if costed_results else 0
    console.print(
        f"[bold]Total cost:[/bold]           [green]${total_cost:.4f}[/green]"
    )
    console.print(f"[bold]Average cost/trial:[/bold]   [cyan]${avg_cost:.4f}[/cyan]")
    if len(costed_results) < total_runs:
        console.print(
            f"[dim]  ({len(costed_results)}/{total_runs} runs have cost data)[/dim]"
        )
    console.print()

    if grades:
        avg_grade = sum(grades) / len(grades)
        std_grade = np.std(grades, ddof=1) if len(grades) > 1 else 0
        best = max(graded_results, key=lambda r: r["grade"])
        worst = min(graded_results, key=lambda r: r["grade"])

        console.print(
            f"[bold]Average grade:[/bold]        [cyan]{avg_grade:.3f}[/cyan] (±{std_grade:.3f} stdev)"
        )
        console.print(
            f"[bold]Best performing:[/bold]      [green]{best['trial']}[/green] ({best['grade']:.3f})"
        )
        console.print(
            f"[bold]Worst performing:[/bold]     [yellow]{worst['trial']}[/yellow] ({worst['grade']:.3f})"
        )

        # Print per-trial scores
        console.print()
        console.print("[bold]Per-trial scores:[/bold]")
        # Sort by trial name/number for easier reading
        sorted_graded = sorted(graded_results, key=lambda r: r["trial"])
        for r in sorted_graded:
            trial_number = r["trial"]
            grade = r["grade"]

            # Color based on performance
            if grade >= avg_grade:
                console.print(f"  {trial_number}: [green]{grade:.3f}[/green]")
            else:
                console.print(f"  {trial_number}: [yellow]{grade:.3f}[/yellow]")

            # Print GitHub link if available
            if "github_link" in r and r["github_link"]:
                console.print(f"    [dim]{r['github_link']}[/dim]")

    # Print full issues for any runs with problems
    runs_with_issues = [r for r in results if r.get("issues")]
    if runs_with_issues:
        console.print()
        console.print(f"[bold red]{'=' * 60}[/bold red]")
        console.print("[bold red]ISSUES[/bold red]")
        console.print(f"[bold red]{'=' * 60}[/bold red]")

        for r in runs_with_issues:
            console.print()

            # Add dir link as subtitle if available
            subtitle_text = None

            if "trial" in r:
                trial_number = r["trial"]
                trial_dir = f"/mnt/polished-lake/data/causal_agent_runs/worktrees/{job_id}/trial_{trial_number}"
                subtitle_text = f"[dim]{trial_dir}[/dim]"

            console.print(
                Panel(
                    Markdown(r["issues"]),
                    title=f"[bold red]Run:[/bold red] {r['trial']}",
                    subtitle=subtitle_text,
                    border_style="red",
                )
            )

    console.print()
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")

    # Get the formatted text with ANSI codes
    summary_text = buffer.getvalue()

    # Write to file
    summary_file = job_dir / "summary.txt"
    summary_file.write_text(summary_text)

    # Also print to stdout (captured in SLURM output)
    print(summary_text)
    print(f"Summary written to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Summarize results from a SLURM job")
    parser.add_argument("job_id", help="SLURM job ID")
    parser.add_argument("expected_count", type=int, help="Expected number of tasks")
    args = parser.parse_args()

    summarize_job(args.job_id, args.expected_count)


if __name__ == "__main__":
    main()
