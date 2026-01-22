"""Command-line interface for Arbiter evaluation framework.

This module provides a CLI for running evaluations without writing Python code.
Enables CI/CD integration, scripting, and accessibility for non-Python users.

Requirements:
    pip install arbiter-ai[cli]

Usage:
    arbiter evaluate --output "text" --reference "ref" --evaluators semantic
    arbiter batch --file inputs.jsonl --evaluators semantic --output results.json
    arbiter compare --output-a "A" --output-b "B" --criteria "accuracy"
    arbiter list-evaluators
    arbiter cost --model gpt-4o-mini --input-tokens 1000 --output-tokens 500
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

try:
    import typer
    from rich.console import Console
    from rich.table import Table
except ImportError:
    raise ImportError(
        "CLI dependencies not installed. Install with: pip install arbiter-ai[cli]"
    )

# Initialize Typer app and Rich console
app = typer.Typer(
    name="arbiter",
    help="Arbiter - Production-grade LLM evaluation framework",
    add_completion=False,
)
console = Console()


def _run_async(coro: Any) -> Any:
    """Run an async coroutine in a sync context."""
    return asyncio.run(coro)


@app.command()
def evaluate(
    output: str = typer.Option(
        ..., "--output", "-o", help="The LLM output to evaluate"
    ),
    reference: Optional[str] = typer.Option(
        None, "--reference", "-r", help="Reference text for comparison"
    ),
    criteria: Optional[str] = typer.Option(
        None, "--criteria", "-c", help="Evaluation criteria (for custom_criteria)"
    ),
    evaluators: str = typer.Option(
        "semantic", "--evaluators", "-e", help="Comma-separated evaluator names"
    ),
    model: str = typer.Option(
        "gpt-4o-mini", "--model", "-m", help="LLM model for evaluation"
    ),
    threshold: float = typer.Option(
        0.7, "--threshold", "-t", help="Score threshold for pass/fail"
    ),
    format_output: str = typer.Option(
        "default", "--format", "-f", help="Output format: default, json, quiet"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Evaluate a single LLM output."""
    from arbiter_ai import evaluate as api_evaluate

    evaluator_list = [e.strip() for e in evaluators.split(",")]

    async def run_evaluation() -> None:
        result = await api_evaluate(
            output=output,
            reference=reference,
            criteria=criteria,
            evaluators=evaluator_list,
            model=model,
            threshold=threshold,
        )

        if format_output == "json":
            # JSON output
            data = {
                "overall_score": result.overall_score,
                "passed": result.passed,
                "scores": {s.name: s.value for s in result.scores},
                "cost": await result.total_llm_cost(),
            }
            console.print_json(json.dumps(data))
        elif format_output == "quiet":
            # Minimal output - just the score
            console.print(f"{result.overall_score:.2f}")
        else:
            # Default human-readable output
            status = "[green]PASSED[/green]" if result.passed else "[red]FAILED[/red]"
            console.print("\n[bold]Evaluation Results[/bold]")
            console.print("=" * 40)
            console.print(f"Overall Score: {result.overall_score:.2f}  {status}")
            console.print(f"Threshold: {threshold}")

            if result.scores:
                console.print("\n[bold]Scores:[/bold]")
                for score in result.scores:
                    conf = (
                        f" (confidence: {score.confidence:.2f})"
                        if score.confidence
                        else ""
                    )
                    console.print(f"  - {score.name}: {score.value:.2f}{conf}")

            cost = await result.total_llm_cost()
            console.print(f"\n[dim]Cost: ${cost:.6f}[/dim]")

            if verbose:
                console.print(f"[dim]LLM Calls: {len(result.interactions)}[/dim]")
                console.print(
                    f"[dim]Processing Time: {result.processing_time:.2f}s[/dim]"
                )

    _run_async(run_evaluation())


@app.command()
def batch(
    file: Path = typer.Option(
        ..., "--file", "-f", help="Input file (JSONL, JSON, or CSV)"
    ),
    evaluators: str = typer.Option(
        "semantic", "--evaluators", "-e", help="Comma-separated evaluator names"
    ),
    model: str = typer.Option(
        "gpt-4o-mini", "--model", "-m", help="LLM model for evaluation"
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path (JSON format)"
    ),
    max_concurrency: int = typer.Option(
        10, "--concurrency", help="Maximum parallel evaluations"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show progress"),
) -> None:
    """Evaluate multiple LLM outputs from a file."""
    from arbiter_ai import batch_evaluate

    # Load input file
    items = _load_input_file(file)
    if not items:
        console.print("[red]Error: No items found in input file[/red]")
        raise typer.Exit(1)

    evaluator_list = [e.strip() for e in evaluators.split(",")]

    async def run_batch() -> None:
        def progress_callback(completed: int, total: int, result: Any) -> None:
            if verbose:
                console.print(f"Progress: {completed}/{total}", end="\r")

        result = await batch_evaluate(
            items=items,
            evaluators=evaluator_list,
            model=model,
            max_concurrency=max_concurrency,
            progress_callback=progress_callback if verbose else None,
        )

        if verbose:
            console.print()  # Clear progress line

        # Output results
        if output_path:
            # Use the built-in file writing capability
            result.to_json(path=str(output_path), indent=2)
            console.print(f"[green]Results written to {output_path}[/green]")
        else:
            # Print to stdout
            json_str = result.to_json(indent=2)
            if json_str is not None:
                console.print_json(json_str)

        # Summary
        console.print(f"\n[bold]Summary:[/bold] {result.summary()}")

    _run_async(run_batch())


@app.command()
def compare(
    output_a: str = typer.Option(
        ..., "--output-a", "-a", help="First output to compare"
    ),
    output_b: str = typer.Option(
        ..., "--output-b", "-b", help="Second output to compare"
    ),
    criteria: Optional[str] = typer.Option(
        None, "--criteria", "-c", help="Comparison criteria"
    ),
    reference: Optional[str] = typer.Option(
        None, "--reference", "-r", help="Reference context (e.g., user question)"
    ),
    model: str = typer.Option(
        "gpt-4o-mini", "--model", "-m", help="LLM model for comparison"
    ),
    format_output: str = typer.Option(
        "default", "--format", "-f", help="Output format: default, json"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Compare two LLM outputs to determine which is better."""
    from arbiter_ai import compare as api_compare

    async def run_comparison() -> None:
        result = await api_compare(
            output_a=output_a,
            output_b=output_b,
            criteria=criteria,
            reference=reference,
            model=model,
        )

        if format_output == "json":
            data = {
                "winner": result.winner,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "aspect_scores": result.aspect_scores,
            }
            console.print_json(json.dumps(data))
        else:
            # Human-readable output
            winner_display = (
                "Tie"
                if result.winner == "tie"
                else result.winner.replace("_", " ").title()
            )
            color = "yellow" if result.winner == "tie" else "green"

            console.print("\n[bold]Comparison Results[/bold]")
            console.print("=" * 40)
            console.print(f"Winner: [{color}]{winner_display}[/{color}]")
            console.print(f"Confidence: {result.confidence:.2f}")

            if verbose:
                console.print("\n[bold]Reasoning:[/bold]")
                console.print(result.reasoning)

                if result.aspect_scores:
                    console.print("\n[bold]Aspect Scores:[/bold]")
                    for aspect, scores in result.aspect_scores.items():
                        a_score = scores.get("output_a", 0.0)
                        b_score = scores.get("output_b", 0.0)
                        console.print(f"  {aspect}: A={a_score:.2f}, B={b_score:.2f}")

    _run_async(run_comparison())


@app.command("list-evaluators")
def list_evaluators() -> None:
    """List all available evaluators."""
    from arbiter_ai.core.registry import get_available_evaluators

    evaluators = get_available_evaluators()

    table = Table(title="Available Evaluators")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="dim")

    descriptions = {
        "semantic": "Semantic similarity between output and reference",
        "factuality": "Factual accuracy of claims in the output",
        "groundedness": "Whether output is grounded in provided sources",
        "relevance": "Relevance of output to the query/criteria",
        "custom_criteria": "Evaluate against custom criteria",
        "pairwise": "Compare two outputs directly",
    }

    for name in sorted(evaluators):
        desc = descriptions.get(name, "Custom evaluator")
        table.add_row(name, desc)

    console.print(table)


@app.command()
def cost(
    model: str = typer.Option(
        ..., "--model", "-m", help="Model name (e.g., gpt-4o-mini)"
    ),
    input_tokens: int = typer.Option(
        ..., "--input-tokens", "-i", help="Number of input tokens"
    ),
    output_tokens: int = typer.Option(
        ..., "--output-tokens", "-o", help="Number of output tokens"
    ),
    cached_tokens: int = typer.Option(
        0, "--cached-tokens", help="Number of cached tokens"
    ),
) -> None:
    """Calculate cost for a given model and token count."""
    from arbiter_ai.core.cost_calculator import get_cost_calculator

    calc = get_cost_calculator()
    cost_value = calc.calculate_cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
    )

    console.print("\n[bold]Cost Estimate[/bold]")
    console.print("=" * 40)
    console.print(f"Model: {model}")
    console.print(f"Input tokens: {input_tokens:,}")
    console.print(f"Output tokens: {output_tokens:,}")
    if cached_tokens > 0:
        console.print(f"Cached tokens: {cached_tokens:,}")
    console.print(f"\n[green]Estimated cost: ${cost_value:.6f}[/green]")


def _load_input_file(file_path: Path) -> list[dict[str, Any]]:
    """Load input items from a file (JSONL, JSON, or CSV)."""
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    suffix = file_path.suffix.lower()
    items: list[dict[str, Any]] = []

    try:
        if suffix == ".jsonl":
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
        elif suffix == ".json":
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    items = data
                else:
                    items = [data]
        elif suffix == ".csv":
            import csv

            with open(file_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                items = list(reader)
        else:
            console.print(f"[red]Error: Unsupported file format: {suffix}[/red]")
            console.print("Supported formats: .jsonl, .json, .csv")
            raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing JSON: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        raise typer.Exit(1)

    return items


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
