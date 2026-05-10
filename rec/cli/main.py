"""
Main CLI entry point for Rec Engine.

Usage:
    rec <command> [options]

Commands:
    ingest          Ingest datasets and knowledge files
    map             Run autonomous semantic mapping
    bml             Business Meaning Layer orchestration
    review          Human-in-the-loop review workflow
    build-config    Generate rec_config.json
    features        Feature engineering pipeline
    train-retrieval Train retrieval model (three-tower)
    train-ranker    Train ranking model
    train-dlrm      Train DLRM model
    build-index     Build ANN vector index
    recommend       Generate recommendations
    rerank          Re-rank recommendations
    evaluate        Evaluate model performance
    feedback        Process feedback data
    retrain         Incremental retraining
    explain         Explain recommendations
    export          Export models and artifacts
"""

import click
from rich.console import Console
from loguru import logger

from .commands import (
    ingest,
    mapping,
    bml,
    hitl,
    config,
    stubs,
)

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="rec")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool):
    """
    Autonomous CLI-Only Recommendation Engine Platform
    
    A production-grade system for generating personalized recommendations
    from arbitrary datasets entirely through local execution.
    """
    if verbose:
        logger.remove()
        logger.add(
            lambda msg: console.print(msg, style="dim"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="DEBUG",
        )
    else:
        logger.remove()
        logger.add(
            lambda msg: console.print(msg, style="dim"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO",
        )


# Register commands
cli.add_command(ingest.ingest)
cli.add_command(mapping.map)
cli.add_command(bml.bml)
cli.add_command(hitl.review)
cli.add_command(config.build_config)
cli.add_command(stubs.features)
cli.add_command(stubs.train_retrieval)
cli.add_command(stubs.train_ranker)
cli.add_command(stubs.train_dlrm)
cli.add_command(stubs.build_index)
cli.add_command(stubs.recommend)
cli.add_command(stubs.rerank)
cli.add_command(stubs.evaluate)
cli.add_command(stubs.feedback)
cli.add_command(stubs.retrain)
cli.add_command(stubs.explain)
cli.add_command(stubs.export)


@cli.command()
def info():
    """Display system information and configuration."""
    console.print("\n[bold cyan]Rec Engine Information[/bold cyan]\n")
    
    info_table = [
        ("Version", "0.1.0"),
        ("Architecture", "CLI-Only, Local-First"),
        ("Supported Formats", "CSV, JSON, JSONL, Parquet, SQLite, PostgreSQL dumps"),
        ("ML Framework", "PyTorch 2.x + PyTorch Lightning"),
        ("Embedding Models", "all-MiniLM-L6-v2, stsb-distilroberta-base"),
        ("Vector Index", "FAISS, hnswlib"),
        ("Data Processing", "Polars, DuckDB, PyArrow"),
    ]
    
    for key, value in info_table:
        console.print(f"[bold]{key}:[/bold] {value}")
    
    console.print("\n[bold cyan]Quick Start[/bold cyan]\n")
    console.print("1. [green]rec ingest --source ./datasets --workspace ./workspace[/green]")
    console.print("2. [green]rec map --workspace ./workspace[/green]")
    console.print("3. [green]rec bml --workspace ./workspace[/green]")
    console.print("4. [green]rec review --workspace ./workspace[/green]")
    console.print("5. [green]rec build-config --workspace ./workspace[/green]")
    console.print("\nSee README.md for complete documentation.")


if __name__ == "__main__":
    cli()
