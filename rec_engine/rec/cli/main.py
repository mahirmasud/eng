"""
CLI Main Entry Point
Autonomous Recommendation Engine Platform
"""

import click
from typing import Optional
import sys
import os

from rec.cli.commands import (
    ingest_cmd,
    map_cmd,
    bml_cmd,
    review_cmd,
    build_config_cmd,
    features_cmd,
    train_retrieval_cmd,
    train_ranker_cmd,
    train_dlrm_cmd,
    build_index_cmd,
    recommend_cmd,
    rerank_cmd,
    evaluate_cmd,
    feedback_cmd,
    retrain_cmd,
    explain_cmd,
    export_cmd,
)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Autonomous Recommendation Engine Platform
    
    A fully local, CLI-only recommendation system that can:
    - Ingest arbitrary datasets
    - Autonomously understand schemas
    - Generate embeddings and build models
    - Retrieve and rank recommendations
    - Learn from feedback
    
    All operations run locally without requiring backend infrastructure.
    """
    pass


# Register all commands
cli.add_command(ingest_cmd, name="ingest")
cli.add_command(map_cmd, name="map")
cli.add_command(bml_cmd, name="bml")
cli.add_command(review_cmd, name="review")
cli.add_command(build_config_cmd, name="build-config")
cli.add_command(features_cmd, name="features")
cli.add_command(train_retrieval_cmd, name="train-retrieval")
cli.add_command(train_ranker_cmd, name="train-ranker")
cli.add_command(train_dlrm_cmd, name="train-dlrm")
cli.add_command(build_index_cmd, name="build-index")
cli.add_command(recommend_cmd, name="recommend")
cli.add_command(rerank_cmd, name="rerank")
cli.add_command(evaluate_cmd, name="evaluate")
cli.add_command(feedback_cmd, name="feedback")
cli.add_command(retrain_cmd, name="retrain")
cli.add_command(explain_cmd, name="explain")
cli.add_command(export_cmd, name="export")


if __name__ == "__main__":
    cli()
