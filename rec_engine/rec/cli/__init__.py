"""
CLI Commands Module
Autonomous Recommendation Engine Platform
"""

from rec.cli.commands.ingest import ingest_cmd
from rec.cli.commands.mapping import map_cmd
from rec.cli.commands.bml import bml_cmd
from rec.cli.commands.hitl import review_cmd
from rec.cli.commands.config import build_config_cmd
from rec.cli.commands.stubs import (
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

__all__ = [
    "ingest_cmd",
    "map_cmd",
    "bml_cmd",
    "review_cmd",
    "build_config_cmd",
    "features_cmd",
    "train_retrieval_cmd",
    "train_ranker_cmd",
    "train_dlrm_cmd",
    "build_index_cmd",
    "recommend_cmd",
    "rerank_cmd",
    "evaluate_cmd",
    "feedback_cmd",
    "retrain_cmd",
    "explain_cmd",
    "export_cmd",
]
