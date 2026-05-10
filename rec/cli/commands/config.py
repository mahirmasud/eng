"""Configuration Generation Command."""

import json
from pathlib import Path
from datetime import datetime

import click
from rich.console import Console

console = Console()


@click.command("build-config")
@click.option("--workspace", "-w", required=True, help="Path to workspace directory")
@click.option("--output", "-o", default=None, help="Output path for rec_config.json")
def build_config(workspace: str, output: str):
    """
    Generate rec_config.json configuration file.
    
    Creates comprehensive configuration including entity mappings,
    feature roles, retrieval settings, ranking settings, DLRM settings,
    vector index settings, training settings, cold start settings,
    and feedback settings.
    """
    console.print("\n[bold cyan]⚙️ Building Configuration[/bold cyan]\n")
    
    workspace_path = Path(workspace)
    
    if not workspace_path.exists():
        console.print(f"[red]Error:[/red] Workspace does not exist: {workspace}")
        return
    
    # Load all metadata
    metadata = load_workspace_metadata(workspace_path)
    
    # Build configuration
    config = build_rec_config(metadata)
    
    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = workspace_path / "rec_config.json"
    
    # Save configuration
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    
    console.print(f"[green]✓[/green] rec_config.json saved to: {output_path}")
    
    # Display summary
    display_config_summary(config)
    
    console.print("\n[bold green]✓ Configuration complete![/bold green]")
    console.print(f"\nNext step: [cyan]rec features --workspace {workspace}[/cyan]")


def load_workspace_metadata(workspace_path: Path) -> dict:
    """Load all available metadata from workspace."""
    metadata = {}
    
    files_to_load = [
        ("source_manifest", "metadata/source_manifest.json"),
        ("dataset_profile", "metadata/dataset_profile.json"),
        ("schema_fingerprint", "metadata/schema_fingerprint.json"),
        ("domain_configs", "metadata/domain_configs.json"),
        ("resolved_mappings", "metadata/resolved_mappings.json"),
        ("semantic_roles", "metadata/semantic_roles.json"),
        ("entity_graph", "metadata/entity_graph.json"),
        ("feature_catalog", "metadata/feature_catalog.json"),
        ("human_feedback", "metadata/human_feedback.json"),
        ("corrected_mappings", "metadata/corrected_mappings.json"),
    ]
    
    for name, rel_path in files_to_load:
        file_path = workspace_path / rel_path
        if file_path.exists():
            with open(file_path, "r") as f:
                metadata[name] = json.load(f)
            console.print(f"[green]✓[/green] Loaded: {rel_path}")
        else:
            console.print(f"[dim]⊘ Skipped (not found): {rel_path}[/dim]")
    
    return metadata


def build_rec_config(metadata: dict) -> dict:
    """Build comprehensive rec_config.json."""
    
    # Extract entity mappings
    entity_mappings = extract_entity_mappings(metadata)
    
    # Extract feature roles
    feature_roles = extract_feature_roles(metadata)
    
    # Build configuration
    config = {
        "version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "workspace_schema_version": "0.1.0",
        
        "entity_mappings": entity_mappings,
        
        "feature_roles": feature_roles,
        
        "retrieval": {
            "enabled": True,
            "model_type": "three_tower",
            "embedding_dim": 384,
            "num_candidates": 100,
            "ann_index": {
                "type": "faiss",
                "index_type": "IVF1024,PQ64",
                "nprobe": 32,
                "use_gpu": False,
            },
            "similarity_metric": "cosine",
        },
        
        "ranking": {
            "enabled": True,
            "model_type": "mlp_ranker",
            "hidden_layers": [256, 128, 64],
            "dropout": 0.1,
            "activation": "relu",
            "features": {
                "user_features": feature_roles.get("user_features", []),
                "item_features": feature_roles.get("item_features", []),
                "context_features": feature_roles.get("context_features", []),
                "interaction_features": feature_roles.get("interaction_features", []),
            },
        },
        
        "dlrm": {
            "enabled": True,
            "dense_arch": {
                "hidden_layers": [512, 256, 128],
                "activation": "relu",
                "dropout": 0.1,
            },
            "sparse_arch": {
                "embedding_dim": 64,
                "num_embeddings": {},  # Will be computed during feature engineering
            },
            "interaction_layer": {
                "type": "dot_interaction",
                "num_interactions": "auto",
            },
            "prediction_head": {
                "hidden_layers": [256, 128, 64, 1],
                "activation": "relu",
                "output_activation": "sigmoid",
            },
        },
        
        "vector_index": {
            "type": "faiss",
            "index_params": {
                "nlist": 1024,
                "m": 64,
                "nbits": 8,
            },
            "training_params": {
                "min_points_per_cluster": 256,
                "max_training_points": 1000000,
            },
            "metadata_filtering": True,
        },
        
        "training": {
            "batch_size": 2048,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "num_epochs": 10,
            "early_stopping": {
                "enabled": True,
                "patience": 3,
                "min_delta": 0.0001,
            },
            "optimizer": "adamw",
            "scheduler": {
                "type": "cosine_annealing",
                "warmup_epochs": 1,
            },
            "mixed_precision": True,
            "gradient_accumulation_steps": 1,
        },
        
        "cold_start": {
            "enabled": True,
            "strategies": {
                "new_users": {
                    "method": "popularity_prior",
                    "fallback_items": 50,
                    "content_based": True,
                },
                "new_items": {
                    "method": "content_embedding",
                    "similarity_threshold": 0.7,
                    "boost_factor": 1.2,
                },
                "sparse_interactions": {
                    "method": "hybrid",
                    "min_interactions": 5,
                    "confidence_decay": 0.9,
                },
            },
            "metadata_embedding_model": "all-MiniLM-L6-v2",
        },
        
        "feedback": {
            "enabled": True,
            "signal_types": {
                "click": {"weight": 1.0, "decay_days": 30},
                "view": {"weight": 0.5, "decay_days": 14},
                "purchase": {"weight": 5.0, "decay_days": 90},
                "rating": {"weight": 2.0, "decay_days": 60},
                "dwell_time": {"weight": 1.5, "decay_days": 30},
            },
            "negative_sampling": {
                "enabled": True,
                "ratio": 4.0,
                "strategy": "in_batch",
            },
            "replay_buffer": {
                "enabled": True,
                "max_size": 1000000,
                "sample_ratio": 0.2,
            },
        },
        
        "reranking": {
            "enabled": True,
            "diversity": {
                "enabled": True,
                "algorithm": "mmr",
                "lambda_param": 0.5,
            },
            "novelty": {
                "enabled": True,
                "penalize_seen": True,
                "time_window_days": 30,
            },
            "freshness": {
                "enabled": True,
                "boost_new_items": True,
                "decay_days": 7,
            },
            "business_rules": {
                "enabled": True,
                "rules_file": "domain_pack/rules.yaml",
            },
            "fairness": {
                "enabled": False,
                "constraints": [],
            },
        },
        
        "evaluation": {
            "metrics": [
                "recall@10",
                "recall@50",
                "precision@10",
                "precision@50",
                "ndcg@10",
                "map@10",
                "mrr",
                "coverage",
                "diversity",
            ],
            "split_strategy": {
                "type": "temporal",
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
            },
            "offline_evaluation": True,
            "online_ab_testing_ready": False,
        },
        
        "monitoring": {
            "mlflow_tracking": True,
            "experiment_name": "rec_engine_experiment",
            "log_params": True,
            "log_metrics": True,
            "log_models": True,
            "drift_detection": {
                "enabled": True,
                "schema_drift": True,
                "embedding_drift": True,
                "threshold": 0.1,
            },
        },
        
        "export": {
            "formats": ["onnx", "torchscript", "saved_model"],
            "include_metadata": True,
            "quantization": {
                "enabled": False,
                "method": "dynamic",
            },
        },
    }
    
    return config


def extract_entity_mappings(metadata: dict) -> dict:
    """Extract entity mappings from metadata."""
    mappings = {
        "user_id": None,
        "item_id": None,
        "interaction_signal": None,
        "temporal": [],
        "session": [],
        "contextual": [],
    }
    
    # Use corrected mappings if available, otherwise resolved mappings
    mapping_data = metadata.get("corrected_mappings", metadata.get("resolved_mappings", {}))
    
    for dataset in mapping_data.get("datasets", []):
        dataset_name = dataset["dataset"]
        
        for mapping in dataset.get("mappings", []):
            role = mapping["best_match"]["role"]
            column = mapping["column"]
            
            if role == "USER_ID" and not mappings["user_id"]:
                mappings["user_id"] = {
                    "column": column,
                    "dataset": dataset_name,
                }
            elif role == "ITEM_ID" and not mappings["item_id"]:
                mappings["item_id"] = {
                    "column": column,
                    "dataset": dataset_name,
                }
            elif role == "INTERACTION_SIGNAL" and not mappings["interaction_signal"]:
                mappings["interaction_signal"] = {
                    "column": column,
                    "dataset": dataset_name,
                }
            elif role == "TEMPORAL":
                mappings["temporal"].append({
                    "column": column,
                    "dataset": dataset_name,
                })
            elif role == "SESSION":
                mappings["session"].append({
                    "column": column,
                    "dataset": dataset_name,
                })
            elif role == "CONTEXTUAL":
                mappings["contextual"].append({
                    "column": column,
                    "dataset": dataset_name,
                })
    
    return mappings


def extract_feature_roles(metadata: dict) -> dict:
    """Extract feature roles from metadata."""
    feature_catalog = metadata.get("feature_catalog", {})
    
    return {
        "user_features": [f["name"] for f in feature_catalog.get("user_features", [])],
        "item_features": [f["name"] for f in feature_catalog.get("item_features", [])],
        "interaction_features": [f["name"] for f in feature_catalog.get("interaction_features", [])],
        "context_features": [f["name"] for f in feature_catalog.get("contextual_features", [])],
    }


def display_config_summary(config: dict):
    """Display configuration summary."""
    console.print("\n[bold]Configuration Summary:[/bold]\n")
    
    sections = [
        ("Entity Mappings", config.get("entity_mappings", {})),
        ("Feature Roles", config.get("feature_roles", {})),
        ("Retrieval", config.get("retrieval", {})),
        ("Ranking", config.get("ranking", {})),
        ("DLRM", config.get("dlrm", {})),
        ("Cold Start", config.get("cold_start", {})),
        ("Feedback", config.get("feedback", {})),
    ]
    
    for name, section in sections:
        enabled_status = ""
        if isinstance(section, dict):
            if "enabled" in section:
                status = "✓" if section["enabled"] else "✗"
                color = "green" if section["enabled"] else "red"
                enabled_status = f" [{color}]{status}[/{color}]"
        
        console.print(f"   [bold]{name}:[/bold]{enabled_status}")
    
    console.print("\n[green]✓ Configuration ready for training pipeline[/green]")
