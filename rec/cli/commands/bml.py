"""Business Meaning Layer Command."""

import json
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command("bml")
@click.option("--workspace", "-w", required=True, help="Path to workspace directory")
def bml(workspace: str):
    """
    Run Business Meaning Layer orchestration.
    
    Detects users, items, interaction events, recommendation signals,
    and infers behavioral meaning.
    """
    console.print("\n[bold cyan]🎯 Running Business Meaning Layer[/bold cyan]\n")
    
    workspace_path = Path(workspace)
    
    if not workspace_path.exists():
        console.print(f"[red]Error:[/red] Workspace does not exist: {workspace}")
        return
    
    # Load resolved mappings
    mappings_path = workspace_path / "metadata" / "resolved_mappings.json"
    if not mappings_path.exists():
        console.print("[red]Error:[/red] No resolved mappings found. Run 'rec map' first.")
        return
    
    with open(mappings_path, "r") as f:
        resolved_mappings = json.load(f)
    
    # Generate semantic roles
    semantic_roles = generate_semantic_roles(resolved_mappings)
    
    # Generate entity graph
    entity_graph = generate_entity_graph(semantic_roles)
    
    # Generate feature catalog
    feature_catalog = generate_feature_catalog(resolved_mappings)
    
    # Save outputs
    output_dir = workspace_path / "metadata"
    
    with open(output_dir / "semantic_roles.json", "w") as f:
        json.dump(semantic_roles, f, indent=2, default=str)
    console.print("[green]✓[/green] semantic_roles.json created")
    
    with open(output_dir / "entity_graph.json", "w") as f:
        json.dump(entity_graph, f, indent=2, default=str)
    console.print("[green]✓[/green] entity_graph.json created")
    
    with open(output_dir / "feature_catalog.json", "w") as f:
        json.dump(feature_catalog, f, indent=2, default=str)
    console.print("[green]✓[/green] feature_catalog.json created")
    
    console.print("\n[bold green]✓ BML complete![/bold green]")
    console.print(f"\nNext step: [cyan]rec review --workspace {workspace}[/cyan]")


def generate_semantic_roles(resolved_mappings: dict) -> dict:
    """Generate semantic role classifications."""
    roles = {
        "IDENTITY": [],
        "SIGNAL": [],
        "FEATURE": [],
        "EVENT": [],
        "SESSION": [],
        "TEMPORAL": [],
        "CONTEXTUAL": [],
    }
    
    for dataset in resolved_mappings.get("datasets", []):
        dataset_name = dataset["dataset"]
        
        for mapping in dataset.get("mappings", []):
            role = mapping["best_match"]["role"]
            column = mapping["column"]
            
            if role in ["USER_ID", "ITEM_ID"]:
                roles["IDENTITY"].append({
                    "dataset": dataset_name,
                    "column": column,
                    "role_type": "USER_ID" if role == "USER_ID" else "ITEM_ID",
                })
            elif role == "INTERACTION_SIGNAL":
                roles["SIGNAL"].append({
                    "dataset": dataset_name,
                    "column": column,
                })
            elif role in ["ITEM_FEATURE", "USER_FEATURE"]:
                roles["FEATURE"].append({
                    "dataset": dataset_name,
                    "column": column,
                    "feature_type": role,
                })
            elif role == "TEMPORAL":
                roles["TEMPORAL"].append({
                    "dataset": dataset_name,
                    "column": column,
                })
            elif role == "SESSION":
                roles["SESSION"].append({
                    "dataset": dataset_name,
                    "column": column,
                })
            elif role == "CONTEXTUAL":
                roles["CONTEXTUAL"].append({
                    "dataset": dataset_name,
                    "column": column,
                })
    
    return {
        "roles": roles,
        "summary": {k: len(v) for k, v in roles.items()},
    }


def generate_entity_graph(semantic_roles: dict) -> dict:
    """Generate entity relationship graph."""
    graph = {
        "nodes": [],
        "edges": [],
    }
    
    # Add identity nodes
    for identity in semantic_roles.get("roles", {}).get("IDENTITY", []):
        graph["nodes"].append({
            "id": f"{identity['dataset']}:{identity['column']}",
            "type": identity["role_type"],
            "dataset": identity["dataset"],
        })
    
    # Add edges from interactions
    for signal in semantic_roles.get("roles", {}).get("SIGNAL", []):
        graph["edges"].append({
            "source": "user",
            "target": "item",
            "signal_column": signal["column"],
            "dataset": signal["dataset"],
        })
    
    return graph


def generate_feature_catalog(resolved_mappings: dict) -> dict:
    """Generate feature catalog."""
    catalog = {
        "user_features": [],
        "item_features": [],
        "interaction_features": [],
        "contextual_features": [],
    }
    
    for dataset in resolved_mappings.get("datasets", []):
        dataset_name = dataset["dataset"]
        
        for mapping in dataset.get("mappings", []):
            role = mapping["best_match"]["role"]
            column = mapping["column"]
            
            if role == "USER_FEATURE":
                catalog["user_features"].append({
                    "name": column,
                    "dataset": dataset_name,
                })
            elif role == "ITEM_FEATURE":
                catalog["item_features"].append({
                    "name": column,
                    "dataset": dataset_name,
                })
            elif role == "INTERACTION_SIGNAL":
                catalog["interaction_features"].append({
                    "name": column,
                    "dataset": dataset_name,
                })
            elif role == "CONTEXTUAL":
                catalog["contextual_features"].append({
                    "name": column,
                    "dataset": dataset_name,
                })
    
    return catalog
