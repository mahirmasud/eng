"""
Business Meaning Layer CLI Command
Semantic Orchestration Layer
"""

import click
import json
from pathlib import Path
from typing import Optional

from rec.bml.orchestrator import BMLOrchestrator
from rec.utils.workspace import WorkspaceManager


@click.command("bml")
@click.option(
    "--workspace",
    "-w",
    default="./workspace",
    type=click.Path(exists=True),
    help="Path to workspace directory"
)
@click.option(
    "--auto-detect",
    is_flag=True,
    default=True,
    help="Automatically detect entity types"
)
@click.option(
    "--min-confidence",
    default=0.5,
    type=float,
    help="Minimum confidence for auto-classification"
)
def bml_cmd(workspace: str, auto_detect: bool, min_confidence: float):
    """
    Run Business Meaning Layer orchestration.
    
    This command analyzes mapped columns and assigns business meaning:
    - Detects users, items, and interactions
    - Classifies features by role (Identity, Signal, Feature, Event, etc.)
    - Generates entity graphs and feature catalogs
    
    Example:
        rec bml --workspace ./workspace
    """
    click.echo("=" * 60)
    click.echo("AUTONOMOUS RECOMMENDATION ENGINE - BUSINESS MEANING LAYER")
    click.echo("=" * 60)
    
    # Initialize workspace
    ws_manager = WorkspaceManager(workspace)
    click.echo(f"\n[1/5] Loading workspace at {workspace}...")
    
    # Check for required files
    mappings_data = ws_manager.load_json("resolved_mappings.json")
    if not mappings_data:
        click.echo("      ✗ Error: No resolved mappings found.")
        click.echo("      Please run 'rec map' first.")
        return {"status": "error", "message": "No resolved mappings found"}
    
    click.echo(f"      ✓ Found {len(mappings_data.get('mappings', []))} mappings")
    
    # Initialize BML orchestrator
    click.echo(f"\n[2/5] Initializing BML orchestrator...")
    orchestrator = BMLOrchestrator(
        workspace_path=Path(workspace),
        min_confidence=min_confidence,
    )
    click.echo(f"      ✓ BML ready")
    
    # Load mappings
    click.echo(f"\n[3/5] Analyzing semantic roles...")
    mappings = mappings_data.get("mappings", [])
    orchestrator.load_mappings(mappings)
    
    # Classify entities
    click.echo(f"\n[4/5] Classifying entities and features...")
    semantic_roles = orchestrator.classify_all()
    
    # Generate outputs
    click.echo(f"\n[5/5] Generating output files...")
    
    # Save semantic roles
    ws_manager.save_json("semantic_roles.json", {
        "roles": semantic_roles,
        "metadata": {
            "auto_detect": auto_detect,
            "min_confidence": min_confidence,
        }
    })
    
    # Generate entity graph
    entity_graph = orchestrator.generate_entity_graph()
    ws_manager.save_json("entity_graph.json", entity_graph)
    
    # Generate feature catalog
    feature_catalog = orchestrator.generate_feature_catalog()
    ws_manager.save_json("feature_catalog.json", feature_catalog)
    
    click.echo(f"      ✓ semantic_roles.json")
    click.echo(f"      ✓ entity_graph.json")
    click.echo(f"      ✓ feature_catalog.json")
    
    # Display summary
    click.echo("\n" + "=" * 60)
    click.echo("BUSINESS MEANING SUMMARY")
    click.echo("-" * 60)
    
    # Count by classification
    classifications = {}
    for role in semantic_roles:
        cls = role.get("classification", "UNKNOWN")
        classifications[cls] = classifications.get(cls, 0) + 1
    
    for cls, count in sorted(classifications.items()):
        click.echo(f"  {cls}: {count}")
    
    # Show detected entities
    click.echo("\nDetected Entities:")
    if entity_graph.get("users"):
        click.echo(f"  • Users: {entity_graph['users']}")
    if entity_graph.get("items"):
        click.echo(f"  • Items: {entity_graph['items']}")
    if entity_graph.get("interactions"):
        click.echo(f"  • Interactions: {entity_graph['interactions']}")
    
    click.echo("\n" + "=" * 60)
    click.echo(f"Ready for next step: rec review --workspace {workspace}")
    click.echo("=" * 60)
    
    return {
        "status": "success",
        "semantic_roles": semantic_roles,
        "entity_graph": entity_graph,
        "feature_catalog": feature_catalog,
    }
