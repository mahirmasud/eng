"""
Config Generation CLI Command
Generates rec_config.json for the recommendation engine
"""

import click
import json
from pathlib import Path
from datetime import datetime

from rec.utils.workspace import WorkspaceManager
from rec.config.generator import ConfigGenerator


@click.command("build-config")
@click.option(
    "--workspace",
    "-w",
    default="./workspace",
    type=click.Path(exists=True),
    help="Path to workspace directory"
)
@click.option(
    "--output",
    "-o",
    default="rec_config.json",
    type=click.Path(),
    help="Output config file path"
)
def build_config_cmd(workspace: str, output: str):
    """
    Generate rec_config.json from analyzed mappings.
    
    This command generates the complete configuration file
    needed for training and inference.
    
    Example:
        rec build-config --workspace ./workspace
    """
    click.echo("=" * 60)
    click.echo("AUTONOMOUS RECOMMENDATION ENGINE - CONFIG GENERATION")
    click.echo("=" * 60)
    
    ws_manager = WorkspaceManager(workspace)
    
    # Load required files
    semantic_roles = ws_manager.load_json("semantic_roles.json")
    entity_graph = ws_manager.load_json("entity_graph.json")
    feature_catalog = ws_manager.load_json("feature_catalog.json")
    profiles = ws_manager.load_json("dataset_profiles.json")
    
    if not all([semantic_roles, entity_graph, feature_catalog]):
        click.echo("      ✗ Error: Missing required analysis files.")
        click.echo("      Please run 'rec bml' first.")
        return {"status": "error", "message": "Missing required files"}
    
    click.echo(f"\n[1/4] Initializing config generator...")
    generator = ConfigGenerator(workspace_path=Path(workspace))
    click.echo(f"      ✓ Generator ready")
    
    click.echo(f"\n[2/4] Building configuration...")
    
    config = generator.build_config(
        semantic_roles=semantic_roles,
        entity_graph=entity_graph,
        feature_catalog=feature_catalog,
        dataset_profiles=profiles,
    )
    
    click.echo(f"      ✓ Configuration built")
    
    click.echo(f"\n[3/4] Validating configuration...")
    validation = generator.validate_config(config)
    
    if validation["is_valid"]:
        click.echo(f"      ✓ Configuration valid")
    else:
        for issue in validation.get("issues", []):
            click.echo(f"      ⚠ {issue}")
    
    click.echo(f"\n[4/4] Saving configuration...")
    
    # Save to workspace
    ws_manager.save_json(output, config)
    
    click.echo(f"      ✓ Saved to {workspace}/{output}")
    
    # Display summary
    click.echo("\n" + "=" * 60)
    click.echo("CONFIGURATION SUMMARY")
    click.echo("-" * 60)
    click.echo(f"User Features:      {config['features']['user_features']}")
    click.echo(f"Item Features:      {config['features']['item_features']}")
    click.echo(f"Interaction Signals: {config['training']['target_signals']}")
    click.echo(f"Retrieval Method:   {config['retrieval']['method']}")
    click.echo(f"Ranking Model:      {config['ranking']['model_type']}")
    click.echo(f"DLRM Enabled:       {config['dlrm']['enabled']}")
    click.echo(f"Cold Start:         {config['cold_start']['enabled']}")
    click.echo("-" * 60)
    click.echo(f"\nReady for next steps:")
    click.echo(f"  • rec features --workspace {workspace}")
    click.echo(f"  • rec train-retrieval --workspace {workspace}")
    click.echo(f"  • rec train-ranker --workspace {workspace}")
    click.echo("=" * 60)
    
    return {
        "status": "success",
        "config_path": str(Path(workspace) / output),
        "validation": validation,
    }
