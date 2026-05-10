"""
Autonomous Mapping Engine CLI Command
Semantic Schema Understanding Pipeline
"""

import click
import json
from pathlib import Path
from typing import Optional

from rec.mapping.engine import AutonomousMappingEngine
from rec.utils.workspace import WorkspaceManager


@click.command("map")
@click.option(
    "--workspace",
    "-w",
    default="./workspace",
    type=click.Path(exists=True),
    help="Path to workspace directory"
)
@click.option(
    "--domain-pack",
    "-d",
    default="./domain_pack",
    type=click.Path(exists=True),
    help="Path to domain pack directory"
)
@click.option(
    "--embedding-model",
    "-e",
    default="all-MiniLM-L6-v2",
    help="Sentence embedding model for semantic matching"
)
@click.option(
    "--cross-encoder-model",
    "-c",
    default="stsb-distilroberta-base",
    help="Cross encoder model for confidence refinement"
)
@click.option(
    "--confidence-threshold",
    "-t",
    default=0.5,
    type=float,
    help="Minimum confidence threshold for mappings"
)
@click.option(
    "--top-k",
    "-k",
    default=5,
    type=int,
    help="Number of top matches to consider"
)
def map_cmd(workspace: str, domain_pack: str, embedding_model: str,
            cross_encoder_model: str, confidence_threshold: float, top_k: int):
    """
    Run autonomous semantic mapping engine.
    
    This command analyzes dataset schemas and automatically maps columns
    to recommendation-relevant semantic roles using:
    - Domain knowledge loading
    - Embedding-based semantic matching
    - Logic boosting with YAML rules
    - Cross-encoder validation
    - Global conflict resolution
    
    Example:
        rec map --workspace ./workspace
    """
    click.echo("=" * 60)
    click.echo("AUTONOMOUS RECOMMENDATION ENGINE - SEMANTIC MAPPING")
    click.echo("=" * 60)
    
    # Initialize workspace
    ws_manager = WorkspaceManager(workspace)
    click.echo(f"\n[1/7] Loading workspace at {workspace}...")
    
    # Check for required files
    profiles = ws_manager.load_json("dataset_profiles.json")
    if not profiles:
        click.echo("      ✗ Error: No dataset profiles found.")
        click.echo("      Please run 'rec ingest' first.")
        return {"status": "error", "message": "No dataset profiles found"}
    
    click.echo(f"      ✓ Found {len(profiles.get('profiles', []))} dataset(s)")
    
    # Initialize mapping engine
    click.echo(f"\n[2/7] Initializing mapping engine...")
    engine = AutonomousMappingEngine(
        workspace_path=Path(workspace),
        domain_pack_path=Path(domain_pack),
        embedding_model_name=embedding_model,
        cross_encoder_model_name=cross_encoder_model,
        confidence_threshold=confidence_threshold,
        top_k=top_k,
    )
    click.echo(f"      ✓ Embedding model: {embedding_model}")
    click.echo(f"      ✓ Cross-encoder model: {cross_encoder_model}")
    
    # Load domain knowledge
    click.echo(f"\n[3/7] Loading domain knowledge...")
    engine.load_domain_knowledge()
    click.echo(f"      ✓ Loaded keywords, rules, and schemas")
    
    # Generate embeddings
    click.echo(f"\n[4/7] Generating column embeddings...")
    column_embeddings = engine.generate_embeddings()
    click.echo(f"      ✓ Generated {len(column_embeddings)} embeddings")
    
    # Semantic matching
    click.echo(f"\n[5/7] Running semantic matching...")
    matches = engine.semantic_match()
    click.echo(f"      ✓ Found {len(matches)} potential matches")
    
    # Cross-encoder refinement
    click.echo(f"\n[6/7] Refining with cross-encoder...")
    refined = engine.cross_encode_refine(matches)
    click.echo(f"      ✓ Refined {len(refined)} matches")
    
    # Global resolution
    click.echo(f"\n[7/7] Resolving conflicts and generating output...")
    resolved = engine.global_resolve(refined)
    
    # Save results
    ws_manager.save_json("resolved_mappings.json", {
        "mappings": resolved,
        "metadata": {
            "embedding_model": embedding_model,
            "cross_encoder_model": cross_encoder_model,
            "confidence_threshold": confidence_threshold,
            "top_k": top_k,
        }
    })
    
    # Display summary
    click.echo("\n" + "=" * 60)
    click.echo("MAPPING RESULTS SUMMARY")
    click.echo("-" * 60)
    
    high_confidence = [m for m in resolved if m["confidence"] >= 0.8]
    medium_confidence = [m for m in resolved if 0.5 <= m["confidence"] < 0.8]
    low_confidence = [m for m in resolved if m["confidence"] < 0.5]
    
    click.echo(f"High Confidence (≥80%):   {len(high_confidence)}")
    click.echo(f"Medium Confidence (50-80%): {len(medium_confidence)}")
    click.echo(f"Low Confidence (<50%):    {len(low_confidence)}")
    click.echo("-" * 60)
    
    if resolved:
        click.echo("\nTop Mappings:")
        for m in sorted(resolved, key=lambda x: x["confidence"], reverse=True)[:10]:
            click.echo(f"  • {m['column']} → {m['target_role']} ({m['confidence']:.1%})")
    
    click.echo("\n" + "=" * 60)
    click.echo(f"Output: {workspace}/resolved_mappings.json")
    click.echo(f"Ready for next step: rec bml --workspace {workspace}")
    click.echo("=" * 60)
    
    return {
        "status": "success",
        "mappings": resolved,
        "summary": {
            "total": len(resolved),
            "high_confidence": len(high_confidence),
            "medium_confidence": len(medium_confidence),
            "low_confidence": len(low_confidence),
        }
    }
