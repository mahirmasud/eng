"""
Data & Knowledge Ingestion CLI Command
Autonomous Recommendation Engine Platform
"""

import click
import json
import os
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from rec.ingestion.loader import DataIngester
from rec.utils.workspace import WorkspaceManager


@click.command("ingest")
@click.option(
    "--source",
    "-s",
    required=True,
    type=click.Path(exists=True),
    help="Path to source data directory or file"
)
@click.option(
    "--domain-pack",
    "-d",
    default="./domain_pack",
    type=click.Path(exists=True),
    help="Path to domain pack directory"
)
@click.option(
    "--workspace",
    "-w",
    default="./workspace",
    type=click.Path(),
    help="Path to workspace directory"
)
@click.option(
    "--file-types",
    "-t",
    multiple=True,
    default=["csv", "json", "jsonl", "parquet", "sqlite"],
    help="File types to ingest"
)
@click.option(
    "--validate",
    is_flag=True,
    default=True,
    help="Validate datasets during ingestion"
)
def ingest_cmd(source: str, domain_pack: str, workspace: str, 
               file_types: tuple, validate: bool):
    """
    Ingest arbitrary datasets into the recommendation engine.
    
    This command detects file types, loads datasets, profiles schemas,
    validates metadata, computes statistics, and generates lineage metadata.
    
    Supported formats:
    - CSV
    - JSON / JSONL
    - Parquet
    - SQLite
    - PostgreSQL dumps
    - Event logs
    - Clickstream exports
    
    Example:
        rec ingest --source ./datasets --domain-pack ./domain_pack --workspace ./workspace
    """
    click.echo("=" * 60)
    click.echo("AUTONOMOUS RECOMMENDATION ENGINE - DATA INGESTION")
    click.echo("=" * 60)
    
    # Initialize workspace
    ws_manager = WorkspaceManager(workspace)
    ws_manager.initialize()
    
    click.echo(f"\n[1/6] Initializing workspace at {workspace}...")
    click.echo(f"      ✓ Workspace ready")
    
    # Initialize ingester
    click.echo(f"\n[2/6] Initializing data ingester...")
    ingester = DataIngester(
        source_path=Path(source),
        domain_pack_path=Path(domain_pack),
        workspace_path=ws_manager.path,
        supported_types=list(file_types)
    )
    click.echo(f"      ✓ Ingester ready")
    
    # Detect and load datasets
    click.echo(f"\n[3/6] Detecting datasets in {source}...")
    detected_files = ingester.detect_files()
    click.echo(f"      ✓ Found {len(detected_files)} dataset(s)")
    
    for f in detected_files:
        click.echo(f"        - {f['name']} ({f['type']}, {f['size_mb']:.2f} MB)")
    
    # Profile and validate
    click.echo(f"\n[4/6] Profiling datasets...")
    profiles = []
    for file_info in detected_files:
        click.echo(f"      Processing: {file_info['name']}")
        profile = ingester.profile_dataset(file_info)
        profiles.append(profile)
        click.echo(f"        ✓ Rows: {profile['num_rows']:,}")
        click.echo(f"        ✓ Columns: {profile['num_columns']}")
        click.echo(f"        ✓ Memory: {profile['memory_mb']:.2f} MB")
    
    # Generate manifests
    click.echo(f"\n[5/6] Generating manifests and metadata...")
    
    source_manifest = {
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "source_path": str(source),
        "domain_pack_path": str(domain_pack),
        "workspace_path": str(workspace),
        "datasets": [p["dataset_name"] for p in profiles],
        "total_files": len(detected_files),
        "total_rows": sum(p["num_rows"] for p in profiles),
        "total_columns": sum(p["num_columns"] for p in profiles),
        "file_types": list(file_types),
    }
    
    # Save manifests
    ws_manager.save_json("source_manifest.json", source_manifest)
    ws_manager.save_json("dataset_profiles.json", {"profiles": profiles})
    
    # Generate schema fingerprint
    schema_fingerprint = ingester.generate_schema_fingerprint(profiles)
    ws_manager.save_json("schema_fingerprint.json", schema_fingerprint)
    
    click.echo(f"      ✓ source_manifest.json")
    click.echo(f"      ✓ dataset_profiles.json")
    click.echo(f"      ✓ schema_fingerprint.json")
    
    # Summary
    click.echo(f"\n[6/6] Ingestion Complete!")
    click.echo("-" * 60)
    click.echo(f"Total Datasets:     {len(detected_files)}")
    click.echo(f"Total Rows:         {sum(p['num_rows'] for p in profiles):,}")
    click.echo(f"Total Columns:      {sum(p['num_columns'] for p in profiles)}")
    click.echo(f"Total Memory:       {sum(p['memory_mb'] for p in profiles):.2f} MB")
    click.echo(f"Schema Fingerprint: {schema_fingerprint['fingerprint'][:16]}...")
    click.echo("-" * 60)
    click.echo(f"\nWorkspace: {workspace}")
    click.echo(f"Ready for next step: rec map --workspace {workspace}")
    click.echo("=" * 60)
    
    return {
        "status": "success",
        "manifest": source_manifest,
        "profiles": profiles,
        "fingerprint": schema_fingerprint
    }
