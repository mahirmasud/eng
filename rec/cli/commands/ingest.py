"""
Data & Knowledge Ingestion Command

Ingests datasets and knowledge files, detects file types, loads datasets,
profiles schemas, validates metadata, computes statistics, and generates
lineage metadata.
"""

import json
import os
from pathlib import Path
from typing import Optional

import click
import polars as pl
import duckdb
from rich.console import Console
from rich.table import Table
from loguru import logger

console = Console()


@click.command("ingest")
@click.option("--source", "-s", required=True, help="Path to source datasets directory")
@click.option("--domain-pack", "-d", default=None, help="Path to domain pack directory")
@click.option("--workspace", "-w", required=True, help="Path to workspace directory")
@click.option("--file-types", "-t", default=None, help="Comma-separated list of file types to ingest")
def ingest(source: str, domain_pack: Optional[str], workspace: str, file_types: Optional[str]):
    """
    Ingest datasets and knowledge files.
    
    Detects file types, loads datasets, profiles schemas, validates metadata,
    computes statistics, and generates lineage metadata.
    
    Examples:
        rec ingest --source ./datasets --workspace ./workspace
        rec ingest --source ./data --domain-pack ./domain_pack --workspace ./ws
    """
    console.print("\n[bold cyan]🚀 Starting Data Ingestion[/bold cyan]\n")
    
    # Validate paths
    source_path = Path(source)
    workspace_path = Path(workspace)
    domain_pack_path = Path(domain_pack) if domain_pack else None
    
    if not source_path.exists():
        console.print(f"[red]Error:[/red] Source path does not exist: {source}")
        return
    
    # Create workspace directories
    workspace_path.mkdir(parents=True, exist_ok=True)
    (workspace_path / "raw").mkdir(exist_ok=True)
    (workspace_path / "processed").mkdir(exist_ok=True)
    (workspace_path / "metadata").mkdir(exist_ok=True)
    (workspace_path / "models").mkdir(exist_ok=True)
    (workspace_path / "indexes").mkdir(exist_ok=True)
    
    console.print(f"[green]✓[/green] Workspace initialized: {workspace_path}")
    
    # Parse file types filter
    allowed_types = None
    if file_types:
        allowed_types = [ft.strip().lower() for ft in file_types.split(",")]
    
    # Discover files
    console.print("\n[bold]Discovering files...[/bold]")
    discovered_files = discover_files(source_path, allowed_types)
    
    if not discovered_files:
        console.print("[yellow]⚠ No files found to ingest[/yellow]")
        return
    
    console.print(f"[green]✓[/green] Found {len(discovered_files)} files")
    
    # Process each file
    manifests = []
    profiles = []
    
    for file_info in discovered_files:
        console.print(f"\n[bold]Processing:[/bold] {file_info['path'].name}")
        
        try:
            # Load and profile
            df, profile = load_and_profile(file_info["path"], file_info["type"])
            
            # Save to workspace
            output_path = workspace_path / "raw" / f"{file_info['path'].stem}.parquet"
            df.write_parquet(output_path)
            
            console.print(f"[green]✓[/green] Saved to: {output_path}")
            console.print(f"   Rows: {profile['num_rows']:,}, Columns: {profile['num_columns']}")
            
            manifests.append({
                "original_path": str(file_info["path"]),
                "workspace_path": str(output_path),
                "file_type": file_info["type"],
                "size_bytes": file_info["size"],
            })
            
            profiles.append(profile)
            
        except Exception as e:
            console.print(f"[red]✗ Error processing {file_info['path'].name}: {str(e)}[/red]")
            logger.error(f"Ingestion error: {e}")
    
    # Generate manifests
    console.print("\n[bold]Generating manifests...[/bold]")
    
    from datetime import datetime
    
    source_manifest = {
        "source_directory": str(source_path),
        "domain_pack": str(domain_pack_path) if domain_pack_path else None,
        "ingestion_timestamp": datetime.now().isoformat(),
        "total_files": len(manifests),
        "files": manifests,
    }
    
    with open(workspace_path / "metadata" / "source_manifest.json", "w") as f:
        json.dump(source_manifest, f, indent=2, default=str)
    
    console.print(f"[green]✓[/green] source_manifest.json created")
    
    # Generate dataset profile
    dataset_profile = {
        "profiles": profiles,
        "summary": {
            "total_datasets": len(profiles),
            "total_rows": sum(p.get("num_rows", 0) for p in profiles),
            "total_columns": sum(p.get("num_columns", 0) for p in profiles),
        }
    }
    
    with open(workspace_path / "metadata" / "dataset_profile.json", "w") as f:
        json.dump(dataset_profile, f, indent=2, default=str)
    
    console.print(f"[green]✓[/green] dataset_profile.json created")
    
    # Generate schema fingerprint
    schema_fingerprint = generate_schema_fingerprint(profiles)
    
    with open(workspace_path / "metadata" / "schema_fingerprint.json", "w") as f:
        json.dump(schema_fingerprint, f, indent=2, default=str)
    
    console.print(f"[green]✓[/green] schema_fingerprint.json created")
    
    # Process domain pack if provided
    if domain_pack_path and domain_pack_path.exists():
        console.print("\n[bold]Processing domain pack...[/bold]")
        process_domain_pack(domain_pack_path, workspace_path)
    
    # Display summary
    display_ingestion_summary(manifests, profiles)
    
    console.print("\n[bold green]✓ Ingestion complete![/bold green]")
    console.print(f"\nNext step: [cyan]rec map --workspace {workspace}[/cyan]")


def discover_files(path: Path, allowed_types: Optional[list] = None) -> list:
    """Discover all supported files in a directory."""
    supported_extensions = {
        ".csv": "csv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".parquet": "parquet",
        ".db": "sqlite",
        ".sqlite": "sqlite",
        ".sql": "sql_dump",
    }
    
    files = []
    
    for ext, file_type in supported_extensions.items():
        if allowed_types and file_type not in allowed_types:
            continue
        
        for file_path in path.rglob(f"*{ext}"):
            if file_path.is_file():
                files.append({
                    "path": file_path,
                    "type": file_type,
                    "size": file_path.stat().st_size,
                })
    
    # Also check for YAML config files
    if not allowed_types or "yaml" in allowed_types:
        for yaml_file in path.rglob("*.yaml"):
            if yaml_file.name in ["schema.yaml", "keywords.yaml", "rules.yaml", 
                                  "metrics.yaml", "validations.yaml", "questions.yaml"]:
                files.append({
                    "path": yaml_file,
                    "type": "yaml_config",
                    "size": yaml_file.stat().st_size,
                })
    
    return files


def load_and_profile(file_path: Path, file_type: str) -> tuple:
    """Load a file and generate its profile."""
    
    if file_type == "csv":
        df = pl.read_csv(str(file_path), infer_schema_length=1000)
    elif file_type == "json":
        df = pl.read_json(str(file_path))
    elif file_type == "jsonl":
        df = pl.read_ndjson(str(file_path))
    elif file_type == "parquet":
        df = pl.read_parquet(str(file_path))
    elif file_type == "sqlite":
        conn = duckdb.connect(str(file_path))
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if tables:
            df = conn.execute(f"SELECT * FROM {tables[0][0]} LIMIT 100000").fetchdf()
            df = pl.from_pandas(df)
        else:
            df = pl.DataFrame()
    elif file_type == "sql_dump":
        # For SQL dumps, we'd need to parse and execute
        # Simplified for now
        df = pl.DataFrame()
    elif file_type == "yaml_config":
        df = pl.DataFrame()
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    # Generate profile
    profile = {
        "file_name": file_path.name,
        "file_type": file_type,
        "num_rows": len(df) if len(df) > 0 else 0,
        "num_columns": len(df.columns) if len(df.columns) > 0 else 0,
        "columns": [],
        "semantic_candidates": [],
    }
    
    if len(df.columns) > 0:
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].null_count()) if hasattr(df[col], 'null_count') else 0,
                "unique_count": int(df[col].n_unique()) if hasattr(df[col], 'n_unique') else 0,
            }
            profile["columns"].append(col_info)
            
            # Detect semantic candidates
            semantic_type = detect_semantic_type(col, df[col])
            if semantic_type:
                profile["semantic_candidates"].append({
                    "column": col,
                    "semantic_type": semantic_type,
                })
    
    return df, profile


def detect_semantic_type(column_name: str, column_series) -> Optional[str]:
    """Detect semantic type of a column based on name and content."""
    name_lower = column_name.lower()
    
    # User-related
    if any(x in name_lower for x in ["user", "uid", "customer", "client", "account"]):
        return "USER_ID"
    
    # Item-related
    if any(x in name_lower for x in ["item", "product", "article", "document", "video", "movie", "song"]):
        return "ITEM_ID"
    
    # Interaction signals
    if any(x in name_lower for x in ["click", "view", "purchase", "rating", "score", "engagement"]):
        return "INTERACTION_SIGNAL"
    
    # Temporal
    if any(x in name_lower for x in ["time", "date", "timestamp", "created", "updated"]):
        return "TEMPORAL"
    
    # Session
    if any(x in name_lower for x in ["session", "visit", "context"]):
        return "SESSION"
    
    return None


def generate_schema_fingerprint(profiles: list) -> dict:
    """Generate a fingerprint of all schemas for drift detection."""
    fingerprint = {
        "schemas": {},
        "hash": "",
    }
    
    for profile in profiles:
        schema_key = profile["file_name"]
        columns_signature = sorted([
            f"{col['name']}:{col['dtype']}" 
            for col in profile["columns"]
        ])
        fingerprint["schemas"][schema_key] = {
            "columns": columns_signature,
            "row_count": profile["num_rows"],
        }
    
    # Simple hash based on schema signatures
    import hashlib
    schema_str = str(sorted(fingerprint["schemas"].items()))
    fingerprint["hash"] = hashlib.md5(schema_str.encode()).hexdigest()[:16]
    
    return fingerprint


def process_domain_pack(domain_pack_path: Path, workspace_path: Path):
    """Process domain pack configuration files."""
    domain_configs = {}
    
    for yaml_file in domain_pack_path.glob("*.yaml"):
        try:
            import yaml
            with open(yaml_file, "r") as f:
                domain_configs[yaml_file.stem] = yaml.safe_load(f)
            console.print(f"[green]✓[/green] Loaded: {yaml_file.name}")
        except Exception as e:
            console.print(f"[yellow]⚠ Warning loading {yaml_file.name}: {e}[/yellow]")
    
    if domain_configs:
        with open(workspace_path / "metadata" / "domain_configs.json", "w") as f:
            json.dump(domain_configs, f, indent=2, default=str)
        console.print(f"[green]✓[/green] Domain configurations saved")


def display_ingestion_summary(manifests: list, profiles: list):
    """Display ingestion summary table."""
    table = Table(title="Ingestion Summary")
    table.add_column("Dataset", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Rows", justify="right", style="green")
    table.add_column("Columns", justify="right", style="blue")
    table.add_column("Semantic Candidates", justify="right", style="yellow")
    
    for i, manifest in enumerate(manifests):
        profile = profiles[i] if i < len(profiles) else {}
        table.add_row(
            Path(manifest["original_path"]).name,
            manifest["file_type"],
            f"{profile.get('num_rows', 0):,}",
            str(profile.get('num_columns', 0)),
            str(len(profile.get('semantic_candidates', []))),
        )
    
    console.print("\n")
    console.print(table)
