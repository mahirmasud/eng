"""
Autonomous Mapping Engine Command

Runs semantic schema understanding pipeline with:
- Domain Loader
- Unified Target Registry
- Embedding Matcher (all-MiniLM-L6-v2)
- Logic Booster
- Cross Encoder (stsb-distilroberta-base)
- Hybrid Scoring
- Global Resolver
"""

import json
from pathlib import Path
from typing import Optional, Dict, List

import click
import polars as pl
from rich.console import Console
from rich.table import Table
from loguru import logger

console = Console()


@click.command("map")
@click.option("--workspace", "-w", required=True, help="Path to workspace directory")
@click.option("--embedding-model", "-e", default="all-MiniLM-L6-v2", help="Sentence embedding model")
@click.option("--cross-encoder-model", "-c", default="stsb-distilroberta-base", help="Cross encoder model")
@click.option("--top-k", "-k", default=5, help="Number of top candidates to consider")
def map(workspace: str, embedding_model: str, cross_encoder_model: str, top_k: int):
    """
    Run autonomous semantic mapping.
    
    Performs semantic schema understanding using embedding matching,
    logic boosting, cross-encoder validation, and global resolution.
    
    Examples:
        rec map --workspace ./workspace
        rec map --workspace ./ws --top-k 10
    """
    console.print("\n[bold cyan]🗺️ Starting Autonomous Mapping[/bold cyan]\n")
    
    workspace_path = Path(workspace)
    
    if not workspace_path.exists():
        console.print(f"[red]Error:[/red] Workspace does not exist: {workspace}")
        return
    
    # Load metadata
    metadata_path = workspace_path / "metadata"
    if not (metadata_path / "dataset_profile.json").exists():
        console.print("[red]Error:[/red] No dataset profile found. Run 'rec ingest' first.")
        return
    
    console.print("[green]✓[/green] Loading dataset profiles...")
    with open(metadata_path / "dataset_profile.json", "r") as f:
        dataset_profile = json.load(f)
    
    # Load domain configs if available
    domain_configs = {}
    if (metadata_path / "domain_configs.json").exists():
        with open(metadata_path / "domain_configs.json", "r") as f:
            domain_configs = json.load(f)
        console.print("[green]✓[/green] Domain configurations loaded")
    
    # Define target schema roles
    target_roles = get_target_schema_roles()
    console.print(f"[green]✓[/green] Target schema roles defined: {len(target_roles)} roles")
    
    # Initialize embedding model
    console.print(f"\n[bold]Loading embedding model: {embedding_model}[/bold]")
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model_instance = SentenceTransformer(embedding_model)
        console.print(f"[green]✓[/green] Embedding model loaded")
    except Exception as e:
        console.print(f"[yellow]⚠ Warning: Could not load embedding model: {e}[/yellow]")
        console.print("[yellow]Using rule-based matching only[/yellow]")
        embedding_model_instance = None
    
    # Process each dataset
    all_mappings = []
    
    for profile in dataset_profile.get("profiles", []):
        console.print(f"\n[bold]Mapping dataset:[/bold] {profile['file_name']}")
        
        columns = profile.get("columns", [])
        semantic_candidates = profile.get("semantic_candidates", [])
        
        if not columns:
            console.print("[yellow]⚠ No columns to map[/yellow]")
            continue
        
        # Generate embeddings for column names
        column_embeddings = {}
        if embedding_model_instance:
            column_names = [col["name"] for col in columns]
            try:
                embeddings = embedding_model_instance.encode(column_names, convert_to_numpy=True)
                for i, col in enumerate(columns):
                    column_embeddings[col["name"]] = embeddings[i]
                console.print(f"[green]✓[/green] Generated embeddings for {len(column_names)} columns")
            except Exception as e:
                console.print(f"[yellow]⚠ Warning generating embeddings: {e}[/yellow]")
        
        # Map each column
        column_mappings = []
        
        for col in columns:
            mapping_result = map_column(
                column=col,
                target_roles=target_roles,
                domain_configs=domain_configs,
                column_embedding=column_embeddings.get(col["name"]),
                embedding_model=embedding_model_instance,
                top_k=top_k,
            )
            column_mappings.append(mapping_result)
            
            confidence = mapping_result.get("best_match", {}).get("confidence", 0.0)
            confidence_color = "green" if confidence > 0.7 else "yellow"
            console.print(
                f"   {col['name']} → [bold]{mapping_result['best_match']['role']}[/bold] "
                f"[{confidence_color}]({confidence:.1%})[/{confidence_color}]"
            )
        
        all_mappings.append({
            "dataset": profile["file_name"],
            "mappings": column_mappings,
        })
    
    # Generate resolved mappings
    console.print("\n[bold]Resolving conflicts and optimizing global assignments...[/bold]")
    resolved_mappings = resolve_global_mappings(all_mappings)
    
    # Save results
    output_path = workspace_path / "metadata" / "resolved_mappings.json"
    with open(output_path, "w") as f:
        json.dump(resolved_mappings, f, indent=2, default=str)
    
    console.print(f"[green]✓[/green] Resolved mappings saved to: {output_path}")
    
    # Display summary
    display_mapping_summary(resolved_mappings)
    
    console.print("\n[bold green]✓ Mapping complete![/bold green]")
    console.print(f"\nNext step: [cyan]rec bml --workspace {workspace}[/cyan]")


def get_target_schema_roles() -> Dict[str, Dict]:
    """Define target schema roles for recommendation systems."""
    return {
        "USER_ID": {
            "description": "Unique identifier for users",
            "synonyms": ["user", "uid", "customer", "client", "account", "member", "visitor"],
            "priority": 1,
        },
        "ITEM_ID": {
            "description": "Unique identifier for items/products",
            "synonyms": ["item", "product", "article", "document", "video", "movie", "song", "track", "book"],
            "priority": 1,
        },
        "INTERACTION_SIGNAL": {
            "description": "User interaction signal (click, view, purchase, etc.)",
            "synonyms": ["click", "view", "purchase", "buy", "rating", "score", "engagement", "interaction", "event"],
            "priority": 2,
        },
        "TEMPORAL": {
            "description": "Timestamp or date information",
            "synonyms": ["time", "date", "timestamp", "datetime", "created", "updated", "occurred"],
            "priority": 3,
        },
        "SESSION": {
            "description": "Session or visit identifier",
            "synonyms": ["session", "visit", "context", "browse_session"],
            "priority": 3,
        },
        "CONTEXTUAL": {
            "description": "Contextual features (location, device, etc.)",
            "synonyms": ["location", "device", "platform", "browser", "os", "country", "city"],
            "priority": 4,
        },
        "ITEM_FEATURE": {
            "description": "Feature describing an item",
            "synonyms": ["category", "genre", "brand", "price", "title", "description", "tags"],
            "priority": 4,
        },
        "USER_FEATURE": {
            "description": "Feature describing a user",
            "synonyms": ["age", "gender", "segment", "tier", "membership"],
            "priority": 4,
        },
    }


def map_column(
    column: Dict,
    target_roles: Dict,
    domain_configs: Dict,
    column_embedding=None,
    embedding_model=None,
    top_k: int = 5,
) -> Dict:
    """Map a single column to the best target role."""
    
    column_name = column["name"]
    column_dtype = column.get("dtype", "unknown")
    
    scores = {}
    reasoning = {}
    
    # Rule-based scoring (Logic Booster)
    rule_scores = apply_logic_booster(column_name, target_roles, domain_configs)
    
    # Embedding-based scoring
    embedding_scores = {}
    if embedding_model and column_embedding is not None:
        embedding_scores = apply_embedding_matcher(
            column_name, column_embedding, target_roles, embedding_model, top_k
        )
    
    # Cross-encoder refinement
    cross_encoder_scores = {}
    if embedding_model:
        try:
            cross_encoder_scores = apply_cross_encoder(
                column_name, target_roles, embedding_model
            )
        except Exception:
            pass
    
    # Hybrid scoring
    for role in target_roles.keys():
        hybrid_score = 0.0
        
        # Weight different signals
        rule_weight = rule_scores.get(role, 0.0)
        embed_weight = embedding_scores.get(role, 0.0)
        ce_weight = cross_encoder_scores.get(role, 0.0)
        
        # Combine scores
        if rule_weight > 0 and embed_weight > 0:
            hybrid_score = 0.4 * rule_weight + 0.4 * embed_weight + 0.2 * ce_weight
        elif rule_weight > 0:
            hybrid_score = 0.7 * rule_weight + 0.3 * ce_weight
        elif embed_weight > 0:
            hybrid_score = 0.7 * embed_weight + 0.3 * ce_weight
        else:
            hybrid_score = max(rule_weight, embed_weight, ce_weight)
        
        scores[role] = hybrid_score
        
        # Build reasoning
        reasons = []
        if rule_weight > 0.3:
            reasons.append(f"rule-match:{rule_weight:.2f}")
        if embed_weight > 0.3:
            reasons.append(f"semantic:{embed_weight:.2f}")
        if ce_weight > 0.3:
            reasons.append(f"context:{ce_weight:.2f}")
        
        reasoning[role] = ", ".join(reasons) if reasons else "low-confidence"
    
    # Find best match
    if scores:
        best_role = max(scores, key=scores.get)
        best_score = scores[best_role]
        top_matches = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    else:
        best_role = "UNKNOWN"
        best_score = 0.0
        top_matches = []
    
    return {
        "column": column_name,
        "dtype": column_dtype,
        "best_match": {
            "role": best_role,
            "confidence": best_score,
        },
        "top_matches": [
            {"role": role, "score": score} for role, score in top_matches
        ],
        "reasoning": reasoning,
    }


def apply_logic_booster(column_name: str, target_roles: Dict, domain_configs: Dict) -> Dict[str, float]:
    """Apply YAML-driven boosting rules."""
    scores = {}
    name_lower = column_name.lower()
    
    for role, config in target_roles.items():
        synonyms = config.get("synonyms", [])
        priority = config.get("priority", 5)
        
        # Exact match
        if name_lower == role.lower():
            scores[role] = 1.0
            continue
        
        # Synonym match
        max_synonym_score = 0.0
        for synonym in synonyms:
            if synonym in name_lower:
                # Longer matches are more confident
                synonym_score = len(synonym) / len(name_lower) * 0.9
                max_synonym_score = max(max_synonym_score, synonym_score)
        
        if max_synonym_score > 0:
            # Apply priority boost
            priority_boost = 1.0 - (priority - 1) * 0.1
            scores[role] = max_synonym_score * priority_boost
    
    # Apply domain-specific boosts
    if "keywords" in domain_configs:
        keywords_config = domain_configs["keywords"]
        for keyword_config in keywords_config.get("boosts", []):
            if keyword_config.get("pattern", "").lower() in name_lower:
                boost_role = keyword_config.get("target_role")
                boost_value = keyword_config.get("value", 0.2)
                if boost_role in scores:
                    scores[boost_role] = min(1.0, scores[boost_role] + boost_value)
                else:
                    scores[boost_role] = boost_value
    
    return scores


def apply_embedding_matcher(
    column_name: str,
    column_embedding,
    target_roles: Dict,
    embedding_model,
    top_k: int,
) -> Dict[str, float]:
    """Apply semantic embedding matching."""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    scores = {}
    
    # Encode target role descriptions
    role_texts = [
        f"{role}: {config['description']}" 
        for role, config in target_roles.items()
    ]
    
    try:
        role_embeddings = embedding_model.encode(role_texts, convert_to_numpy=True)
        
        # Compute cosine similarity
        similarities = cosine_similarity([column_embedding], role_embeddings)[0]
        
        for i, role in enumerate(target_roles.keys()):
            # Normalize to 0-1 range
            normalized_sim = (similarities[i] + 1) / 2
            scores[role] = float(normalized_sim)
    
    except Exception as e:
        logger.warning(f"Embedding matching failed: {e}")
    
    return scores


def apply_cross_encoder(
    column_name: str,
    target_roles: Dict,
    embedding_model,
) -> Dict[str, float]:
    """Apply cross-encoder for contextual validation."""
    from sentence_transformers import CrossEncoder
    
    scores = {}
    
    try:
        # Initialize cross-encoder (cache it in production)
        cross_encoder = CrossEncoder("stsb-distilroberta-base")
        
        # Create pairs
        pairs = [
            [column_name, f"{role}: {config['description']}"]
            for role, config in target_roles.items()
        ]
        
        # Get scores
        ce_scores = cross_encoder.predict(pairs)
        
        for i, role in enumerate(target_roles.keys()):
            scores[role] = float(ce_scores[i])
    
    except Exception as e:
        logger.warning(f"Cross-encoder failed: {e}")
    
    return scores


def resolve_global_mappings(all_mappings: List[Dict]) -> Dict:
    """Resolve conflicts and enforce 1:1 integrity."""
    resolved = {
        "datasets": [],
        "global_integrity": {
            "user_id_assigned": False,
            "item_id_assigned": False,
            "interaction_signal_assigned": False,
        },
        "conflicts_resolved": 0,
    }
    
    # Track assigned roles across datasets
    assigned_roles = {}
    
    for dataset_mapping in all_mappings:
        dataset_name = dataset_mapping["dataset"]
        resolved_dataset = {
            "dataset": dataset_name,
            "mappings": [],
        }
        
        # Sort mappings by confidence
        sorted_mappings = sorted(
            dataset_mapping["mappings"],
            key=lambda x: x["best_match"]["confidence"],
            reverse=True,
        )
        
        for mapping in sorted_mappings:
            role = mapping["best_match"]["role"]
            column = mapping["column"]
            
            # Check for conflicts
            if role in ["USER_ID", "ITEM_ID"]:
                if role in assigned_roles:
                    # Conflict: same critical role assigned to multiple columns
                    existing = assigned_roles[role]
                    
                    # Keep higher confidence, mark other as conflict
                    if mapping["best_match"]["confidence"] > existing["confidence"]:
                        # New one wins
                        resolved_dataset["mappings"].append({
                            **mapping,
                            "status": "assigned",
                            "conflict_with": existing["column"],
                        })
                        assigned_roles[role] = {
                            "column": column,
                            "confidence": mapping["best_match"]["confidence"],
                            "dataset": dataset_name,
                        }
                        resolved["conflicts_resolved"] += 1
                    else:
                        # Existing wins
                        resolved_dataset["mappings"].append({
                            **mapping,
                            "status": "conflict_rejected",
                            "conflict_with": existing["column"],
                        })
                        resolved["conflicts_resolved"] += 1
                        continue
                else:
                    assigned_roles[role] = {
                        "column": column,
                        "confidence": mapping["best_match"]["confidence"],
                        "dataset": dataset_name,
                    }
                    resolved["global_integrity"][f"{role.lower()}_assigned"] = True
            
            resolved_dataset["mappings"].append({
                **mapping,
                "status": "assigned" if role not in [m["best_match"]["role"] for m in resolved_dataset["mappings"]] else "duplicate",
            })
        
        resolved["datasets"].append(resolved_dataset)
    
    # Add summary statistics
    resolved["summary"] = {
        "total_datasets": len(all_mappings),
        "total_columns_mapped": sum(len(d["mappings"]) for d in resolved["datasets"]),
        "critical_roles_found": sum(1 for v in resolved["global_integrity"].values() if v),
    }
    
    return resolved


def display_mapping_summary(resolved_mappings: Dict):
    """Display mapping summary table."""
    table = Table(title="Mapping Summary")
    table.add_column("Dataset", style="cyan")
    table.add_column("Column", style="white")
    table.add_column("Role", style="green")
    table.add_column("Confidence", justify="right", style="yellow")
    table.add_column("Status", style="magenta")
    
    for dataset in resolved_mappings.get("datasets", []):
        dataset_name = dataset["dataset"]
        for mapping in dataset["mappings"][:10]:  # Show top 10 per dataset
            role = mapping["best_match"]["role"]
            confidence = mapping["best_match"]["confidence"]
            status = mapping.get("status", "assigned")
            
            conf_color = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"
            
            table.add_row(
                dataset_name,
                mapping["column"],
                role,
                f"[{conf_color}]{confidence:.1%}[/{conf_color}]",
                status,
            )
    
    console.print("\n")
    console.print(table)
    
    # Print integrity status
    integrity = resolved_mappings.get("global_integrity", {})
    console.print("\n[bold]Global Integrity Status:[/bold]")
    for role, assigned in integrity.items():
        icon = "✓" if assigned else "✗"
        color = "green" if assigned else "red"
        console.print(f"   [{color}]{icon} {role}[/{color}]")
