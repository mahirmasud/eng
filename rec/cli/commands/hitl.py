"""Human-in-the-Loop Review Command."""

import json
from pathlib import Path

import click
from rich.console import Console
from rich.prompt import Prompt, Confirm

console = Console()


@click.command("review")
@click.option("--workspace", "-w", required=True, help="Path to workspace directory")
@click.option("--auto-accept", "-a", is_flag=True, help="Auto-accept high confidence mappings (>90%)")
@click.option("--min-confidence", "-c", default=0.5, help="Minimum confidence threshold for review")
def review(workspace: str, auto_accept: bool, min_confidence: float):
    """
    Human-in-the-loop review workflow.
    
    Interactive terminal-based review of autonomous mappings with
    ambiguity review, correction persistence, and confidence validation.
    
    Examples:
        rec review --workspace ./workspace
        rec review --workspace ./ws --auto-accept
    """
    console.print("\n[bold cyan]👥 Starting Human-in-the-Loop Review[/bold cyan]\n")
    
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
    
    corrected_mappings = {
        "datasets": [],
        "corrections": [],
        "accepted": [],
        "rejected": [],
    }
    
    total_reviewed = 0
    total_corrected = 0
    
    for dataset in resolved_mappings.get("datasets", []):
        dataset_name = dataset["dataset"]
        console.print(f"\n[bold]Reviewing dataset:[/bold] {dataset_name}")
        
        corrected_dataset = {
            "dataset": dataset_name,
            "mappings": [],
        }
        
        for mapping in dataset.get("mappings", []):
            column = mapping["column"]
            role = mapping["best_match"]["role"]
            confidence = mapping["best_match"]["confidence"]
            
            # Skip low confidence items if below threshold
            if confidence < min_confidence:
                console.print(f"   [dim]Skipping {column} (confidence {confidence:.1%} < {min_confidence:.1%})[/dim]")
                corrected_dataset["mappings"].append(mapping)
                continue
            
            total_reviewed += 1
            
            # Auto-accept high confidence
            if auto_accept and confidence >= 0.9:
                console.print(f"   [green]✓[/green] {column} → {role} ({confidence:.1%}) [dim](auto-accepted)[/dim]")
                corrected_dataset["mappings"].append(mapping)
                corrected_mappings["accepted"].append({
                    "dataset": dataset_name,
                    "column": column,
                    "role": role,
                    "confidence": confidence,
                })
                continue
            
            # Interactive review
            console.print(f"\n   Column: [bold cyan]{column}[/bold cyan]")
            console.print(f"   Detected as: [bold green]{role}[/bold green]")
            console.print(f"   Confidence: {confidence:.1%}")
            
            if mapping.get("reasoning"):
                reasoning = mapping["reasoning"].get(role, "")
                if reasoning:
                    console.print(f"   Reasoning: [dim]{reasoning}[/dim]")
            
            # Ask for confirmation
            accept = Confirm.ask(f"   Accept this mapping?", default=True)
            
            if accept:
                console.print(f"   [green]✓ Accepted[/green]")
                corrected_dataset["mappings"].append(mapping)
                corrected_mappings["accepted"].append({
                    "dataset": dataset_name,
                    "column": column,
                    "role": role,
                    "confidence": confidence,
                })
            else:
                # Show alternatives
                top_matches = mapping.get("top_matches", [])
                if top_matches:
                    console.print("\n   Alternatives:")
                    for i, match in enumerate(top_matches[1:6], 1):
                        console.print(f"     {i}. {match['role']} ({match['score']:.1%})")
                    
                    choice = Prompt.ask(
                        "   Select alternative (number or type new role)",
                        default="1"
                    )
                    
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(top_matches):
                            new_role = top_matches[idx]["role"]
                        else:
                            new_role = choice.upper()
                    except ValueError:
                        new_role = choice.upper()
                    
                    mapping["best_match"]["role"] = new_role
                    mapping["best_match"]["confidence"] = 1.0  # Human confirmed
                    mapping["status"] = "human_corrected"
                    
                    corrected_dataset["mappings"].append(mapping)
                    corrected_mappings["corrections"].append({
                        "dataset": dataset_name,
                        "column": column,
                        "original_role": role,
                        "corrected_role": new_role,
                        "original_confidence": confidence,
                    })
                    total_corrected += 1
                    
                    console.print(f"   [yellow]✓ Corrected to {new_role}[/yellow]")
                else:
                    new_role = Prompt.ask("   Enter correct role")
                    mapping["best_match"]["role"] = new_role.upper()
                    mapping["best_match"]["confidence"] = 1.0
                    mapping["status"] = "human_corrected"
                    
                    corrected_dataset["mappings"].append(mapping)
                    corrected_mappings["corrections"].append({
                        "dataset": dataset_name,
                        "column": column,
                        "original_role": role,
                        "corrected_role": new_role.upper(),
                        "original_confidence": confidence,
                    })
                    total_corrected += 1
                    
                    console.print(f"   [yellow]✓ Corrected to {new_role.upper()}[/yellow]")
        
        corrected_mappings["datasets"].append(corrected_dataset)
    
    # Save human feedback
    feedback_path = workspace_path / "metadata" / "human_feedback.json"
    with open(feedback_path, "w") as f:
        json.dump(corrected_mappings, f, indent=2, default=str)
    
    console.print(f"\n[green]✓[/green] human_feedback.json saved")
    
    # Save corrected mappings
    corrected_path = workspace_path / "metadata" / "corrected_mappings.json"
    with open(corrected_path, "w") as f:
        json.dump(corrected_mappings, f, indent=2, default=str)
    
    console.print(f"[green]✓[/green] corrected_mappings.json saved")
    
    # Display summary
    console.print("\n[bold]Review Summary:[/bold]")
    console.print(f"   Total reviewed: {total_reviewed}")
    console.print(f"   Auto-accepted: {len(corrected_mappings['accepted']) - total_corrected}")
    console.print(f"   Manually corrected: {total_corrected}")
    console.print(f"   Correction rate: {total_corrected / max(total_reviewed, 1):.1%}")
    
    console.print("\n[bold green]✓ Review complete![/bold green]")
    console.print(f"\nNext step: [cyan]rec build-config --workspace {workspace}[/cyan]")
