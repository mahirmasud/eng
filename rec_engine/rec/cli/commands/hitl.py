"""
Human-in-the-Loop CLI Command
Terminal-based review workflow
"""

import click
import json
from pathlib import Path
from typing import Optional

from rec.utils.workspace import WorkspaceManager


@click.command("review")
@click.option(
    "--workspace",
    "-w",
    default="./workspace",
    type=click.Path(exists=True),
    help="Path to workspace directory"
)
@click.option(
    "--auto-accept",
    is_flag=True,
    default=False,
    help="Auto-accept high confidence mappings"
)
@click.option(
    "--threshold",
    default=0.8,
    type=float,
    help="Confidence threshold for auto-accept"
)
def review_cmd(workspace: str, auto_accept: bool, threshold: float):
    """
    Review and validate semantic mappings interactively.
    
    This command presents ambiguous or low-confidence mappings
    for human review and correction.
    
    Example:
        rec review --workspace ./workspace
    """
    click.echo("=" * 60)
    click.echo("AUTONOMOUS RECOMMENDATION ENGINE - HUMAN REVIEW")
    click.echo("=" * 60)
    
    ws_manager = WorkspaceManager(workspace)
    
    # Load semantic roles
    roles_data = ws_manager.load_json("semantic_roles.json")
    if not roles_data:
        click.echo("      ✗ Error: No semantic roles found.")
        click.echo("      Please run 'rec bml' first.")
        return {"status": "error", "message": "No semantic roles found"}
    
    roles = roles_data.get("roles", [])
    
    # Filter for review (low confidence or ambiguous)
    needs_review = [r for r in roles if r.get("is_ambiguous", False) or r.get("confidence", 1.0) < threshold]
    
    if not needs_review:
        click.echo("\n✓ All mappings have sufficient confidence!")
        click.echo("  No review needed.")
        return {"status": "success", "reviewed": 0, "corrected": 0}
    
    click.echo(f"\nFound {len(needs_review)} mappings requiring review.\n")
    
    corrections = []
    reviewed = 0
    
    for role in needs_review:
        column = role["column"]
        target = role["target_role"]
        confidence = role.get("confidence", 0.0)
        
        click.echo("-" * 40)
        click.echo(f"Column: \"{column}\"")
        click.echo(f"Detected as: {target}")
        click.echo(f"Confidence: {confidence:.1%}")
        click.echo()
        
        if auto_accept and confidence >= threshold:
            click.echo("  [AUTO-ACCEPTED]")
            reviewed += 1
            continue
        
        # Interactive prompt
        try:
            response = click.prompt(
                "Accept? [Y/n/c]",
                type=click.Choice(["y", "n", "c"], case_sensitive=False),
                default="y"
            )
            
            reviewed += 1
            
            if response.lower() == "n":
                # Override
                new_role = click.prompt("Enter correct role", type=str)
                corrections.append({
                    "column": column,
                    "original_role": target,
                    "corrected_role": new_role,
                })
                click.echo(f"  ✓ Corrected to: {new_role}")
            elif response.lower() == "c":
                click.echo("  Skipped (manual correction later)")
            else:
                click.echo("  ✓ Accepted")
                
        except click.exceptions.Abort:
            break
    
    # Save feedback
    feedback = {
        "reviewed_count": reviewed,
        "corrections": corrections,
        "auto_accept_threshold": threshold,
    }
    ws_manager.save_json("human_feedback.json", feedback)
    
    # Apply corrections
    if corrections:
        corrected_roles = apply_corrections(roles, corrections)
        ws_manager.save_json("corrected_mappings.json", {"roles": corrected_roles})
        click.echo(f"\n✓ Saved {len(corrections)} corrections")
    
    click.echo("\n" + "=" * 60)
    click.echo(f"Reviewed: {reviewed}, Corrections: {len(corrections)}")
    click.echo(f"Ready for next step: rec build-config --workspace {workspace}")
    click.echo("=" * 60)
    
    return {
        "status": "success",
        "reviewed": reviewed,
        "corrected": len(corrections),
    }


def apply_corrections(roles, corrections):
    """Apply corrections to roles."""
    correction_map = {c["column"]: c["corrected_role"] for c in corrections}
    
    corrected = []
    for role in roles:
        new_role = role.copy()
        if role["column"] in correction_map:
            new_role["target_role"] = correction_map[role["column"]]
            new_role["human_corrected"] = True
        corrected.append(new_role)
    
    return corrected
