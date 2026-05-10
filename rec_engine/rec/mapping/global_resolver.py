"""
Global Resolver - Conflict resolution and 1:1 integrity enforcement
Autonomous Recommendation Engine Platform
"""

from typing import Dict, List, Any, Optional, Set
from collections import defaultdict


class GlobalResolver:
    """
    Resolve conflicts and enforce global integrity constraints.
    
    Responsibilities:
    - Ensure 1:1 column-to-role mapping
    - Resolve conflicting assignments
    - Optimize global confidence
    - Handle many-to-many scenarios
    """
    
    def __init__(self):
        self.priority_order = [
            "USER_ID",
            "ITEM_ID",
            "INTERACTION_EVENT",
            "ENGAGEMENT_SIGNAL",
            "TEMPORAL_FEATURE",
            "SESSION_ID",
            "CONTEXTUAL_FEATURE",
            "ITEM_CATEGORY",
            "ITEM_TITLE",
            "ITEM_DESCRIPTION",
            "USER_AGE",
            "USER_GENDER",
            "USER_LOCATION",
            "PRICE",
            "RATING",
            "CLICK",
            "VIEW",
            "PURCHASE",
            "DWELL_TIME",
            "IMPRESSION",
        ]
    
    def optimize(
        self, 
        mappings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Optimize mappings for global consistency.
        
        Ensures each column maps to exactly one role and each role
        is assigned to at most one column (when possible).
        """
        if not mappings:
            return []
        
        # Group by column
        column_groups = defaultdict(list)
        for m in mappings:
            column_groups[m["column"]].append(m)
        
        resolved = []
        used_roles: Set[str] = set()
        
        # Process columns in order of their best confidence
        all_best = []
        for col, col_mappings in column_groups.items():
            best = max(col_mappings, key=lambda x: x["confidence"])
            all_best.append((col, best))
        
        # Sort by confidence descending
        all_best.sort(key=lambda x: x[1]["confidence"], reverse=True)
        
        for col, best_mapping in all_best:
            role = best_mapping["target_role"]
            
            # Check if role already used
            if role in used_roles:
                # Find alternative role for this column
                alternatives = [
                    m for m in column_groups[col]
                    if m["target_role"] not in used_roles
                ]
                
                if alternatives:
                    # Pick best alternative
                    best_alt = max(alternatives, key=lambda x: x["confidence"])
                    resolved.append(best_alt)
                    used_roles.add(best_alt["target_role"])
                # else: skip this column (no valid assignment)
            else:
                resolved.append(best_mapping)
                used_roles.add(role)
        
        return resolved
    
    def resolve_conflicts(
        self,
        mappings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Resolve conflicts where multiple columns claim the same role."""
        # Group by role
        role_groups = defaultdict(list)
        for m in mappings:
            role_groups[m["target_role"]].append(m)
        
        resolved = []
        
        for role, role_mappings in role_groups.items():
            if len(role_mappings) == 1:
                resolved.append(role_mappings[0])
            else:
                # Multiple columns claim same role
                # Sort by confidence and take best
                sorted_mappings = sorted(
                    role_mappings,
                    key=lambda x: x["confidence"],
                    reverse=True
                )
                
                # Take the best one
                resolved.append(sorted_mappings[0])
                
                # Mark others as conflicts
                for m in sorted_mappings[1:]:
                    m["conflict_resolved"] = True
                    m["lost_to"] = sorted_mappings[0]["column"]
        
        return resolved
    
    def enforce_one_to_one(
        self,
        mappings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enforce strict 1:1 mapping between columns and roles."""
        seen_columns = set()
        seen_roles = set()
        result = []
        
        # Sort by confidence
        sorted_mappings = sorted(
            mappings,
            key=lambda x: x["confidence"],
            reverse=True
        )
        
        for mapping in sorted_mappings:
            col = mapping["column"]
            role = mapping["target_role"]
            
            if col not in seen_columns and role not in seen_roles:
                result.append(mapping)
                seen_columns.add(col)
                seen_roles.add(role)
        
        return result
    
    def get_role_priority(self, role: str) -> int:
        """Get priority index for a role (lower is higher priority)."""
        try:
            return self.priority_order.index(role)
        except ValueError:
            return len(self.priority_order)
    
    def balance_assignments(
        self,
        mappings: List[Dict[str, Any]],
        required_roles: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Balance assignments to ensure required roles are covered."""
        if required_roles is None:
            required_roles = ["USER_ID", "ITEM_ID", "INTERACTION_EVENT"]
        
        assigned_roles = {m["target_role"] for m in mappings}
        missing_roles = set(required_roles) - assigned_roles
        
        result = list(mappings)
        
        # Log missing required roles
        for role in missing_roles:
            result.append({
                "target_role": role,
                "column": None,
                "confidence": 0.0,
                "missing_required": True,
            })
        
        return result
    
    def compute_global_confidence(
        self,
        mappings: List[Dict[str, Any]]
    ) -> float:
        """Compute overall confidence score for the mapping set."""
        if not mappings:
            return 0.0
        
        # Weighted average based on role importance
        total_weight = 0
        weighted_sum = 0
        
        for m in mappings:
            role = m["target_role"]
            confidence = m.get("confidence", 0)
            priority = self.get_role_priority(role)
            
            # Higher priority roles get more weight
            weight = 1.0 / (priority + 1)
            
            weighted_sum += confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def generate_reasoning(
        self,
        mappings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate reasoning signals for each mapping decision."""
        reasoned = []
        
        for m in mappings:
            reasoning = {
                "column": m["column"],
                "target_role": m["target_role"],
                "confidence": m["confidence"],
                "reasons": [],
            }
            
            # High confidence reason
            if m["confidence"] >= 0.8:
                reasoning["reasons"].append("High semantic similarity")
            elif m["confidence"] >= 0.5:
                reasoning["reasons"].append("Moderate semantic match")
            else:
                reasoning["reasons"].append("Low confidence - review recommended")
            
            # Priority-based reason
            priority = self.get_role_priority(m["target_role"])
            if priority < 5:
                reasoning["reasons"].append(f"High-priority role ({m['target_role']})")
            
            # Conflict resolution reason
            if m.get("conflict_resolved"):
                reasoning["reasons"].append(
                    f"Won conflict against other columns for {m['target_role']}"
                )
            
            reasoning["reasoning_summary"] = "; ".join(reasoning["reasons"])
            reasoned.append(reasoning)
        
        return reasoned
