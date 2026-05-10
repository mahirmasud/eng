"""
BML Orchestrator - Main orchestrator for Business Meaning Layer
Autonomous Recommendation Engine Platform
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from rec.bml.classifier import RoleClassifier
from rec.bml.entity_graph import EntityGraphBuilder
from rec.bml.feature_catalog import FeatureCatalog


class BMLOrchestrator:
    """
    Main orchestrator for the Business Meaning Layer.
    
    Responsibilities:
    - Classify columns into business roles
    - Detect users, items, and interactions
    - Build entity graphs
    - Generate feature catalogs
    """
    
    # Classification categories
    CLASSIFICATIONS = [
        "IDENTITY",      # User ID, Item ID
        "SIGNAL",        # Click, View, Purchase, Rating
        "FEATURE",       # User/Item attributes
        "EVENT",         # Interaction events
        "SESSION",       # Session identifiers
        "TEMPORAL",      # Time-related features
        "CONTEXTUAL",    # Context features
    ]
    
    def __init__(
        self,
        workspace_path: Path,
        min_confidence: float = 0.5,
    ):
        self.workspace_path = Path(workspace_path)
        self.min_confidence = min_confidence
        
        self.classifier = RoleClassifier()
        self.entity_builder = EntityGraphBuilder()
        self.feature_catalog = FeatureCatalog()
        
        self.mappings = []
        self.semantic_roles = []
    
    def load_mappings(self, mappings: List[Dict[str, Any]]) -> None:
        """Load semantic mappings from previous step."""
        self.mappings = mappings
    
    def classify_all(self) -> List[Dict[str, Any]]:
        """Classify all mapped columns into business roles."""
        self.semantic_roles = []
        
        for mapping in self.mappings:
            role = self._classify_mapping(mapping)
            self.semantic_roles.append(role)
        
        return self.semantic_roles
    
    def _classify_mapping(
        self, 
        mapping: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify a single mapping into business role."""
        column = mapping["column"]
        target_role = mapping["target_role"]
        confidence = mapping.get("confidence", 0.0)
        
        # Determine classification based on target role
        classification = self.classifier.classify_target(target_role)
        
        # Determine if this is user-side, item-side, or interaction
        side = self.classifier.determine_side(target_role)
        
        # Check for ambiguity
        is_ambiguous = confidence < self.min_confidence
        
        return {
            "column": column,
            "target_role": target_role,
            "classification": classification,
            "side": side,
            "confidence": confidence,
            "is_ambiguous": is_ambiguous,
            "dtype": mapping.get("dtype", "unknown"),
        }
    
    def generate_entity_graph(self) -> Dict[str, Any]:
        """Generate entity graph from classified roles."""
        if not self.semantic_roles:
            self.classify_all()
        
        # Extract entities
        users = []
        items = []
        interactions = []
        sessions = []
        features = []
        
        for role in self.semantic_roles:
            target = role["target_role"]
            column = role["column"]
            
            if target in ["USER_ID", "USER_AGE", "USER_GENDER", "USER_LOCATION"]:
                users.append(column)
            elif target in ["ITEM_ID", "ITEM_CATEGORY", "ITEM_TITLE", "ITEM_DESCRIPTION"]:
                items.append(column)
            elif target in ["CLICK", "VIEW", "PURCHASE", "RATING", "INTERACTION_EVENT", 
                           "ENGAGEMENT_SIGNAL", "IMPRESSION"]:
                interactions.append(column)
            elif target == "SESSION_ID":
                sessions.append(column)
            else:
                features.append(column)
        
        # Build graph structure
        graph = self.entity_builder.build(
            users=users,
            items=items,
            interactions=interactions,
            sessions=sessions,
            features=features,
        )
        
        return graph
    
    def generate_feature_catalog(self) -> Dict[str, Any]:
        """Generate feature catalog from classified roles."""
        if not self.semantic_roles:
            self.classify_all()
        
        catalog = self.feature_catalog.build(self.semantic_roles)
        return catalog
    
    def get_user_features(self) -> List[Dict[str, Any]]:
        """Get all user-side features."""
        return [
            r for r in self.semantic_roles
            if r.get("side") == "user"
        ]
    
    def get_item_features(self) -> List[Dict[str, Any]]:
        """Get all item-side features."""
        return [
            r for r in self.semantic_roles
            if r.get("side") == "item"
        ]
    
    def get_interaction_signals(self) -> List[Dict[str, Any]]:
        """Get all interaction signals."""
        return [
            r for r in self.semantic_roles
            if r.get("classification") == "SIGNAL"
        ]
    
    def detect_ambiguities(self) -> List[Dict[str, Any]]:
        """Detect ambiguous classifications that need review."""
        return [
            r for r in self.semantic_roles
            if r.get("is_ambiguous", False)
        ]
    
    def validate_minimum_requirements(self) -> Dict[str, Any]:
        """Validate minimum requirements for recommendation system."""
        required = {
            "has_user_id": False,
            "has_item_id": False,
            "has_interaction": False,
        }
        
        for role in self.semantic_roles:
            target = role["target_role"]
            if target == "USER_ID":
                required["has_user_id"] = True
            elif target == "ITEM_ID":
                required["has_item_id"] = True
            elif role.get("classification") == "SIGNAL":
                required["has_interaction"] = True
        
        all_present = all(required.values())
        
        return {
            "is_valid": all_present,
            "requirements": required,
            "missing": [k for k, v in required.items() if not v],
        }
