"""
Autonomous Mapping Engine
Semantic Schema Understanding Pipeline
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import numpy as np

from rec.mapping.domain_loader import DomainLoader
from rec.mapping.embedding_matcher import EmbeddingMatcher
from rec.mapping.logic_booster import LogicBooster
from rec.mapping.cross_encoder import CrossEncoderScorer
from rec.mapping.global_resolver import GlobalResolver


class AutonomousMappingEngine:
    """
    Main autonomous mapping engine orchestrator.
    
    Components:
    1. Domain Loader - Loads domain knowledge from YAML files
    2. Embedding Matcher - Semantic matching using sentence embeddings
    3. Logic Booster - YAML-driven boosting and synonym weighting
    4. Cross Encoder - Contextual validation and confidence refinement
    5. Global Resolver - Conflict resolution and 1:1 integrity enforcement
    """
    
    # Target roles for recommendation systems
    TARGET_ROLES = [
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
    
    def __init__(
        self,
        workspace_path: Path,
        domain_pack_path: Path,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        cross_encoder_model_name: str = "stsb-distilroberta-base",
        confidence_threshold: float = 0.5,
        top_k: int = 5,
    ):
        self.workspace_path = Path(workspace_path)
        self.domain_pack_path = Path(domain_pack_path)
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        
        # Initialize components
        self.domain_loader = DomainLoader(domain_pack_path)
        self.embedding_matcher = EmbeddingMatcher(embedding_model_name)
        self.logic_booster = LogicBooster()
        self.cross_encoder = CrossEncoderScorer(cross_encoder_model_name)
        self.global_resolver = GlobalResolver()
        
        # State
        self.columns = []
        self.column_embeddings = {}
        self.target_embeddings = {}
        self.domain_knowledge = {}
    
    def load_domain_knowledge(self) -> Dict[str, Any]:
        """Load domain knowledge from YAML files."""
        self.domain_knowledge = self.domain_loader.load_all()
        
        # Get target role descriptions
        target_descriptions = self.domain_knowledge.get("keywords", {}).get(
            "target_roles", {}
        )
        
        # Generate embeddings for target roles
        self.target_embeddings = self.embedding_matcher.generate_embeddings(
            list(target_descriptions.keys())
        )
        
        return self.domain_knowledge
    
    def set_columns(self, columns: List[Dict[str, Any]]) -> None:
        """Set the columns to be mapped."""
        self.columns = columns
    
    def load_columns_from_profiles(self) -> List[Dict[str, Any]]:
        """Load columns from dataset profiles in workspace."""
        profiles_path = self.workspace_path / "dataset_profiles.json"
        if not profiles_path.exists():
            raise FileNotFoundError("dataset_profiles.json not found")
        
        with open(profiles_path, "r") as f:
            profiles_data = json.load(f)
        
        columns = []
        seen_columns = set()
        
        for profile in profiles_data.get("profiles", []):
            for col in profile.get("columns", []):
                col_key = col["name"].lower()
                if col_key not in seen_columns:
                    columns.append({
                        "name": col["name"],
                        "dtype": col["dtype"],
                        "semantic_type": col.get("semantic_type", "UNKNOWN"),
                        "dataset": profile["dataset_name"],
                    })
                    seen_columns.add(col_key)
        
        self.columns = columns
        return columns
    
    def generate_embeddings(self) -> Dict[str, np.ndarray]:
        """Generate embeddings for all columns."""
        if not self.columns:
            self.load_columns_from_profiles()
        
        column_names = [col["name"] for col in self.columns]
        self.column_embeddings = self.embedding_matcher.generate_embeddings(
            column_names
        )
        
        return self.column_embeddings
    
    def semantic_match(self) -> List[Dict[str, Any]]:
        """Run semantic matching between columns and target roles."""
        if not self.column_embeddings:
            self.generate_embeddings()
        
        matches = []
        
        for col in self.columns:
            col_name = col["name"]
            col_emb = self.column_embeddings.get(col_name)
            
            if col_emb is None:
                continue
            
            # Find top-k similar target roles
            similarities = self.embedding_matcher.find_top_k(
                col_emb, 
                self.target_embeddings, 
                self.top_k
            )
            
            for target_role, similarity in similarities:
                matches.append({
                    "column": col_name,
                    "dtype": col["dtype"],
                    "target_role": target_role,
                    "embedding_score": float(similarity),
                    "confidence": float(similarity),
                    "method": "embedding",
                })
        
        return matches
    
    def apply_logic_boosting(
        self, 
        matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply logic boosting rules to matches."""
        boosted_matches = []
        
        for match in matches:
            boosted = match.copy()
            
            # Apply keyword-based boosts
            boost_factor = self.logic_booster.get_boost(
                match["column"], 
                match["target_role"]
            )
            
            boosted["logic_boost"] = boost_factor
            boosted["confidence"] = min(1.0, match["confidence"] * (1 + boost_factor))
            boosted["boost_reasons"] = self.logic_booster.get_boost_reasons(
                match["column"], 
                match["target_role"]
            )
            
            boosted_matches.append(boosted)
        
        return boosted_matches
    
    def cross_encode_refine(
        self, 
        matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Refine matches using cross-encoder."""
        if not matches:
            return []
        
        # Prepare pairs for cross-encoding
        pairs = [(m["column"], m["target_role"]) for m in matches]
        
        # Get cross-encoder scores
        ce_scores = self.cross_encoder.score_pairs(pairs)
        
        refined_matches = []
        for i, match in enumerate(matches):
            refined = match.copy()
            ce_score = ce_scores[i] if i < len(ce_scores) else 0.0
            
            # Combine embedding score and cross-encoder score
            combined_score = (
                0.4 * match["embedding_score"] + 
                0.6 * ce_score
            )
            
            refined["cross_encoder_score"] = float(ce_score)
            refined["confidence"] = float(combined_score)
            refined["methods"] = ["embedding", "cross_encoder"]
            
            refined_matches.append(refined)
        
        return refined_matches
    
    def global_resolve(
        self, 
        matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Resolve conflicts and enforce 1:1 integrity."""
        # Group by column
        column_matches = {}
        for match in matches:
            col = match["column"]
            if col not in column_matches:
                column_matches[col] = []
            column_matches[col].append(match)
        
        resolved = []
        
        for col, col_match_list in column_matches.items():
            # Sort by confidence
            sorted_matches = sorted(
                col_match_list, 
                key=lambda x: x["confidence"], 
                reverse=True
            )
            
            # Take best match per column-role pair
            seen_roles = set()
            for match in sorted_matches:
                role = match["target_role"]
                if role not in seen_roles and match["confidence"] >= self.confidence_threshold:
                    resolved.append(match)
                    seen_roles.add(role)
                    break  # Only one role per column
        
        # Additional global optimization
        resolved = self.global_resolver.optimize(resolved)
        
        return resolved
    
    def run_full_pipeline(self) -> List[Dict[str, Any]]:
        """Run the complete mapping pipeline."""
        # Load domain knowledge
        self.load_domain_knowledge()
        
        # Load columns
        self.load_columns_from_profiles()
        
        # Generate embeddings
        self.generate_embeddings()
        
        # Semantic matching
        matches = self.semantic_match()
        
        # Logic boosting
        boosted = self.apply_logic_boosting(matches)
        
        # Cross-encoder refinement
        refined = self.cross_encode_refine(boosted)
        
        # Global resolution
        resolved = self.global_resolve(refined)
        
        return resolved
