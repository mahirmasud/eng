"""
Config Generator - Generates rec_config.json
Autonomous Recommendation Engine Platform
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class ConfigGenerator:
    """Generate complete recommendation engine configuration."""
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = Path(workspace_path)
    
    def build_config(
        self,
        semantic_roles: Dict[str, Any],
        entity_graph: Dict[str, Any],
        feature_catalog: Dict[str, Any],
        dataset_profiles: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the complete rec_config.json."""
        
        roles = semantic_roles.get("roles", [])
        
        # Extract column mappings
        user_id_col = self._find_column(roles, "USER_ID")
        item_id_col = self._find_column(roles, "ITEM_ID")
        interaction_cols = self._find_columns_by_classification(roles, "SIGNAL")
        temporal_col = self._find_column(roles, "TEMPORAL_FEATURE")
        session_col = self._find_column(roles, "SESSION_ID")
        
        config = {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "workspace_path": str(self.workspace_path),
            
            "entities": {
                "user_id_column": user_id_col,
                "item_id_column": item_id_col,
                "session_id_column": session_col,
                "timestamp_column": temporal_col,
            },
            
            "features": {
                "user_features": feature_catalog.get("user_features", []),
                "item_features": feature_catalog.get("item_features", []),
                "context_features": feature_catalog.get("context_features", []),
                "interaction_features": feature_catalog.get("interaction_features", []),
            },
            
            "training": {
                "target_signals": [s["role"] for s in interaction_cols],
                "batch_size": 256,
                "num_epochs": 10,
                "learning_rate": 0.001,
                "embedding_dim": 64,
                "validation_split": 0.1,
                "test_split": 0.1,
            },
            
            "retrieval": {
                "method": "faiss_ann",
                "index_type": "IVF32",
                "nprobe": 8,
                "top_k": 100,
                "embedding_model": "all-MiniLM-L6-v2",
            },
            
            "ranking": {
                "model_type": "two_tower",
                "hidden_layers": [128, 64, 32],
                "activation": "relu",
                "dropout": 0.1,
                "top_k": 10,
            },
            
            "dlrm": {
                "enabled": True,
                "bottom_mlp": [64, 32],
                "top_mlp": [128, 64, 32, 1],
                "embedding_sizes": {},
            },
            
            "cold_start": {
                "enabled": True,
                "strategy": "hybrid",
                "content_embeddings": True,
                "popularity_prior": True,
                "min_interactions": 5,
            },
            
            "reranking": {
                "enabled": True,
                "diversity_weight": 0.1,
                "freshness_weight": 0.05,
                "business_rules": [],
            },
            
            "feedback": {
                "enabled": True,
                "log_path": str(self.workspace_path / "feedback"),
                "learning_rate": 0.01,
                "decay_factor": 0.95,
            },
            
            "monitoring": {
                "enabled": True,
                "metrics": ["recall@k", "precision@k", "ndcg", "map", "mrr"],
                "log_path": str(self.workspace_path / "logs"),
            },
            
            "metadata": {
                "total_rows": sum(
                    p.get("num_rows", 0) 
                    for p in dataset_profiles.get("profiles", [])
                ),
                "total_columns": sum(
                    p.get("num_columns", 0) 
                    for p in dataset_profiles.get("profiles", [])
                ),
                "schema_fingerprint": "",
            },
        }
        
        return config
    
    def _find_column(
        self, 
        roles: List[Dict[str, Any]], 
        target_role: str
    ) -> str:
        """Find column name for a target role."""
        for role in roles:
            if role.get("target_role") == target_role:
                return role.get("column", "")
        return ""
    
    def _find_columns_by_classification(
        self,
        roles: List[Dict[str, Any]],
        classification: str,
    ) -> List[Dict[str, Any]]:
        """Find all columns with a specific classification."""
        return [
            r for r in roles 
            if r.get("classification") == classification
        ]
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the generated configuration."""
        issues = []
        warnings = []
        
        # Check required fields
        if not config["entities"]["user_id_column"]:
            issues.append("Missing USER_ID column mapping")
        
        if not config["entities"]["item_id_column"]:
            issues.append("Missing ITEM_ID column mapping")
        
        if not config["training"]["target_signals"]:
            warnings.append("No interaction signals detected")
        
        # Check feature counts
        if len(config["features"]["user_features"]) == 0:
            warnings.append("No user features configured")
        
        if len(config["features"]["item_features"]) == 0:
            warnings.append("No item features configured")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }
