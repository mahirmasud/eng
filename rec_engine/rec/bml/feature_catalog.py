"""
Feature Catalog - Generates feature catalogs for ML pipelines
Autonomous Recommendation Engine Platform
"""

from typing import Dict, List, Any


class FeatureCatalog:
    """Build and manage feature catalogs for recommendation systems."""
    
    def build(
        self, 
        semantic_roles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build a feature catalog from semantic roles."""
        
        catalog = {
            "user_features": [],
            "item_features": [],
            "interaction_features": [],
            "context_features": [],
            "target_variables": [],
            "metadata": {},
        }
        
        for role in semantic_roles:
            feature = {
                "name": role["column"],
                "role": role["target_role"],
                "classification": role.get("classification", "FEATURE"),
                "dtype": role.get("dtype", "unknown"),
                "confidence": role.get("confidence", 0.0),
            }
            
            side = role.get("side", "unknown")
            
            if side == "user":
                catalog["user_features"].append(feature)
            elif side == "item":
                catalog["item_features"].append(feature)
            elif side == "interaction":
                if role.get("classification") == "SIGNAL":
                    catalog["target_variables"].append(feature)
                else:
                    catalog["interaction_features"].append(feature)
            elif side == "context":
                catalog["context_features"].append(feature)
            else:
                catalog["item_features"].append(feature)  # Default to item
        
        # Add metadata
        catalog["metadata"] = {
            "total_features": sum(len(v) for k, v in catalog.items() if k != "metadata"),
            "user_feature_count": len(catalog["user_features"]),
            "item_feature_count": len(catalog["item_features"]),
            "interaction_feature_count": len(catalog["interaction_features"]),
            "context_feature_count": len(catalog["context_features"]),
            "target_variable_count": len(catalog["target_variables"]),
        }
        
        # Generate feature processing hints
        catalog["processing_hints"] = self._generate_processing_hints(catalog)
        
        return catalog
    
    def _generate_processing_hints(
        self, 
        catalog: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate hints for feature processing."""
        hints = {
            "categorical_features": [],
            "numerical_features": [],
            "temporal_features": [],
            "embedding_candidates": [],
        }
        
        all_features = (
            catalog["user_features"] +
            catalog["item_features"] +
            catalog["context_features"]
        )
        
        for feature in all_features:
            dtype = feature.get("dtype", "").lower()
            name = feature["name"].lower()
            
            # Categorical detection
            if any(t in dtype for t in ["str", "utf8", "string", "object", "category"]):
                hints["categorical_features"].append(feature["name"])
            
            # Numerical detection
            if any(t in dtype for t in ["int", "float", "double"]):
                hints["numerical_features"].append(feature["name"])
            
            # Temporal detection
            if any(t in name for t in ["date", "time", "timestamp"]):
                hints["temporal_features"].append(feature["name"])
            
            # Embedding candidates (text-like features)
            if any(t in name for t in ["title", "description", "text", "content", "name"]):
                hints["embedding_candidates"].append(feature["name"])
        
        return hints
    
    def get_feature_names(self, category: str) -> List[str]:
        """Get feature names for a specific category."""
        valid_categories = [
            "user_features",
            "item_features", 
            "interaction_features",
            "context_features",
            "target_variables",
        ]
        
        if category not in valid_categories:
            return []
        
        # This would need the full catalog, simplified here
        return []
    
    def validate_catalog(self, catalog: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the feature catalog."""
        issues = []
        warnings = []
        
        # Check minimum requirements
        if not catalog["user_features"]:
            warnings.append("No user features detected")
        
        if not catalog["item_features"]:
            warnings.append("No item features detected")
        
        if not catalog["target_variables"]:
            issues.append("No target variables (interaction signals) detected")
        
        # Check for duplicate features
        all_names = []
        for key in ["user_features", "item_features", "interaction_features", "context_features"]:
            all_names.extend([f["name"] for f in catalog.get(key, [])])
        
        duplicates = set([x for x in all_names if all_names.count(x) > 1])
        if duplicates:
            warnings.append(f"Duplicate feature names: {duplicates}")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }
