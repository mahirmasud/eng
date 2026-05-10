"""Feature definitions and management for automated feature engineering."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl


@dataclass
class FeatureSpec:
    """Specification for a single feature."""
    name: str
    feature_type: str  # numeric, categorical, temporal, aggregation, interaction
    source_columns: List[str]
    transformation: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


class FeatureDefinitionManager:
    """Manages feature definitions based on semantic roles and entity graph."""
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.semantic_roles = {}
        self.entity_graph = {}
        self.feature_catalog = {}
        self.feature_specs: List[FeatureSpec] = []
        
    def load_metadata(self):
        """Load semantic roles and entity graph from workspace."""
        metadata_dir = self.workspace_path / "metadata"
        
        if (metadata_dir / "semantic_roles.json").exists():
            with open(metadata_dir / "semantic_roles.json", "r") as f:
                self.semantic_roles = json.load(f)
                
        if (metadata_dir / "entity_graph.json").exists():
            with open(metadata_dir / "entity_graph.json", "r") as f:
                self.entity_graph = json.load(f)
                
        if (metadata_dir / "feature_catalog.json").exists():
            with open(metadata_dir / "feature_catalog.json", "r") as f:
                self.feature_catalog = json.load(f)
                
    def generate_feature_specs(self) -> List[FeatureSpec]:
        """Generate feature specifications based on detected entities and roles."""
        self.feature_specs = []
        
        # Generate user features
        user_features = self._generate_user_features()
        self.feature_specs.extend(user_features)
        
        # Generate item features
        item_features = self._generate_item_features()
        self.feature_specs.extend(item_features)
        
        # Generate interaction features
        interaction_features = self._generate_interaction_features()
        self.feature_specs.extend(interaction_features)
        
        # Generate temporal features
        temporal_features = self._generate_temporal_features()
        self.feature_specs.extend(temporal_features)
        
        # Generate session features
        session_features = self._generate_session_features()
        self.feature_specs.extend(session_features)
        
        # Generate aggregation features
        aggregation_features = self._generate_aggregation_features()
        self.feature_specs.extend(aggregation_features)
        
        return self.feature_specs
    
    def _generate_user_features(self) -> List[FeatureSpec]:
        """Generate user-level features."""
        features = []
        user_id_col = self._get_column_for_role("USER_ID")
        
        if user_id_col:
            # User activity count
            features.append(FeatureSpec(
                name="user_interaction_count",
                feature_type="aggregation",
                source_columns=[user_id_col],
                transformation="count",
                description="Total number of interactions by user",
                parameters={"group_by": user_id_col}
            ))
            
            # User engagement score
            features.append(FeatureSpec(
                name="user_engagement_score",
                feature_type="aggregation",
                source_columns=[user_id_col],
                transformation="weighted_sum",
                description="Weighted engagement score for user",
                parameters={"group_by": user_id_col, "weight_column": "signal_value"}
            ))
            
            # User recency
            features.append(FeatureSpec(
                name="user_last_activity_days",
                feature_type="temporal",
                source_columns=[user_id_col],
                transformation="days_since_last",
                description="Days since user's last activity",
                parameters={"group_by": user_id_col}
            ))
            
        return features
    
    def _generate_item_features(self) -> List[FeatureSpec]:
        """Generate item-level features."""
        features = []
        item_id_col = self._get_column_for_role("ITEM_ID")
        
        if item_id_col:
            # Item popularity
            features.append(FeatureSpec(
                name="item_interaction_count",
                feature_type="aggregation",
                source_columns=[item_id_col],
                transformation="count",
                description="Total number of interactions for item",
                parameters={"group_by": item_id_col}
            ))
            
            # Item average rating/signal
            features.append(FeatureSpec(
                name="item_avg_signal",
                feature_type="aggregation",
                source_columns=[item_id_col],
                transformation="mean",
                description="Average signal value for item",
                parameters={"group_by": item_id_col, "agg_column": "signal_value"}
            ))
            
            # Item age
            features.append(FeatureSpec(
                name="item_age_days",
                feature_type="temporal",
                source_columns=[item_id_col],
                transformation="days_since_creation",
                description="Age of item in days",
                parameters={"group_by": item_id_col}
            ))
            
        return features
    
    def _generate_interaction_features(self) -> List[FeatureSpec]:
        """Generate interaction-level features."""
        features = []
        user_id_col = self._get_column_for_role("USER_ID")
        item_id_col = self._get_column_for_role("ITEM_ID")
        
        if user_id_col and item_id_col:
            # User-item interaction history
            features.append(FeatureSpec(
                name="user_item_interaction_history",
                feature_type="interaction",
                source_columns=[user_id_col, item_id_col],
                transformation="historical_interactions",
                description="Historical interaction count between user and item",
                parameters={}
            ))
            
            # Signal strength
            features.append(FeatureSpec(
                name="interaction_signal_strength",
                feature_type="numeric",
                source_columns=["signal_value"] if "signal_value" in self._get_all_columns() else [],
                transformation="normalize",
                description="Normalized interaction signal strength",
                parameters={"method": "minmax"}
            ))
            
        return features
    
    def _generate_temporal_features(self) -> List[FeatureSpec]:
        """Generate temporal features."""
        features = []
        timestamp_col = self._get_column_for_role("TIMESTAMP")
        
        if timestamp_col:
            # Hour of day
            features.append(FeatureSpec(
                name="interaction_hour",
                feature_type="temporal",
                source_columns=[timestamp_col],
                transformation="extract_hour",
                description="Hour of day when interaction occurred",
                parameters={}
            ))
            
            # Day of week
            features.append(FeatureSpec(
                name="interaction_day_of_week",
                feature_type="temporal",
                source_columns=[timestamp_col],
                transformation="extract_day_of_week",
                description="Day of week when interaction occurred",
                parameters={}
            ))
            
            # Is weekend
            features.append(FeatureSpec(
                name="interaction_is_weekend",
                feature_type="categorical",
                source_columns=[timestamp_col],
                transformation="is_weekend",
                description="Whether interaction occurred on weekend",
                parameters={}
            ))
            
        return features
    
    def _generate_session_features(self) -> List[FeatureSpec]:
        """Generate session-level features."""
        features = []
        session_col = self._get_column_for_role("SESSION_ID")
        timestamp_col = self._get_column_for_role("TIMESTAMP")
        user_id_col = self._get_column_for_role("USER_ID")
        
        if session_col or (user_id_col and timestamp_col):
            # Session length
            features.append(FeatureSpec(
                name="session_length",
                feature_type="aggregation",
                source_columns=[session_col] if session_col else [user_id_col],
                transformation="session_duration",
                description="Duration of the session",
                parameters={"time_column": timestamp_col} if timestamp_col else {}
            ))
            
            # Session position
            features.append(FeatureSpec(
                name="session_position",
                feature_type="numeric",
                source_columns=[session_col] if session_col else [user_id_col],
                transformation="row_number",
                description="Position of interaction within session",
                parameters={"order_by": timestamp_col} if timestamp_col else {}
            ))
            
        return features
    
    def _generate_aggregation_features(self) -> List[FeatureSpec]:
        """Generate statistical aggregation features."""
        features = []
        user_id_col = self._get_column_for_role("USER_ID")
        item_id_col = self._get_column_for_role("ITEM_ID")
        
        if user_id_col:
            # User mean signal
            features.append(FeatureSpec(
                name="user_mean_signal",
                feature_type="aggregation",
                source_columns=[user_id_col],
                transformation="mean",
                description="Mean signal value for user",
                parameters={"group_by": user_id_col, "agg_column": "signal_value"}
            ))
            
            # User signal std
            features.append(FeatureSpec(
                name="user_signal_std",
                feature_type="aggregation",
                source_columns=[user_id_col],
                transformation="std",
                description="Standard deviation of signal for user",
                parameters={"group_by": user_id_col, "agg_column": "signal_value"}
            ))
            
        if item_id_col:
            # Item signal variance
            features.append(FeatureSpec(
                name="item_signal_variance",
                feature_type="aggregation",
                source_columns=[item_id_col],
                transformation="variance",
                description="Variance of signal values for item",
                parameters={"group_by": item_id_col, "agg_column": "signal_value"}
            ))
            
        return features
    
    def _get_column_for_role(self, role: str) -> Optional[str]:
        """Get column name for a given semantic role."""
        if "column_roles" in self.semantic_roles:
            for col, info in self.semantic_roles["column_roles"].items():
                if info.get("role") == role:
                    return col
        return None
    
    def _get_all_columns(self) -> List[str]:
        """Get all available columns."""
        if "column_roles" in self.semantic_roles:
            return list(self.semantic_roles["column_roles"].keys())
        return []
    
    def save_feature_specs(self, output_path: Optional[Path] = None):
        """Save feature specifications to JSON."""
        if output_path is None:
            output_path = self.workspace_path / "metadata" / "feature_specs.json"
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        specs_data = [
            {
                "name": spec.name,
                "feature_type": spec.feature_type,
                "source_columns": spec.source_columns,
                "transformation": spec.transformation,
                "description": spec.description,
                "parameters": spec.parameters
            }
            for spec in self.feature_specs
        ]
        
        with open(output_path, "w") as f:
            json.dump(specs_data, f, indent=2)
            
        return output_path
