"""Main feature engineering engine using Featuretools for automated feature generation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class FeatureEngineeringEngine:
    """
    Automated feature engineering engine using Featuretools.
    
    This engine transforms temporal and relational datasets into feature matrices
    for machine learning using automated feature engineering techniques.
    """
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.metadata_dir = workspace_path / "metadata"
        self.processed_dir = workspace_path / "processed"
        self.semantic_roles = {}
        self.entity_graph = {}
        self.feature_catalog = {}
        self.feature_specs = []
        self.es = None  # EntitySet for Featuretools
        self.feature_matrix = None
        
    def load_metadata(self):
        """Load semantic roles, entity graph, and feature catalog from workspace."""
        logger.info("Loading metadata from workspace...")
        
        if (self.metadata_dir / "semantic_roles.json").exists():
            with open(self.metadata_dir / "semantic_roles.json", "r") as f:
                self.semantic_roles = json.load(f)
            logger.info("Loaded semantic roles")
                
        if (self.metadata_dir / "entity_graph.json").exists():
            with open(self.metadata_dir / "entity_graph.json", "r") as f:
                self.entity_graph = json.load(f)
            logger.info("Loaded entity graph")
                
        if (self.metadata_dir / "feature_catalog.json").exists():
            with open(self.metadata_dir / "feature_catalog.json", "r") as f:
                self.feature_catalog = json.load(f)
            logger.info("Loaded feature catalog")
            
    def load_data(self) -> pl.DataFrame:
        """Load processed data from workspace."""
        # Try to load from multiple possible locations
        possible_paths = [
            self.processed_dir / "interactions.parquet",
            self.workspace_path / "raw" / "interactions.parquet",
            self.processed_dir / "data.parquet",
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Loading data from {path}")
                return pl.read_parquet(path)
                
        # If no parquet found, try CSV in raw
        raw_dir = self.workspace_path / "raw"
        if raw_dir.exists():
            csv_files = list(raw_dir.glob("*.csv"))
            if csv_files:
                logger.info(f"Loading data from {csv_files[0]}")
                return pl.read_csv(csv_files[0])
                
        raise FileNotFoundError("No processed data found. Run 'rec ingest' first.")
    
    def prepare_entityset(self, df: pl.DataFrame) -> bool:
        """
        Prepare EntitySet for Featuretools.
        
        Returns True if successful, False if Featuretools is not available.
        """
        try:
            import featuretools as ft
        except ImportError:
            logger.warning("Featuretools not installed. Using basic feature engineering.")
            return False
            
        # Create EntitySet
        self.es = ft.EntitySet(id="recommendation_data")
        
        # Get column mappings
        user_id_col = self._get_column_for_role("USER_ID")
        item_id_col = self._get_column_for_role("ITEM_ID")
        timestamp_col = self._get_column_for_role("TIMESTAMP")
        session_col = self._get_column_for_role("SESSION_ID")
        signal_col = self._get_column_for_role("ENGAGEMENT_SIGNAL")
        
        # Convert Polars to Pandas for Featuretools
        pdf = df.to_pandas()
        
        # Add transaction entity (interactions)
        transaction_columns = []
        if user_id_col:
            transaction_columns.append(user_id_col)
        if item_id_col:
            transaction_columns.append(item_id_col)
        if timestamp_col:
            transaction_columns.append(timestamp_col)
        if signal_col:
            transaction_columns.append(signal_col)
        if session_col:
            transaction_columns.append(session_col)
            
        # Add all other columns
        for col in df.columns:
            if col not in transaction_columns:
                transaction_columns.append(col)
                
        self.es.add_dataframe(
            dataframe_name="transactions",
            dataframe=pdf[transaction_columns],
            index="transaction_id",
            make_index=True,
            time_index=timestamp_col if timestamp_col else None,
        )
        
        # Add user entity if user_id exists
        if user_id_col:
            user_df = pdf[[user_id_col]].drop_duplicates()
            self.es.add_dataframe(
                dataframe_name="users",
                dataframe=user_df,
                index=user_id_col,
            )
            
            # Add relationship
            self.es.add_relationship("users", user_id_col, "transactions", user_id_col)
            
        # Add item entity if item_id exists
        if item_id_col:
            item_df = pdf[[item_id_col]].drop_duplicates()
            self.es.add_dataframe(
                dataframe_name="items",
                dataframe=item_df,
                index=item_id_col,
            )
            
            # Add relationship
            self.es.add_relationship("items", item_id_col, "transactions", item_id_col)
            
        logger.info(f"EntitySet created with {len(self.es.dataframes)} entities")
        return True
    
    def generate_features(self, use_featuretools: bool = True) -> pl.DataFrame:
        """
        Generate features using automated feature engineering.
        
        Args:
            use_featuretools: Whether to use Featuretools for deep feature synthesis
            
        Returns:
            DataFrame with generated features
        """
        logger.info("Starting feature generation...")
        
        # Load data
        df = self.load_data()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Task 1: Basic statistical features
            task = progress.add_task("[cyan]Generating basic statistical features...", total=1)
            df = self._generate_basic_features(df)
            progress.update(task, advance=1)
            
            # Task 2: Temporal features
            task = progress.add_task("[cyan]Generating temporal features...", total=1)
            df = self._generate_temporal_features(df)
            progress.update(task, advance=1)
            
            # Task 3: Aggregation features
            task = progress.add_task("[cyan]Generating aggregation features...", total=1)
            df = self._generate_aggregation_features(df)
            progress.update(task, advance=1)
            
            # Task 4: Interaction features
            task = progress.add_task("[cyan]Generating interaction features...", total=1)
            df = self._generate_interaction_features(df)
            progress.update(task, advance=1)
            
            # Task 5: Session features
            task = progress.add_task("[cyan]Generating session features...", total=1)
            df = self._generate_session_features(df)
            progress.update(task, advance=1)
            
            # Task 6: Deep feature synthesis with Featuretools (if available)
            if use_featuretools:
                task = progress.add_task("[cyan]Running Deep Feature Synthesis...", total=1)
                try:
                    df = self._run_deep_feature_synthesis(df)
                except Exception as e:
                    logger.warning(f"Deep feature synthesis failed: {e}. Using basic features only.")
                progress.update(task, advance=1)
                
            # Task 7: Feature selection and cleanup
            task = progress.add_task("[cyan]Finalizing feature matrix...", total=1)
            df = self._finalize_features(df)
            progress.update(task, advance=1)
            
        logger.info(f"Generated {len(df.columns)} features")
        self.feature_matrix = df
        return df
    
    def _generate_basic_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate basic statistical features."""
        logger.debug("Generating basic features...")
        
        # Get signal column
        signal_col = self._get_column_for_role("ENGAGEMENT_SIGNAL")
        
        if signal_col and signal_col in df.columns:
            # Normalize signal if numeric
            if df[signal_col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                min_val = df[signal_col].min()
                max_val = df[signal_col].max()
                if max_val > min_val:
                    df = df.with_columns(
                        ((pl.col(signal_col) - min_val) / (max_val - min_val)).alias("signal_normalized")
                    )
                    
        return df
    
    def _generate_temporal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate temporal features from timestamp."""
        logger.debug("Generating temporal features...")
        
        timestamp_col = self._get_column_for_role("TIMESTAMP")
        
        if timestamp_col and timestamp_col in df.columns:
            # Ensure timestamp is datetime
            if df[timestamp_col].dtype == pl.Utf8:
                df = df.with_columns(pl.col(timestamp_col).str.to_datetime().alias(timestamp_col))
                
            # Extract temporal components
            df = df.with_columns([
                pl.col(timestamp_col).dt.hour().alias("hour_of_day"),
                pl.col(timestamp_col).dt.weekday().alias("day_of_week"),
                pl.col(timestamp_col).dt.day().alias("day_of_month"),
                pl.col(timestamp_col).dt.month().alias("month"),
                pl.col(timestamp_col).dt.year().alias("year"),
                (pl.col(timestamp_col).dt.weekday() > 5).alias("is_weekend"),
            ])
            
            # Is business hours (9 AM - 5 PM)
            df = df.with_columns(
                ((pl.col("hour_of_day") >= 9) & (pl.col("hour_of_day") <= 17)).alias("is_business_hours")
            )
            
        return df
    
    def _generate_aggregation_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate aggregation features for users and items."""
        logger.debug("Generating aggregation features...")
        
        user_id_col = self._get_column_for_role("USER_ID")
        item_id_col = self._get_column_for_role("ITEM_ID")
        signal_col = self._get_column_for_role("ENGAGEMENT_SIGNAL")
        timestamp_col = self._get_column_for_role("TIMESTAMP")
        
        agg_features = {}
        
        # User aggregations
        if user_id_col and user_id_col in df.columns:
            user_agg = df.group_by(user_id_col).agg([
                pl.len().alias("user_interaction_count"),
                pl.col(signal_col).mean().alias("user_avg_signal") if signal_col else pl.lit(0).alias("user_avg_signal"),
                pl.col(signal_col).std().alias("user_signal_std") if signal_col else pl.lit(0).alias("user_signal_std"),
                pl.col(timestamp_col).max().alias("user_last_activity") if timestamp_col else pl.lit(None).alias("user_last_activity"),
            ])
            
            # Calculate days since last activity
            if timestamp_col and "user_last_activity" in user_agg.columns:
                now = datetime.now()
                user_agg = user_agg.with_columns(
                    ((now - pl.col("user_last_activity")).dt.total_days()).alias("user_days_since_last_activity")
                )
                
            for col in user_agg.columns:
                if col != user_id_col:
                    agg_features[col] = user_agg.select([user_id_col, col])
                    
        # Item aggregations
        if item_id_col and item_id_col in df.columns:
            item_agg = df.group_by(item_id_col).agg([
                pl.len().alias("item_interaction_count"),
                pl.col(signal_col).mean().alias("item_avg_signal") if signal_col else pl.lit(0).alias("item_avg_signal"),
                pl.col(signal_col).std().alias("item_signal_std") if signal_col else pl.lit(0).alias("item_signal_std"),
            ])
            
            for col in item_agg.columns:
                if col != item_id_col:
                    agg_features[col] = item_agg.select([item_id_col, col])
                    
        # Join aggregations back to main dataframe
        for feat_name, feat_df in agg_features.items():
            if user_id_col and user_id_col in feat_df.columns:
                df = df.join(feat_df, on=user_id_col, how="left")
            elif item_id_col and item_id_col in feat_df.columns:
                df = df.join(feat_df, on=item_id_col, how="left")
                
        return df
    
    def _generate_interaction_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate user-item interaction features."""
        logger.debug("Generating interaction features...")
        
        user_id_col = self._get_column_for_role("USER_ID")
        item_id_col = self._get_column_for_role("ITEM_ID")
        
        if user_id_col and item_id_col and both_in_df(user_id_col, item_id_col, df):
            # Count historical interactions between user-item pairs
            interaction_counts = df.group_by([user_id_col, item_id_col]).agg(
                pl.len().alias("user_item_interaction_history")
            )
            
            df = df.join(interaction_counts, on=[user_id_col, item_id_col], how="left")
            
        return df
    
    def _generate_session_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate session-level features."""
        logger.debug("Generating session features...")
        
        session_col = self._get_column_for_role("SESSION_ID")
        user_id_col = self._get_column_for_role("USER_ID")
        timestamp_col = self._get_column_for_role("TIMESTAMP")
        
        group_col = session_col if (session_col and session_col in df.columns) else user_id_col
        
        if group_col and group_col in df.columns:
            if timestamp_col and timestamp_col in df.columns:
                # Session duration
                session_stats = df.group_by(group_col).agg([
                    (pl.col(timestamp_col).max() - pl.col(timestamp_col).min()).dt.total_seconds().alias("session_duration_seconds"),
                    pl.len().alias("session_length"),
                ])
            else:
                session_stats = df.group_by(group_col).agg([
                    pl.len().alias("session_length"),
                ])
                
            df = df.join(session_stats, on=group_col, how="left")
            
            # Session position
            if timestamp_col and timestamp_col in df.columns:
                df = df.with_columns(
                    pl.col(timestamp_col).rank(method="ordinal").over(group_col).alias("session_position")
                )
                
        return df
    
    def _run_deep_feature_synthesis(self, df: pl.DataFrame) -> pl.DataFrame:
        """Run Deep Feature Synthesis using Featuretools."""
        logger.info("Running Deep Feature Synthesis with Featuretools...")
        
        import featuretools as ft
        
        if self.es is None:
            self.prepare_entityset(df)
            
        # Run DFS
        feature_matrix, feature_defs = ft.dfs(
            entityset=self.es,
            target_dataframe_name="transactions",
            max_depth=2,
            agg_primitives=["count", "mean", "sum", "std", "min", "max"],
            trans_primitives=["absolute", "divide_by_feature", "multiply_boolean"],
            verbose=True,
        )
        
        logger.info(f"Generated {len(feature_defs)} features via DFS")
        
        # Convert back to Polars
        df_ft = pl.from_pandas(feature_matrix.reset_index())
        
        # Keep original columns and add new features
        original_cols = [c for c in df.columns if c in df_ft.columns]
        new_cols = [c for c in df_ft.columns if c not in df.columns]
        
        # Select relevant columns
        df_result = df_ft[original_cols + new_cols[:50]]  # Limit to 50 new features to avoid explosion
        
        return df_result
    
    def _finalize_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Finalize feature matrix by cleaning and selecting features."""
        logger.debug("Finalizing feature matrix...")
        
        # Remove columns with too many nulls
        null_threshold = 0.5
        cols_to_keep = []
        for col in df.columns:
            null_ratio = df[col].null_count() / len(df)
            if null_ratio < null_threshold:
                cols_to_keep.append(col)
                
        df = df.select(cols_to_keep)
        
        # Fill remaining nulls
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        if numeric_cols:
            df = df.with_columns([
                pl.col(c).fill_null(0) for c in numeric_cols
            ])
            
        categorical_cols = df.select(pl.col(pl.Utf8)).columns
        if categorical_cols:
            df = df.with_columns([
                pl.col(c).fill_null("UNKNOWN") for c in categorical_cols
            ])
            
        # Remove duplicate columns
        df = df.unique()
        
        return df
    
    def save_features(self, output_path: Optional[Path] = None) -> Path:
        """Save feature matrix to disk."""
        if output_path is None:
            output_path = self.processed_dir / "features.parquet"
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.feature_matrix is not None:
            self.feature_matrix.write_parquet(output_path)
            logger.info(f"Features saved to {output_path}")
        else:
            logger.warning("No feature matrix to save")
            
        return output_path
    
    def save_feature_metadata(self, output_path: Optional[Path] = None) -> Path:
        """Save feature metadata and statistics."""
        if output_path is None:
            output_path = self.metadata_dir / "feature_metadata.json"
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "num_features": len(self.feature_matrix.columns) if self.feature_matrix is not None else 0,
            "num_samples": len(self.feature_matrix) if self.feature_matrix is not None else 0,
            "feature_statistics": {},
        }
        
        if self.feature_matrix is not None:
            for col in self.feature_matrix.columns:
                col_stats = {
                    "dtype": str(self.feature_matrix[col].dtype),
                    "null_count": int(self.feature_matrix[col].null_count()),
                    "unique_count": int(self.feature_matrix[col].n_unique()),
                }
                
                if self.feature_matrix[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    col_stats["min"] = float(self.feature_matrix[col].min()) if self.feature_matrix[col].min() is not None else None
                    col_stats["max"] = float(self.feature_matrix[col].max()) if self.feature_matrix[col].max() is not None else None
                    col_stats["mean"] = float(self.feature_matrix[col].mean()) if self.feature_matrix[col].mean() is not None else None
                    
                metadata["feature_statistics"][col] = col_stats
                
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
            
        logger.info(f"Feature metadata saved to {output_path}")
        return output_path
    
    def _get_column_for_role(self, role: str) -> Optional[str]:
        """Get column name for a given semantic role."""
        if "column_roles" in self.semantic_roles:
            for col, info in self.semantic_roles["column_roles"].items():
                if info.get("role") == role:
                    return col
        return None


def both_in_df(col1: str, col2: str, df: pl.DataFrame) -> bool:
    """Check if both columns are in dataframe."""
    return col1 in df.columns and col2 in df.columns
