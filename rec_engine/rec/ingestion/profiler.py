"""
Dataset Profiler
Autonomous Recommendation Engine Platform
"""

import polars as pl
from typing import Dict, List, Any
from datetime import datetime


class DatasetProfiler:
    """Generate comprehensive profiles for datasets."""
    
    def generate_profile(
        self, 
        df: pl.DataFrame, 
        file_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a complete profile for a dataset."""
        
        # Basic statistics
        num_rows = len(df)
        num_columns = len(df.columns)
        memory_bytes = df.estimated_size()
        memory_mb = memory_bytes / (1024 * 1024)
        
        # Column profiles
        column_profiles = []
        for col_name in df.columns:
            col_profile = self._profile_column(df, col_name)
            column_profiles.append(col_profile)
        
        # Data quality metrics
        total_cells = num_rows * num_columns
        null_count = sum(df[col].null_count() for col in df.columns)
        null_percentage = (null_count / total_cells * 100) if total_cells > 0 else 0
        
        # Duplicate detection
        try:
            duplicate_count = num_rows - len(df.unique())
        except Exception:
            duplicate_count = 0
        
        profile = {
            "dataset_name": file_info["name"],
            "source_path": file_info["path"],
            "file_type": file_info["type"],
            "num_rows": num_rows,
            "num_columns": num_columns,
            "memory_bytes": memory_bytes,
            "memory_mb": memory_mb,
            "columns": column_profiles,
            "data_quality": {
                "total_cells": total_cells,
                "null_count": null_count,
                "null_percentage": round(null_percentage, 2),
                "duplicate_rows": duplicate_count,
            },
            "profiled_at": datetime.now().isoformat(),
        }
        
        return profile
    
    def _profile_column(self, df: pl.DataFrame, col_name: str) -> Dict[str, Any]:
        """Profile a single column."""
        col = df[col_name]
        
        dtype = str(col.dtype)
        null_count = col.null_count()
        null_percentage = (null_count / len(df) * 100) if len(df) > 0 else 0
        
        # Type-specific statistics
        stats = {}
        
        try:
            if dtype in ["Int64", "Float64", "Int32", "Float32"]:
                stats = {
                    "min": float(col.min()) if col.min() is not None else None,
                    "max": float(col.max()) if col.max() is not None else None,
                    "mean": float(col.mean()) if col.mean() is not None else None,
                    "std": float(col.std()) if col.std() is not None else None,
                    "unique_count": col.n_unique(),
                }
            elif dtype in ["Utf8", "String"]:
                stats = {
                    "unique_count": col.n_unique(),
                    "avg_length": float(col.str.len_chars().mean()) if col.str.len_chars().mean() is not None else None,
                    "max_length": int(col.str.len_chars().max()) if col.str.len_chars().max() is not None else None,
                }
            elif dtype in ["Boolean"]:
                true_count = col.sum()
                stats = {
                    "true_count": int(true_count) if true_count is not None else 0,
                    "false_count": len(df) - null_count - (int(true_count) if true_count is not None else 0),
                }
        except Exception:
            pass
        
        # Infer semantic type
        semantic_type = self._infer_semantic_type(col, col_name)
        
        return {
            "name": col_name,
            "dtype": dtype,
            "null_count": null_count,
            "null_percentage": round(null_percentage, 2),
            "unique_count": col.n_unique() if dtype not in ["Boolean"] else 2,
            "stats": stats,
            "semantic_type": semantic_type,
        }
    
    def _infer_semantic_type(
        self, 
        col: pl.Series, 
        col_name: str
    ) -> str:
        """Infer the semantic type of a column."""
        col_lower = col_name.lower()
        dtype = str(col.dtype)
        
        # ID patterns
        if any(p in col_lower for p in ["id", "uid", "uuid", "_id"]):
            return "IDENTIFIER"
        
        # Temporal patterns
        if any(p in col_lower for p in ["date", "time", "timestamp", "created", "updated"]):
            return "TEMPORAL"
        
        # Numeric patterns
        if any(p in col_lower for p in ["score", "rating", "price", "amount", "count"]):
            return "NUMERIC_METRIC"
        
        # Categorical patterns
        if any(p in col_lower for p in ["category", "type", "status", "name"]):
            return "CATEGORICAL"
        
        # Text patterns
        if any(p in col_lower for p in ["description", "text", "content", "title"]):
            return "TEXT"
        
        # Boolean patterns
        if any(p in col_lower for p in ["is_", "has_", "flag", "bool"]):
            return "BOOLEAN"
        
        # Default based on dtype
        if dtype in ["Int64", "Float64", "Int32", "Float32"]:
            unique_ratio = col.n_unique() / len(col) if len(col) > 0 else 0
            if unique_ratio < 0.1:
                return "CATEGORICAL_NUMERIC"
            return "NUMERIC"
        elif dtype in ["Utf8", "String"]:
            return "TEXT"
        elif dtype in ["Boolean"]:
            return "BOOLEAN"
        
        return "UNKNOWN"
