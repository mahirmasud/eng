"""
Data Ingester - Main ingestion orchestrator
Autonomous Recommendation Engine Platform
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import polars as pl
import duckdb

from rec.ingestion.detector import FileTypeDetector
from rec.ingestion.profiler import DatasetProfiler
from rec.ingestion.validator import DataValidator


class DataIngester:
    """
    Main data ingestion orchestrator.
    
    Responsibilities:
    - Detect file types
    - Load datasets from various formats
    - Profile schemas
    - Validate metadata
    - Compute statistics
    - Generate lineage metadata
    - Detect semantic column candidates
    """
    
    SUPPORTED_FORMATS = {
        "csv": ".csv",
        "json": ".json",
        "jsonl": ".jsonl",
        "parquet": ".parquet",
        "sqlite": ".db",
        "duckdb": ".duckdb",
    }
    
    def __init__(
        self,
        source_path: Path,
        domain_pack_path: Path,
        workspace_path: Path,
        supported_types: Optional[List[str]] = None
    ):
        self.source_path = Path(source_path)
        self.domain_pack_path = Path(domain_pack_path)
        self.workspace_path = Path(workspace_path)
        self.supported_types = supported_types or list(self.SUPPORTED_FORMATS.keys())
        
        self.detector = FileTypeDetector()
        self.profiler = DatasetProfiler()
        self.validator = DataValidator()
        
        # Ensure workspace directories exist
        (self.workspace_path / "data").mkdir(parents=True, exist_ok=True)
        (self.workspace_path / "metadata").mkdir(parents=True, exist_ok=True)
        (self.workspace_path / "models").mkdir(parents=True, exist_ok=True)
        (self.workspace_path / "indexes").mkdir(parents=True, exist_ok=True)
        (self.workspace_path / "feedback").mkdir(parents=True, exist_ok=True)
        (self.workspace_path / "logs").mkdir(parents=True, exist_ok=True)
    
    def detect_files(self) -> List[Dict[str, Any]]:
        """Detect all supported dataset files in source path."""
        detected = []
        
        if self.source_path.is_file():
            files = [self.source_path]
        else:
            files = list(self.source_path.rglob("*"))
        
        for file_path in files:
            if not file_path.is_file():
                continue
            
            file_type = self.detector.detect_type(file_path)
            if file_type and file_type in self.supported_types:
                stat = file_path.stat()
                detected.append({
                    "path": str(file_path.absolute()),
                    "name": file_path.name,
                    "type": file_type,
                    "size_bytes": stat.st_size,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
        
        return detected
    
    def load_dataset(self, file_info: Dict[str, Any]) -> pl.DataFrame:
        """Load a dataset into a Polars DataFrame."""
        file_path = Path(file_info["path"])
        file_type = file_info["type"]
        
        if file_type == "csv":
            df = pl.read_csv(file_path, infer_schema_length=10000)
        elif file_type == "json":
            df = pl.read_json(file_path)
        elif file_type == "jsonl":
            df = pl.read_ndjson(file_path)
        elif file_type == "parquet":
            df = pl.read_parquet(file_path)
        elif file_type == "sqlite":
            conn = duckdb.connect(str(file_path))
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            # Load first table or union all
            if tables:
                df = pl.read_database(f"SELECT * FROM {tables[0][0]}", conn)
            else:
                df = pl.DataFrame()
            conn.close()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return df
    
    def profile_dataset(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Profile a dataset and generate statistics."""
        df = self.load_dataset(file_info)
        
        profile = self.profiler.generate_profile(df, file_info)
        
        # Save profiled data to workspace
        output_path = self.workspace_path / "data" / f"{file_info['name']}.parquet"
        df.write_parquet(output_path)
        profile["output_path"] = str(output_path)
        
        return profile
    
    def validate_dataset(
        self, 
        file_info: Dict[str, Any], 
        profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate dataset against rules."""
        validation_result = self.validator.validate(file_info, profile)
        return validation_result
    
    def generate_schema_fingerprint(
        self, 
        profiles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a unique fingerprint for the schema."""
        # Combine all column names and types
        schema_data = []
        for profile in profiles:
            for col in profile.get("columns", []):
                schema_data.append(f"{col['name']}:{col['dtype']}")
        
        schema_string = "|".join(sorted(schema_data))
        fingerprint = hashlib.sha256(schema_string.encode()).hexdigest()
        
        return {
            "fingerprint": fingerprint,
            "schema_hash": fingerprint[:32],
            "column_count": len(schema_data),
            "generated_at": datetime.now().isoformat(),
            "profiles_count": len(profiles),
        }
    
    def detect_semantic_candidates(
        self, 
        profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect columns that are likely semantic matches."""
        candidates = []
        
        # Common recommendation-related column patterns
        user_patterns = ["user", "uid", "customer", "member", "account"]
        item_patterns = ["item", "product", "article", "document", "content", "movie", "book"]
        interaction_patterns = ["click", "view", "purchase", "rating", "score", "interaction"]
        temporal_patterns = ["time", "date", "timestamp", "created", "updated"]
        session_patterns = ["session", "visit", "browse"]
        
        for col in profile.get("columns", []):
            col_lower = col["name"].lower()
            
            candidate_type = None
            confidence = 0.5
            
            for pattern in user_patterns:
                if pattern in col_lower:
                    candidate_type = "USER_ID"
                    confidence = max(confidence, 0.8)
                    break
            
            for pattern in item_patterns:
                if pattern in col_lower:
                    candidate_type = "ITEM_ID"
                    confidence = max(confidence, 0.8)
                    break
            
            for pattern in interaction_patterns:
                if pattern in col_lower:
                    candidate_type = "INTERACTION"
                    confidence = max(confidence, 0.7)
                    break
            
            for pattern in temporal_patterns:
                if pattern in col_lower:
                    candidate_type = "TEMPORAL"
                    confidence = max(confidence, 0.6)
                    break
            
            if candidate_type:
                candidates.append({
                    "column": col["name"],
                    "dtype": col["dtype"],
                    "candidate_type": candidate_type,
                    "confidence": confidence,
                })
        
        return candidates
