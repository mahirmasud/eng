"""
Workspace Manager Utility
Autonomous Recommendation Engine Platform
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class WorkspaceManager:
    """Manage workspace directory structure and files."""
    
    def __init__(self, workspace_path: str):
        self.path = Path(workspace_path)
        self.subdirs = [
            "data",
            "metadata",
            "models",
            "indexes",
            "feedback",
            "logs",
            "config",
            "temp",
        ]
    
    def initialize(self) -> None:
        """Initialize workspace directory structure."""
        for subdir in self.subdirs:
            (self.path / subdir).mkdir(parents=True, exist_ok=True)
        
        # Create workspace metadata
        workspace_info = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "path": str(self.path.absolute()),
        }
        
        workspace_file = self.path / "workspace.json"
        if not workspace_file.exists():
            with open(workspace_file, "w") as f:
                json.dump(workspace_info, f, indent=2)
    
    def save_json(self, filename: str, data: Dict[str, Any]) -> Path:
        """Save JSON data to workspace."""
        filepath = self.path / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath
    
    def load_json(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load JSON data from workspace."""
        filepath = self.path / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                return json.load(f)
        return None
    
    def save_parquet(self, filename: str, df) -> Path:
        """Save Polars DataFrame to parquet."""
        filepath = self.path / "data" / filename
        df.write_parquet(filepath)
        return filepath
    
    def load_parquet(self, filename: str):
        """Load Polars DataFrame from parquet."""
        import polars as pl
        filepath = self.path / "data" / filename
        if filepath.exists():
            return pl.read_parquet(filepath)
        return None
    
    def get_file_path(self, filename: str, subdir: str = "") -> Path:
        """Get absolute path for a file in workspace."""
        if subdir:
            return self.path / subdir / filename
        return self.path / filename
    
    def file_exists(self, filename: str, subdir: str = "") -> bool:
        """Check if a file exists in workspace."""
        return self.get_file_path(filename, subdir).exists()
    
    def list_files(self, subdir: str = "", pattern: str = "*") -> list:
        """List files in a workspace subdirectory."""
        target_dir = self.path / subdir if subdir else self.path
        if target_dir.exists():
            return list(target_dir.glob(pattern))
        return []
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get workspace metadata."""
        workspace_file = self.path / "workspace.json"
        if workspace_file.exists():
            with open(workspace_file, "r") as f:
                return json.load(f)
        return {}
    
    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """Update workspace metadata."""
        metadata = self.get_metadata()
        metadata.update(updates)
        metadata["updated_at"] = datetime.now().isoformat()
        self.save_json("workspace.json", metadata)
