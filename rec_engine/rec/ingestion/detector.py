"""
File Type Detector
Autonomous Recommendation Engine Platform
"""

from pathlib import Path
from typing import Optional, Dict, Any


class FileTypeDetector:
    """Detect file types based on extension and content inspection."""
    
    EXTENSION_MAP = {
        ".csv": "csv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".parquet": "parquet",
        ".db": "sqlite",
        ".sqlite": "sqlite",
        ".duckdb": "duckdb",
    }
    
    MAGIC_BYTES = {
        b"PAR1": "parquet",
        b'{"': "json",
        b"[{": "json",
    }
    
    def detect_type(self, file_path: Path) -> Optional[str]:
        """Detect the file type of a given file."""
        # First try extension-based detection
        ext = file_path.suffix.lower()
        if ext in self.EXTENSION_MAP:
            return self.EXTENSION_MAP[ext]
        
        # Fall back to magic bytes inspection
        try:
            with open(file_path, "rb") as f:
                header = f.read(4)
                
                for magic, file_type in self.MAGIC_BYTES.items():
                    if header.startswith(magic):
                        return file_type
                
                # Check for CSV-like content
                f.seek(0)
                first_line = f.readline().decode("utf-8", errors="ignore")
                if "," in first_line or "\t" in first_line:
                    return "csv"
                    
        except Exception:
            pass
        
        return None
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get detailed file information."""
        stat = file_path.stat()
        file_type = self.detect_type(file_path)
        
        return {
            "path": str(file_path.absolute()),
            "name": file_path.name,
            "extension": file_path.suffix,
            "type": file_type,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "is_supported": file_type is not None,
        }
