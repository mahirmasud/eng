"""
Data Ingestion Module
Autonomous Recommendation Engine Platform
"""

from rec.ingestion.loader import DataIngester
from rec.ingestion.detector import FileTypeDetector
from rec.ingestion.profiler import DatasetProfiler
from rec.ingestion.validator import DataValidator

__all__ = [
    "DataIngester",
    "FileTypeDetector",
    "DatasetProfiler",
    "DataValidator",
]
