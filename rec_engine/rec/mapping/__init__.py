"""
Mapping Engine Module
Autonomous Recommendation Engine Platform
"""

from rec.mapping.engine import AutonomousMappingEngine
from rec.mapping.domain_loader import DomainLoader
from rec.mapping.embedding_matcher import EmbeddingMatcher
from rec.mapping.logic_booster import LogicBooster
from rec.mapping.cross_encoder import CrossEncoderScorer
from rec.mapping.global_resolver import GlobalResolver

__all__ = [
    "AutonomousMappingEngine",
    "DomainLoader",
    "EmbeddingMatcher",
    "LogicBooster",
    "CrossEncoderScorer",
    "GlobalResolver",
]
