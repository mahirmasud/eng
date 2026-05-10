"""
Cross Encoder Scorer - Contextual validation and confidence refinement
Autonomous Recommendation Engine Platform
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class CrossEncoderScorer:
    """
    Use cross-encoder models for fine-grained semantic validation.
    
    Uses stsb-distilroberta-base for contextual similarity scoring.
    Cross-encoders provide more accurate but slower scoring than 
    bi-encoders (embedding models).
    """
    
    def __init__(self, model_name: str = "stsb-distilroberta-base"):
        self.model_name = model_name
        self.model = None
        self.device = None
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self.model
    
    def score_pair(self, text1: str, text2: str) -> float:
        """Score a single pair of texts."""
        model = self._load_model()
        score = model.predict([[text1, text2]])[0]
        return float(score)
    
    def score_pairs(
        self, 
        pairs: List[Tuple[str, str]],
        batch_size: int = 32
    ) -> List[float]:
        """Score multiple pairs of texts."""
        if not pairs:
            return []
        
        model = self._load_model()
        
        # Convert tuples to list format expected by model
        pair_list = [[p[0], p[1]] for p in pairs]
        
        scores = model.predict(
            pair_list,
            batch_size=batch_size,
            show_progress_bar=len(pairs) > 10,
        )
        
        return [float(s) for s in scores]
    
    def score_with_normalization(
        self,
        pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """Score pairs and normalize to [0, 1] range."""
        raw_scores = self.score_pairs(pairs)
        
        if not raw_scores:
            return []
        
        # Sigmoid normalization for cross-encoder outputs
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        
        if max_score == min_score:
            return [0.5] * len(raw_scores)
        
        normalized = [
            (s - min_score) / (max_score - min_score)
            for s in raw_scores
        ]
        
        return normalized
    
    def validate_mapping(
        self,
        column: str,
        target_role: str,
        threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """Validate if a column-target mapping is semantically consistent."""
        # Create descriptive prompts for better context
        column_prompt = f"Database column named '{column}'"
        role_prompt = f"Represents {target_role.replace('_', ' ').lower()} in a recommendation system"
        
        score = self.score_pair(column_prompt, role_prompt)
        is_valid = score >= threshold
        
        return is_valid, score
    
    def rank_candidates(
        self,
        column: str,
        candidates: List[str]
    ) -> List[Tuple[str, float]]:
        """Rank candidate roles for a given column."""
        pairs = [(column, candidate) for candidate in candidates]
        scores = self.score_pairs(pairs)
        
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked
    
    def get_confidence_adjustment(
        self,
        column: str,
        target_role: str,
        initial_confidence: float
    ) -> float:
        """Get adjusted confidence using cross-encoder validation."""
        is_valid, ce_score = self.validate_mapping(column, target_role)
        
        # Blend initial confidence with cross-encoder score
        # Weight cross-encoder higher as it's more accurate
        adjusted = 0.3 * initial_confidence + 0.7 * ce_score
        
        return adjusted
    
    def batch_validate(
        self,
        mappings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Batch validate a list of mappings."""
        pairs = [(m["column"], m["target_role"]) for m in mappings]
        scores = self.score_pairs(pairs)
        
        validated = []
        for i, mapping in enumerate(mappings):
            updated = mapping.copy()
            updated["cross_encoder_score"] = scores[i] if i < len(scores) else 0.0
            updated["is_validated"] = updated["cross_encoder_score"] >= 0.5
            validated.append(updated)
        
        return validated
