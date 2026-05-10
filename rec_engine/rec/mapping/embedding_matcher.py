"""
Embedding Matcher - Semantic matching using sentence embeddings
Autonomous Recommendation Engine Platform
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class EmbeddingMatcher:
    """
    Generate and match embeddings for semantic schema understanding.
    
    Uses SentenceTransformers with all-MiniLM-L6-v2 model for efficient
    semantic similarity computation.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.cache_dir = Path.home() / ".rec_engine" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self.model
    
    def generate_embeddings(
        self, 
        texts: List[str],
        batch_size: int = 32
    ) -> Dict[str, np.ndarray]:
        """Generate embeddings for a list of texts."""
        model = self._load_model()
        
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        return {text: emb for text, emb in zip(texts, embeddings)}
    
    def find_top_k(
        self,
        query_embedding: np.ndarray,
        target_embeddings: Dict[str, np.ndarray],
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find top-k most similar target embeddings."""
        # Compute cosine similarities
        similarities = {}
        for target, target_emb in target_embeddings.items():
            sim = self.cosine_similarity(query_embedding, target_emb)
            similarities[target] = sim
        
        # Sort and return top-k
        sorted_sims = sorted(
            similarities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:k]
        
        return sorted_sims
    
    def cosine_similarity(
        self, 
        a: np.ndarray, 
        b: np.ndarray
    ) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def batch_cosine_similarity(
        self,
        query_embeddings: np.ndarray,
        target_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute batch cosine similarity matrix."""
        # Normalize
        query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        target_norm = target_embeddings / (np.linalg.norm(target_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Matrix multiplication
        return np.dot(query_norm, target_norm.T)
    
    def semantic_search(
        self,
        query: str,
        candidates: List[str],
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """Perform semantic search for a query against candidates."""
        # Generate query embedding
        query_emb = self.generate_embeddings([query])
        query_vec = query_emb[query]
        
        # Generate candidate embeddings
        candidate_embs = self.generate_embeddings(candidates)
        
        # Find top-k
        return self.find_top_k(query_vec, candidate_embs, k)
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension of the model."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()
    
    def save_embeddings(
        self, 
        embeddings: Dict[str, np.ndarray], 
        filepath: Path
    ) -> None:
        """Save embeddings to disk."""
        np.savez(
            filepath,
            keys=list(embeddings.keys()),
            values=np.array(list(embeddings.values()))
        )
    
    def load_embeddings(self, filepath: Path) -> Dict[str, np.ndarray]:
        """Load embeddings from disk."""
        data = np.load(filepath)
        keys = data["keys"]
        values = data["values"]
        return {key: vec for key, vec in zip(keys, values)}
