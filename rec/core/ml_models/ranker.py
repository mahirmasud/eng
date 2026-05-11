"""
MLP Ranker Implementation.

Implements a pairwise/listwise ranking model for re-ranking candidates.
Supports multiple loss functions: BPR, Hinge, Softmax cross-entropy.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


class MLPRanker(pl.LightningModule):
    """
    MLP-based Ranking Model using PyTorch Lightning.
    
    Supports:
    - Pairwise ranking (BPR loss, Hinge loss)
    - Listwise ranking (Softmax cross-entropy, ListNet)
    - Pointwise ranking (BCE loss)
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_layers: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        activation: str = "relu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        ranking_loss: str = "bpr",
        margin: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ranking_loss = ranking_loss
        self.margin = margin
        
        # Build MLP scoring network
        layers = []
        prev_dim = feature_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
            
        # Output layer (single score)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.scoring_network = nn.Sequential(*layers)
        
    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass producing ranking scores.
        
        Args:
            features: Input features [batch, feature_dim]
            
        Returns:
            Ranking scores [batch, 1]
        """
        return self.scoring_network(features)
    
    def compute_pairwise_loss(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise ranking loss."""
        if self.ranking_loss == "bpr":
            # Bayesian Personalized Ranking loss
            loss = -F.logsigmoid(positive_scores - negative_scores).mean()
        elif self.ranking_loss == "hinge":
            # Hinge loss with margin
            loss = torch.relu(self.margin - (positive_scores - negative_scores)).mean()
        else:
            raise ValueError(f"Unknown pairwise loss: {self.ranking_loss}")
            
        return loss
    
    def compute_listwise_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute listwise ranking loss."""
        if self.ranking_loss == "softmax":
            # Softmax cross-entropy over list
            log_probs = F.log_softmax(scores, dim=1)
            loss = -(labels * log_probs).sum(dim=1).mean()
        elif self.ranking_loss == "listnet":
            # ListNet loss
            p_label = F.softmax(labels, dim=1)
            p_score = F.log_softmax(scores, dim=1)
            loss = -(p_label * p_score).sum(dim=1).mean()
        else:
            raise ValueError(f"Unknown listwise loss: {self.ranking_loss}")
            
        return loss
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step with pairwise or listwise loss."""
        if batch.get("is_pairwise", False):
            # Pairwise training
            positive_features = batch["positive_features"]
            negative_features = batch["negative_features"]
            
            positive_scores = self(positive_features)
            negative_scores = self(negative_features)
            
            loss = self.compute_pairwise_loss(positive_scores, negative_scores)
            
            # Compute accuracy (positive should score higher)
            with torch.no_grad():
                correct = (positive_scores > negative_scores).float().mean()
                self.log("pairwise_accuracy", correct, prog_bar=True, on_epoch=True)
        else:
            # Listwise training
            features = batch["features"]
            labels = batch["labels"]
            
            scores = self(features)
            scores = scores.view(batch["num_items"], -1).t()  # [batch, num_items]
            labels = labels.view(batch["num_items"], -1).t().float()
            
            loss = self.compute_listwise_loss(scores, labels)
            
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        if batch.get("is_pairwise", False):
            positive_features = batch["positive_features"]
            negative_features = batch["negative_features"]
            
            positive_scores = self(positive_features)
            negative_scores = self(negative_features)
            
            loss = self.compute_pairwise_loss(positive_scores, negative_scores)
            
            with torch.no_grad():
                correct = (positive_scores > negative_scores).float().mean()
                self.log("val_accuracy", correct, prog_bar=True, on_epoch=True)
        else:
            features = batch["features"]
            labels = batch["labels"]
            
            scores = self(features)
            scores = scores.view(batch["num_items"], -1).t()
            labels = labels.view(batch["num_items"], -1).t().float()
            
            loss = self.compute_listwise_loss(scores, labels)
            
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=10, eta_min=1e-6
            ),
            "interval": "epoch",
        }
        
        return [optimizer], [scheduler]
    
    def score(self, features: np.ndarray) -> np.ndarray:
        """Score items for ranking."""
        self.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            scores = self(features_tensor)
            
        return scores.cpu().numpy().flatten()
    
    def rank(
        self,
        features: np.ndarray,
        top_k: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rank items by score.
        
        Returns:
            Tuple of (sorted_indices, sorted_scores)
        """
        scores = self.score(features)
        sorted_indices = np.argsort(-scores)  # Descending order
        sorted_scores = scores[sorted_indices]
        
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
            sorted_scores = sorted_scores[:top_k]
            
        return sorted_indices, sorted_scores
    
    def save_model(self, path: str):
        """Save model checkpoint and configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch Lightning checkpoint
        torch.save(self.state_dict(), path / "ranker.pt")
        
        # Save config
        config = {
            "feature_dim": self.hparams.feature_dim,
            "hidden_layers": self.hparams.hidden_layers,
            "dropout": self.hparams.dropout,
            "activation": self.hparams.activation,
            "ranking_loss": self.hparams.ranking_loss,
            "margin": self.hparams.margin,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def load_model(cls, path: str, **kwargs):
        """Load model from checkpoint."""
        path = Path(path)
        
        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)
            
        # Update config with any overrides
        config.update(kwargs)
        
        # Initialize model
        model = cls(**config)
        
        # Load weights
        model.load_state_dict(torch.load(path / "ranker.pt", map_location="cpu"))
        
        return model


class PairwiseRankingDataset(Dataset):
    """Dataset for pairwise ranking training."""
    
    def __init__(
        self,
        positive_features: np.ndarray,
        negative_features: np.ndarray,
    ):
        self.positive_features = positive_features
        self.negative_features = negative_features
        
        assert len(positive_features) == len(negative_features)
        
    def __len__(self):
        return len(self.positive_features)
    
    def __getitem__(self, idx: int):
        return {
            "positive_features": torch.FloatTensor(self.positive_features[idx]),
            "negative_features": torch.FloatTensor(self.negative_features[idx]),
            "is_pairwise": True,
        }


class ListwiseRankingDataset(Dataset):
    """Dataset for listwise ranking training."""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        num_items: int = 10,
    ):
        self.features = features
        self.labels = labels
        self.num_items = num_items
        
        # Group into lists
        self.num_lists = len(features) // num_items
        
    def __len__(self):
        return self.num_lists
    
    def __getitem__(self, idx: int):
        start_idx = idx * self.num_items
        end_idx = start_idx + self.num_items
        
        return {
            "features": torch.FloatTensor(self.features[start_idx:end_idx]),
            "labels": torch.FloatTensor(self.labels[start_idx:end_idx]),
            "num_items": self.num_items,
            "is_pairwise": False,
        }


def pairwise_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for pairwise ranking."""
    return {
        "positive_features": torch.stack([item["positive_features"] for item in batch]),
        "negative_features": torch.stack([item["negative_features"] for item in batch]),
        "is_pairwise": True,
    }


def listwise_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for listwise ranking."""
    features = torch.cat([item["features"] for item in batch], dim=0)
    labels = torch.cat([item["labels"] for item in batch], dim=0)
    
    return {
        "features": features,
        "labels": labels,
        "num_items": batch[0]["num_items"],
        "is_pairwise": False,
    }
