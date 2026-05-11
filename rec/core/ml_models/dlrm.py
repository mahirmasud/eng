"""
DLRM (Deep Learning Recommendation Model) Implementation.

Implements the DLRM architecture with:
- Dense arch: MLP for continuous features
- Sparse arch: Embedding tables for categorical features
- Interaction layer: Feature interactions
- Prediction head: Final scoring layer
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


class DLRMModel(pl.LightningModule):
    """
    Deep Learning Recommendation Model (DLRM).
    
    Architecture:
    1. Dense Arch: MLP processing continuous features
    2. Sparse Arch: Embedding lookup for categorical features
    3. Interaction Layer: Dot product / outer product interactions
    4. Prediction Head: MLP for final score prediction
    """
    
    def __init__(
        self,
        dense_feature_dim: int,
        sparse_feature_dims: Dict[str, int],
        embedding_dim: int = 64,
        dense_hidden_layers: List[int] = [512, 256, 128],
        prediction_hidden_layers: List[int] = [256, 128, 64],
        activation: str = "relu",
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        interaction_type: str = "dot",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.interaction_type = interaction_type
        
        # Dense arch - MLP for continuous features
        self.dense_mlp = self._build_mlp(
            input_dim=dense_feature_dim,
            hidden_layers=dense_hidden_layers,
            output_dim=None,  # No final layer, output is intermediate
            dropout=dropout,
            activation=activation,
        )
        dense_output_dim = dense_hidden_layers[-1] if dense_hidden_layers else dense_feature_dim
        
        # Sparse arch - Embedding tables for each categorical feature
        self.embedding_tables = nn.ModuleDict()
        for feature_name, vocab_size in sparse_feature_dims.items():
            self.embedding_tables[feature_name] = nn.Embedding(
                vocab_size, embedding_dim
            )
            
        # Calculate interaction output dimension
        # Dense output + sum of all embeddings
        num_sparse = len(sparse_feature_dims)
        interaction_input_dim = dense_output_dim + (num_sparse * embedding_dim)
        
        # Interaction layer - creates feature crosses
        self.interaction_layer = None
        if interaction_type == "outer":
            # Outer product creates quadratic features
            self.interaction_proj = nn.Linear(interaction_input_dim, interaction_input_dim)
            interaction_output_dim = interaction_input_dim + (interaction_input_dim ** 2)
        else:
            # Dot product interaction (default)
            interaction_output_dim = interaction_input_dim
            
        # Prediction head
        pred_layers = []
        prev_dim = interaction_output_dim
        
        for hidden_dim in prediction_hidden_layers[:-1]:
            pred_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
            
        # Final output layer (single score)
        pred_layers.append(nn.Linear(prev_dim, 1))
        pred_layers.append(nn.Sigmoid())
        
        self.prediction_head = nn.Sequential(*pred_layers)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(activation, nn.ReLU())
    
    def _build_mlp(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> nn.Sequential:
        """Build MLP layers."""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
            
        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))
            
        return nn.Sequential(*layers)
    
    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through DLRM.
        
        Args:
            dense_features: Continuous features [batch, dense_feature_dim]
            sparse_features: Dict of categorical feature indices
            
        Returns:
            Prediction scores [batch, 1]
        """
        # Process dense features through MLP
        dense_out = self.dense_mlp(dense_features)
        
        # Get embeddings for sparse features
        sparse_embeddings = []
        for feature_name, indices in sparse_features.items():
            if feature_name in self.embedding_tables:
                emb = self.embedding_tables[feature_name](indices)
                sparse_embeddings.append(emb)
                
        # Concatenate dense and sparse representations
        if sparse_embeddings:
            sparse_concat = torch.cat(sparse_embeddings, dim=-1)
            combined = torch.cat([dense_out, sparse_concat], dim=-1)
        else:
            combined = dense_out
            
        # Apply interaction layer
        if self.interaction_layer is not None:
            combined = self.interaction_layer(combined)
        elif hasattr(self, 'interaction_proj'):
            # Outer product interaction
            projected = self.interaction_proj(combined)
            outer = (combined.unsqueeze(-1) * projected.unsqueeze(-2)).view(
                combined.size(0), -1
            )
            combined = torch.cat([combined, outer], dim=-1)
            
        # Prediction head
        output = self.prediction_head(combined)
        
        return output.squeeze(-1)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        dense_features = batch["dense_features"]
        sparse_features = batch["sparse_features"]
        labels = batch["labels"]
        
        predictions = self(dense_features, sparse_features)
        
        loss = self.criterion(predictions, labels.float())
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Compute AUC approximation
        with torch.no_grad():
            preds_binary = (predictions > 0.5).float()
            accuracy = (preds_binary == labels).float().mean()
            self.log("train_accuracy", accuracy, prog_bar=True, on_epoch=True)
            
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        dense_features = batch["dense_features"]
        sparse_features = batch["sparse_features"]
        labels = batch["labels"]
        
        predictions = self(dense_features, sparse_features)
        loss = self.criterion(predictions, labels.float())
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        
        # Compute accuracy
        with torch.no_grad():
            preds_binary = (predictions > 0.5).float()
            accuracy = (preds_binary == labels).float().mean()
            self.log("val_accuracy", accuracy, prog_bar=True, on_epoch=True)
            
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
    
    def predict(self, dense_features: np.ndarray, sparse_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        self.eval()
        
        with torch.no_grad():
            dense_tensor = torch.FloatTensor(dense_features)
            sparse_tensors = {
                k: torch.LongTensor(v) for k, v in sparse_features.items()
            }
            
            output = self(dense_tensor.to(self.device), sparse_tensors)
            
        return output.cpu().numpy()
    
    def save_model(self, path: str):
        """Save model checkpoint and configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch Lightning checkpoint
        torch.save(self.state_dict(), path / "dlrm.pt")
        
        # Save config
        config = {
            "dense_feature_dim": self.hparams.dense_feature_dim,
            "sparse_feature_dims": self.hparams.sparse_feature_dims,
            "embedding_dim": self.hparams.embedding_dim,
            "dense_hidden_layers": self.hparams.dense_hidden_layers,
            "prediction_hidden_layers": self.hparams.prediction_hidden_layers,
            "activation": self.hparams.activation,
            "dropout": self.hparams.dropout,
            "interaction_type": self.hparams.interaction_type,
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
        model.load_state_dict(torch.load(path / "dlrm.pt", map_location="cpu"))
        
        return model


class DLRMDataset(Dataset):
    """Dataset for DLRM training."""
    
    def __init__(
        self,
        dense_features: np.ndarray,
        sparse_features: Dict[str, np.ndarray],
        labels: np.ndarray,
    ):
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.labels = labels
        
        self.num_samples = len(dense_features)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int):
        batch = {
            "dense_features": torch.FloatTensor(self.dense_features[idx]),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        
        for key, value in self.sparse_features.items():
            batch[f"sparse_{key}"] = torch.tensor(value[idx], dtype=torch.long)
            
        return batch


def dlrm_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for DLRM DataLoader."""
    result = {
        "dense_features": torch.stack([item["dense_features"] for item in batch]),
        "labels": torch.FloatTensor([item["labels"] for item in batch]),
        "sparse_features": {},
    }
    
    # Collect sparse features
    sparse_keys = [k for k in batch[0].keys() if k.startswith("sparse_")]
    for key in sparse_keys:
        feature_name = key.replace("sparse_", "")
        result["sparse_features"][feature_name] = torch.LongTensor([
            item[key] for item in batch
        ])
        
    return result
