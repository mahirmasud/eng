"""
Three-Tower Retrieval Model Implementation.

Implements a three-tower architecture for semantic retrieval:
- User Tower: Encodes user features into embeddings
- Item Tower: Encodes item features into embeddings  
- Candidate Tower: Encodes candidate/context features for scoring

Uses contrastive loss for training with in-batch negatives.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer


class UserTower(nn.Module):
    """User encoding tower that produces user embeddings."""
    
    def __init__(
        self,
        user_feature_dim: int,
        embedding_dim: int = 384,
        hidden_layers: List[int] = [512, 256, 128],
        dropout: float = 0.1,
        activation: str = "relu",
        use_text_embeddings: bool = False,
        use_id_embedding: bool = False,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_text_embeddings = use_text_embeddings
        self.use_id_embedding = use_id_embedding
        
        # Text embedding model for categorical features
        if use_text_embeddings:
            self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            text_embed_dim = 384
        else:
            text_embed_dim = 0
            
        # ID embedding (optional — only included when use_id_embedding=True)
        categorical_embed_dim = 64
        if use_id_embedding:
            self.user_id_embedding = nn.Embedding(100000, categorical_embed_dim)
        else:
            self.user_id_embedding = None
            categorical_embed_dim = 0
        
        total_input_dim = user_feature_dim + categorical_embed_dim + text_embed_dim
        
        # Build MLP layers
        layers = []
        prev_dim = total_input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
            
        # Output projection
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.mlp = nn.Sequential(*layers)
        
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
        user_features: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
        text_features: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Forward pass through user tower.
        
        Args:
            user_features: Dense user features [batch, user_feature_dim]
            user_ids: User ID indices [batch]
            text_features: Optional text descriptions for embedding
            
        Returns:
            User embeddings [batch, embedding_dim]
        """
        components = [user_features]
        
        # Add user ID embedding (only when enabled at build time)
        if self.use_id_embedding and user_ids is not None:
            user_id_emb = self.user_id_embedding(user_ids)
            components.append(user_id_emb)
            
        # Add text embeddings
        if text_features is not None and self.use_text_embeddings:
            with torch.no_grad():
                text_emb = self.text_encoder.encode(
                    text_features, 
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
            components.append(text_emb)
            
        # Concatenate all components
        x = torch.cat(components, dim=-1)
        
        # Pass through MLP
        output = self.mlp(x)
        
        # L2 normalize for cosine similarity
        output = F.normalize(output, p=2, dim=-1)
        
        return output


class ItemTower(nn.Module):
    """Item encoding tower that produces item embeddings."""
    
    def __init__(
        self,
        item_feature_dim: int,
        embedding_dim: int = 384,
        hidden_layers: List[int] = [512, 256, 128],
        dropout: float = 0.1,
        activation: str = "relu",
        use_text_embeddings: bool = False,
        use_id_embedding: bool = False,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_text_embeddings = use_text_embeddings
        self.use_id_embedding = use_id_embedding
        
        # Text embedding model for item titles/descriptions
        if use_text_embeddings:
            self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            text_embed_dim = 384
        else:
            text_embed_dim = 0
            
        # Item ID embedding (optional — only included when use_id_embedding=True)
        categorical_embed_dim = 64
        if use_id_embedding:
            self.item_id_embedding = nn.Embedding(100000, categorical_embed_dim)
        else:
            self.item_id_embedding = None
            categorical_embed_dim = 0
        
        total_input_dim = item_feature_dim + categorical_embed_dim + text_embed_dim
        
        # Build MLP layers
        layers = []
        prev_dim = total_input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
            
        # Output projection
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.mlp = nn.Sequential(*layers)
        
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
        item_features: torch.Tensor,
        item_ids: Optional[torch.Tensor] = None,
        text_features: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Forward pass through item tower.
        
        Args:
            item_features: Dense item features [batch, item_feature_dim]
            item_ids: Item ID indices [batch]
            text_features: Optional text descriptions (titles, etc.)
            
        Returns:
            Item embeddings [batch, embedding_dim]
        """
        components = [item_features]
        
        # Add item ID embedding (only when enabled at build time)
        if self.use_id_embedding and item_ids is not None:
            item_id_emb = self.item_id_embedding(item_ids)
            components.append(item_id_emb)
            
        # Add text embeddings
        if text_features is not None and self.use_text_embeddings:
            with torch.no_grad():
                text_emb = self.text_encoder.encode(
                    text_features,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
            components.append(text_emb)
            
        # Concatenate all components
        x = torch.cat(components, dim=-1)
        
        # Pass through MLP
        output = self.mlp(x)
        
        # L2 normalize for cosine similarity
        output = F.normalize(output, p=2, dim=-1)
        
        return output


class CandidateTower(nn.Module):
    """Candidate encoding tower for context-aware scoring."""
    
    def __init__(
        self,
        context_feature_dim: int,
        embedding_dim: int = 384,
        hidden_layers: List[int] = [256, 128],
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Build MLP layers
        layers = []
        prev_dim = context_feature_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
            
        # Output projection
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, context_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through candidate tower.
        
        Args:
            context_features: Context features [batch, context_feature_dim]
            
        Returns:
            Candidate embeddings [batch, embedding_dim]
        """
        x = self.mlp(context_features)
        output = F.normalize(x, p=2, dim=-1)
        return output


class ThreeTowerRetriever(pl.LightningModule):
    """
    Three-Tower Retrieval Model using PyTorch Lightning.
    
    Trains with contrastive loss using in-batch negatives.
    """
    
    def __init__(
        self,
        user_feature_dim: int,
        item_feature_dim: int,
        context_feature_dim: int = 0,
        embedding_dim: int = 384,
        user_hidden_layers: List[int] = [512, 256, 128],
        item_hidden_layers: List[int] = [512, 256, 128],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        
        # Initialize towers
        self.user_tower = UserTower(
            user_feature_dim=user_feature_dim,
            embedding_dim=embedding_dim,
            hidden_layers=user_hidden_layers,
        )
        
        self.item_tower = ItemTower(
            item_feature_dim=item_feature_dim,
            embedding_dim=embedding_dim,
            hidden_layers=item_hidden_layers,
        )
        
        self.candidate_tower = None
        if context_feature_dim > 0:
            self.candidate_tower = CandidateTower(
                context_feature_dim=context_feature_dim,
                embedding_dim=embedding_dim,
            )
            
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
        item_ids: Optional[torch.Tensor] = None,
        user_text: Optional[List[str]] = None,
        item_text: Optional[List[str]] = None,
        context_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing user and item embeddings.
        
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        user_emb = self.user_tower(
            user_features, user_ids, user_text
        )
        item_emb = self.item_tower(
            item_features, item_ids, item_text
        )
        
        return user_emb, item_emb
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step with contrastive loss."""
        user_features = batch["user_features"]
        item_features = batch["item_features"]
        user_ids = batch.get("user_ids")
        item_ids = batch.get("item_ids")
        user_text = batch.get("user_text")
        item_text = batch.get("item_text")
        
        # Get embeddings
        user_emb, item_emb = self(
            user_features, item_features,
            user_ids, item_ids,
            user_text, item_text,
        )
        
        # Compute similarity matrix
        # [batch_size, batch_size]
        sim_matrix = torch.matmul(user_emb, item_emb.t()) / self.temperature
        
        # Create labels for contrastive loss
        # Positive pairs are on the diagonal
        batch_size = sim_matrix.size(0)
        labels = torch.arange(batch_size, device=self.device)
        
        # Compute loss (symmetric)
        loss_user = self.criterion(sim_matrix, labels)
        loss_item = self.criterion(sim_matrix.t(), labels)
        loss = (loss_user + loss_item) / 2
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Compute recall@K metrics
        with torch.no_grad():
            # Recall@10
            _, topk_indices = sim_matrix.topk(10, dim=1)
            matches = (topk_indices == labels.unsqueeze(1)).any(dim=1)
            recall_10 = matches.float().mean()
            self.log("recall@10", recall_10, prog_bar=True, on_epoch=True)
            
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        user_features = batch["user_features"]
        item_features = batch["item_features"]
        user_ids = batch.get("user_ids")
        item_ids = batch.get("item_ids")
        user_text = batch.get("user_text")
        item_text = batch.get("item_text")
        
        user_emb, item_emb = self(
            user_features, item_features,
            user_ids, item_ids,
            user_text, item_text,
        )
        
        sim_matrix = torch.matmul(user_emb, item_emb.t()) / self.temperature
        batch_size = sim_matrix.size(0)
        labels = torch.arange(batch_size, device=self.device)
        
        loss_user = self.criterion(sim_matrix, labels)
        loss_item = self.criterion(sim_matrix.t(), labels)
        loss = (loss_user + loss_item) / 2
        
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
    
    def get_user_embedding(
        self,
        user_features: np.ndarray,
        user_id: Optional[int] = None,
        user_text: Optional[str] = None,
    ) -> np.ndarray:
        """Get embedding for a single user."""
        self.eval()
        
        with torch.no_grad():
            user_feat_tensor = torch.FloatTensor(user_features).unsqueeze(0)
            user_id_tensor = torch.LongTensor([user_id]) if user_id is not None else None
            user_text_list = [user_text] if user_text is not None else None
            
            user_emb = self.user_tower(
                user_feat_tensor.to(self.device),
                user_id_tensor.to(self.device) if user_id_tensor is not None else None,
                user_text_list,
            )
            
        return user_emb.cpu().numpy()[0]
    
    def get_item_embedding(
        self,
        item_features: np.ndarray,
        item_id: Optional[int] = None,
        item_text: Optional[str] = None,
    ) -> np.ndarray:
        """Get embedding for a single item."""
        self.eval()
        
        with torch.no_grad():
            item_feat_tensor = torch.FloatTensor(item_features).unsqueeze(0)
            item_id_tensor = torch.LongTensor([item_id]) if item_id is not None else None
            item_text_list = [item_text] if item_text is not None else None
            
            item_emb = self.item_tower(
                item_feat_tensor.to(self.device),
                item_id_tensor.to(self.device) if item_id_tensor is not None else None,
                item_text_list,
            )
            
        return item_emb.cpu().numpy()[0]
    
    def save_model(self, path: str):
        """Save model checkpoint and configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch Lightning checkpoint
        torch.save(self.state_dict(), path / "three_tower.pt")
        
        # Save config
        config = {
            "user_feature_dim": self.hparams.user_feature_dim,
            "item_feature_dim": self.hparams.item_feature_dim,
            "context_feature_dim": self.hparams.context_feature_dim,
            "embedding_dim": self.hparams.embedding_dim,
            "user_hidden_layers": self.hparams.user_hidden_layers,
            "item_hidden_layers": self.hparams.item_hidden_layers,
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
        model.load_state_dict(torch.load(path / "three_tower.pt", map_location="cpu"))
        
        return model


class RetrievalDataset(Dataset):
    """Dataset for retrieval training."""
    
    def __init__(
        self,
        user_features: np.ndarray,
        item_features: np.ndarray,
        user_ids: Optional[np.ndarray] = None,
        item_ids: Optional[np.ndarray] = None,
        user_text: Optional[List[str]] = None,
        item_text: Optional[List[str]] = None,
    ):
        self.user_features = user_features
        self.item_features = item_features
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.user_text = user_text
        self.item_text = item_text
        
        assert len(user_features) == len(item_features)
        
    def __len__(self):
        return len(self.user_features)
    
    def __getitem__(self, idx: int):
        batch = {
            "user_features": torch.FloatTensor(self.user_features[idx]),
            "item_features": torch.FloatTensor(self.item_features[idx]),
        }
        
        if self.user_ids is not None:
            batch["user_ids"] = torch.LongTensor([self.user_ids[idx]])
            
        if self.item_ids is not None:
            batch["item_ids"] = torch.LongTensor([self.item_ids[idx]])
            
        if self.user_text is not None:
            batch["user_text"] = self.user_text[idx]
            
        if self.item_text is not None:
            batch["item_text"] = self.item_text[idx]
            
        return batch


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for DataLoader."""
    result = {
        "user_features": torch.stack([item["user_features"] for item in batch]),
        "item_features": torch.stack([item["item_features"] for item in batch]),
    }
    
    if "user_ids" in batch[0]:
        result["user_ids"] = torch.cat([item["user_ids"] for item in batch])
        
    if "item_ids" in batch[0]:
        result["item_ids"] = torch.cat([item["item_ids"] for item in batch])
        
    if "user_text" in batch[0]:
        result["user_text"] = [item["user_text"] for item in batch]
        
    if "item_text" in batch[0]:
        result["item_text"] = [item["item_text"] for item in batch]
        
    return result
