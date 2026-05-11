"""Training Pipeline for recommendation models."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import polars as pl

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl_trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from .three_tower import (
    ThreeTowerRetriever, 
    RetrievalDataset, 
    collate_fn as retrieval_collate_fn,
)
from .dlrm import DLRMModel, DLRMDataset, dlrm_collate_fn
from .ranker import (
    MLPRanker, 
    PairwiseRankingDataset, 
    ListwiseRankingDataset,
    pairwise_collate_fn,
    listwise_collate_fn,
)


class TrainingPipeline:
    """Unified training pipeline for all recommendation models."""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.config = self._load_config()
        self.models_dir = self.workspace_path / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from workspace."""
        config_path = self.workspace_path / "rec_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
            
        with open(config_path, "r") as f:
            return json.load(f)
    
    def _load_features(self) -> Dict[str, Any]:
        """Load features from workspace."""
        features_path = self.workspace_path / "processed" / "features.parquet"
        
        if not features_path.exists():
            # Try alternative paths
            possible_paths = [
                self.workspace_path / "features" / "features.parquet",
                self.workspace_path / "processed" / "interactions.parquet",
            ]
            
            for path in possible_paths:
                if path.exists():
                    features_path = path
                    break
                    
        if not features_path.exists():
            raise FileNotFoundError(f"Features not found at {features_path}")
            
        df = pl.read_parquet(features_path)
        return df.to_pandas()
    
    def train_retrieval(
        self,
        epochs: int = 10,
        batch_size: int = 256,
        gpu: bool = False,
        **kwargs,
    ) -> ThreeTowerRetriever:
        """
        Train the three-tower retrieval model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            gpu: Whether to use GPU
            **kwargs: Additional model/training parameters
            
        Returns:
            Trained ThreeTowerRetriever model
        """
        print("\n[bold cyan]🎯 Training Retrieval Model (Three-Tower)[/bold cyan]")
        
        # Load features
        data = self._load_features()
        
        # Get feature dimensions from config
        retrieval_config = self.config.get("retrieval", {})
        embedding_dim = retrieval_config.get("embedding_dim", 384)
        training_config = self.config.get("training", {})
        
        # Determine feature columns
        feature_roles = self.config.get("feature_roles", {})
        user_feature_cols = feature_roles.get("user_features", [])
        item_feature_cols = feature_roles.get("item_features", [])
        
        # For now, use simple numeric features
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Split into user and item features (simplified)
        # In production, this would use proper entity resolution
        n_samples = len(data)
        half = n_samples // 2
        
        user_features = data[numeric_cols].iloc[:half].values
        item_features = data[numeric_cols].iloc[half:].values
        
        # Ensure equal lengths by duplicating if needed
        min_len = min(len(user_features), len(item_features))
        user_features = user_features[:min_len]
        item_features = item_features[:min_len]
        
        # Create dataset
        dataset = RetrievalDataset(
            user_features=user_features.astype(np.float32),
            item_features=item_features.astype(np.float32),
        )
        
        # Create dataloader
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=retrieval_collate_fn,
        )
        
        # Initialize model
        feature_dim = user_features.shape[1]
        model = ThreeTowerRetriever(
            user_feature_dim=feature_dim,
            item_feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            learning_rate=training_config.get("learning_rate", 1e-3),
            weight_decay=training_config.get("weight_decay", 1e-5),
        )
        
        # Setup training
        callbacks = [
            ModelCheckpoint(
                dirpath=self.models_dir / "retrieval",
                filename="three_tower-{epoch:02d}-{train_loss:.4f}",
                monitor="train_loss_epoch",
                mode="min",
                save_top_k=3,
            ),
            EarlyStopping(
                monitor="train_loss_epoch",
                patience=training_config.get("early_stopping", {}).get("patience", 3),
                mode="min",
            ),
        ]
        
        logger = CSVLogger(
            save_dir=self.workspace_path / "logs",
            name="retrieval",
        )
        
        trainer = pl_trainer.Trainer(
            max_epochs=epochs,
            accelerator="gpu" if gpu else "cpu",
            devices=1,
            # batch_size argument removed (handled via DataLoader)
            callbacks=callbacks,
            logger=logger,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
        )
        
        # Train
        print(f"\nConfiguration:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Embedding dim: {embedding_dim}")
        print(f"  - Device: {'GPU' if gpu else 'CPU'}")
        print(f"\nStarting training...")
        
        trainer.fit(model, train_loader)
        
        # Save model
        model_save_path = self.models_dir / "retrieval"
        model.save_model(model_save_path)
        
        print(f"\n[green]✓ Retrieval training complete![/green]")
        print(f"  - Model saved to: {model_save_path}")
        print(f"  - Best val_loss: {trainer.callback_metrics.get('val_loss', 'N/A')}")
        
        return model
    
    def train_dlrm(
        self,
        epochs: int = 10,
        batch_size: int = 1024,
        gpu: bool = False,
        **kwargs,
    ) -> DLRMModel:
        """
        Train the DLRM model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            gpu: Whether to use GPU
            **kwargs: Additional model/training parameters
            
        Returns:
            Trained DLRMModel
        """
        print("\n[bold cyan]🧠 Training DLRM Model[/bold cyan]")
        
        # Load features
        data = self._load_features()
        
        # Get DLRM config
        dlrm_config = self.config.get("dlrm", {})
        dense_arch = dlrm_config.get("dense_arch", {})
        sparse_arch = dlrm_config.get("sparse_arch", {})
        training_config = self.config.get("training", {})
        
        # Identify dense and sparse features
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column if present
        target_col = None
        for col in ['label', 'target', 'rating', 'click']:
            if col in numeric_cols:
                target_col = col
                numeric_cols.remove(col)
                break
                
        if target_col is None and len(numeric_cols) > 0:
            # Use last numeric column as target
            target_col = numeric_cols.pop()
        
        # Prepare features
        dense_features = data[numeric_cols].fillna(0).values.astype(np.float32)
        
        # Encode categorical features
        sparse_features = {}
        sparse_dims = {}
        for col in categorical_cols[:10]:  # Limit to 10 categorical features
            unique_vals = data[col].unique()
            mapping = {v: i for i, v in enumerate(unique_vals)}
            sparse_features[col] = data[col].map(mapping).fillna(0).values.astype(np.int64)
            sparse_dims[col] = len(unique_vals)
        
        # Prepare labels
        if target_col:
            labels = (data[target_col] > 0).astype(np.float32).values
        else:
            # Generate pseudo-labels for demo
            labels = np.random.randint(0, 2, len(dense_features)).astype(np.float32)
        
        # Create dataset
        dataset = DLRMDataset(
            dense_features=dense_features,
            sparse_features=sparse_features,
            labels=labels,
        )
        
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dlrm_collate_fn,
        )
        
        # Initialize model
        model = DLRMModel(
            dense_feature_dim=dense_features.shape[1],
            sparse_feature_dims=sparse_dims or {"dummy": 100},
            embedding_dim=sparse_arch.get("embedding_dim", 64),
            dense_hidden_layers=dense_arch.get("hidden_layers", [512, 256, 128]),
            prediction_hidden_layers=dlrm_config.get("prediction_head", {}).get("hidden_layers", [256, 128, 64]),
            activation=dense_arch.get("activation", "relu"),
            dropout=dense_arch.get("dropout", 0.1),
            learning_rate=training_config.get("learning_rate", 1e-3),
        )
        
        # Setup training
        callbacks = [
            ModelCheckpoint(
                dirpath=self.models_dir / "dlrm",
                filename="dlrm-{epoch:02d}-{train_loss:.4f}",
                monitor="train_loss_epoch",
                mode="min",
                save_top_k=3,
            ),
        ]
        
        logger = CSVLogger(
            save_dir=self.workspace_path / "logs",
            name="dlrm",
        )
        
        trainer = pl_trainer.Trainer(
            max_epochs=epochs,
            accelerator="gpu" if gpu else "cpu",
            devices=1,
            callbacks=callbacks,
            logger=logger,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
        )
        
        # Train
        print(f"\nConfiguration:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Dense arch: {dense_arch}")
        print(f"  - Sparse arch: {sparse_arch}")
        print(f"\nStarting training...")
        
        trainer.fit(model, train_loader)
        
        # Save model
        model_save_path = self.models_dir / "dlrm"
        model.save_model(model_save_path)
        
        print(f"\n[green]✓ DLRM training complete![/green]")
        print(f"  - Model saved to: {model_save_path}")
        
        return model
    
    def train_ranker(
        self,
        epochs: int = 10,
        batch_size: int = 512,
        gpu: bool = False,
        ranking_loss: str = "bpr",
        **kwargs,
    ) -> MLPRanker:
        """
        Train the MLP ranker model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            gpu: Whether to use GPU
            ranking_loss: Loss function ('bpr', 'hinge', 'softmax')
            **kwargs: Additional model/training parameters
            
        Returns:
            Trained MLPRanker model
        """
        print("\n[bold cyan]📊 Training Ranking Model[/bold cyan]")
        
        # Load features
        data = self._load_features()
        
        # Get ranking config
        ranking_config = self.config.get("ranking", {})
        training_config = self.config.get("training", {})
        
        # Identify feature columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if present
        target_col = None
        for col in ['label', 'target', 'rating', 'click']:
            if col in numeric_cols:
                target_col = col
                numeric_cols.remove(col)
                break
        
        # Prepare features
        features = data[numeric_cols].fillna(0).values.astype(np.float32)
        
        # Prepare positive/negative pairs for pairwise ranking
        if target_col:
            ratings = data[target_col].values
            positive_mask = ratings > ratings.median()
            negative_mask = ~positive_mask
            
            positive_features = features[positive_mask]
            negative_features = features[negative_mask]
            
            # Balance classes
            min_count = min(len(positive_features), len(negative_features))
            positive_features = positive_features[:min_count]
            negative_features = negative_features[:min_count]
        else:
            # Generate random pairs for demo
            positive_features = features[:len(features)//2]
            negative_features = features[len(features)//2:len(features)]
            min_count = min(len(positive_features), len(negative_features))
            positive_features = positive_features[:min_count]
            negative_features = negative_features[:min_count]
        
        # Create dataset
        dataset = PairwiseRankingDataset(
            positive_features=positive_features,
            negative_features=negative_features,
        )
        
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=pairwise_collate_fn,
        )
        
        # Initialize model
        model = MLPRanker(
            feature_dim=features.shape[1],
            hidden_layers=ranking_config.get("hidden_layers", [256, 128, 64]),
            dropout=ranking_config.get("dropout", 0.1),
            activation=ranking_config.get("activation", "relu"),
            learning_rate=training_config.get("learning_rate", 1e-3),
            ranking_loss=ranking_loss,
        )
        
        # Setup training
        callbacks = [
            ModelCheckpoint(
                dirpath=self.models_dir / "ranker",
                filename="ranker-{epoch:02d}-{train_loss:.4f}",
                monitor="train_loss_epoch",
                mode="min",
                save_top_k=3,
            ),
        ]
        
        logger = CSVLogger(
            save_dir=self.workspace_path / "logs",
            name="ranker",
        )
        
        trainer = pl_trainer.Trainer(
            max_epochs=epochs,
            accelerator="gpu" if gpu else "cpu",
            devices=1,
            callbacks=callbacks,
            logger=logger,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
        )
        
        # Train
        print(f"\nConfiguration:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Model: mlp_ranker")
        print(f"  - Ranking loss: {ranking_loss}")
        print(f"\nStarting training...")
        
        trainer.fit(model, train_loader)
        
        # Save model
        model_save_path = self.models_dir / "ranker"
        model.save_model(model_save_path)
        
        print(f"\n[green]✓ Ranker training complete![/green]")
        print(f"  - Model saved to: {model_save_path}")
        
        return model
