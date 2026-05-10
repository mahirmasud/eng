# Autonomous Recommendation Engine (rec)

A **fully local, CLI-only** autonomous recommendation engine platform that trains and generates personalized recommendations from **any dataset** without requiring backend servers, web services, or cloud infrastructure.

## 🚀 Key Features

- **Autonomous Schema Understanding**: Automatically infers business meaning from arbitrary datasets
- **Semantic Mapping**: Uses embedding models to match columns to recommendation concepts
- **Human-in-the-Loop**: Interactive terminal review for validation and corrections
- **Three-Tower Architecture**: User, Item, and Candidate towers for semantic retrieval
- **DLRM Support**: Deep Learning Recommendation Models for personalized scoring
- **Cold Start Handling**: Metadata embeddings and semantic bootstrapping for new users/items
- **Local Vector Indexing**: FAISS/hnswlib for billion-scale retrieval on local hardware
- **GPU Acceleration**: Optional CUDA support for training and inference
- **Feedback Learning**: Continual learning from clicks, ratings, and engagement signals
- **Fully Offline**: No runtime dependencies on external services

## 📋 Requirements

- Python 3.9+
- CUDA-compatible GPU (optional, for acceleration)
- 16GB+ RAM recommended for large datasets
- 50GB+ free disk space for models and indices

## 🛠️ Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd rec_engine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Installation

```bash
rec --help
```

## 📚 Quick Start Guide

### Step 1: Prepare Your Data

Place your dataset(s) in a `datasets/` directory. Supported formats:
- CSV, JSON, JSONL, Parquet
- SQLite databases
- Event logs and clickstream exports

Create a `domain_pack/` directory with optional knowledge files:
```yaml
# domain_pack/schema.yaml
entities:
  user: ["user_id", "uid", "customer_id"]
  item: ["item_id", "product_id", "movie_id"]
  interaction: ["rating", "click", "purchase", "view_time"]

# domain_pack/keywords.yaml
synonyms:
  user: ["customer", "member", "account", "profile"]
  item: ["product", "content", "article", "video", "song"]
  engagement: ["interaction", "activity", "behavior", "signal"]
```

### Step 2: Ingest Data

```bash
rec ingest \
  --source ./datasets \
  --domain-pack ./domain_pack \
  --workspace ./workspace
```

**Output:**
- `workspace/source_manifest.json` - Detected files and schemas
- `workspace/dataset_profile.json` - Statistical profiles
- `workspace/schema_fingerprint.json` - Schema metadata

### Step 3: Autonomous Semantic Mapping

```bash
rec map --workspace ./workspace
```

This step:
- Loads domain knowledge
- Generates embeddings for column names
- Matches columns to semantic concepts using `all-MiniLM-L6-v2`
- Validates matches with cross-encoder `stsb-distilroberta-base`
- Resolves conflicts for 1:1 mapping integrity

**Output:**
- `workspace/resolved_mappings.json` - Column-to-concept mappings
- Confidence scores and reasoning signals

### Step 4: Business Meaning Layer

```bash
rec bml --workspace ./workspace
```

Identifies:
- **Identity**: User IDs, Item IDs
- **Signal**: Ratings, clicks, purchases
- **Feature**: Demographics, categories, timestamps
- **Event**: Interaction events
- **Session**: Session boundaries
- **Temporal**: Time-based features
- **Contextual**: Location, device, context

**Output:**
- `workspace/semantic_roles.json` - Role classifications
- `workspace/entity_graph.json` - Entity relationships
- `workspace/feature_catalog.json` - Feature definitions

### Step 5: Human-in-the-Loop Review

```bash
rec review --workspace ./workspace
```

Interactive terminal prompts:
```
? Column "uid" detected as USER_ID with 81% confidence.
Accept? [Y/n] y

? Column "watch_time" classified as ENGAGEMENT_SIGNAL.
Override? > dwell_time

? Multiple candidates for ITEM_ID detected. Select primary:
  1) product_id (92%)
  2) item_sku (78%)
  3) content_id (65%)
Choice [1]: 1
```

**Output:**
- `workspace/human_feedback.json` - Review decisions
- `workspace/corrected_mappings.json` - Final validated mappings

### Step 6: Generate Configuration

```bash
rec build-config --workspace ./workspace
```

Generates `workspace/rec_config.json` with:
- Entity mappings and feature roles
- Retrieval and ranking parameters
- DLRM architecture settings
- Vector index configuration
- Training hyperparameters
- Cold start strategies
- Feedback processing rules

### Step 7: Feature Engineering

```bash
rec features --workspace ./workspace
```

Automatically generates:
- User features (aggregations, statistics)
- Item features (popularity, categories)
- Interaction features (frequency, recency)
- Temporal features (hour, day, season)
- Session features (sequence, duration)
- Graph features (connectivity, centrality)

**Output:**
- `workspace/features/user_features.parquet`
- `workspace/features/item_features.parquet`
- `workspace/features/interaction_features.parquet`

### Step 8: Train Retrieval Model (Three-Tower)

```bash
rec train-retrieval \
  --workspace ./workspace \
  --epochs 10 \
  --batch-size 256 \
  --gpu  # Optional: use GPU
```

Trains user, item, and candidate towers for semantic retrieval.

**Output:**
- `workspace/models/retrieval_user_tower.pt`
- `workspace/models/retrieval_item_tower.pt`
- `workspace/models/retrieval_candidate_tower.pt`

### Step 9: Train Ranking Model

```bash
rec train-ranker \
  --workspace ./workspace \
  --epochs 5 \
  --batch-size 512 \
  --gpu
```

Trains pairwise/listwise ranking model for candidate scoring.

**Output:**
- `workspace/models/ranker.pt`

### Step 10: Train DLRM Model

```bash
rec train-dlrm \
  --workspace ./workspace \
  --epochs 15 \
  --batch-size 1024 \
  --embedding-dim 64 \
  --mlp-layers "256,128,64" \
  --gpu
```

Trains Deep Learning Recommendation Model with sparse embeddings.

**Output:**
- `workspace/models/dlrm.pt`

### Step 11: Build Vector Index

```bash
rec build-index \
  --workspace ./workspace \
  --index-type faiss \
  --nlist 4096 \
  --m 32  # For HNSW
```

Builds ANN index for fast candidate retrieval.

**Output:**
- `workspace/indices/item_index.faiss`
- `workspace/indices/user_index.faiss`
- `workspace/indices/metadata.parquet`

### Step 12: Generate Recommendations

```bash
rec recommend \
  --workspace ./workspace \
  --user-id "user_123" \
  --top-k 20 \
  --output ./recommendations.json
```

**Output:**
```json
{
  "user_id": "user_123",
  "recommendations": [
    {"item_id": "item_456", "score": 0.94, "reason": "Based on your viewing history"},
    {"item_id": "item_789", "score": 0.89, "reason": "Trending in your category"}
  ],
  "metadata": {
    "retrieval_time_ms": 45,
    "ranking_time_ms": 120,
    "model_version": "v1.0"
  }
}
```

### Step 13: Re-rank Results

```bash
rec rerank \
  --workspace ./workspace \
  --input ./recommendations.json \
  --diversity 0.3 \
  --novelty 0.2 \
  --freshness 0.1 \
  --output ./reranked_recommendations.json
```

Applies diversity, novelty, freshness, and business rule adjustments.

### Step 14: Evaluate Model Performance

```bash
rec evaluate \
  --workspace ./workspace \
  --test-split 0.2 \
  --metrics recall@10,recall@20,ndcg@10,map,mrr
```

**Output:**
```
Evaluation Results:
===================
Recall@10: 0.342
Recall@20: 0.458
NDCG@10: 0.287
MAP: 0.234
MRR: 0.412

Report saved to: workspace/reports/evaluation_report.json
```

### Step 15: Process Feedback

```bash
rec feedback \
  --workspace ./workspace \
  --feedback-file ./new_interactions.csv \
  --update-interval daily
```

Ingests new interaction data for continual learning.

### Step 16: Incremental Retraining

```bash
rec retrain \
  --workspace ./workspace \
  --models retrieval,ranker \
  --incremental \
  --gpu
```

Updates models with new feedback data.

### Step 17: Explain Recommendations

```bash
rec explain \
  --workspace ./workspace \
  --user-id "user_123" \
  --item-id "item_456"
```

**Output:**
```
Explanation for recommending "item_456" to "user_123":
======================================================
Primary Factors:
  - User watched similar items (weight: 0.45)
  - Item trending in user's location (weight: 0.28)
  - High engagement from similar users (weight: 0.18)
  - Recent release bonus (weight: 0.09)

Feature Contributions:
  - genre_match: +0.32
  - watch_history_overlap: +0.28
  - popularity_score: +0.15
  - recency_bonus: +0.09
```

### Step 18: Export Artifacts

```bash
rec export \
  --workspace ./workspace \
  --output ./export_package \
  --include models,indices,config
```

Packages all artifacts for deployment or sharing.

## 📁 Workspace Structure

After running the full pipeline, your workspace will contain:

```
workspace/
├── source_manifest.json
├── dataset_profile.json
├── schema_fingerprint.json
├── resolved_mappings.json
├── semantic_roles.json
├── entity_graph.json
├── feature_catalog.json
├── human_feedback.json
├── corrected_mappings.json
├── rec_config.json
├── features/
│   ├── user_features.parquet
│   ├── item_features.parquet
│   └── interaction_features.parquet
├── models/
│   ├── retrieval_user_tower.pt
│   ├── retrieval_item_tower.pt
│   ├── retrieval_candidate_tower.pt
│   ├── ranker.pt
│   └── dlrm.pt
├── indices/
│   ├── item_index.faiss
│   ├── user_index.faiss
│   └── metadata.parquet
├── reports/
│   ├── evaluation_report.json
│   └── training_logs.json
└── logs/
    └── rec_engine.log
```

## 🔧 Advanced Configuration

### Custom Domain Pack

Create a comprehensive `domain_pack/`:

```yaml
# domain_pack/rules.yaml
boost_rules:
  - condition: "column_name contains 'premium'"
    boost: 0.2
    target_concept: "HIGH_VALUE_SIGNAL"
  - condition: "data_type == 'timestamp' AND column_name contains 'last'"
    boost: 0.15
    target_concept: "RECENCY_FEATURE"

# domain_pack/metrics.yaml
business_metrics:
  primary: "purchase_rate"
  secondary: ["click_through_rate", "dwell_time", "conversion_rate"]
  constraints:
    min_diversity: 0.3
    max_repeat_items: 0.1

# domain_pack/validations.yaml
schema_validations:
  - entity: "USER"
    required_columns: ["user_id"]
    optional_columns: ["age", "gender", "location"]
  - entity: "ITEM"
    required_columns: ["item_id"]
    optional_columns: ["category", "price", "release_date"]

# domain_pack/questions.yaml
clarification_questions:
  - pattern: "multiple_user_candidates"
    question: "Multiple columns detected as potential USER_ID. Which is primary?"
  - pattern: "missing_interaction_signal"
    question: "No explicit rating found. Use implicit signals (clicks, views) instead?"
```

### GPU Optimization

Enable GPU acceleration in `rec_config.json`:

```json
{
  "training": {
    "device": "cuda",
    "mixed_precision": true,
    "gradient_checkpointing": true
  },
  "inference": {
    "device": "cuda",
    "batch_size": 1024
  }
}
```

### Distributed Local Training

Use Ray for multi-core/multi-GPU training:

```bash
rec train-dlrm \
  --workspace ./workspace \
  --distributed \
  --num-workers 4 \
  --gpus-per-worker 1
```

## 📊 Monitoring and Logging

All operations are logged to `workspace/logs/rec_engine.log`.

View training metrics:
```bash
rec logs --workspace ./workspace --component training --tail 100
```

Export MLflow-compatible metrics:
```bash
rec export-metrics --workspace ./workspace --format json
```

## 🔍 Troubleshooting

### Common Issues

**Issue: Out of Memory during training**
```bash
# Reduce batch size
rec train-dlrm --batch-size 256 --gradient-accumulation 4

# Enable gradient checkpointing
# Add to rec_config.json: "gradient_checkpointing": true
```

**Issue: Low mapping confidence**
```bash
# Enhance domain pack with more synonyms
# Run mapping with custom threshold
rec map --workspace ./workspace --confidence-threshold 0.6
```

**Issue: Slow retrieval**
```bash
# Use smaller index or quantization
rec build-index --index-type hnsw --m 16 --ef-construction 200
```

## 🔄 Future Backend Integration

The architecture is designed for future backend integration:

```python
# Example: Future FastAPI integration hook
from rec.core.ml_models import ThreeTowerRetriever

class RecommendationService:
    def __init__(self, workspace_path):
        self.retriever = ThreeTowerRetriever.load(workspace_path)
        self.ranker = RankerModel.load(workspace_path)
    
    async def get_recommendations(self, user_id, top_k=20):
        candidates = await self.retriever.retrieve(user_id, k=top_k*5)
        ranked = await self.ranker.rank(user_id, candidates)
        return ranked[:top_k]
```

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## 📞 Support

For issues and questions:
- GitHub Issues: [link]
- Documentation: [link]
- Community Forum: [link]
