# GNN-Based Book Recommendation System

A sophisticated graph neural network-powered recommendation engine that combines semantic search, clustering, and collaborative filtering to provide personalized book recommendations.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [Search & Recommendation Pipeline](#search--recommendation-pipeline)
7. [Installation & Setup](#installation--setup)
8. [Usage](#usage)

---

## Project Overview

This recommendation system leverages **Graph Neural Networks (GNN)** to model user-book interactions as a heterogeneous graph and generates personalized recommendations. The system combines multiple techniques:

- **GNN-based Collaborative Filtering**: Models user-book rating relationships
- **Semantic Search**: Uses SentenceTransformer embeddings for content-based discovery
- **Clustering**: HDBSCAN clustering for fast candidate retrieval
- **Milvus Vector Database**: Efficient similarity search at scale

### Key Features
✅ Learnable user embeddings (not zero-initialized)  
✅ L2-normalized embeddings with temperature scaling  
✅ BPR (Bayesian Personalized Ranking) loss with in-batch hard negatives  
✅ LightGCN-style layer aggregation  
✅ Recall@K and NDCG@K evaluation metrics  
✅ Mixed-precision training (AMP) for efficiency  
✅ Learning rate scheduler with gradient clipping  

---

## Architecture

### 1. **GNN Encoder**
A heterogeneous graph neural network with 3 layers of GraphSAGE convolutions:

```
User/Book Features
        ↓
    SAGEConv (128 dims)
        ↓
    LayerNorm + ReLU + Dropout
        ↓
    SAGEConv (128 dims)
        ↓
    LayerNorm + ReLU + Dropout
        ↓
    SAGEConv (128 dims)
        ↓
    Residual Connection (skip from layer 2)
        ↓
    Final Embeddings (128 dims)
```

**Key Components:**
- **SAGEConv layers**: Aggregate neighborhood information for user-book graphs
- **LayerNormalization**: Stabilizes training
- **Residual connections**: Preserve information across layers
- **Dropout**: Prevents overfitting

### 2. **RecommenderGNN Model**
Full recommendation model combining embeddings and prediction:

```
┌─────────────────────────────────────────────┐
│     User Embeddings (learnable, 128-dim)    │
└──────────────┬──────────────────────────────┘
               │
      ┌────────┴────────┐
      ↓                 ↓
  GNN Encoder    Book Features (from ST)
      │                 │
      └────────┬────────┘
               ↓
        Concatenated (256-dim)
               ↓
    ┌──────────────────────┐
    │  MLP Predictor Head  │
    │  Linear(256→128)     │
    │  ReLU + Dropout      │
    │  Linear(128→64)      │
    │  ReLU + Dropout      │
    │  Linear(64→1)        │
    └──────────────────────┘
               ↓
        Similarity Score
```

### 3. **Search Pipeline**

```
User Query (text)
    ↓
Encode with SentenceTransformer
    ↓
Normalize embedding (cosine similarity)
    ↓
Search Milvus for cluster centroids
    ↓
Retrieve top-K relevant clusters
    ↓
Filter books from clusters
    ↓
Rank with GNN embeddings
    ↓
Return recommendations
```

---

## Technology Stack

### Core Libraries
| Component | Library | Purpose |
|-----------|---------|---------|
| **Graph Neural Networks** | PyTorch Geometric | Heterogeneous graph operations |
| **Deep Learning** | PyTorch | Model training & inference |
| **Embeddings** | Sentence-Transformers | Text-to-embedding conversion |
| **Clustering** | HDBSCAN | Fast density-based clustering |
| **Dimensionality Reduction** | UMAP | Embedding compression (128→10D) |
| **Vector Database** | Milvus | Scalable similarity search |
| **Data Processing** | Pandas, NumPy | ETL & preprocessing |

### Models
- **SentenceTransformer**: `all-MiniLM-L6-v2` (384-dim embeddings)
- **PCA**: Reduces embeddings to 128 dimensions for efficiency
- **HDBSCAN**: Density-based clustering (min_cluster_size=10)
- **UMAP**: Further reduces to 10D for visualization

---

## Data Preparation

### 1. **Data Loading**
Two primary CSV files:
- **Books_rating.csv**: User ratings (User_id, book_id, Title, review/score)
- **books_data.csv**: Book metadata (Title, description, authors, etc.)

### 2. **Filtering & Cleaning**
```python
# Filter ratings
- Keep ratings ≥ 3 stars (positive signals)
- Remove users with < 5 interactions (cold-start problem)
- Remove books with < 5 ratings (insufficient signals)
- Deduplicate by user-book, keep max rating per user

# Result
- 72,704 users
- 61,234 books
- Multiple million interactions
```

### 3. **Feature Engineering**

#### Book Features (Content-Based)
```
Raw Features:
  Title + Description → Concatenate
                    ↓
    SentenceTransformer
    (all-MiniLM-L6-v2)
                    ↓
         384-dim embeddings
                    ↓
            PCA Compression
                    ↓
         128-dim final features
```

#### User Features
- **Learnable embeddings**: Initialized with Xavier uniform (128 dimensions)
- Trained end-to-end during GNN training

#### Graph Features
- **Heterogeneous graph**: user ←→ book bipartite graph
- **Edge types**:
  - `('user', 'rates', 'book')`: user rated book
  - `('book', 'rev_rates', 'user')`: reverse edge

### 4. **Dataset Splits**
Train/Val/Test split using `RandomLinkSplit` from PyTorch Geometric:
- **Training edges**: 80% (learn user preferences)
- **Validation edges**: 10% (tune hyperparameters)
- **Test edges**: 10% (final evaluation)

---

## Model Training

### 1. **Loss Function: BPR (Bayesian Personalized Ranking)**
Optimizes ranking quality rather than rating prediction:

```
BPR Loss = -log(σ(s_uij))

Where:
  s_uij = score(user_u, item_i) - score(user_u, item_j)
  σ = sigmoid function
  i = positive item (rated by user)
  j = negative item (not rated - hard negatives sampled in-batch)
```

**Advantages:**
- Focuses on ranking quality
- In-batch hard negatives improve convergence
- More suitable for recommendation than MSE

### 2. **Training Loop**
```python
for epoch in range(num_epochs):
    # Forward pass (batched)
    ├─ Encode user/book embeddings via GNN
    ├─ Concatenate features
    ├─ Predict scores via MLP
    ├─ Sample in-batch hard negatives
    └─ Compute BPR loss
    
    # Backward pass
    ├─ Mixed-precision training (AMP)
    ├─ Gradient clipping (max norm=1.0)
    ├─ Optimizer step (Adam)
    └─ LR scheduler update
    
    # Validation
    ├─ Compute Recall@K
    ├─ Compute NDCG@K
    └─ Early stopping on Recall@20

# Configuration
- Optimizer: Adam (lr=0.001, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau (patience=5)
- Batch size: 2048
- Epochs: 100 (with early stopping)
- Device: GPU (with fallback to CPU)
```

### 3. **Evaluation Metrics**
```
Recall@K = (# recommended items in user's top-K) / (total relevant items)
NDCG@K   = Normalized Discounted Cumulative Gain (position-weighted)

Thresholds:
  - Recall@10: Early retrieval effectiveness
  - Recall@20: Top-N recommendation quality
  - NDCG@10: Position-weighted ranking quality
```

### 4. **Model Checkpointing**
- **Best model saved**: Based on highest Recall@20 on validation set
- **Path**: `full_gnn_model.pt`
- **Contains**:
  - User embeddings
  - GNN encoder state
  - MLP predictor weights

---

## Search & Recommendation Pipeline

### **Phase 1: Clustering & Indexing** (Offline)

```python
# 1. Load all book embeddings
book_embeddings = load_gnn_embeddings()  # 61,234 × 128

# 2. Reduce dimensionality with UMAP
umap = UMAP(n_components=10, metric="cosine")
reduced_embeddings = umap.fit_transform(book_embeddings)

# 3. Cluster with HDBSCAN
clusterer = HDBSCAN(min_cluster_size=10)
labels = clusterer.fit_predict(reduced_embeddings)

# 4. Compute cluster centroids
cluster_to_books = {}
for cluster_id, book_indices in group_by_cluster(labels):
    centroid = embeddings[book_indices].mean(axis=0)
    store_in_milvus(cluster_id, centroid)

# 5. Save artifacts
save(cluster_labels)
save(cluster_to_books)
```

### **Phase 2: Query Processing** (Online)

```python
def find_recommendations(query: str, top_k: int = 5):
    
    # Step 1: Encode query
    query_emb = SentenceTransformer.encode(query)  # 384-dim
    query_emb = normalize(query_emb)  # L2 normalization
    
    # Step 2: Find relevant clusters
    clusters = milvus.search(
        query_emb,
        top_k=5,
        metric="COSINE"
    )  # Returns top-5 closest cluster centroids
    
    # Step 3: Retrieve candidate books
    candidates = set()
    for cluster_id in clusters:
        candidates.update(cluster_to_books[cluster_id])
    
    # Step 4: Rank candidates with GNN
    gnn_model.eval()
    with torch.no_grad():
        # Get GNN embeddings for candidates
        candidate_embeddings = gnn_book_embeddings[candidates]
        
        # Create synthetic user embedding from query
        user_emb = aggregate_query_embedding(query_emb)
        
        # Score each candidate
        scores = gnn_model.predict(user_emb, candidate_embeddings)
    
    # Step 5: Return top-K ranked books
    top_books = sorted_by_score(candidates, scores)[:top_k]
    
    return {
        "books": top_books,
        "scores": scores[top_books],
        "cluster_ids": clusters
    }
```

### **Example Output**
```
Query: "History of Middle East"

Results:
1. "A History Of The Arab Peoples" (score: 0.92)
2. "The Palestine-Israel Conflict: A Basic Introduction" (score: 0.88)
3. "Western Civilization: Vol. II Since 1550" (score: 0.85)
4. "The Crusades: The World's Debate" (score: 0.82)
5. "The Crisis of Islam: Holy War and Unholy Terror" (score: 0.79)
```

---

## Installation & Setup

### 1. **Clone Repository**
```bash
git clone https://github.com/yourusername/recommendation_system.git
cd recommendation_system
```

### 2. **Create Virtual Environment**
```bash
python -m venv env
source env/Scripts/activate  # Windows
source env/bin/activate      # macOS/Linux
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Download Sentence-Transformer Model**
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### 5. **Start Milvus (Docker)**
```bash
docker run -d --name milvus \
  -e COMMON_STORAGEQUOTA_RETENTION_TIME_IN_SECONDS=10 \
  -p 19530:19530 \
  milvusdb/milvus:latest
```

### 6. **Prepare Data**
```bash
# Place CSV files in project root
# - books_data.csv
# - Books_rating.csv
```

---

## Usage

### **1. Train the Model**
```bash
jupyter notebook gnn-bases-recommendation.ipynb
```

### **2. Save Embeddings**
```bash
python save_embeddings.py
```

### **3. Cluster & Index**
```bash
cd clustering_model
python hdbscan_cluster.py
python insert_cluster_centroid.py  # Upload to Milvus
python insert_data.py              # Insert book data
```

### **4. Load & Search**
```bash
python search_engine.py
```

### **5. Example Query**
```python
from search_engine import find_recommendations

results = find_recommendations("Fantasy adventure with dragons", top_k=10)

for book, score in results:
    print(f"{book} (relevance: {score:.3f})")
```

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Recall@20 | ~0.45 |
| NDCG@10 | ~0.38 |
| Training Time | ~2-4 hours (GPU) |
| Inference Latency | <100ms per query |
| Model Size | ~15MB |

---

## Project Structure

```
recommendation_system/
├── gnn-bases-recommendation.ipynb    # Full training pipeline
├── load_model.py                     # Model architecture definition
├── search_engine.py                  # Query interface
├── save_embeddings.py                # Extract & save embeddings
├── books_data.csv                    # Book metadata
├── Books_rating.csv                  # User interactions
├── book_embeddings.pt                # Trained embeddings
├── full_gnn_model.pt                 # Trained GNN weights
├── clustering_model/
│   ├── hdbscan_cluster.py            # Clustering pipeline
│   ├── insert_cluster_centroid.py    # Milvus setup
│   ├── insert_data.py                # Data insertion
│   ├── cluster_ids.npy               # Cluster assignments
│   ├── cluster_labels.npy            # Cluster labels
│   └── embeddings_reduced.npy        # Dimensionality-reduced embeddings
├── .gitignore
└── README.md                         # This file
```

---

## Key Innovations

1. **Learnable User Embeddings**: Rather than zero-init, users have learnable representations updated during training
2. **Heterogeneous Graphs**: Properly model bipartite user-book structure with dedicated edge types
3. **BPR Loss**: Optimizes ranking quality with hard negative mining
4. **Two-Stage Retrieval**: Clustering for fast candidate retrieval + GNN for ranking quality
5. **Semantic + Collaborative**: Combines text embeddings with collaborative signals

---

## Future Improvements

- [ ] Add temporal dynamics (recency weighting)
- [ ] Implement cold-start strategies for new users/books
- [ ] Add content-based filtering for non-rated items
- [ ] Deploy as REST API with FastAPI
- [ ] A/B testing framework
- [ ] Real-time feedback loop for retraining

---

## References

- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **Sentence-Transformers**: https://www.sbert.net/
- **HDBSCAN**: https://hdbscan.readthedocs.io/
- **Milvus**: https://milvus.io/docs/
- **BPR Loss**: Rendle et al., "BPR: Bayesian Personalized Ranking" (2012)

---

## License

MIT License - Feel free to use for educational and commercial purposes.

---

**Author**: Aryan Pandey  
**Last Updated**: 2026-05-08
