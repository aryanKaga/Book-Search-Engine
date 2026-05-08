import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import SAGEConv, to_hetero


# ============================================================
# CONFIG
# ============================================================

NUM_USERS = 72704
NUM_BOOKS = 61234

HIDDEN_DIM = 128
DEVICE = "cpu"


# ============================================================
# GNN ENCODER
# ============================================================

class GNNEncoder(nn.Module):

    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()

        self.conv1 = SAGEConv((-1, -1), hidden_dim)
        self.conv2 = SAGEConv((-1, -1), hidden_dim)
        self.conv3 = SAGEConv((-1, -1), hidden_dim)

        self.drop = nn.Dropout(dropout)

        self.bn1 = nn.LayerNorm(hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):

        x1 = self.bn1(
            self.conv1(x, edge_index).relu()
        )

        x1 = self.drop(x1)

        x2 = self.bn2(
            self.conv2(x1, edge_index).relu()
        )

        x2 = self.drop(x2)

        x3 = self.conv3(x2, edge_index)

        return x3 + x2


# ============================================================
# RECOMMENDER MODEL
# ============================================================

class RecommenderGNN(nn.Module):

    def __init__(
        self,
        num_users,
        num_books,
        hidden_dim=256,
        dropout=0.3
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # ====================================================
        # USER EMBEDDINGS
        # ====================================================

        self.user_emb = nn.Embedding(
            num_users,
            128
        )

        nn.init.xavier_uniform_(
            self.user_emb.weight
        )

        # ====================================================
        # GNN BACKBONE
        # ====================================================

        base_encoder = GNNEncoder(
            hidden_dim,
            dropout
        )

        metadata = (
            ['user', 'book'],
            [
                ('user', 'rates', 'book'),
                ('book', 'rev_rates', 'user')
            ]
        )

        # IMPORTANT:
        # checkpoint expects "encoder.*"
        self.encoder = to_hetero(
            base_encoder,
            metadata=metadata,
            aggr='sum'
        )

        # ====================================================
        # PREDICTOR
        # ====================================================

        # IMPORTANT:
        # checkpoint expects "predictor.*"
        self.predictor = nn.Module()

        self.predictor.mlp = nn.Sequential(

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim // 2,
                1
            )
        )

    def forward(self):
        pass


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating model...")

def return_model():
    model = RecommenderGNN(
        num_users=NUM_USERS,
        num_books=NUM_BOOKS,
        hidden_dim=HIDDEN_DIM
    )

    return model