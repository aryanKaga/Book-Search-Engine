import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero


class GNNEncoder(nn.Module):
    """
    3-layer GraphSAGE with residual connections, layer normalization, and dropout.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_dim)
        self.conv2 = SAGEConv((-1, -1), hidden_dim)
        self.conv3 = SAGEConv((-1, -1), hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        x1 = self.bn1(self.conv1(x, edge_index).relu())
        x1 = self.drop(x1)
        x2 = self.bn2(self.conv2(x1, edge_index).relu())
        x2 = self.drop(x2)
        x3 = self.conv3(x2, edge_index)
        return x3 + x2


class RecommenderGNN(nn.Module):
    """
    Heterogeneous user-book recommender with GraphSAGE backbone.
    """

    def __init__(
        self,
        num_users: int,
        num_books: int,
        hidden_dim: int = 256,
        use_mlp_head: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.user_emb = nn.Embedding(num_users, 128)
        nn.init.xavier_uniform_(self.user_emb.weight)

        metadata = (
            ["user", "book"],
            [("user", "rates", "book"), ("book", "rev_rates", "user")],
        )
        encoder = GNNEncoder(hidden_dim, dropout)
        self.gnn = to_hetero(encoder, metadata=metadata, aggr="sum")

        self.use_mlp_head = use_mlp_head
        if self.use_mlp_head:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        self.log_temp = nn.Parameter(torch.zeros(1))

    def encode(self, hetero_data):
        x_dict = {
            "user": self.user_emb(
                torch.arange(hetero_data["user"].x.size(0), device=hetero_data["user"].x.device)
            ),
            "book": hetero_data["book"].x,
        }
        edge_index_dict = {
            ("user", "rates", "book"): hetero_data["user", "rates", "book"].edge_index,
            ("book", "rev_rates", "user"): hetero_data["book", "rev_rates", "user"].edge_index,
        }
        out = self.gnn(x_dict, edge_index_dict)
        user_emb = F.normalize(out["user"], dim=-1)
        book_emb = F.normalize(out["book"], dim=-1)
        return user_emb, book_emb

    def predict(self, user_emb, book_emb, user_idx, book_idx):
        u = user_emb[user_idx]
        b = book_emb[book_idx]
        if self.use_mlp_head:
            return self.mlp(torch.cat([u, b], dim=-1)).squeeze(-1)
        temp = self.log_temp.exp().clamp(0.01, 10.0)
        return (u * b).sum(dim=-1) / temp

    def forward(self, hetero_data, pos_users, pos_books, neg_books):
        user_emb, book_emb = self.encode(hetero_data)
        pos_scores = self.predict(user_emb, book_emb, pos_users, pos_books)
        neg_scores = self.predict(user_emb, book_emb, pos_users, neg_books)
        return pos_scores, neg_scores


def bpr_loss(pos_scores, neg_scores, l2_reg: float = 1e-4, params=None):
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    if l2_reg > 0 and params is not None:
        reg = sum(p.pow(2).sum() for p in params)
        loss = loss + l2_reg * reg
    return loss


def ssm_loss(user_emb, pos_book_emb, neg_book_embs, temperature: float = 0.07):
    pos = (user_emb * pos_book_emb).sum(-1, keepdim=True)
    neg = torch.bmm(neg_book_embs, user_emb.unsqueeze(-1)).squeeze(-1)
    logits = torch.cat([pos, neg], dim=1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
