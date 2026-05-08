import math
import random
from typing import Dict, Optional, Set, List, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import HeteroData

try:
    from .data_prep import (
        build_positives,
        build_user_to_seen,
        prepare_data,
        sample_negative,
    )
    from .model import RecommenderGNN, bpr_loss
except ImportError:
    from model.data_prep import (
        build_positives,
        build_user_to_seen,
        prepare_data,
        sample_negative,
    )
    from model.model import RecommenderGNN, bpr_loss


def evaluate(
    model: RecommenderGNN,
    split_data: HeteroData,
    positives_dict: Dict[int, Set[int]],
    seen_dict: Dict[int, Set[int]],
    K: int = 10,
    max_users: int = 2000,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Compute Recall@K and NDCG@K over users with positive targets."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    split_data = split_data.to(device)
    user_emb, book_emb = model.encode(split_data)

    users_with_pos = [u for u in positives_dict if len(positives_dict[u]) > 0]
    if len(users_with_pos) > max_users:
        users_with_pos = random.sample(users_with_pos, max_users)

    recalls: List[float] = []
    ndcgs: List[float] = []
    for u in users_with_pos:
        pos_books = positives_dict[u]
        all_seen = seen_dict.get(u, set())

        u_emb = user_emb[u].unsqueeze(0)
        scores = (u_emb @ book_emb.T).squeeze(0)

        mask_ids = list(all_seen - pos_books)
        if mask_ids:
            scores[torch.tensor(mask_ids, dtype=torch.long, device=device)] = -1e9

        top_k = scores.topk(K).indices.tolist()
        hits = [1 if b in pos_books else 0 for b in top_k]

        recalls.append(sum(hits) / min(len(pos_books), K))
        dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(pos_books), K)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(recalls)), float(np.mean(ndcgs))


def sample_train_batch(
    train_data: HeteroData,
    user_to_seen_train: Dict[int, Set[int]],
    num_books: int,
    batch_size: int = 2048,
    num_neg: int = 1,
    top_popular: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ed = train_data["user", "rates", "book"]
    n_pos = ed.edge_index.size(1)
    idx = torch.randint(n_pos, (batch_size,), device=ed.edge_index.device)
    pos_u = ed.edge_index[0, idx]
    pos_b = ed.edge_index[1, idx]

    neg_b = torch.tensor(
        [sample_negative(u.item(), user_to_seen_train, num_books, top_popular) for u in pos_u],
        dtype=torch.long,
        device=ed.edge_index.device,
    )
    return pos_u, pos_b, neg_b


def run_training(
    ratings_path: str = "Books_rating.csv",
    books_path: str = "books_data.csv",
    min_score: int = 3,
    min_interactions: int = 5,
    pca_components: int = 128,
    embedder_name: str = "all-MiniLM-L6-v2",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    top_popular_k: int = 500,
    hidden_dim: int = 256,
    batch_size: int = 4096,
    num_neg: int = 1,
    l2_reg: float = 1e-5,
    lr: float = 3e-4,
    epochs: int = 50,
    eval_every: int = 5,
    steps_per_epoch: int = 300,
    use_mlp_head: bool = False,
) -> Tuple[RecommenderGNN, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_bundle = prepare_data(
        ratings_path=ratings_path,
        books_path=books_path,
        min_score=min_score,
        min_interactions=min_interactions,
        pca_components=pca_components,
        embedder_name=embedder_name,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        top_popular_k=top_popular_k,
    )

    train_data = data_bundle["train_data"].to(device)
    val_data = data_bundle["val_data"].to(device)
    test_data = data_bundle["test_data"].to(device)
    num_users = data_bundle["num_users"]
    num_books = data_bundle["num_books"]
    user_to_seen_train = data_bundle["user_to_seen_train"]
    val_positives = data_bundle["val_positives"]
    test_positives = data_bundle["test_positives"]
    top_popular = data_bundle["top_popular"]

    model = RecommenderGNN(
        num_users=num_users,
        num_books=num_books,
        hidden_dim=hidden_dim,
        use_mlp_head=use_mlp_head,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_recall = 0.0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    print("\n===== TRAINING =====")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            pos_u, pos_b, neg_b = sample_train_batch(
                train_data,
                user_to_seen_train,
                num_books,
                batch_size=batch_size,
                num_neg=num_neg,
                top_popular=top_popular,
            )

            optimizer.zero_grad()
            with autocast(enabled=device.type == "cuda"):
                pos_scores, neg_scores = model(train_data, pos_u, pos_b, neg_b)
                emb_params = list(model.user_emb.parameters())
                loss = bpr_loss(pos_scores, neg_scores, l2_reg=l2_reg, params=emb_params)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / steps_per_epoch

        if epoch == 1 or epoch % eval_every == 0:
            recall, ndcg = evaluate(
                model,
                val_data,
                val_positives,
                user_to_seen_train,
                K=10,
                device=device,
            )
            print(
                f"Epoch {epoch:3d} | Loss {avg_loss:.4f} | "
                f"Val Recall@10 {recall:.4f} | Val NDCG@10 {ndcg:.4f}"
            )
            if recall > best_recall:
                best_recall = recall
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            print(f"Epoch {epoch:3d} | Loss {avg_loss:.4f}")

    print("\n===== TEST EVALUATION =====")
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    user_to_seen_test = build_user_to_seen(data_bundle["val_data"], num_users)
    for u, books in user_to_seen_train.items():
        user_to_seen_test[u] = user_to_seen_test.get(u, set()) | books

    for K in [5, 10, 20]:
        recall, ndcg = evaluate(
            model,
            test_data,
            test_positives,
            user_to_seen_test,
            K=K,
            device=device,
        )
        print(f"  Test Recall@{K}: {recall:.4f}  |  NDCG@{K}: {ndcg:.4f}")

    return model, best_recall


if __name__ == "__main__":
    run_training()
