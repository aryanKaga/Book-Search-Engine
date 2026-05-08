from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit


def load_raw_data(
    ratings_path: str = "Books_rating.csv",
    books_path: str = "books_data.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings_path = Path(ratings_path)
    books_path = Path(books_path)

    df_ratings = pd.read_csv(ratings_path)
    df_books = pd.read_csv(books_path)
    df_books["description"] = df_books["description"].fillna("")
    return df_ratings, df_books


def filter_positive_ratings(
    df_ratings: pd.DataFrame,
    min_score: int = 3,
    min_interactions: int = 5,
) -> pd.DataFrame:
    df = df_ratings[df_ratings["review/score"] >= min_score].copy()
    user_counts = df["User_id"].value_counts()
    book_counts = df["Title"].value_counts()

    df = df[
        df["User_id"].isin(user_counts[user_counts >= min_interactions].index)
        & df["Title"].isin(book_counts[book_counts >= min_interactions].index)
    ]

    df = df.groupby(["User_id", "Id", "Title"], as_index=False)["review/score"].max()
    df = df[["User_id", "Id", "Title"]].dropna()
    return df


def build_id_maps(
    df_ratings: pd.DataFrame,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str], pd.DataFrame]:
    unique_users = df_ratings["User_id"].unique()
    unique_books = df_ratings["Title"].unique()

    user2id = {u: i for i, u in enumerate(unique_users)}
    book2id = {b: i for i, b in enumerate(unique_books)}
    id2user = {i: u for u, i in user2id.items()}
    id2book = {i: b for b, i in book2id.items()}

    df_ratings = df_ratings.copy()
    df_ratings["user_id"] = df_ratings["User_id"].map(user2id)
    df_ratings["book_id"] = df_ratings["Title"].map(book2id)

    return user2id, book2id, id2user, id2book, df_ratings


def build_book_features(
    df_books: pd.DataFrame,
    book2id: Dict[str, int],
    n_components: int = 128,
    embedder_name: str = "all-MiniLM-L6-v2",
) -> Tuple[torch.Tensor, PCA]:
    df_books = df_books.copy()
    df_books["text"] = df_books["Title"].fillna("") + " " + df_books["description"]
    df_books = df_books[df_books["Title"].isin(book2id)].copy()
    df_books["book_id"] = df_books["Title"].map(book2id)

    df_books = df_books.drop_duplicates("book_id").sort_values("book_id").reset_index(drop=True)
    num_books = len(book2id)

    missing_ids = set(range(num_books)) - set(df_books["book_id"].tolist())
    if missing_ids:
        missing_df = pd.DataFrame({"book_id": list(missing_ids), "text": [""] * len(missing_ids)})
        df_books = pd.concat([df_books, missing_df], ignore_index=True).sort_values("book_id").reset_index(drop=True)

    embedder = SentenceTransformer(embedder_name)
    book_embeddings = embedder.encode(
        df_books["text"].tolist(),
        convert_to_tensor=False,
        show_progress_bar=True,
        batch_size=256,
    )

    pca = PCA(n_components=n_components)
    book_x_np = pca.fit_transform(book_embeddings)
    book_x = torch.tensor(book_x_np, dtype=torch.float)

    assert book_x.shape[0] == num_books, (
        f"Book feature rows {book_x.shape[0]} != num_books {num_books}"
    )
    return book_x, pca


def build_hetero_data(
    df_ratings: pd.DataFrame,
    book_x: torch.Tensor,
    num_users: int,
) -> HeteroData:
    data = HeteroData()
    data["book"].x = book_x
    data["user"].x = torch.zeros(num_users, 128)

    user_indices = torch.tensor(df_ratings["user_id"].values, dtype=torch.long)
    book_indices = torch.tensor(df_ratings["book_id"].values, dtype=torch.long)
    edge_index = torch.stack([user_indices, book_indices], dim=0)

    data["user", "rates", "book"].edge_index = edge_index
    data["book", "rev_rates", "user"].edge_index = edge_index.flip(0)
    return data


def create_train_val_test_split(
    data: HeteroData,
    num_val: float = 0.1,
    num_test: float = 0.1,
) -> Tuple[HeteroData, HeteroData, HeteroData]:
    transform = RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
        edge_types=[("user", "rates", "book")],
        rev_edge_types=[("book", "rev_rates", "user")],
    )
    return transform(data)


def build_user_to_seen(split_data: HeteroData, num_users: int) -> Dict[int, Set[int]]:
    seen = {u: set() for u in range(num_users)}
    ed = split_data["user", "rates", "book"]
    for u, v in zip(ed.edge_index[0].tolist(), ed.edge_index[1].tolist()):
        seen[u].add(v)
    if hasattr(ed, "edge_label_index") and ed.edge_label_index is not None:
        pos_mask = ed.edge_label == 1
        pos_ei = ed.edge_label_index[:, pos_mask]
        for u, v in zip(pos_ei[0].tolist(), pos_ei[1].tolist()):
            seen[u].add(v)
    return seen


def build_positives(split_data: HeteroData) -> Dict[int, Set[int]]:
    pos = {}
    ed = split_data["user", "rates", "book"]
    if hasattr(ed, "edge_label_index") and ed.edge_label_index is not None:
        pos_mask = ed.edge_label == 1
        pos_ei = ed.edge_label_index[:, pos_mask]
        for u, v in zip(pos_ei[0].tolist(), pos_ei[1].tolist()):
            pos.setdefault(u, set()).add(v)
    return pos


def build_popularity_index(
    train_data: HeteroData,
    num_books: int,
    top_k: int = 500,
) -> List[int]:
    book_interaction_counts = torch.zeros(num_books, dtype=torch.long)
    for v in train_data["user", "rates", "book"].edge_index[1].tolist():
        book_interaction_counts[v] += 1
    return book_interaction_counts.topk(min(top_k, num_books)).indices.tolist()


def sample_negative(
    user_id: int,
    user_to_seen_train: Dict[int, Set[int]],
    num_books: int,
    top_popular: Optional[List[int]] = None,
    hard_ratio: float = 0.5,
) -> int:
    seen = user_to_seen_train.get(user_id, set())
    use_hard = top_popular is not None and random.random() < hard_ratio
    pool = top_popular if use_hard else None
    for _ in range(100):
        neg = int(random.choice(pool) if pool else random.randrange(num_books))
        if neg not in seen:
            return neg
    while True:
        neg = random.randrange(num_books)
        if neg not in seen:
            return neg


def prepare_data(
    ratings_path: str = "Books_rating.csv",
    books_path: str = "books_data.csv",
    min_score: int = 3,
    min_interactions: int = 5,
    pca_components: int = 128,
    embedder_name: str = "all-MiniLM-L6-v2",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    top_popular_k: int = 500,
) -> Dict[str, object]:
    df_ratings, df_books = load_raw_data(ratings_path, books_path)
    df_ratings = filter_positive_ratings(df_ratings, min_score, min_interactions)
    user2id, book2id, id2user, id2book, df_ratings = build_id_maps(df_ratings)
    num_users = len(user2id)
    num_books = len(book2id)

    book_x, pca = build_book_features(df_books, book2id, pca_components, embedder_name)
    data = build_hetero_data(df_ratings, book_x, num_users)
    train_data, val_data, test_data = create_train_val_test_split(data, val_ratio, test_ratio)

    user_to_seen_train = build_user_to_seen(train_data, num_users)
    val_positives = build_positives(val_data)
    test_positives = build_positives(test_data)
    top_popular = build_popularity_index(train_data, num_books, top_popular_k)

    return {
        "data": data,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "user2id": user2id,
        "book2id": book2id,
        "id2user": id2user,
        "id2book": id2book,
        "num_users": num_users,
        "num_books": num_books,
        "user_to_seen_train": user_to_seen_train,
        "val_positives": val_positives,
        "test_positives": test_positives,
        "top_popular": top_popular,
        "book_x": book_x,
        "pca": pca,
    }

