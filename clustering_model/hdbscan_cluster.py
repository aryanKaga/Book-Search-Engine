import numpy as np
import hdbscan
from pymilvus import Collection, connections
import joblib

def get_clusters(collection_name="book_embeddings"):
    connections.connect(alias="default", host="127.0.0.1", port="19530")

    collection = Collection(collection_name)
    collection.load()

    results = collection.query(
        expr="book_id > 0",
        output_fields=["book_id", "embedding"]
    )

    ids = []
    embeddings = []

    for r in results:
        ids.append(r["book_id"])
        embeddings.append(r["embedding"])

    embeddings = np.array(embeddings)

    # =========================
    # STEP 1: UMAP (FIT + SAVE)
    # =========================
    from umap import UMAP

    umap_model = UMAP(
        n_components=10,
        n_neighbors=30,
        min_dist=0.0,
        metric="cosine"
    )

    

    joblib.dump(umap_model, "umap.pkl")   # 🔥 SAVE UMAP MODEL

    # =========================
    # STEP 2: HDBSCAN
    # =========================
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric="euclidean"
    )

    labels = clusterer.fit_predict(embeddings)

    # =========================
    # SAVE OUTPUTS
    # =========================
    np.save("cluster_labels.npy", labels)
    np.save("cluster_ids.npy", ids)
    np.save("embeddings_reduced.npy", embeddings)

    return {
        "ids": ids,
        "labels": labels,
        "embeddings": embeddings
    }


get_clusters("book_embeddings")