import numpy as np
from load_model import return_model
import joblib
from pymilvus import (
    connections,
    Collection
)
import torch
from sentence_transformers import SentenceTransformer


# ============================================
# CONNECT TO MILVUS
# ============================================

connections.connect(
    alias="default",
    host="127.0.0.1",
    port="19530"
)

print("✅ Connected to Milvus")


# ============================================
# LOAD MODEL
# ============================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("✅ Embedding model loaded")


# ============================================
# LOAD COLLECTION
# ============================================

collection = Collection("centroids")

collection.load()

print("✅ Cluster centroid collection loaded")


# ============================================
# FIND CLUSTERS
# ============================================

#load model

gnn_model = return_model()
cluster_to_books = joblib.load("./clustering_model/cluster_to_books.pkl")



def find_relevant_cluster(query, top_k=5):

    # ----------------------------------------
    # CREATE QUERY EMBEDDING
    # ----------------------------------------

    query_embedding = model.encode([query])[0]

    # normalize for cosine similarity
    query_embedding = (
        query_embedding /
        np.linalg.norm(query_embedding)
    )

    # ----------------------------------------
    # SEARCH
    # ----------------------------------------

    results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="centroid",
        param={
            "metric_type": "COSINE",
            "params": {
                "nprobe": 10
            }
        },
        limit=top_k,
        output_fields=["cluster_id"]
    )

    # ----------------------------------------
    # FORMAT RESULTS
    # ----------------------------------------

    clusters = []

    for hit in results[0]:

        clusters.append({
            "cluster_id": hit.entity.get("cluster_id"),
            "score": hit.score
        })

    return clusters


# ============================================
# MAIN LOOP
# ============================================

print("\n✅ Ready")

def graphical_reranking():
    pass




# ============================================
# USER INTERACTION

user_id = int(input("\nEnter User ID: "))
user_embedding = torch.load('./user_emb', map_location='cpu')[user_id]
print(user_embedding.shape)

while True:

    query = input("\nEnter Query: ")

    if query.lower() == "exit":
        break

    clusters = find_relevant_cluster(query)
    
    print("\nRelevant Clusters:\n")
    cluster_ids = []
    book_ids = []
    bookid_to_score = {}
    for c in clusters:
        cluster_ids.append(c["cluster_id"])

        for book_id in cluster_to_books[c["cluster_id"]]:
            book_ids.append(book_id)


        book_embeddings = [torch.load('./book_emb', map_location='cpu')[book_id] for book_id in book_ids]
        for book_id, book_emb in zip(book_ids, book_embeddings):
            score = torch.cosine_similarity(
                user_embedding,
                book_emb,
                dim=0
            ).item()

            bookid_to_score[book_id] = score


    sorted_books = sorted(bookid_to_score.items(), key=lambda x: x[1], reverse=True)
    id2book = joblib.load('./id2book.pkl')
    books = [id2book[book_id] for book_id, _ in sorted_books[:10]]
    with open('results.txt', 'w') as f:
        for book in books:
            f.write(f"{book}\n")
            

            



        