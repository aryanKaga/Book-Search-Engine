import time
import torch
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)
import numpy as np
import joblib
from collections import defaultdict

# ================================
# 1. CONNECT TO MILVUS
# ================================
for i in range(20):
    try:
        connections.connect(alias="default", host="127.0.0.1", port="19530")
        print(" Connected to Milvus")
        break
    except Exception as e:
        print(f" Waiting for Milvus... ({i+1}/20)")
        time.sleep(3)
else:
    raise Exception(" Could not connect to Milvus")

# ================================
# 2. LOAD EMBEDDINGS
# ================================
centroids = {}
# ================================
# 3. DROP OLD COLLECTION
# ================================


ids = np.load("cluster_ids.npy")
labels = np.load("cluster_labels.npy")
embeddings = np.load('embeddings_reduced.npy')
print(len(ids), len(labels), len(embeddings))

cluster_to_books = defaultdict(list)

for book_id,cluster in zip(ids,labels):
    if cluster==-1:
        continue
    cluster_to_books[cluster].append(book_id)


cluster_vectors = defaultdict(list)


for emb,cluster in zip(embeddings,labels):
    if cluster==-1:
        continue
    cluster_vectors[int(cluster)].append(emb)


centroid = {}

for cluster,vectors in cluster_vectors.items():
    centroid[cluster] = np.mean(vectors, axis=0)
    

print('centroids computed')
joblib.dump(centroid, 'centroid.pkl')
joblib.dump(cluster_to_books, 'cluster_to_books.pkl')


collection_name = "centroids"
if collection_name in utility.list_collections():
    utility.drop_collection(collection_name)
    print("🗑 Old centroid collection deleted")

embedding_size = embeddings.shape[1]

print('embedding size:', embedding_size)
fields = [
    FieldSchema(
        name="cluster_id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=False
    ),

    FieldSchema(
        name="centroid",
        dtype=DataType.FLOAT_VECTOR,
        dim=embedding_size
    )
]

schema = CollectionSchema(
    fields,
    description="Cluster centroids"
)
collection = Collection(name = collection_name,schema = schema)
print("✅ New centroid collection created")

cluster_ids = []
centroid_vectors = []


for cluster_id,centroid_vector in centroid.items():
    cluster_ids.append(cluster_id)
    centroid_vectors.append(centroid_vector)


data = [
    cluster_ids,
    centroid_vectors
]
print(len(data[0]), len(data[1]))
collection.insert(data)

collection.flush()

print("✅ Centroids inserted into Milvus")


index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {
        "M": 16,
        "efConstruction": 200
    }
}

collection.create_index(
    field_name="centroid",
    index_params=index_params
)