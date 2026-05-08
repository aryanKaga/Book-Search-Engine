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
embeddings = torch.load("book_embeddings.pt", map_location="cpu")

# FIX: don’t overwrite torch.device
if hasattr(embeddings, "numpy"):
    embeddings = embeddings.numpy()

embedding_size = embeddings.shape[1]
print(f" Embedding size: {embedding_size}")

# ================================
# 3. DROP OLD COLLECTION
# ================================
collection_name = "book_embeddings"

if collection_name in utility.list_collections():
    utility.drop_collection(collection_name)
    print(" Old collection deleted")

# ================================
# 4. CREATE COLLECTION
# ================================
fields = [
    FieldSchema(name="book_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_size)
]

schema = CollectionSchema(fields, description="Book embeddings")

collection = Collection(name=collection_name, schema=schema)
print(" New collection created")

# ================================
# 5. INSERT DATA (BATCHED)
# ================================
batch_size = 1000  # safe for 384-dim vectors

for i in range(0, len(embeddings), batch_size):
    batch = embeddings[i:i + batch_size]

    data = [
        list(range(i, i + len(batch))),     # book_ids
        batch.tolist()                     # embeddings
    ]

    collection.insert(data)
    print(f" Inserted {i} → {i + len(batch)}")

collection.flush()
print("🎉 All data inserted successfully")

# ================================
# 6. CREATE INDEX
# ================================
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {
        "M": 16,
        "efConstruction": 200
    }
}

collection.create_index(field_name="embedding", index_params=index_params)
print(" Index created")

# ================================
# 7. LOAD COLLECTION FOR SEARCH
# ================================
collection.load()
print(" Collection loaded and ready for search")

# ================================
# 8. OPTIONAL TEST SEARCH
# ================================
query_vector = embeddings[0].tolist()

results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 50}},
    limit=5
)

print("\n🔎 Sample search results:")
for hit in results[0]:
    print(f"ID: {hit.id}, Score: {hit.distance}")