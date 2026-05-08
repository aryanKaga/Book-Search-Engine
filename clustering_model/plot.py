import numpy as np
import pandas as pd
import plotly.express as px
from umap import UMAP

from hdbscan_cluster import get_clusters   # your function


# =========================
# 1. GET CLUSTER OUTPUT
# =========================
data = get_clusters()

ids = data["ids"]
labels = data["labels"]
embeddings = data["embeddings"]


# =========================
# 2. UMAP REDUCTION
# =========================
umap = UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=80,
    min_dist=0.1
)

reduced = umap.fit_transform(embeddings)


# =========================
# 3. BUILD DATAFRAME
# =========================
df = pd.DataFrame(reduced, columns=["x", "y"])

df["cluster"] = labels.astype(str)
df["id"] = ids


# remove noise
df = df[df["cluster"] != "-1"]


# =========================
# 4. PLOT
# =========================
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="cluster",
    hover_data={"id": True}
)

fig.show()