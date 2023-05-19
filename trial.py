# %%
# Part 1: Initialize database
# Part 2: Fill database
# Attach image link in metadata
# Current status: Succesfull
# Next to do: Visualization function
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pinecone
from pinecone.index import Index

from settings import API_KEY
from settings import ENVIRONMENT
from settings import INDEX_NAME
from src.embedding.embedding import Embedder
from src.extract.extract import extract_images_with_metadata
from src.extract.extract import LabelPath

N_LABELS: int = 10
N_IMAGES: int = 1000

train_data = extract_images_with_metadata(N_LABELS, N_IMAGES, LabelPath.train)

embedder = Embedder()

vector_dimension = len(embedder.embed(train_data[0]))
pinecone.init(api_key=API_KEY, environment=ENVIRONMENT)
index = pinecone.Index(INDEX_NAME)


def does_id_exist(id: str, index: Index) -> bool:
    id_search_result = index.fetch(ids=[id])
    return len(id_search_result["vectors"]) != 0


# %%
for i, image_with_metadata in enumerate(train_data):
    vector_id = f"vec{i}"
    if not does_id_exist(vector_id, index):
        upsert_response = index.upsert(
            vectors=[
                {
                    "id": vector_id,
                    "values": embedder.embed(image_with_metadata).tolist(),
                    "metadata": image_with_metadata.summary(),
                }
            ],
        )

# %%
# Creating visualization functions
# Envisioned goal: a collage of similar images
# Expected input, a list of image paths

# Get random id in the index:
available_ids = [
    f"vec{i}" for i in range(0, 5000, 50) if does_id_exist(f"vec{i}", index)
]

# %%

N_SQUARE_SIZE: int = 4

random_id = np.random.choice(available_ids)

fetch_random_vector = index.fetch(ids=[random_id])
random_vector = fetch_random_vector["vectors"][random_id]["values"]

fetch_similar_vectors = index.query(
    vector=random_vector, top_k=N_SQUARE_SIZE**2, include_metadata=True
)
similar_vectors = fetch_similar_vectors["matches"]
# vector_list = fetch_response['vectors']
image_paths = [path["metadata"]["image_path"] for path in similar_vectors]

# %%

fig, axs = plt.subplots(nrows=N_SQUARE_SIZE, ncols=N_SQUARE_SIZE)
for i, (col, row) in enumerate(product(range(N_SQUARE_SIZE), range(N_SQUARE_SIZE))):
    axs[col, row].imshow(plt.imread(image_paths[i]))

for ax in axs.flat:
    ax.set_axis_off()
