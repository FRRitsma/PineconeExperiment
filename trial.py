# %%
import time

import numpy as np
import pinecone
import torch
from pinecone.index import Index

from settings import API_KEY
from settings import ENVIRONMENT
from settings import INDEX_NAME
from src.database_functions.database_functions import create_list_of_upload_chunks
from src.embedding.embedding import Embedder
from src.extract.extract import extract_images_with_metadata
from src.extract.extract import LabelPath
from src.visualization.visualization import visualize_similarities

embedder = Embedder()
device = torch.device("cuda")


# %%

N_LABELS: int = 10
N_IMAGES: int = 50

train_data = extract_images_with_metadata(N_LABELS, N_IMAGES, LabelPath.train)
chunk_list = create_list_of_upload_chunks(35, train_data)

# %%


embedder = Embedder()
# device = torch.device("cuda")
# embedder.resnet.to(device)

t0 = time.time()
for i in range(10):
    embedder.embed(train_data[0])
t1 = time.time()
print(t1 - t0)
# %%

vector_dimension = len(embedder.embed(train_data[0]))

pinecone.init(api_key=API_KEY, environment=ENVIRONMENT)
index = pinecone.Index(INDEX_NAME)


def does_id_exist(id: str, index: Index) -> bool:
    id_search_result = index.fetch(ids=[id])
    return len(id_search_result["vectors"]) != 0


def upsert_batches(chunk_list) -> list:
    with pinecone.Index(INDEX_NAME, pool_threads=30) as index:
        async_results = [
            index.upsert(vectors=chunk, async_req=True) for chunk in chunk_list
        ]
        results = [async_result.get() for async_result in async_results]
    return results


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

visualize_similarities(4, image_paths)
