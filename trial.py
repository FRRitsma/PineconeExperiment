# %%
# Part 1: Initialize database
# Part 2: Fill database
# Attach image link in metadata
import numpy as np
import pinecone

from settings import API_KEY
from settings import ENVIRONMENT
from settings import INDEX_NAME
from src.embedding.embedding import Embedder
from src.extract.extract import extract_images_with_metadata
from src.extract.extract import ImageWithMetadata
from src.extract.extract import LabelPath


N_LABELS: int = 5
N_IMAGES: int = 10

train_data = extract_images_with_metadata(N_LABELS, N_IMAGES, LabelPath.train)
embedder = Embedder()

vector_dimension = len(embedder.embed(train_data[0].image))
pinecone.init(api_key=API_KEY, environment=ENVIRONMENT)


def create_index_no_overwrite(index_name: str, vector_dimension: int) -> None:
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(index_name, vector_dimension, metric="euclidean")


def create_index_overwrite(index_name: str, vector_dimension: int) -> None:
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
    pinecone.create_index(index_name, vector_dimension, metric="euclidean")


index = pinecone.Index(INDEX_NAME)
vector = embedder.embed(train_data[0].image)


def process_images_with_metadata_list(
    images_with_metadata_list: list[ImageWithMetadata],
) -> list[tuple]:
    embedder = Embedder()
    return [
        (
            vector_from_image_with_metadata(iwm, embedder),
            dictionary_from_image_with_metadata(iwm),
        )
        for iwm in images_with_metadata_list
    ]


def vector_from_image_with_metadata(
    image_with_metadata: ImageWithMetadata, embedder: Embedder
) -> np.ndarray:
    return embedder.embed(image_with_metadata.image)


def dictionary_from_image_with_metadata(
    image_with_metadata: ImageWithMetadata,
) -> dict[str, str]:
    return {
        "image_path": str(image_with_metadata.image_path),
        "label": image_with_metadata.label,
    }


testeroni = process_images_with_metadata_list(train_data)
# %%
# upsert_response = index.upsert(
#     vectors=[
#         {
#             "id": "vec1",
#             "values": vector,
#         }
#     ]
# )

# upsert_response = index.upsert(
#     vectors=[
#         {
#         'id':'vec1',
#         'values':[0.1, 0.2, 0.3, 0.4],
#         'metadata':{'genre': 'drama'},
#            'sparse_values':
#            {'indices': [10, 45, 16],
#            'values':  [0.5, 0.5, 0.2]}},
#         {'id':'vec2',
#         'values':[0.2, 0.3, 0.4, 0.5],
#         'metadata':{'genre': 'action'},
#            'sparse_values':
#            {'indices': [15, 40, 11],
#            'values':  [0.4, 0.5, 0.2]}}
#     ],
#     namespace='example-namespace'
# )
