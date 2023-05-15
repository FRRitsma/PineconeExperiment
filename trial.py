# %%
# Part 1: Initialize database
# Part 2: Fill database
# Attach image link in metadata
import pinecone

from settings import API_KEY
from settings import ENVIRONMENT
from settings import INDEX_NAME
from src.embedding.embedding import Embedder
from src.extract.extract import extract_images_with_metadata
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


# %%

# TEST:
# Can I upsert torch tensors directly?
index = pinecone.Index(INDEX_NAME)
# index.upsert([
#     ("A", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
#     ("B", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
#     ("C", [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
#     ("D", [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
#     ("E", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
# ])

vector = embedder.embed(train_data[0].image)
vector_np = vector.detach().numpy()
# %%
upsert_response = index.upsert(
    vectors=[
        {
            "id": "vec1",
            "values": vector_np,
        }
    ]
)

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
