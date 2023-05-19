import pinecone

from settings import API_KEY
from settings import ENVIRONMENT

pinecone.init(api_key=API_KEY, environment=ENVIRONMENT)


def create_index_no_overwrite(index_name: str, vector_dimension: int) -> None:
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, vector_dimension, metric="euclidean")
    else:
        raise ValueError("Index has already been initialized")


def create_index_overwrite(index_name: str, vector_dimension: int) -> None:
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
    pinecone.create_index(index_name, vector_dimension, metric="euclidean")
