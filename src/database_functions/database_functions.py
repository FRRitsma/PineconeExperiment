from math import ceil

import pinecone
import structlog

from settings import API_KEY
from settings import ENVIRONMENT
from src.embedding.embedding import Embedder
from src.extract.extract import ImageWithMetadata

logger = structlog.get_logger(__name__)

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


def create_list_of_upload_chunks(
    chunk_size: int, train_data: list[ImageWithMetadata]
) -> list[dict]:
    embedder = Embedder()
    chunk_list: list = [[] for _ in range(0, len(train_data), chunk_size)]
    for i, image_with_metadata in enumerate(train_data):
        chunk_list_index = int(i / chunk_size)
        upsert_data = {
            "id": f"vec{i}",
            "values": embedder.embed(image_with_metadata).tolist(),
            "metadata": image_with_metadata.summary(),
        }
        chunk_list[chunk_list_index].append(upsert_data)
        if i % chunk_size == 0:
            logger.info(f"Chunk {chunk_list_index}/{ceil(len(train_data)/chunk_size)}")
    logger.info("Chunking has completed")

    return chunk_list


def upsert_batches(chunk_list: list[list], index_name: str) -> list:
    with pinecone.Index(index_name, pool_threads=30) as index:
        async_results = [
            index.upsert(vectors=chunk, async_req=True) for chunk in chunk_list
        ]
        results = [async_result.get() for async_result in async_results]
    return results
