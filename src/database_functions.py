from math import ceil
from pathlib import Path
from typing import Any
from typing import Tuple

import numpy as np
import pinecone
import structlog

from settings import API_KEY
from settings import ENVIRONMENT
from settings import INDEX_NAME
from src.embedding import Embedder
from src.extract import ImageWithMetadata

logger = structlog.get_logger(__name__)

pinecone.init(api_key=API_KEY, environment=ENVIRONMENT)


def index_exists() -> bool:
    return INDEX_NAME in pinecone.list_indexes()


def create_index_no_overwrite(vector_dimension: int) -> None:
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(INDEX_NAME, vector_dimension, metric="euclidean")
    else:
        raise ValueError("Index has already been initialized")


def create_index_overwrite(index_name: str, vector_dimension: int) -> None:
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
    pinecone.create_index(index_name, vector_dimension, metric="euclidean")


def divide_list_into_chunks(chunk_size: int, input_list: list) -> list[list]:
    chunk_list: list = [
        (
            input_list[i: i + chunk_size],
            logger.info(
                f"Encoding Chunk {i // chunk_size + 1}/{ceil(len(input_list)/chunk_size)}"
            ),
        )[0]
        for i in range(0, len(input_list), chunk_size)
    ]
    return chunk_list


def convert_data_to_pinecone_format(
    input_list: list[ImageWithMetadata], id_offset: int
) -> list[dict]:
    embedder = Embedder()
    upload_data: list = [
        {
            "id": f"vec{i + id_offset}",
            "values": embedder.embed(image_with_metadata).tolist(),
            "metadata": image_with_metadata.summary(),
        }
        for i, image_with_metadata in enumerate(input_list)
    ]
    return upload_data


def create_list_of_upload_chunks(
    chunk_size: int, raw_data: list[Any]
) -> list[list[Any]]:
    chunked_raw_data = divide_list_into_chunks(chunk_size, raw_data)
    chunked_upload_data = [
        convert_data_to_pinecone_format(input_list, i * chunk_size)
        for i, input_list in enumerate(chunked_raw_data)
    ]
    return chunked_upload_data


def upsert_batches(chunk_list: list[list], index_name: str) -> list:
    with pinecone.Index(index_name, pool_threads=30) as index:
        async_results = [
            index.upsert(vectors=chunk, async_req=True) for chunk in chunk_list
        ]
        results = [async_result.get() for async_result in async_results]
    return results


def return_random_vector_and_its_nearest_neighbors(top_k) -> Tuple[dict, list[dict]]:
    with pinecone.Index(INDEX_NAME, pool_threads=30) as index:
        vector_count: int = index.describe_index_stats()["total_vector_count"]
        random_id: str = f"vec{np.random.randint(vector_count)}"
        random_vector: dict = index.fetch(ids=[random_id]).to_dict()["vectors"][
            random_id
        ]
        nearest_vectors = index.query(
            top_k=top_k,
            include_values=True,
            include_metadata=True,
            vector=random_vector["values"],
        )["matches"]
    return random_vector, nearest_vectors


def extract_image_paths_from_database_query(random_vector: dict, nearest_vectors: list[dict]):
    list_image_paths = [Path(random_vector["metadata"]["image_path"])]
    for fetch_dictionary in nearest_vectors[1:]:
        list_image_paths.append(fetch_dictionary["metadata"]["image_path"])
    return list_image_paths
