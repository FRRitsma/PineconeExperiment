from itertools import chain

import numpy as np
import pytest

from src.database_functions import convert_data_to_pinecone_format
from src.database_functions import create_list_of_upload_chunks
from src.database_functions import divide_list_into_chunks
from src.embedding import Embedder
from src.extract import extract_images_with_metadata
from src.extract import LabelPath

CHUNK_SIZE: int = 50
N_LABELS: int = 10
N_IMAGES: int = 6


@pytest.fixture()
def list_of_train_data():
    train_data = extract_images_with_metadata(N_LABELS, N_IMAGES, LabelPath.train)
    return train_data


def test_create_list_of_upload_chunks_happy_flow(list_of_train_data):
    upload_chunks_list = create_list_of_upload_chunks(CHUNK_SIZE, list_of_train_data)
    assert isinstance(upload_chunks_list, list)


def test_divide_list_into_chunks_first_chunk_size():
    input_list = [i for i in range(77)]
    chunked_list = divide_list_into_chunks(10, input_list)
    assert len(chunked_list[0]) == 10


def test_divide_list_into_chunks_total_size():
    input_list = [i for i in range(77)]
    chunked_list = divide_list_into_chunks(10, input_list)
    flat_list = list(chain.from_iterable(chunked_list))
    assert len(flat_list) == len(input_list)


def test_divide_list_into_chunks_content_check():
    input_list = [i for i in range(77)]
    chunked_list = divide_list_into_chunks(10, input_list)
    assert max(chunked_list[1]) == 19


def test_create_upload_data_happy_flow():
    train_data = extract_images_with_metadata(2, 1, LabelPath.train)
    upload_data = convert_data_to_pinecone_format(train_data, 0)
    assert isinstance(upload_data[0], dict)


@pytest.fixture()
def list_of_upload_chunks(mocker, list_of_train_data):
    RESNET_EMBEDDING_SIZE: int = Embedder().embedding_dimension
    mock_embedder = mocker.Mock(spec=Embedder)
    mock_embedder.embed.return_value = np.random.rand(RESNET_EMBEDDING_SIZE)
    mocker.patch("src.database_functions.Embedder", return_value=mock_embedder)
    upload_chunks_list = create_list_of_upload_chunks(CHUNK_SIZE, list_of_train_data)
    return upload_chunks_list


def test_chunk_function_total_data_count(list_of_upload_chunks, list_of_train_data):
    flat_list = list(chain.from_iterable(list_of_upload_chunks))
    assert len(flat_list) == len(list_of_train_data)


def test_chunk_function_every_id_is_unique(list_of_upload_chunks, list_of_train_data):
    flat_list = list(chain.from_iterable(list_of_upload_chunks))
    all_index_names = [i["id"] for i in flat_list]
    assert len(set(all_index_names)) == len(all_index_names) == len(list_of_train_data)
