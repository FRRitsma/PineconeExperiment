from itertools import chain

import numpy as np
import pytest

from src.embedding.embedding import create_list_of_upload_chunks
from src.embedding.embedding import Embedder
from src.extract.extract import extract_images_with_metadata
from src.extract.extract import LabelPath


@pytest.fixture()
def list_of_train_data():
    N_LABELS: int = 10
    N_IMAGES: int = 1000
    train_data = extract_images_with_metadata(N_LABELS, N_IMAGES, LabelPath.train)
    return train_data


@pytest.fixture()
def list_of_upload_chunks(mocker, list_of_train_data):
    RESNET_EMBEDDING_SIZE: int = 2048

    mock_embedder = mocker.Mock(spec=Embedder)
    mock_embedder.embed.return_value = np.random.rand(RESNET_EMBEDDING_SIZE)
    mocker.patch("src.embedding.embedding.Embedder", return_value=mock_embedder)
    upload_chunks_list = create_list_of_upload_chunks(100, list_of_train_data)
    return upload_chunks_list


def test_chunk_function_total_data_count(list_of_upload_chunks, list_of_train_data):
    flat_list = list(chain.from_iterable(list_of_upload_chunks))
    assert len(flat_list) == len(list_of_train_data)


def test_chunk_function_every_id_is_unique(list_of_upload_chunks, list_of_train_data):
    flat_list = list(chain.from_iterable(list_of_upload_chunks))
    all_index_names = [i["id"] for i in flat_list]
    assert len(set(all_index_names)) == len(all_index_names) == len(list_of_train_data)
