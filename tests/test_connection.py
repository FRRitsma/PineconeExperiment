import pinecone
import pytest
from pinecone import ApiException
from pinecone import UnauthorizedException

from settings import API_KEY
from settings import ENVIRONMENT
from settings import INDEX_NAME
from src.database_functions.database_functions import create_index_no_overwrite


def test_overwrite_error():
    pinecone.init(api_key=API_KEY, environment=ENVIRONMENT)
    with pytest.raises(ValueError):
        create_index_no_overwrite(INDEX_NAME, 8)


def test_uninitialized_api():
    with pytest.raises(ApiException):
        pinecone.create_index("nonsense", dimension=8, metric="euclidean")


def test_connect_with_incorrect_key():
    pinecone.init(api_key="NONSENSE", environment=ENVIRONMENT)
    with pytest.raises(UnauthorizedException):
        pinecone.create_index("NONSENSE", dimension=8, metric="euclidean")
