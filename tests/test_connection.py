import pinecone
import pytest
from pinecone import UnauthorizedException

from settings import API_KEY
from settings import ENVIRONMENT


def test_incorrect_api_key():
    pinecone.init(api_key="NONSENSE", environment=ENVIRONMENT)
    with pytest.raises(UnauthorizedException):
        pinecone.create_index("NONSENSE", dimension=8, metric="euclidean")


def test_connection():
    pinecone.init(api_key=API_KEY, environment=ENVIRONMENT)
