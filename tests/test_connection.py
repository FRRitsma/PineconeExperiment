import pinecone
import pytest
from pinecone import ApiException
from pinecone import UnauthorizedException

from settings import ENVIRONMENT


@pytest.mark.needs_internet
def test_uninitialized_api():
    with pytest.raises(ApiException):
        pinecone.create_index("nonsense", dimension=8, metric="euclidean")


@pytest.mark.needs_internet
def test_connect_with_incorrect_key():
    pinecone.init(api_key="NONSENSE", environment=ENVIRONMENT)
    with pytest.raises(UnauthorizedException):
        pinecone.create_index("NONSENSE", dimension=8, metric="euclidean")
