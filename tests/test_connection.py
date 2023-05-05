import pinecone

from settings import API_KEY
from settings import ENVIRONMENT


def test_connection():
    pinecone.init(api_key=API_KEY, environment=ENVIRONMENT)
