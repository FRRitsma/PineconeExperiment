from settings import API_KEY


def test_api_key():
    assert isinstance(API_KEY, str)
