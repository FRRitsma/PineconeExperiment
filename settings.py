import os

from src.download_dataset import FileSize

API_KEY = os.getenv("API_KEY")
ENVIRONMENT: str = "us-west4-gcp"
INDEX_NAME: str = "first-index"


class DownloadSettings(enumerate):
    large_size: FileSize = FileSize(
        "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    )
    medium_size: FileSize = FileSize(
        "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    )
    small_size: FileSize = FileSize(
        "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    )
