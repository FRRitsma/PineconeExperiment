from pathlib import Path

import pytest

from settings import DownloadSettings
from src.download_dataset import download_imagenette2_dataset
from src.download_dataset import download_via_url_write_to_file_with_name
from src.download_dataset import FileSize
from src.download_dataset import open_tar_file_on_source_write_to_target


class MockRequestsGet:
    def __init__(self):
        with open(Path(__file__).parent / "imagenette2-160.tgz", "rb") as fs:
            content_not_bytes = fs.read()
        self.content = bytes(content_not_bytes)


@pytest.fixture(scope="session")
def temporary_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("nonexistent")
    return path


@pytest.fixture(scope="session")
def filesize():
    return DownloadSettings.small_size


def test_requests_mock(mocker, tmp_path):
    mocker.patch("requests.get", return_value=MockRequestsGet())
    url = "https://www.facebook.com/favicon.ico"
    save_path = tmp_path
    save_name = "favicon.ico"
    download_via_url_write_to_file_with_name(url, save_path, save_name)


def test_filesize_class():
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    filesize = FileSize(url)
    assert filesize.url == url
    assert filesize.name == "imagenette2.tgz"
    assert filesize.name_no_extension == "imagenette2"


def test_download_settings_class():
    assert isinstance(DownloadSettings.small_size, FileSize)
    assert isinstance(DownloadSettings.medium_size, FileSize)
    assert isinstance(DownloadSettings.large_size, FileSize)


@pytest.mark.needs_internet
def test_download_via_url_write_to_file_with_name(tmp_path):
    url = "https://www.facebook.com/favicon.ico"
    save_path = tmp_path
    save_name = "favicon.ico"
    download_via_url_write_to_file_with_name(url, save_path, save_name)
    assert Path.is_file(save_path / save_name)


def test_download_image_dataset(mocker, temporary_path, filesize):
    mocker.patch("requests.get", return_value=MockRequestsGet())
    save_path = temporary_path
    save_name = filesize.name
    download_via_url_write_to_file_with_name(filesize.url, save_path, save_name)
    assert Path.is_file(save_path / save_name)


def test_open_tar_file(temporary_path, filesize):
    source_path = temporary_path / filesize.name
    unpacked_name = filesize.name_no_extension
    target_location = temporary_path / "data"
    open_tar_file_on_source_write_to_target(source_path, target_location)
    assert Path.is_dir(target_location / unpacked_name)


def test_download_imagenette2_dataset(mocker, tmp_path, filesize):
    mocker.patch("requests.get", return_value=MockRequestsGet())
    download_imagenette2_dataset(filesize, tmp_path)
    assert Path(tmp_path / filesize.name_no_extension).is_dir()
    assert not Path(tmp_path / filesize.name).is_dir()
