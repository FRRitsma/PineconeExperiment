import tarfile
from pathlib import Path

import requests  # type: ignore


class FileSize:
    def __init__(self, url: str):
        self.url: str = url
        self.name: str = url.split("/")[-1]
        self.name_no_extension = str(Path(self.name).with_suffix(""))


def download_via_url_write_to_file_with_name(
    url: str, save_path: Path, save_name: Path | str
) -> None:
    response = requests.get(url, allow_redirects=True)
    with open(save_path / save_name, "wb") as fs:
        fs.write(response.content)


def open_tar_file_on_source_write_to_target(
    source_path: Path, target_path: Path
) -> None:
    with tarfile.open(source_path) as file:
        file.extractall(target_path)


def download_imagenette2_dataset(filesize: FileSize, target_location: Path) -> None:
    download_via_url_write_to_file_with_name(
        filesize.url, target_location, filesize.name
    )
    source_path = target_location / filesize.name
    open_tar_file_on_source_write_to_target(source_path, target_location)
    source_path.unlink()
