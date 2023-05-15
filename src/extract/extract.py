# %%
"""
Loading train images bundled with their respective labels
"""
import os
from itertools import product
from pathlib import Path

from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile

here: Path = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent


class LabelPath:
    train: Path = Path.joinpath(here, "data", "imagenette2", "train")
    val: Path = Path.joinpath(here, "data", "imagenette2", "val")


class ImageWithMetadata:
    image: JpegImageFile
    image_path: Path
    label: str

    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.label = image_path.parent.name


def select_label_directories(labels_path: Path, n_labels: int) -> list[Path]:
    all_label_directories: list[str] = os.listdir(labels_path)
    selected_label_directories: list[str] = all_label_directories[:n_labels]
    selected_label_directories_full_path = [
        Path.joinpath(labels_path, i) for i in selected_label_directories
    ]
    return selected_label_directories_full_path


def select_images(label_directories: list, n_images: int) -> list[Path]:
    images_full_path: list = []
    for label_directory, i in product(label_directories, range(n_images)):
        all_images_in_label_directory = os.listdir(label_directory)
        images_full_path.append(
            Path.joinpath(label_directory, all_images_in_label_directory[i])
        )
    return images_full_path


def extract_images_with_metadata(
    n_labels: int, n_images: int, train_or_val_path: Path
) -> list[ImageWithMetadata]:
    """
    Extracting images and metadata from the "imagenette2" dataset, as a list of "ImageWithMetaData" classes
    """

    label_directories = select_label_directories(train_or_val_path, n_labels)
    image_paths = select_images(label_directories, n_images)
    images_with_metadata: list[ImageWithMetadata] = [
        ImageWithMetadata(path) for path in image_paths
    ]

    return images_with_metadata
