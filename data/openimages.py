# %%
"""
Loading train images bundled with their respective labels
"""
import os
from itertools import product
from pathlib import Path

from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile

here: Path = Path(os.path.dirname(os.path.abspath(__file__)))


class SetPath(enumerate):
    train: Path = Path.joinpath(here, "imagenette2", "train")
    val: Path = Path.joinpath(here, "imagenette2", "val")


def extract_images(n_labels: int, n_images: int, setpath: Path) -> list[dict]:
    """
    Function for extracting images and metadata from the "imagenette2" dataset.

    Args:
        n_labels (int): Amount of labels to be used
        n_images (int): Amount of images to be selected for each label
        setpath (Path): Either "train" or "val"

    Returns:
        list[dict]: A list of images with associated metadata bundled in a dictionary
    """

    all_label_directories: list[str] = os.listdir(setpath)
    selected_label_directories: list[str] = all_label_directories[
        : min(len(all_label_directories), n_labels)
    ]

    output_images_with_metadata: list[dict] = []
    for label, index in product(selected_label_directories, range(n_images)):
        full_path_directory: Path = Path.joinpath(setpath, label)
        image_path: str = os.listdir(full_path_directory)[index]
        image: JpegImageFile = Image.open(
            Path.joinpath(full_path_directory, image_path)
        )

        image_with_meta_data = {
            "image_name": image_path,
            "label": label,
            "image": image,
        }

        output_images_with_metadata.append(image_with_meta_data)

    return output_images_with_metadata
