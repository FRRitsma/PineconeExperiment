# %%
"""
Loading train images bundled with their respective labels
"""
import os
from pathlib import Path


directory_root: Path = Path(__file__).parent.parent


class LabelPath:
    train: Path = directory_root / "data" / "imagenette2-160" / "train"
    val: Path = directory_root / "data" / "imagenette2-160" / "val"


class ImageWithMetadata:
    image_path: Path
    label: str

    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path
        self.label = image_path.parent.name

    def summary(self) -> dict[str, str]:
        return {
            "image_path": str(self.image_path),
            "label": self.label,
        }


def select_label_directories(labels_path: Path, n_labels: int) -> list[Path]:
    if n_labels < 1:
        return []
    all_label_directories: list[Path] = list(
        path for path in labels_path.iterdir() if path.is_dir()
    )
    selected_label_directories: list[Path] = all_label_directories[:n_labels]
    return selected_label_directories


def select_images(label_directories: list, n_images: int) -> list[Path]:
    images_full_path: list = []
    for label_directory in label_directories:
        all_images_in_label_directory = os.listdir(label_directory)
        for image_path in all_images_in_label_directory[:n_images]:
            images_full_path.append(Path.joinpath(label_directory, image_path))
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
