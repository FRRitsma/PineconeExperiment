# %%
"""
Loading train images bundled with their respective labels
"""
# TODO: Add notation of specific file name
import os
from itertools import product
from pathlib import Path

from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile


class setchoicestr(str):
    pass


class SetChoice(enumerate):
    train: setchoicestr = setchoicestr("train")
    val: setchoicestr = setchoicestr("val")


def extract_images(n_labels: int, n_images: int, set_type: setchoicestr) -> list[dict]:
    here: Path = Path(os.path.dirname(os.path.abspath(__file__)))
    path_to_labels: Path = Path.joinpath(
        here,
        "imagenette2",
        set_type,
    )

    all_label_directories: list[str] = os.listdir(path_to_labels)
    selected_label_directories: list[str] = all_label_directories[
        : min(len(all_label_directories), n_labels)
    ]

    output_images_with_metadata: list[dict] = []
    for label, index in product(selected_label_directories, range(n_images)):
        full_path_directory: Path = Path.joinpath(path_to_labels, label)
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
