from typing import Union

import pytest
from PIL.JpegImagePlugin import JpegImageFile as JpegImageFile
from PIL.PngImagePlugin import PngImageFile

from src.extract.extract import extract_images_with_metadata
from src.extract.extract import ImageWithMetadata
from src.extract.extract import LabelPath


def test_difference_between_train_and_val_path():
    assert LabelPath.train != LabelPath.val


@pytest.mark.parametrize("path", (LabelPath.train, LabelPath.val))
def test_extract_images_with_metadata_returns_list(path):
    n_labels: int = 5
    n_images: int = 7
    images_with_metadata = extract_images_with_metadata(n_labels, n_images, path)
    assert isinstance(images_with_metadata, list)


@pytest.mark.parametrize("path", (LabelPath.train, LabelPath.val))
def test_extract_images_with_metadata_list_contains_class(path):
    n_labels: int = 5
    n_images: int = 7
    images_with_metadata = extract_images_with_metadata(n_labels, n_images, path)
    assert isinstance(images_with_metadata[0], ImageWithMetadata)


def test_image_with_metadata_class():
    n_labels: int = 1
    n_images: int = 1
    images_with_metadata = extract_images_with_metadata(
        n_labels, n_images, LabelPath.train
    )
    image_with_metadata = images_with_metadata[0]
    assert isinstance(image_with_metadata.image, Union[JpegImageFile, PngImageFile])


@pytest.mark.parametrize("path", (LabelPath.train, LabelPath.val))
def test_extract_images_with_metadata_list_correct_length(path):
    n_labels: int = 5
    n_images: int = 7
    images_with_metadata = extract_images_with_metadata(n_labels, n_images, path)
    assert len(images_with_metadata) == n_labels * n_images


@pytest.mark.parametrize("path", (LabelPath.train, LabelPath.val))
def test_extract_images_with_metadata_list_correct_length_for_more_labels_than_exist(
    path,
):
    n_labels: int = 20
    n_images: int = 50
    images_with_metadata = extract_images_with_metadata(n_labels, n_images, path)
    assert len(images_with_metadata) == 10 * n_images
