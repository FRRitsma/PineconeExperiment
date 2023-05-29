from pathlib import Path

import pytest

from src.extract.extract import extract_images_with_metadata
from src.extract.extract import ImageWithMetadata
from src.extract.extract import LabelPath


@pytest.fixture
def collected_images_with_metadata():
    N_LABELS: int = 5
    N_IMAGES: int = 7
    return {
        "train": extract_images_with_metadata(N_LABELS, N_IMAGES, LabelPath.train),
        "val": extract_images_with_metadata(N_LABELS, N_IMAGES, LabelPath.val),
    }


def test_difference_between_train_and_val_path():
    assert LabelPath.train != LabelPath.val


@pytest.mark.local
@pytest.mark.parametrize("path", ["train", "val"])
def test_extract_images_with_metadata_returns_list(
    path, collected_images_with_metadata
):
    assert isinstance(collected_images_with_metadata[path], list)


@pytest.mark.local
@pytest.mark.parametrize("path", ["train", "val"])
def test_extract_images_with_metadata_list_contains_class(
    path, collected_images_with_metadata
):
    images_with_metadata = collected_images_with_metadata[path]
    assert isinstance(images_with_metadata[0], ImageWithMetadata)


@pytest.mark.local
@pytest.mark.parametrize("path", ["train", "val"])
def test_image_with_metadata_class(path, collected_images_with_metadata):
    images_with_metadata = collected_images_with_metadata[path]
    image_with_metadata: ImageWithMetadata = images_with_metadata[0]
    assert isinstance(image_with_metadata.label, str)
    assert isinstance(image_with_metadata.image_path, Path)


@pytest.mark.local
@pytest.mark.parametrize("path", (LabelPath.train, LabelPath.val))
def test_extract_images_with_metadata_list_correct_length(path):
    n_labels: int = 5
    n_images: int = 7
    images_with_metadata = extract_images_with_metadata(n_labels, n_images, path)
    assert len(images_with_metadata) == n_labels * n_images


@pytest.mark.local
@pytest.mark.parametrize("path", (LabelPath.train, LabelPath.val))
def test_extract_images_with_metadata_list_correct_length_for_more_labels_than_exist(
    path,
):
    n_labels: int = 20
    n_images: int = 50
    images_with_metadata = extract_images_with_metadata(n_labels, n_images, path)
    assert len(images_with_metadata) == 10 * n_images


@pytest.mark.local
@pytest.mark.parametrize("path", (LabelPath.train, LabelPath.val))
def test_extract_images_with_metadata_list_correct_length_for_more_images_than_exist(
    path,
):
    n_labels: int = 1
    n_images: int = 10000
    images_with_metadata = extract_images_with_metadata(n_labels, n_images, path)
    assert len(images_with_metadata) < n_images
