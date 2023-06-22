from pathlib import Path
from pathlib import WindowsPath

import pytest

from src.extract import extract_images_with_metadata
from src.extract import ImageWithMetadata
from src.extract import LabelPath
from src.extract import select_label_directories


def test_difference_between_train_and_val_path():
    assert LabelPath.train != LabelPath.val


def test_isdir_train_and_val():
    assert LabelPath.train.is_dir()
    assert LabelPath.val.is_dir()


def test_select_label_directories_happy_flow():
    select_n = 5
    path_list = select_label_directories(LabelPath.train, select_n)
    assert isinstance(path_list[0], Path) or isinstance(path_list[0], WindowsPath)
    assert len(path_list) == select_n


def test_select_label_directories_more_than_exist():
    select_n = 50
    path_list = select_label_directories(LabelPath.train, select_n)
    assert len(path_list) == 10


def test_select_label_directories_less_than_exist():
    select_n = -10
    path_list = select_label_directories(LabelPath.train, select_n)
    assert len(path_list) == 0


@pytest.fixture
def collected_images_with_metadata():
    N_LABELS: int = 5
    N_IMAGES: int = 7
    return {
        "train": extract_images_with_metadata(N_LABELS, N_IMAGES, LabelPath.train),
        "val": extract_images_with_metadata(N_LABELS, N_IMAGES, LabelPath.val),
    }


@pytest.mark.parametrize("path", ["train", "val"])
def test_extract_images_with_metadata_returns_list(
    path, collected_images_with_metadata
):
    assert isinstance(collected_images_with_metadata[path], list)


@pytest.mark.parametrize("path", ["train", "val"])
def test_extract_images_with_metadata_list_contains_class(
    path, collected_images_with_metadata
):
    images_with_metadata = collected_images_with_metadata[path]
    assert isinstance(images_with_metadata[0], ImageWithMetadata)


@pytest.mark.parametrize("path", ["train", "val"])
def test_image_with_metadata_class(path, collected_images_with_metadata):
    images_with_metadata = collected_images_with_metadata[path]
    image_with_metadata: ImageWithMetadata = images_with_metadata[0]
    assert isinstance(image_with_metadata.label, str)
    assert isinstance(image_with_metadata.image_path, Path)


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


@pytest.mark.parametrize("path", (LabelPath.train, LabelPath.val))
def test_extract_images_with_metadata_list_correct_length_for_more_images_than_exist(
    path,
):
    n_labels: int = 1
    n_images: int = 10000
    images_with_metadata = extract_images_with_metadata(n_labels, n_images, path)
    assert len(images_with_metadata) < n_images
