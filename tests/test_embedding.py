from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile as JpegImageFile
from PIL.PngImagePlugin import PngImageFile
from torchvision import transforms

from src.embedding import Embedder
from src.embedding import image_transform
from src.embedding import image_transform_one_channel_to_three_channels
from src.embedding import image_transform_pad_to_square
from src.embedding import image_transform_to_tensor_and_resize
from src.embedding import load_image_from_file
from src.extract import extract_images_with_metadata
from src.extract import LabelPath

# This is needed to load files in the directory correctly:
here = Path(__file__).parent


@pytest.fixture
def single_test_image():
    filename = here / "input.jpg"
    img = load_image_from_file(filename)
    return img


@pytest.fixture
def test_image_single_channel():
    path = LabelPath.train / "n02102040" / "n02102040_1408.JPEG"
    img = Image.open(path)
    return img


def test_is_cuda_available():
    assert torch.cuda.is_available(
    ), "Cuda is not available. Note that if you work in a poetry environment\
     you can install cuda with the command 'poe force-cuda'"


def test_load_image_from_file(single_test_image):
    assert isinstance(single_test_image, JpegImageFile) or isinstance(single_test_image, PngImageFile)


def test_image_transform_pad_to_square(single_test_image):
    img_as_array = np.array(single_test_image)
    transformed_img = image_transform_pad_to_square(single_test_image)
    transformed_img_as_array = np.array(transformed_img)
    assert img_as_array.shape[1] != img_as_array.shape[2]
    assert (
        transformed_img_as_array.shape[0]
        == transformed_img_as_array.shape[1]
        == max(img_as_array.shape)
    )


def test_image_transform_to_tensor_and_resize(single_test_image):
    RESIZE: int = 400
    transformed_img = image_transform_to_tensor_and_resize(single_test_image, RESIZE)
    assert transformed_img.shape[0] == 3
    assert min(transformed_img.shape[1:]) == RESIZE
    assert isinstance(transformed_img, torch.Tensor)


def test_image_transform_one_channel_to_three_channels(test_image_single_channel):
    img_before_transform = transforms.ToTensor()(test_image_single_channel)
    img_after_transform = image_transform_one_channel_to_three_channels(
        img_before_transform
    )
    assert min(img_before_transform.shape) == 1
    assert min(img_after_transform.shape) == 3


def test_transform_function(single_test_image):
    transformed_image = image_transform(single_test_image)
    assert isinstance(transformed_image, torch.Tensor)


def test_extract_and_embed():
    n_labels: int = 1
    n_images: int = 1
    images_with_metadata = extract_images_with_metadata(
        n_labels, n_images, LabelPath.train
    )
    image_with_metadata = images_with_metadata[0]

    embedder = Embedder()
    embedded_image = embedder.embed(image_with_metadata)
    assert isinstance(embedded_image, np.ndarray)
