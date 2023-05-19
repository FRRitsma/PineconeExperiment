# %%
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile as JpegImageFile
from torchvision import transforms

from src.embedding.embedding import Embedder
from src.embedding.embedding import image_transform
from src.embedding.embedding import image_transform_one_channel_to_three_channels
from src.embedding.embedding import image_transform_pad_to_square
from src.embedding.embedding import image_transform_to_tensor_and_resize
from src.embedding.embedding import load_image_from_file
from src.extract.extract import extract_images_with_metadata
from src.extract.extract import LabelPath

# This is needed to load files in the directory correctly:
here = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def single_test_image():
    filename = os.path.join(here, "input.jpg")
    img = load_image_from_file(filename)
    return img


def test_load_image_from_file(single_test_image):
    assert isinstance(single_test_image, JpegImageFile)


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


def test_image_transform_one_channel_to_three_channels():
    # TODO: Fix this hardcode
    path = Path(
        """c://Users//frrit//OneDrive//Desktop//Weaviate//Weaviate//data
        //imagenette2//train//n02102040//n02102040_1408.JPEG"""
    )
    img = Image.open(path)
    img_before_transform = transforms.ToTensor()(img)
    img_after_transform = image_transform_one_channel_to_three_channels(
        img_before_transform
    )
    assert min(img_before_transform.shape) == 1
    assert min(img_after_transform.shape) == 3


def test_transform_function():
    filename = os.path.join(here, "input.jpg")
    img = load_image_from_file(filename)
    transformed_image = image_transform(img)
    assert isinstance(transformed_image, torch.Tensor)


def test_embedder_class():
    embedder = Embedder()
    filename = os.path.join(here, "input.jpg")
    img = load_image_from_file(filename)
    embedded_image = embedder.embed(img)
    assert isinstance(embedded_image, np.ndarray)


def test_extract_and_embed():
    n_labels: int = 10
    n_images: int = 20
    embedder = Embedder()
    images_with_metadata = extract_images_with_metadata(
        n_labels, n_images, LabelPath.train
    )
    image_with_metadata = images_with_metadata[0]
    img = image_with_metadata.image
    embedded_image = embedder.embed(img)
    assert isinstance(embedded_image, np.ndarray)
