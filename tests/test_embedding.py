# %%
import os

import torch
from PIL.JpegImagePlugin import JpegImageFile as JpegImageFile

from src.embedding.embedding import Embedder
from src.embedding.embedding import image_transform
from src.embedding.embedding import load_image_from_file
from src.extract.extract import extract_images_with_metadata
from src.extract.extract import LabelPath

# This is needed to load files in the directory correctly:
here = os.path.dirname(os.path.abspath(__file__))


def test_load_image_from_file():
    filename = os.path.join(here, "input.jpg")
    img = load_image_from_file(filename)
    assert isinstance(img, JpegImageFile)


def test_transform_function():
    filename = os.path.join(here, "input.jpg")
    img = load_image_from_file(filename)
    transformed_image = image_transform(img)
    assert isinstance(transformed_image, torch.Tensor)


def test_embedder_class():
    embedder = Embedder()
    assert isinstance(embedder.resnet, torch.nn.modules.container.Sequential)
    filename = os.path.join(here, "input.jpg")
    img = load_image_from_file(filename)
    embedded_image = embedder.embed(img)
    assert isinstance(embedded_image, torch.Tensor)


def test_extract_and_embed():
    n_labels: int = 1
    n_images: int = 3
    embedder = Embedder()
    images_with_metadata = extract_images_with_metadata(
        n_labels, n_images, LabelPath.train
    )
    image_with_metadata = images_with_metadata[0]
    img = image_with_metadata.image
    embedded_image = embedder.embed(img)
    assert isinstance(embedded_image, torch.Tensor)
