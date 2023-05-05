# %%
import os

import torch
from PIL.JpegImagePlugin import JpegImageFile as JpegImageFile

from src.embedding.embedding import Embedder
from src.embedding.embedding import image_transform
from src.embedding.embedding import load_image_from_file

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
