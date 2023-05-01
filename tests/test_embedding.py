# %%
import os

from PIL.JpegImagePlugin import JpegImageFile as JpegImageFile

from src.embedding.encode_img import embed_image
from src.embedding.encode_img import load_image_from_file

# This is needed to load files in the directory correctly:
here = os.path.dirname(os.path.abspath(__file__))


def test_always_passes():
    assert True


def test_load_image_from_file():
    filename = os.path.join(here, "input.jpg")
    img = load_image_from_file(filename)
    assert isinstance(img, JpegImageFile)


def test_embed_image():
    filename = os.path.join(here, "input.jpg")
    img = load_image_from_file(filename)
    embed_image(img)
    assert True
