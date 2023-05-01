# %%
import os

from PIL.JpegImagePlugin import JpegImageFile as JpegImageFile

from embedding.encode_img import embed_image
from embedding.encode_img import load_image_from_file

# This is needed to load files in the directory correctly:
here = os.path.dirname(os.path.abspath(__file__))


def always_passes_test():
    assert True


def load_image_from_file_test():
    filename = os.path.join(here, "input.jpg")
    img = load_image_from_file(filename)
    assert isinstance(img, JpegImageFile)


def embed_image_test():
    filename = os.path.join(here, "input.jpg")
    img = load_image_from_file(filename)
    embed_image(img)
    assert True
