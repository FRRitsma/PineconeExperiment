# %%

from embedding.encode_img import embed_image, load_image_from_file
import os

# This is needed to load files in the directory correctly:
here = os.path.dirname(os.path.abspath(__file__))


def test_always_passes():
    assert True


def test_embed_image():
    filename = os.path.join(here, "input.jpg")
    img = load_image_from_file(filename)
    embed_image(img)
    assert True
