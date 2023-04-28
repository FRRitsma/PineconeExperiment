# %%

from embedding.encode_img import embed_image, load_image_from_file
import os


def test_always_passes():
    assert True


def test_embed_image():
    here = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(here, "input.jpg")
    img = load_image_from_file(filename)
    embed_image(img)
    assert True
