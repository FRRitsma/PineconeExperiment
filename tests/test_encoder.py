# %%

from embedding.encode_img import embed_image, load_image_from_file


def test_always_passes():
    assert True


def test_embed_image():
    img = load_image_from_file("input.jpg")
    embed_image(img)
    assert True
