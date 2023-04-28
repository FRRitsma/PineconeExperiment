# %%
from embedding.encode_img import embed_image
from pathlib import Path
import os

current_dir = Path.cwd()
parent_dir = current_dir.parent
os.chdir(parent_dir)


def test_always_passes():
    assert True


def test_embed_iamge():
    embed_image()
    assert True
