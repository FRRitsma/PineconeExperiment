# %%
# Part 1: Initialize database
# Part 2: Fill database
# Attach image link in metadata
from data.openimages import extract_images
from data.openimages import SetPath

train_images = extract_images(5, 5, SetPath.train)
val_images = extract_images(5, 5, SetPath.val)


def fill_data_base():
    train_images = extract_images(5, 5, SetPath.train)
    print(train_images)
    ...
