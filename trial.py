# %%
# Part 1: Initialize database
# Part 2: Fill database
# Attach image link in metadata
from src.extract.extract import extract_images
from src.extract.extract import SetPath

train_images = extract_images(5, 5, SetPath.train)
val_images = extract_images(5, 5, SetPath.val)


def fill_data_base():
    # Commence database connection:s
    train_images = extract_images(5, 5, SetPath.train)
    print(train_images)
    ...
