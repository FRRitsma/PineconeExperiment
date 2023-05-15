# %%
# Part 1: Initialize database
# Part 2: Fill database
# Attach image link in metadata
from src.embedding.embedding import Embedder
from src.extract.extract import extract_images_with_metadata
from src.extract.extract import LabelPath

train_images = extract_images_with_metadata(5, 5, LabelPath.train)
val_images = extract_images_with_metadata(5, 5, LabelPath.val)

embedder = Embedder()
embedded_image = embedder.embed(train_images[0].image)


# def embed_images_with_metadata(image_list: list[dict]) -> list[dict]:
#     embedded_image_list: list = []
#     for image_with_metadata in image_list:
#         embedded_image_with_metadata = image_with_metadata.copy()
#         embedded_image_with_metadata["image"] = embedder.embed(
#             embedded_image_with_metadata
#         )
#         embedded_image_list.append(embedded_image_with_metadata)
#     return embedded_image_list


# def fill_data_base():
#     # Commence database connection:s
#     train_images = extract_images_with_metadata(5, 5, LabelPath.train)

#     print(train_images)
#     ...
