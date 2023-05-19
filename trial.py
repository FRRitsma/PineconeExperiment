# %%
# Part 1: Initialize database
# Part 2: Fill database
# Attach image link in metadata
import numpy as np
import pinecone
from PIL import Image

from settings import API_KEY
from settings import ENVIRONMENT
from settings import INDEX_NAME
from src.embedding.embedding import Embedder
from src.extract.extract import extract_images_with_metadata
from src.extract.extract import ImageWithMetadata
from src.extract.extract import LabelPath

N_LABELS: int = 10
N_IMAGES: int = 10

train_data = extract_images_with_metadata(N_LABELS, N_IMAGES, LabelPath.train)

embedder = Embedder()
vector_dimension = len(embedder.embed(train_data[0].image))
pinecone.init(api_key=API_KEY, environment=ENVIRONMENT)
index = pinecone.Index(INDEX_NAME)


def process_images_with_metadata_list(
    images_with_metadata_list: list[ImageWithMetadata],
) -> list[tuple]:
    embedder = Embedder()
    return [
        (
            vector_from_image_with_metadata(iwm, embedder),
            dictionary_from_image_with_metadata(iwm),
        )
        for iwm in images_with_metadata_list
    ]


def vector_from_image_with_metadata(
    image_with_metadata: ImageWithMetadata, embedder: Embedder
) -> np.ndarray:
    return embedder.embed(image_with_metadata.image)


def dictionary_from_image_with_metadata(
    image_with_metadata: ImageWithMetadata,
) -> dict[str, str]:
    return {
        "image_path": str(image_with_metadata.image_path),
        "label": image_with_metadata.label,
    }


testeroni = process_images_with_metadata_list(train_data)

for i, (vector, metadata) in enumerate(testeroni):
    upsert_response = index.upsert(
        vectors=[
            {
                "id": f"vec{i}",
                "values": vector.tolist(),
                "metadata": metadata,
            }
        ],
    )

# %%
query_vector = vector.tolist()
query_results = index.query(queries=[query_vector], top_k=10, include_metadata=True)
results = query_results["results"][0]["matches"]

for result in results:
    img = Image.open(result["metadata"]["image_path"])
    img.show()
