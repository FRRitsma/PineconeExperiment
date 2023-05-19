# %%
# Part 1: Initialize database
# Part 2: Fill database
# Attach image link in metadata
import pinecone

from settings import API_KEY
from settings import ENVIRONMENT
from settings import INDEX_NAME
from src.embedding.embedding import Embedder
from src.extract.extract import extract_images_with_metadata
from src.extract.extract import LabelPath

N_LABELS: int = 10
N_IMAGES: int = 1000

train_data = extract_images_with_metadata(N_LABELS, N_IMAGES, LabelPath.train)

embedder = Embedder()

vector_dimension = len(embedder.embed(train_data[0]))
pinecone.init(api_key=API_KEY, environment=ENVIRONMENT)
index = pinecone.Index(INDEX_NAME)

# %%
for i, image_with_metadata in enumerate(train_data):
    upsert_response = index.upsert(
        vectors=[
            {
                "id": f"vec{i}",
                "values": embedder.embed(image_with_metadata).tolist(),
                "metadata": image_with_metadata.summary(),
            }
        ],
    )

# %%
# query_vector = vector.tolist()
# query_results = index.query(queries=[query_vector], top_k=10, include_metadata=True)
# results = query_results["results"][0]["matches"]

# for result in results:
#     img = Image.open(result["metadata"]["image_path"])
#     img.show()
