# %%
import pinecone

from settings import API_KEY
from settings import DIMENSION
from settings import ENVIRONMENT
from settings import INDEX_NAME

# pinecone.init(api_key=API_KEY, environment=ENVIRONMENT)


def find_out():
    print(pinecone.list_indexes())


# %%

# Connect to Pinecone
pinecone.init(api_key=API_KEY, environment=ENVIRONMENT)

# Add index if it does not already exist:
if INDEX_NAME not in pinecone.list_indexes():
    # Add index name to pinecone:
    pinecone.create_index(INDEX_NAME, DIMENSION, metric="euclidean")

index = pinecone.Index(index_name=INDEX_NAME)

index.upsert(
    [
        ("A", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        ("B", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
        ("C", [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
        ("D", [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
        ("E", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    ]
)

# Perform the similarity search
query_vector = [0.155] * DIMENSION
results = index.query(queries=[query_vector], top_k=10)
