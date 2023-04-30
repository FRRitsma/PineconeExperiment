# %%
import pinecone

# import numpy as np
from settings import API_KEY, ENVIRONMENT, INDEX_NAME, DIMENSION


# Connect to Pinecone
pinecone.init(api_key=API_KEY, environment=ENVIRONMENT)

# Add index if it does not already exist:
if INDEX_NAME not in pinecone.list_indexes():

    # Add index name to pinecone:
    pinecone.create_index(INDEX_NAME, DIMENSION)


# %%
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

# # Define the target vector for similarity search
# target_vector = np.array([0.4, 0.3, 0.5, 0.2, 0.1], dtype=np.float16)

# # Insert the target vector into the index
# doc_id = "my_doc_id"
# index.upsert(ids=[doc_id], vectors=[target_vector])

# %%

# # Perform the similarity search
# query_vector = target_vector
# results = index.query(queries=[query_vector], top_k=10)

# # Print the results
# print("Similarity search results:")
# for result in results[0]:
#     print(result.id, result.score)

# # Close the connection to Pinecone
pinecone.deinit()
