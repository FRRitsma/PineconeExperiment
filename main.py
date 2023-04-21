
import pinecone

# Connect to Pinecone
pinecone.init(api_key="YOUR_API_KEY")
index = pinecone.Index(index_name="my_index", mode="w")

# Define the target vector for similarity search
target_vector = [0.4, 0.3, 0.5, 0.2, 0.1]

# Insert the target vector into the index
doc_id = "my_doc_id"
index.upsert(ids=[doc_id], vectors=[target_vector])

# Perform the similarity search
query_vector = target_vector
results = index.query(queries=[query_vector], top_k=10)

# Print the results
print("Similarity search results:")
for result in results[0]:
    print(result.id, result.score)
    
# Close the connection to Pinecone
pinecone.deinit()
