from google import genai
from dotenv import load_dotenv
load_dotenv()
text = "Docker containers package an application with its dependencies."


# Chunking the text into smaller parts for embedding
def chunk_text(text, chunk_size=10):
    words = text.split()
    chunks = []
    for i in range(0,len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

client = genai.Client()
chunks = chunk_text(text,5)

# Generating embeddings for each chunk

embeddings = []
for chunk in chunks:
    embedding_result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=[chunk]
    )
    embeddings.append(embedding_result.embeddings[0])

# Details for chunks and embeddings
# There has to one embedding for each chunk, so the number of embeddings should match the number of chunks

print("Number of chunks:", len(chunks))
print("Number of embeddings:", len(embeddings))

for i, embedding in enumerate(embeddings):
    print(f"Chunk {i+1}: {len(embedding.values)} dimensions")

# Store the embedding vector and its relative chunks for later vector search

vector_database = []

for chunk, embedding in zip(chunks,embeddings):
    vector_database.append({
        "chunk": chunk,
        "embedding": embedding.values
    })

print(vector_database)
