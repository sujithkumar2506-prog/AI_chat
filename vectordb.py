
import chromadb
import hashlib
from pathlib import Path
import os
from google import genai
from dotenv import load_dotenv
load_dotenv()
client = genai.Client()
CHUNK_SIZE = 500
OVERLAP = 100

def initialize_chromadb():
    return chromadb.PersistentClient(path="./chroma_db")

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    # Simple character-based chunking for our first version
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def setup_collection(collection_name="knowledge"):
    client = initialize_chromadb()
    collection = client.get_or_create_collection(name=collection_name)
    document_path = Path("./documents")

    current_files = {
        path.name
        for path in document_path.glob("*.txt")
        if path.is_file()
    }

    # Remove records for source files that no longer exist
    existing = collection.get(include=["metadatas"])
    deleted_ids = [
        record_id
        for record_id, metadata in zip(
            existing["ids"], existing["metadatas"]
        )
        if metadata.get("source") not in current_files
    ]

    if deleted_ids:
        collection.delete(ids=deleted_ids)

    for file_path in document_path.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")
        fingerprint = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        old_records = collection.get(
            where={"source": file_path.name},
            include=["metadatas"]
        )

        if old_records["ids"]:
            old_hashes = {
                metadata.get("content_hash")
                for metadata in old_records["metadatas"]
            }

            if old_hashes == {fingerprint}:
                print(f"Unchanged: {file_path.name}")
                continue

            # Remove stale chunks before writing updated chunks
            collection.delete(ids=old_records["ids"])
            print(f"Updated: {file_path.name}")
        else:
            print(f"New: {file_path.name}")

        chunks = chunk_text(content)
        ids = [
            f"{file_path.name}::chunk_{index}"
            for index in range(len(chunks))
        ]
        metadatas = [
            {
                "source": file_path.name,
                "chunk_index": index,
                "content_hash": fingerprint
            }
            for index in range(len(chunks))
        ]

        if chunks:
            collection.upsert(
                ids=ids,
                documents=chunks,
                metadatas=metadatas
            )

    return collection
def retrieve_context(collection, question, top_k=3):
    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count())
    )

    documents = results["documents"][0]
    return "\n\n".join(documents)


def ask_question(collection, question):
    context = retrieve_context(collection, question)

    if not context:
        return "No relevant documents found."

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    prompt = f"""
Answer the question using only the context below.
If the context does not contain the answer, say you don't
have enough information in the provided documents.

Context:
{context}

Question:
{question}
"""
    model = "gemini-3.8-flash"
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    return response.text


if __name__ == "__main__":
    client = initialize_chromadb()
    collection = client.get_collection(name="knowledge")
    question = input("Ask a question about your documents: ")
    answer = ask_question(collection, question)

    print("\nAnswer:", answer)