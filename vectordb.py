
import chromadb
import hashlib
from pathlib import Path


def initialize_chromadb():
    return chromadb.PersistentClient(path="./chroma_db")


def setup_collection(collection_name):
    client = initialize_chromadb()
    collection = client.get_or_create_collection(
        name=collection_name
    )

    document_path = Path("./documents")

    for file_path in document_path.iterdir():
        if not file_path.is_file() or file_path.suffix != ".txt":
            continue

        # Read the current source file
        content = file_path.read_text(encoding="utf-8")

        # Calculate its current fingerprint
        fingerprint = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        # Look for an existing record
        existing = collection.get(
            ids=[file_path.name],
            include=["metadatas"]
        )

        if existing["ids"]:
            stored_hash = existing["metadatas"][0]["content_hash"]

            if fingerprint == stored_hash:
                print(f"Unchanged: {file_path.name}")
                continue

            print(f"Changed: {file_path.name}")
        else:
            print(f"New: {file_path.name}")

        # Insert new or update changed document
        collection.upsert(
            ids=[file_path.name],
            documents=[content],
            metadatas=[{
                "source": file_path.name,
                "content_hash": fingerprint
            }]
        )

    return collection


def check_unchanged(collection_name):
    client = initialize_chromadb()
    collection = client.get_collection(name=collection_name)

    records = collection.get(
        include=["documents", "metadatas"]
    )

    for doc_id, document, metadata in zip(
        records["ids"],
        records["documents"],
        records["metadatas"]
    ):
        stored_hash = metadata["content_hash"]
        document_hash = hashlib.sha256(
            document.encode("utf-8")
        ).hexdigest()

        if document_hash == stored_hash:
            print(f"Stored content and hash match: {doc_id}")
        else:
            print(f"Stored content and hash mismatch: {doc_id}")


collection = setup_collection("knowledge")
check_unchanged("knowledge")
