import sys
from pathlib import Path

import chromadb
import ollama

from chunker import chunk_text
from loader import load_documents

DOCS_DIR = Path(__file__).parent.parent / "Simple-rag/docs_dir"
DB_DIR = Path(__file__).parent / "chroma_db"
EMBED_MODEL = "nomic-embed-text"
COLLECTION = "documents"


def main():
    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        print(f"No files found in {DOCS_DIR}")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    client.delete_collection(COLLECTION) if COLLECTION in [c.name for c in client.list_collections()] else None
    collection = client.create_collection(COLLECTION)

    docs = load_documents(DOCS_DIR)
    print(f"Loaded {len(docs)} document(s) from {DOCS_DIR}")

    ids, texts, metadatas = [], [], []
    for source, text in docs:
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{source}::{i}")
            texts.append(chunk)
            metadatas.append({"source": source, "chunk": i})

    if not texts:
        print("No text extracted from documents.")
        sys.exit(1)

    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL}...")
    embeddings = [ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"] for t in texts]

    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print(f"Indexed {len(texts)} chunks into {DB_DIR}")


if __name__ == "__main__":
    main()