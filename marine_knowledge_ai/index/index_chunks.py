import os
import json
import chromadb
from chromadb.config import Settings

CHUNKED_EMBEDDINGS_PATH = os.path.join("embeddings", "chunked_embeddings.json")
CHROMA_DB_DIR = os.path.join("data", "vector_db_chunks")

def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def store_chunks_in_chroma(chunks, persist_dir):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name="chunked_marine_docs")

    for chunk in chunks:
        try:
            chunk_id = chunk["chunk_id"]
            embedding = chunk["embedding"]
            text = chunk["chunk_text"]
            metadata = {
                "document_id": chunk.get("document_id", "unknown"),
                "file_name": chunk.get("file_name", "unknown"),
                "chunk_index": chunk.get("chunk_index", -1)
            }

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata]
            )
            print(f"✅ Indexed chunk: {chunk_id}")

        except Exception as e:
            print(f"⚠️ Failed to index chunk: {chunk.get('chunk_id', 'unknown')}")
            print(f"   Reason: {e}")

    print(f"\n💾 Chunked Vector DB saved at: {persist_dir}")

if __name__ == "__main__":
    print("📦 Phase 2.3: Indexing Chunked Embeddings")

    chunks = load_chunks(CHUNKED_EMBEDDINGS_PATH)
    print(f"🔢 Loaded {len(chunks)} chunks from file")

    store_chunks_in_chroma(chunks, CHROMA_DB_DIR)

    print("🎯 Phase 2.3 Complete: Chunk-level embeddings indexed.")
