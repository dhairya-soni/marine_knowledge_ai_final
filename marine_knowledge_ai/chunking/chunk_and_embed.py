# chunking/chunk_and_embed.py

import os
import json
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')  # Only first time

# === CONFIG ===
EXTRACTED_TEXT_FILE = os.path.join("data", "raw_documents", "extracted_text.json")
OUTPUT_CHUNK_FILE = os.path.join("embeddings", "chunked_embeddings.json")
MODEL_NAME = "all-mpnet-base-v2"

CHUNK_SIZE = 300  # words
CHUNK_OVERLAP = 50

# === Load Text ===
def load_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# === Chunk Text ===
def chunk_text(text, size, overlap):
    words = word_tokenize(text)
    chunks = []

    for i in range(0, len(words), size - overlap):
        chunk_words = words[i:i + size]
        chunk_text = " ".join(chunk_words).strip()
        if len(chunk_words) >= 30:  # Avoid very tiny chunks
            chunks.append(chunk_text)
    return chunks

# === Embed Chunks ===
def embed_chunks(documents):
    model = SentenceTransformer(MODEL_NAME)
    chunked_data = []

    for doc in documents:
        doc_id = doc["document_id"]
        title = doc["title"]
        file_name = doc["file_name"]
        text = doc.get("extracted_text", "")

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        embeddings = model.encode(chunks, show_progress_bar=True)

        for i, chunk in enumerate(chunks):
            chunked_data.append({
                "chunk_id": f"{doc_id}_chunk_{i}",
                "document_id": doc_id,
                "title": title,
                "file_name": file_name,
                "chunk_text": chunk,
                "embedding": embeddings[i].tolist()
            })

        print(f"✅ {file_name} → {len(chunks)} chunks")

    return chunked_data

# === Save to File ===
def save_chunks(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved {len(chunks)} chunked embeddings to: {path}")

# === MAIN ===
if __name__ == "__main__":
    print("🔧 Phase 2.2: Chunk-Based Embedding")

    docs = load_documents(EXTRACTED_TEXT_FILE)
    chunks = embed_chunks(docs)
    save_chunks(chunks, OUTPUT_CHUNK_FILE)

    print("✅ Done. Ready for chunk-based indexing!")
