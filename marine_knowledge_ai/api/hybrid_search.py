# api/hybrid_search.py

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import re

# Load model
model = SentenceTransformer("all-mpnet-base-v2")

# Connect to ChromaDB
CHROMA_DB_DIR = "data/vector_db_chunks"
client = chromadb.PersistentClient(path="data/vector_db_chunks")
collection = client.get_or_create_collection(name="chunked_marine_docs")


def highlight_keywords(text, query):
    keywords = re.findall(r"\w+", query.lower())
    for kw in keywords:
        text = re.sub(f"(?i)({re.escape(kw)})", r"<mark>\1</mark>", text)
    return text

def hybrid_search(query, top_k=5):
    embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k * 2,  # fetch extra to allow filtering
        include=["documents", "metadatas", "distances"]
    )

    hybrid_results = []
    query_lower = query.lower()

    for i in range(len(results["documents"][0])):
        doc_text = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        score = 1 - distance  # semantic score
        keyword_hit = query_lower in doc_text.lower() or any(query_lower in str(v).lower() for v in metadata.values())

        if keyword_hit:
            score += 0.25  # bonus for exact match

        preview = highlight_keywords(doc_text[:300], query)

        hybrid_results.append({
            "document": preview,
            "full_text": doc_text,
            "metadata": metadata,
            "score": round(score, 3)
        })

    # Sort by final hybrid score
    sorted_results = sorted(hybrid_results, key=lambda x: x["score"], reverse=True)
    return sorted_results[:top_k]
