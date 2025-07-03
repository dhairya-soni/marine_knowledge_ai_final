import chromadb

client = chromadb.PersistentClient(path="data/vector_db_chunks")
collection = client.get_or_create_collection(name="marine_chunks")

results = collection.get(include=["documents", "metadatas"])
print(f"✅ Total Chunks: {len(results['ids'])}")

for i in range(min(5, len(results["ids"]))):
    print(f"\n🔹 Chunk ID: {results['ids'][i]}")
    print(f"📄 File: {results['metadatas'][i]['file_name']}")
    print(f"📝 Preview: {results['documents'][i][:200]}")
