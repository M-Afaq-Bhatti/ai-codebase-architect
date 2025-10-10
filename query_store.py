# query_store.py
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer


# 1️⃣ Connect to ChromaDB
def connect_chroma(persist_dir="data/chroma", collection_name="codebase_embeddings"):
    client = PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)
    print(f"✅ Connected to ChromaDB collection: '{collection_name}'")
    return collection


# 2️⃣ Load the embedding model (same as used during insertion)
def load_model(model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    print(f"✅ Loaded embedding model: {model_name}")
    return model


# 3️⃣ Query function
def query_codebase(query_text, top_k=5):
    collection = connect_chroma()
    model = load_model()

    # Convert query into embedding
    query_vector = model.encode([query_text])

    # Perform similarity search
    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    print("\n🔍 Top Results:")
    for i, (doc, meta, doc_id) in enumerate(zip(documents, metadatas, ids), start=1):
        print(f"\nResult #{i}")
        snippet = doc[:300] + "..." if doc else "⚠️ No document text found."
        print("📄 Code Snippet:", snippet)
        print("📁 File Path:", meta.get("source", "Unknown") if isinstance(meta, dict) else "Unknown")
        print("🧩 Tokens:", meta.get("tokens", "N/A") if isinstance(meta, dict) else "N/A")
        print("🆔 ID:", doc_id)

    return results


if __name__ == "__main__":
    query = input("🧠 Enter your query: ")
    query_codebase(query, top_k=5)
