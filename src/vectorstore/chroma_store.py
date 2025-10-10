import os
import json
import uuid
import chromadb

# ---------------------------------------
# 1. Connect to Chroma
# ---------------------------------------
def connect_chroma(persist_directory="data/chroma"):
    client = chromadb.PersistentClient(path=persist_directory)
    print(f"✅ Connected to ChromaDB (persist: {persist_directory})")
    return client

# ---------------------------------------
# 2. Get or create collection
# ---------------------------------------
def get_or_create_collection(client, collection_name="codebase_embeddings"):
    collection = client.get_or_create_collection(collection_name)
    print(f"✅ Using collection: '{collection_name}'")
    return collection

# ---------------------------------------
# 3. Load embeddings JSON
# ---------------------------------------
def load_embeddings(json_path):
    print(f"📂 Loading embeddings from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} raw embeddings")
    return data

# ---------------------------------------
# 4. Clean + ensure unique IDs
# ---------------------------------------
def clean_embeddings(data):
    cleaned = []
    seen_ids = set()

    for i, item in enumerate(data):
        if "vector" not in item or not item["vector"]:
            continue

        # Create base ID (prefer path if available)
        base_id = item.get("path", f"id_{i}")
        if base_id in seen_ids:
            # Ensure uniqueness by appending a UUID or index
            unique_id = f"{base_id}_{uuid.uuid4().hex[:8]}"
        else:
            unique_id = base_id

        seen_ids.add(unique_id)

        # ---- Sanitize metadata ----
        meta = item.get("metadata", {})
        safe_meta = {}
        for k, v in meta.items():
            if isinstance(v, (list, dict)):
                safe_meta[k] = json.dumps(v)
            elif v is None:
                safe_meta[k] = "None"
            else:
                safe_meta[k] = str(v)

        cleaned.append({
            "id": unique_id,
            "embedding": item["vector"],
            "metadata": safe_meta,
        })

    print(f"🧹 Cleaned {len(cleaned)} valid embeddings (filtered out {len(data) - len(cleaned)})")
    return cleaned

# ---------------------------------------
# 5. Insert into Chroma
# ---------------------------------------
def insert_into_chroma(collection, cleaned_data, batch_size=200):
    total = len(cleaned_data)
    print(f"🚀 Inserting {total} embeddings into ChromaDB...")

    try:
        for i in range(0, total, batch_size):
            batch = cleaned_data[i:i+batch_size]
            ids = [item["id"] for item in batch]
            vectors = [item["embedding"] for item in batch]
            metadatas = [item["metadata"] for item in batch]

            collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)
            print(f"✅ Inserted batch {i//batch_size + 1} ({len(batch)} items)")
        print(f"🎯 Successfully stored {total} embeddings in ChromaDB")

    except Exception as e:
        raise RuntimeError(f"❌ Failed to insert embeddings: {e}")
