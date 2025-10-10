from src.vectorstore.chroma_store import (
    connect_chroma,
    get_or_create_collection,
    load_embeddings,
    clean_embeddings,
    insert_into_chroma,
)

def main():
    json_path = "data/processed_chunks/embeddings.json"
    persist_directory = "data/chroma"
    collection_name = "codebase_embeddings"

    # Step 1: Connect
    client = connect_chroma(persist_directory)
    # Step 2: Get Collection
    collection = get_or_create_collection(client, collection_name)
    # Step 3: Load JSON
    data = load_embeddings(json_path)
    # Step 4: Clean metadata
    cleaned_data = clean_embeddings(data)
    # Step 5: Insert
    insert_into_chroma(collection, cleaned_data, batch_size=200)


if __name__ == "__main__":
    main()
