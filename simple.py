from src.embeddings.code_embedder import CodeEmbedder

if __name__ == "__main__":
    # For a fast test, use sample_size to only process first 100 items
    embedder = CodeEmbedder(
        input_path="data/processed_chunks/processed_chunks.json",
        output_path="data/processed_chunks/embeddings.json",
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
        sample_size=2000  # set None to process entire file
    )
    snippets = embedder.load_parsed_code()
    embedder.generate_embeddings(snippets)
