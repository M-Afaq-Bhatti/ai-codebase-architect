from src.embeddings.code_embedder import CodeEmbedder

if __name__ == "__main__":
    embedder = CodeEmbedder(
        input_path="data/processed_chunks/processed_chunks.json",
        output_path="data/processed_chunks/embeddings.json",
        # print("🔍 Loading from:", self.input_path)
    )

    parsed_code = embedder.load_parsed_code()
    embeddings = embedder.generate_embeddings(parsed_code)
    embedder.save_embeddings(embeddings)
