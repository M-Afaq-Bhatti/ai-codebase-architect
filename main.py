from src.ingestion.code_ingestor import CodeIngestor

# Example usage:
if __name__ == "__main__":
    ingestor = CodeIngestor(repo_path="D:/gen ai projects/endtoend_proj/medical_chatbot")  
    files = ingestor.collect_files()
    data = ingestor.read_files(files)
    ingestor.save_json(data)
