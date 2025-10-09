# parse_main.py (or put in main.py)
from src.parsing.code_parser import CodeParser

if __name__ == "__main__":
    parser = CodeParser(
        ingested_json="data/raw_code/ingested_code.json",
        out_dir="data/processed_chunks/"
    )
    chunks = parser.parse_all()
    print("Parsed chunks:", len(chunks))
