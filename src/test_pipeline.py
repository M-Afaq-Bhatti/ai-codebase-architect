# src/test_pipeline.py

import os
import shutil
from git import Repo
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings


# ------------------------------
# 1. Clone or use local repo
# ------------------------------
def clone_repo(repo_url, clone_dir="repo_clone"):
    """Clone a GitHub repository into a local folder."""
    if os.path.exists(clone_dir):
        print("🔁 Removing old repo folder...")
        shutil.rmtree(clone_dir)

    print(f"📦 Cloning repository from {repo_url}...")
    Repo.clone_from(repo_url, clone_dir)
    print("✅ Repository cloned successfully!")
    return clone_dir


# ------------------------------
# 2. Load and read code files
# ------------------------------
def load_code_files(repo_dir):
    """Load all code files (Python, JS, TS, Java, etc.) into LangChain documents."""
    print("📂 Loading code files...")
    docs = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith((".py", ".js", ".ts", ".java")):
                path = os.path.join(root, file)
                try:
                    loader = TextLoader(path, encoding="utf-8")
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"⚠️ Skipping file {path}: {e}")
    print(f"✅ Loaded {len(docs)} code files.")
    return docs


# ------------------------------
# 3. Split code into chunks
# ------------------------------
def chunk_documents(docs):
    """Split code documents into overlapping text chunks."""
    print("✂️ Splitting files into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} text chunks.")
    return chunks


# ------------------------------
# 4. Generate embeddings & store in vector DB
# ------------------------------
def create_embeddings(chunks, persist_dir="vector_db"):
    """Create embeddings using all-MiniLM-L6-v2 and store them in Chroma."""
    print("🔢 Creating embeddings with 'all-MiniLM-L6-v2'...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    print(f"✅ Embeddings stored in: {persist_dir}")
    return vectordb


# ------------------------------
# 5. Test retrieval
# ------------------------------
def test_query(vectordb, query="How is data ingestion handled?"):
    """Run a test similarity search query."""
    print(f"🔍 Testing query: {query}")
    results = vectordb.similarity_search(query, k=3)
    for i, r in enumerate(results, 1):
        print(f"\n📘 Result {i}:")
        print(r.page_content[:400])  # print snippet only


# ------------------------------
# 6. Full pipeline
# ------------------------------
if __name__ == "__main__":
    # You can switch between a GitHub repo or local folder easily here:
    USE_LOCAL_FOLDER = False  # change to True if you want to test on your local folder

    if USE_LOCAL_FOLDER:
        repo_dir = "D:/ai_codebase/ai-codebase-architect"  # your local folder path
        print(f"📁 Using local folder: {repo_dir}")
    else:
        REPO_URL = "https://github.com/openai/openai-python.git"  # example public repo
        repo_dir = clone_repo(REPO_URL)

    docs = load_code_files(repo_dir)
    chunks = chunk_documents(docs)
    vectordb = create_embeddings(chunks)
    test_query(vectordb)
