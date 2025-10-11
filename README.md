# 🤖 AI-Powered Codebase Architect

> An intelligent system that **understands, analyzes, documents, and refactors entire software codebases** using multi-agent AI architecture.  
> This repository currently includes **Phase 1 – Core Pipeline Setup & Testing**.

---

## 🧩 Phase 1 Overview — Codebase Understanding Pipeline

Phase 1 establishes the **foundation** of the project: building an automated pipeline that can  
fetch any repository, parse its code, chunk it, generate embeddings, and store them in a vector database for intelligent querying.

### ✅ Objectives Completed
- 📦 **Repository ingestion** — load code either from a local folder or by cloning a public/private GitHub repo.  
- 🧠 **Code parsing** — automatically read and extract source files (`.py`, `.js`, `.ts`, `.java`, etc.).  
- ✂️ **Chunking** — split long code files into semantically meaningful segments for embedding.  
- 🔢 **Embedding generation** — convert each chunk into vector representations using `all-MiniLM-L6-v2`.  
- 🧱 **Vector storage** — persist embeddings locally with **ChromaDB** for efficient semantic search.  
- 🔍 **Retrieval test** — verify that relevant code segments are returned for a given query.

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| Programming Language | **Python 3.10+** |
| Vector DB | **ChromaDB** |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Frameworks / Libs | LangChain Community, Sentence-Transformers, GitPython |
| Environment | Conda / venv |

---

## 🧪 Pipeline Flow
Repo (GitHub / Local)
↓
File Loader
↓
Text Splitter
↓
Embeddings (all-MiniLM-L6-v2)
↓
ChromaDB Storage
↓
Query & Retrieve Results

---

🪄 Summary

Phase 1 establishes a fully functional AI embedding pipeline that transforms codebases into searchable vector representations — forming the foundation for intelligent multi-agent operations in the later stages.

