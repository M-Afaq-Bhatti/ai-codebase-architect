import os
import json
import mimetypes
from git import Repo  # pip install GitPython

class CodeIngestor:
    def __init__(self, repo_path: str, output_path: str = "data/raw_code/"):
        self.repo_path = repo_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def clone_repo(self, github_url: str):
        """Clone GitHub repo into local directory"""
        repo_name = github_url.split("/")[-1].replace(".git", "")
        local_path = os.path.join(self.output_path, repo_name)
        if not os.path.exists(local_path):
            Repo.clone_from(github_url, local_path)
        self.repo_path = local_path
        print(f"✅ Cloned: {repo_name}")

    def collect_files(self):
        """Recursively collect all source code files"""
        supported_ext = [".py", ".js", ".ts", ".java", ".cpp", ".html", ".css", ".json"]
        file_list = []
        for root, _, files in os.walk(self.repo_path):
            if any(x in root for x in ["venv", ".git", "node_modules"]):
                continue
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in supported_ext:
                    file_list.append(os.path.join(root, file))
        return file_list

    def read_files(self, file_list):
        """Read text content and return structured data"""
        data = []
        for path in file_list:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                data.append({
                    "path": path,
                    "content": content,
                    "size": len(content),
                    "type": mimetypes.guess_type(path)[0]
                })
            except Exception as e:
                print(f"⚠️ Skipped {path}: {e}")
        return data

    def save_json(self, data):
        """Save collected data as JSON"""
        out_file = os.path.join(self.output_path, "ingested_code.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved {len(data)} files to {out_file}")

