# src/parsing/code_parser.py
import ast
import json
import os
import hashlib
import re
from typing import List, Dict, Any

class CodeParser:
    def __init__(self,
        ingested_json: str = "data/raw_code/ingested_code.json",
        out_dir: str = "data/processed_chunks/"):
        self.ingested_json = ingested_json
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def detect_language(self, path: str, content: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
        }
        return mapping.get(ext, "unknown")

    def load_ingested(self) -> List[Dict[str, Any]]:
        with open(self.ingested_json, "r", encoding="utf-8") as f:
            return json.load(f)

    def parse_all(self) -> List[Dict[str, Any]]:
        entries = self.load_ingested()
        all_chunks = []
        for e in entries:
            chunks = self.parse_file(e)
            all_chunks.extend(chunks)
        out_file = os.path.join(self.out_dir, "processed_chunks.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2)
        print(f"✅ Saved {len(all_chunks)} chunks to {out_file}")
        return all_chunks

    def parse_file(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        path = entry.get("path", "<unknown>")
        content = entry.get("content", "")
        lang = self.detect_language(path, content)
        if lang == "python":
            return self.parse_python(path, content)
        else:
            return self.generic_parse(path, content, lang)

    # ---------- Python specific parsing ----------
    def parse_python(self, path: str, content: str) -> List[Dict[str, Any]]:
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"⚠️ SyntaxError parsing {path}: {e} — saving whole file as single chunk")
            return [self.make_chunk(path, "module", None, 1, None, content, [], None, "python")]

        # collect imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    imports.append(f"{mod}.{alias.name}" if mod else alias.name)

        chunks = []
        # extract top-level functions & classes as chunks
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = getattr(node, "name", None)
                start = getattr(node, "lineno", 1)
                end = getattr(node, "end_lineno", None)
                snippet = self.get_source_segment(content, node)
                doc = ast.get_docstring(node)
                typ = "class" if isinstance(node, ast.ClassDef) else "function"
                chunk = self.make_chunk(path, typ, name, start, end, snippet, imports, doc, "python")
                chunks.append(chunk)

        # module-level code: everything that's not inside top-level function/class
        module_code = self.get_module_level_code(content, tree)
        if module_code.strip():
            module_doc = ast.get_docstring(tree)
            module_chunk = self.make_chunk(path, "module", None, 1, None, module_code, imports, module_doc, "python")
            chunks.insert(0, module_chunk)

        return chunks

    def get_source_segment(self, source: str, node: ast.AST) -> str:
        # try ast.get_source_segment (py3.8+), otherwise use lineno/end_lineno fallback
        try:
            seg = ast.get_source_segment(source, node)
            if seg:
                return seg
        except Exception:
            pass
        lines = source.splitlines()
        start = max(0, getattr(node, "lineno", 1) - 1)
        end = getattr(node, "end_lineno", start + 1)
        if end is None:
            end = start + 1
        return "\n".join(lines[start:end])

    def get_module_level_code(self, source: str, tree: ast.Module) -> str:
        lines = source.splitlines()
        remove_ranges = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                s = getattr(node, "lineno", 1) - 1
                e = getattr(node, "end_lineno", s + 1)
                if e is None:
                    e = s + 1
                remove_ranges.append((s, e))
        keep = []
        for i, line in enumerate(lines):
            if any(s <= i < e for (s, e) in remove_ranges):
                continue
            keep.append(line)
        return "\n".join(keep)

    # ---------- Generic fallback parser ----------
    def generic_parse(self, path: str, content: str, lang: str) -> List[Dict[str, Any]]:
        # Basic heuristic: try to collect import-like lines and put whole file as one chunk
        imports = []
        if lang in ("javascript", "typescript"):
            imports = re.findall(r"^\s*(?:import .* from ['\"]([^'\"]+)['\"]|const .* = require\(['\"]([^'\"]+)['\"]\))", content, re.MULTILINE)
            # flatten tuple results and remove empty
            imports = [x for tup in imports for x in tup if x]
        else:
            # a few generic heuristics
            imports = re.findall(r"^\s*(?:#|//)?\s*(import|require|using)\s+([^\s;'\"]+)", content, re.MULTILINE)
            imports = [t if isinstance(t, str) else (t[1] if len(t) > 1 else "") for t in imports]

        chunk = self.make_chunk(path, "module", None, 1, None, content, imports, None, lang)
        return [chunk]

    # ---------- utilities ----------
    def compute_cyclomatic(self, code: str) -> int:
        # simple estimator: 1 + number of branching constructs
        try:
            tree = ast.parse(code)
        except Exception:
            return 1
        count = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.AsyncFor, ast.IfExp, ast.Try, ast.ExceptHandler)):
                count += 1
            if isinstance(node, ast.BoolOp):
                # each boolean operator increases branching roughly by number of values-1
                count += max(0, len(node.values) - 1)
        return count

    def make_chunk(self, path, typ, name, start, end, code, imports, doc, lang):
        start = start or 1
        if end is None:
            # estimate end from code
            end = start + code.count("\n")
        loc = max(1, end - start + 1)
        size = len(code)
        uid = hashlib.sha1(f"{path}:{name}:{start}:{end}".encode("utf-8")).hexdigest()
        complexity = self.compute_cyclomatic(code) if lang == "python" else None
        return {
            "id": uid,
            "file_path": path,
            "language": lang or "unknown",
            "type": typ,
            "name": name,
            "start_line": start,
            "end_line": end,
            "loc": loc,
            "size": size,
            "complexity": complexity,
            "imports": imports,
            "docstring": doc,
            "code": code
        }
