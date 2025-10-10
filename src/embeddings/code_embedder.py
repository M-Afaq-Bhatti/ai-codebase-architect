# src/embeddings/code_embedder.py
"""
Robust CodeEmbedder using sentence-transformers (local, free).
- Handles different key names produced by your parser (code/content/body).
- Batches encoding for speed and memory efficiency.
- Streams output to JSON array (writes incrementally) to avoid large memory usage.
- Prints step-by-step status messages for easy debugging.
"""

import os
import json
import hashlib
import time
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError(
        "sentence-transformers not found. Install with: pip install sentence-transformers"
    ) from e


class CodeEmbedder:
    def __init__(
        self,
        input_path: str = "data/processed_chunks/processed_chunks.json",
        output_path: str = "data/processed_chunks/embeddings.json",
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        sample_size: Optional[int] = None,  # set to an int for quick testing
    ):
        """
        Args:
            input_path: path to parser output JSON (list of chunk objects).
            output_path: path where embeddings JSON array will be written.
            model_name: sentence-transformers model to use.
            batch_size: number of items encoded at once.
            sample_size: if set, only process the first N entries (useful for testing).
        """
        self.input_path = input_path
        self.output_path = output_path
        self.batch_size = max(1, batch_size)
        self.sample_size = sample_size

        print(f"🔁 Loading embedding model '{model_name}' (this may take a moment)...")
        t0 = time.time()
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded in {time.time() - t0:.1f}s")

    def load_parsed_code(self) -> List[Dict[str, Any]]:
        """Load the JSON output from the parser and return list of snippets."""
        print(f"📂 Loading parsed chunks from: {self.input_path}")
        with open(self.input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Parsed file must be a JSON list of chunk objects.")
        total = len(data)
        if self.sample_size:
            data = data[: self.sample_size]
            print(f"⚠️ sample_size={self.sample_size} — only using first {len(data)} / {total} entries")
        else:
            print(f"✅ Loaded {total} parsed chunks")
        return data

    def _extract_text(self, snippet: Dict[str, Any]) -> str:
        """
        Try several common keys for the code text:
        'code', 'content', 'body', 'text'
        """
        return (
            (snippet.get("code") or snippet.get("content") or snippet.get("body") or snippet.get("text") or "")
            .strip()
        )

    def _get_path(self, snippet: Dict[str, Any]) -> str:
        return snippet.get("file_path") or snippet.get("path") or snippet.get("filepath") or ""

    def _get_id(self, snippet: Dict[str, Any]) -> str:
        return snippet.get("id") or hashlib.sha1(self._get_path(snippet).encode("utf-8")).hexdigest()

    def generate_embeddings(self, code_snippets: List[Dict[str, Any]]) -> int:
        """
        Encode all code_snippets and stream the results to self.output_path.
        Returns total number of embeddings written.
        """
        total_items = len(code_snippets)
        print(f"▶️ Starting embedding generation for {total_items} items (batch_size={self.batch_size})")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        written = 0
        skipped = 0
        first_write = True

        start_time = time.time()
        with open(self.output_path, "w", encoding="utf-8") as out_f:
            out_f.write("[\n")  # start JSON array

            # Process in batches
            for batch_start in range(0, total_items, self.batch_size):
                batch = code_snippets[batch_start : batch_start + self.batch_size]

                # Prepare texts and metadata for this batch
                texts = []
                metas = []
                for snippet in batch:
                    text = self._extract_text(snippet)
                    if not text:
                        skipped += 1
                        continue
                    texts.append(text)
                    metas.append(snippet)

                if not texts:
                    print(f"ℹ️ Batch {batch_start + 1}-{min(total_items, batch_start + self.batch_size)}: nothing to encode (all empty)")
                    continue

                # Encode the batch
                try:
                    vectors = self.model.encode(texts, show_progress_bar=False)
                except Exception as e:
                    # Fall back to item-wise encoding if batch fails
                    print(f"⚠️ Batch encode failed: {e} — falling back to per-item encoding")
                    vectors = []
                    for t in texts:
                        try:
                            v = self.model.encode(t)
                        except Exception as e2:
                            print(f"⚠️ Single encode failed: {e2} — skipping one item")
                            v = None
                        vectors.append(v)

                # Write encoded vectors to file (streaming, append objects with commas)
                for vec_idx, vec in enumerate(vectors):
                    if vec is None:
                        skipped += 1
                        continue
                    meta = metas[vec_idx]
                    obj = {
                        "id": self._get_id(meta),
                        "path": self._get_path(meta),
                        # ensure vector is plain python list (JSON serializable)
                        "vector": vec.tolist() if hasattr(vec, "tolist") else list(vec),
                        "metadata": {
                            "language": meta.get("language"),
                            "type": meta.get("type"),
                            "name": meta.get("name"),
                            "imports": meta.get("imports"),
                            "loc": meta.get("loc"),
                            "size": meta.get("size"),
                        },
                    }

                    # handle commas / formatting for streaming JSON array
                    if not first_write:
                        out_f.write(",\n")
                    json.dump(obj, out_f, ensure_ascii=False)
                    first_write = False
                    written += 1

                # Progress print for this batch
                processed_up_to = min(total_items, batch_start + self.batch_size)
                print(f"🔁 Processed {processed_up_to}/{total_items} — written so far: {written} (skipped: {skipped})")

            out_f.write("\n]\n")  # close JSON array

        elapsed = time.time() - start_time
        print(f"✅ Finished. Written: {written}, Skipped: {skipped}, Time: {elapsed:.1f}s")
        return written

    def save_embeddings(self, data: List[Dict[str, Any]]):
        """
        Backwards-compatible method if you already have embeddings in-memory and want to save them as a list.
        Prefer generate_embeddings(streaming) for large datasets.
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"💾 Saved {len(data)} embeddings to {self.output_path}")
