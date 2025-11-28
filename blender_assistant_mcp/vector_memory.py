import json
import os
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from . import ollama_adapter

class VectorMemory:
    """Simple vector store using JSON persistence and numpy for similarity."""

    def __init__(self, storage_path: Path, embedding_model: str = "nomic-embed-text"):
        self.storage_path = storage_path
        self.embedding_model = embedding_model
        self.documents: List[Dict[str, Any]] = []
        self.preferences: Dict[str, Any] = {}
        self.vectors: Optional[np.ndarray] = None
        self._load()

    def _load(self):
        """Load documents and vectors from disk."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.documents = data.get("documents", [])
                self.preferences = data.get("preferences", {})
                
                # Reconstruct numpy array from list of lists
                vectors_list = data.get("vectors", [])
                if HAS_NUMPY and vectors_list:
                    self.vectors = np.array(vectors_list, dtype=np.float32)
                elif vectors_list:
                    # Fallback if numpy missing (though unlikely in Blender)
                    self.vectors = vectors_list
        except Exception as e:
            print(f"[VectorMemory] Failed to load memory: {e}")

    def save(self):
        """Save documents and vectors to disk."""
        try:
            vectors_list = []
            if self.vectors is not None:
                if HAS_NUMPY:
                    vectors_list = self.vectors.tolist()
                else:
                    vectors_list = self.vectors

            data = {
                "documents": self.documents,
                "preferences": self.preferences,
                "vectors": vectors_list
            }
            
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[VectorMemory] Failed to save memory: {e}")

    def set_preference(self, key: str, value: Any):
        """Set a user preference."""
        self.preferences[key] = value
        self.save()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return self.preferences.get(key, default)

    def add(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a document to memory."""
        embedding = ollama_adapter.generate_embedding(self.embedding_model, text)
        if not embedding:
            print(f"[VectorMemory] Failed to generate embedding for: {text[:50]}...")
            return False

        doc = {
            "text": text,
            "metadata": metadata or {},
            "timestamp": __import__("time").time()
        }
        self.documents.append(doc)

        if HAS_NUMPY:
            embedding_np = np.array([embedding], dtype=np.float32)
            if self.vectors is None:
                self.vectors = embedding_np
            else:
                self.vectors = np.vstack([self.vectors, embedding_np])
        else:
            if self.vectors is None:
                self.vectors = [embedding]
            else:
                self.vectors.append(embedding)

        self.save()
        return True

    def search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents."""
        if not self.documents or self.vectors is None:
            return []

        query_embedding = ollama_adapter.generate_embedding(self.embedding_model, query)
        if not query_embedding:
            return []

        if HAS_NUMPY:
            # Cosine similarity: (A . B) / (||A|| * ||B||)
            # Assume vectors are already normalized? Probably not.
            # Let's normalize query
            q_vec = np.array(query_embedding, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm

            # Normalize stored vectors
            v_norms = np.linalg.norm(self.vectors, axis=1)
            # Avoid division by zero
            v_norms[v_norms == 0] = 1.0
            
            # Compute similarity
            # (N, D) dot (D,) -> (N,)
            scores = np.dot(self.vectors, q_vec) / v_norms
            
            # Get top k
            # argsort returns indices of sorted elements (ascending)
            # We want descending
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for idx in top_indices:
                results.append((self.documents[idx], float(scores[idx])))
            return results

        else:
            # Slow pure python fallback
            def cosine_similarity(v1, v2):
                dot = sum(a*b for a,b in zip(v1, v2))
                norm1 = math.sqrt(sum(a*a for a in v1))
                norm2 = math.sqrt(sum(a*a for a in v2))
                return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

            scores = []
            for i, vec in enumerate(self.vectors):
                score = cosine_similarity(query_embedding, vec)
                scores.append((i, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            return [(self.documents[i], score) for i, score in scores[:k]]
