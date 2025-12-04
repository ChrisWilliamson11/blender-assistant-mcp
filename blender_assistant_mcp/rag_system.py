"""
RAG (Retrieval-Augmented Generation) system for Blender documentation.

This module provides semantic search over Blender documentation to augment
LLM prompts with relevant context.

Uses a lightweight JSON-based vector database for easy bundling.
"""

import bpy
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import threading


class BlenderRAG:
    """RAG system for Blender documentation."""

    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.enabled = False
        self.db_path = None
        self.last_indices = []
        self.loading = False  # non-blocking init state

    def initialize(self):
        """Initialize the RAG system by loading the vector database."""
        try:
            # Get the extension directory
            extension_dir = Path(__file__).parent

            # Prefer new sharded database under rag_db/index.json
            rag_dir = extension_dir / "rag_db"
            index_path = rag_dir / "index.json"
            loaded_docs = []
            loaded_embs = []

            if index_path.exists():
                try:
                    with open(index_path, 'r', encoding='utf-8') as f:
                        index_data = json.load(f)
                    shards = index_data.get('shards', [])
                    for s in shards:
                        shard_file = rag_dir / s.get('path', '')
                        if shard_file.exists():
                            with open(shard_file, 'r', encoding='utf-8') as sf:
                                shard_data = json.load(sf)
                            docs = shard_data.get('documents', [])
                            embs = shard_data.get('embeddings', [])
                            # Sanity: lengths should match
                            if docs and embs and len(docs) == len(embs):
                                loaded_docs.extend(docs)
                                loaded_embs.extend(embs)
                except Exception as e:
                    print(f"[RAG] Failed to load shards: {e}")

            # Require sharded database; no legacy fallback
            if not loaded_docs:
                print("[RAG] Sharded RAG database not found.")
                self.enabled = False
                return False

            if not loaded_docs or not loaded_embs:
                print("[RAG] Empty database - RAG disabled")
                self.enabled = False
                return False

            self.documents = loaded_docs
            self.embeddings = np.array(loaded_embs, dtype=np.float32)
            # Record DB path for stats
            self.db_path = index_path


            print(f"[RAG] Loaded {len(self.documents)} documents")
            self.enabled = True
            return True

        except Exception as e:
            print(f"[RAG] Failed to initialize RAG system: {e}")
            self.enabled = False
            return False

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text using Ollama."""
        try:
            from . import ollama_adapter as llama_manager

            # Use Ollama's nomic-embed-text model
            # User must have run: ollama pull nomic-embed-text
            embedding_model_name = "nomic-embed-text"

            # Generate embedding using Ollama
            embedding = llama_manager.generate_embedding(
                model_path=embedding_model_name,
                text=text
            )

            if embedding is None:
                print("[RAG] Failed to generate embedding. Make sure 'nomic-embed-text' is installed.")
                print("[RAG] Run: ollama pull nomic-embed-text")
                return None

            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            print(f"[RAG] Failed to get embedding: {e}")
            print(f"[RAG] Make sure Ollama is running and 'nomic-embed-text' is installed")
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def query_documents(self, query: str, n_results: int = 5, prefer_source: Optional[str] = None, exclude_indices: Optional[set] = None) -> List[Dict[str, str]]:
        """
        Query the vector database for relevant documentation.

        Args:
            query: The user's question or search query
            n_results: Number of relevant documents to retrieve
            prefer_source: Optional source bias ("API" or "Manual")
            exclude_indices: Optional set of integer indices to skip (already sent)

        Returns:
            List of dictionaries containing document text and metadata
        """
        if not self.enabled or self.embeddings is None:
            return []

        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                return []

            # Calculate similarities with optional source bias
            similarities = []
            bias = (prefer_source or '').lower()
            for i, doc_embedding in enumerate(self.embeddings):
                if exclude_indices and i in exclude_indices:
                    continue
                sim = self._cosine_similarity(query_embedding, doc_embedding)
                if bias:
                    md = self.documents[i].get('metadata', {})
                    doc_src = str(md.get('source', '')).lower()
                    if (bias == 'api' and doc_src == 'api') or (bias == 'manual' and doc_src == 'manual'):
                        sim *= 1.15  # small boost for preferred source
                similarities.append((i, sim))

            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Get top N results
            docs = []
            for i, similarity in similarities[:n_results]:
                doc = self.documents[i]
                docs.append({
                    'index': i,
                    'content': doc['content'],
                    'metadata': doc.get('metadata', {}),
                    'similarity': float(similarity)
                })

            return docs

        except Exception as e:
            print(f"[RAG] Query failed: {e}")
            return []

    def augment_prompt(self, user_message: str, n_results: int = 5, prefer_source: Optional[str] = None, exclude_indices: Optional[set] = None) -> str:
        """
        Augment a user message with relevant documentation context.

        Args:
            user_message: The user's original message
            n_results: Number of relevant documents to retrieve
            prefer_source: Optional source bias ("API" or "Manual")
            exclude_indices: Optional set of integer indices to skip (already sent)

        Returns:
            Augmented prompt with documentation context
        """
        if not self.enabled:
            return user_message

        # Retrieve relevant documents
        docs = self.query_documents(user_message, n_results=n_results, prefer_source=prefer_source, exclude_indices=exclude_indices)

        if not docs:
            return user_message

        # Remember which indices were used (for de-duplication in follow-ups)
        self.last_indices = [d.get('index') for d in docs if 'index' in d]

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(docs, 1):
            md = doc.get('metadata', {})
            source = md.get('source', 'Unknown')
            file_name = md.get('file') or ''
            url = md.get('url') or ''
            header = f"[Doc {i} - {source}]"
            if file_name:
                header = f"{header} {file_name}"
            if url:
                header = f"{header} ({url})"
            content = doc['content']
            context_parts.append(f"{header}\n{content}")

        context = "\n\n".join(context_parts)

        # Create augmented prompt - emphasize tool calling over explanation
        augmented = f"""You are a Blender automation assistant with access to tools. The user wants you to DO things in Blender, not just explain how.

        INTERNAL REFERENCE - BLENDER API DOCUMENTATION (for your tool calls):
        {context}

        USER REQUEST:
        {user_message}

        IMPORTANT INSTRUCTIONS:
        1. The user wants you to PERFORM the task using your available tools, not explain how to do it
        2. Use the documentation above to understand the correct Blender API calls
        3. Call the appropriate tools to accomplish the user's request
        4. Only provide explanations if the user explicitly asks "how do I..." or "explain..."
        5. Default action: DO the task using tools, don't just describe it

        If the user asks you to create, modify, or manipulate something in Blender, use your tools to do it directly."""

        return augmented

    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the RAG system."""
        if not self.enabled:
            return {
                'enabled': False,
                'document_count': 0,
                'db_path': None
            }

        return {
            'enabled': True,
            'document_count': len(self.documents),
            'db_path': str(self.db_path) if self.db_path else None
        }


# Global RAG instance
_rag_instance: Optional[BlenderRAG] = None
_rag_loader_thread: Optional[threading.Thread] = None


def get_rag_instance() -> BlenderRAG:
    """Get or create the global RAG instance without blocking initialize.

    Use ensure_rag_loaded_async() to start background initialization.
    """
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = BlenderRAG()
    return _rag_instance


def ensure_rag_loaded_async() -> None:
    """Start loading the RAG database in a background thread if not loaded.

    Safe to call from UI draw code; returns immediately.
    """
    global _rag_loader_thread
    rag = get_rag_instance()
    # Already loaded or loading?
    if getattr(rag, "enabled", False):
        return
    if getattr(rag, "loading", False):
        # If we lost the thread reference but still loading, do nothing
        if _rag_loader_thread and _rag_loader_thread.is_alive():
            return
    # Start background loader
    def _worker():
        try:
            rag.loading = True
            rag.initialize()
        finally:
            rag.loading = False
    _rag_loader_thread = threading.Thread(target=_worker, daemon=True)
    _rag_loader_thread.start()


def is_rag_enabled() -> bool:
    """Check if RAG is enabled and ready (non-blocking)."""
    rag = get_rag_instance()
    return bool(getattr(rag, 'enabled', False))


def augment_with_rag(message: str, enabled: bool = True) -> str:
    """
    Augment a message with RAG if enabled.

    Args:
        message: The user's message
        enabled: Whether to use RAG (from preferences)

    Returns:
        Original or augmented message
    """
    if not enabled:
        return message

    rag = get_rag_instance()
    return rag.augment_prompt(message)

