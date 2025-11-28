"""Memory manager for Blender Assistant.

Handles loading, saving, and retrieving persistent memory (preferences, facts, etc.)
from a JSON file in the extension directory.
"""

import json
import os
from typing import Dict, List, Any, Optional

class MemoryManager:
    def __init__(self, memory_file: str = "blender_assistant_vectors.json"):
        import bpy
        
        # Use Blender's user config directory for persistence
        config_dir = bpy.utils.user_resource("CONFIG")
        self.file_path = os.path.join(config_dir, memory_file)
        
        # Initialize Vector Memory
        from pathlib import Path
        from .vector_memory import VectorMemory
        self.vector_store = VectorMemory(Path(self.file_path))

    def remember_preference(self, key: str, value: Any):
        """Store a user preference."""
        self.vector_store.set_preference(key, value)

    def remember_fact(self, fact: str, category: str = "general"):
        """Store a general fact."""
        self.vector_store.add(fact, {"category": category, "type": "fact"})

    def remember_learning(self, topic: str, insight: str):
        """Store a technical learning, pitfall, or version quirk."""
        content = f"[{topic}] {insight}"
        self.vector_store.add(content, {"category": "learning", "type": "learning", "topic": topic})

    def search_memory(self, query: str, limit: int = 5) -> List[str]:
        """Semantic search over memory."""
        results = self.vector_store.search(query, k=limit)
        return [doc["text"] for doc, score in results]

    def get_context(self) -> str:
        """Get a formatted string of relevant memory for the system prompt."""
        lines = []
        
        prefs = self.vector_store.preferences
        if prefs:
            lines.append("User Preferences:")
            for k, v in prefs.items():
                lines.append(f"- {k}: {v}")
        
        # Show recent learnings/facts from vector store to keep it fresh
        # We can just take the last 5 documents
        recent_docs = self.vector_store.documents[-5:] if self.vector_store.documents else []
        
        if recent_docs:
            lines.append("Recent Learnings & Facts:")
            for doc in recent_docs:
                lines.append(f"- {doc['text']}")
                
        return "\n".join(lines)

    def clear(self):
        """Clear all memory."""
        # Reset vector store
        self.vector_store.documents = []
        self.vector_store.preferences = {}
        self.vector_store.vectors = None
        self.vector_store.save()
