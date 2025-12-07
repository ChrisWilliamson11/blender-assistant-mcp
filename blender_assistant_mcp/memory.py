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

    # --- Memory Abstracts ---
    
    def get_abstracts_path(self) -> str:
        import bpy
        config_dir = bpy.utils.user_resource("CONFIG")
        return os.path.join(config_dir, "blender_assistant_abstracts.json")

    def get_abstracts(self) -> List[str]:
        """Load all session abstracts."""
        path = self.get_abstracts_path()
        if not os.path.exists(path):
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def store_abstract(self, abstract: str):
        """Store a new abstract."""
        abstracts = self.get_abstracts()
        abstracts.append(abstract)
        path = self.get_abstracts_path()
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(abstracts, f, indent=2)
        except Exception as e:
            print(f"Failed to save abstract: {e}")

    def create_abstract(self, session_history: List[Dict[str, str]], model_name: str = "gpt-oss:20b") -> Optional[str]:
        """Generate an abstract from session history using LLM."""
        from . import ollama_adapter
        
        if not session_history:
            return None
            
        # Format history for summarization
        history_text = ""
        for msg in session_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n"
            
        prompt = f"""Summarize the following conversation into a concise abstract (1-2 sentences).
Focus on key decisions, learnings, or user preferences.
Ignore trivial chit-chat.

Conversation:
{history_text}

Abstract:"""

        response = ollama_adapter.chat_completion(
            model_path=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=128
        )
        
        abstract = response.get("content", "").strip()
        if abstract:
            self.store_abstract(abstract)
            return abstract
        return None
